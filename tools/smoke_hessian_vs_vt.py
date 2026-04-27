"""Option C smoke test: compare v_t (Adam second moment EMA) vs H_diag (Hessian diagonal
via Hutchinson) as importance-score components, on a toy MoE-like model.

Rationale: CLAUDE.md score = v̂_t × δ². Proposal to swap v̂_t → H_ii. Question: how
different are the two in practice, and how much does it change Top-K selection?

This smoke does NOT touch Megatron/DeepSpeed. It builds a small nn.Module with a
mix of "expert"-labeled params and non-expert params, trains with AdamW for N
iters, then at iter=N dumps:

  - v_t (from optimizer.state["exp_avg_sq"])
  - H_diag (Hutchinson with K samples on a fresh batch)
  - cosine / Spearman / Pearson correlation between the two (per-expert-param)
  - Jaccard overlap of Top-0.1% and Top-1% selections
  - per-expert summary to stdout, JSONL to log file
  - histograms + scatter plots to <out>/plots/

Usage:
    python3 tools/smoke_hessian_vs_vt.py \
        --out /home/asdf1234/Megatron-DeepSpeed/results/smoke_hessian \
        --iters 200 --hidden 128 --experts 4 --seq 32 --batch 32 --k-samples 4
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import numpy as np
    _HAS_NP = True
except ImportError:
    _HAS_NP = False


# --------------------------------------------------------------------------- #
# Toy MoE-ish model: token-level routing over K experts per MoE layer.
# --------------------------------------------------------------------------- #
class ToyExpertFFN(nn.Module):
    def __init__(self, hidden: int, ffn: int):
        super().__init__()
        self.w1 = nn.Linear(hidden, ffn, bias=True)
        self.w2 = nn.Linear(ffn, hidden, bias=True)

    def forward(self, x):
        return self.w2(F.gelu(self.w1(x)))


class ToyMoELayer(nn.Module):
    """Softmax-routed MoE layer. Each token picks Top-1 expert (hard) in smoke."""

    def __init__(self, hidden: int, ffn: int, n_experts: int):
        super().__init__()
        self.gate = nn.Linear(hidden, n_experts, bias=False)
        self.experts = nn.ModuleList(
            [ToyExpertFFN(hidden, ffn) for _ in range(n_experts)]
        )
        for exp in self.experts:
            for p in exp.parameters():
                # Tag so our detector classifies them as "expert" params.
                p.is_expert = True

    def forward(self, x):
        # x: [B, S, H]
        logits = self.gate(x)                       # [B, S, N]
        probs = F.softmax(logits, dim=-1)
        idx = probs.argmax(dim=-1)                  # [B, S]
        out = torch.zeros_like(x)
        for e, exp in enumerate(self.experts):
            mask = (idx == e).unsqueeze(-1).to(x.dtype)  # [B, S, 1]
            if mask.sum() == 0:
                continue
            out = out + mask * exp(x)
        return out


class ToyBlock(nn.Module):
    def __init__(self, hidden: int, n_experts: int, is_moe: bool):
        super().__init__()
        self.ln = nn.LayerNorm(hidden)
        self.attn = nn.Linear(hidden, hidden, bias=True)   # trivial attention surrogate
        if is_moe:
            self.mlp = ToyMoELayer(hidden, hidden * 4, n_experts)
        else:
            self.mlp = ToyExpertFFN(hidden, hidden * 4)    # dense MLP (non-expert)

    def forward(self, x):
        return x + self.mlp(self.ln(x + self.attn(x)))


class ToyModel(nn.Module):
    """4-block transformer stand-in with interleaved MoE at blocks 1, 3."""

    def __init__(self, vocab: int, hidden: int, n_experts: int, n_layers: int = 4):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.blocks = nn.ModuleList([
            ToyBlock(hidden, n_experts, is_moe=(i % 2 == 1))
            for i in range(n_layers)
        ])
        self.out = nn.Linear(hidden, vocab, bias=False)

    def forward(self, x):
        h = self.embed(x)
        for blk in self.blocks:
            h = blk(h)
        return self.out(h)


# --------------------------------------------------------------------------- #
# Hutchinson Hessian diagonal estimator.
# --------------------------------------------------------------------------- #
def hutchinson_h_diag(
    loss: torch.Tensor,
    params: List[torch.Tensor],
    k_samples: int,
    generator: torch.Generator,
) -> Dict[int, torch.Tensor]:
    """Estimate diag(H) for each param via Hutchinson's trick:

        H_ii ≈ E_z[ z_i · (Hz)_i ]

    where z is Rademacher. K samples are averaged.

    Returns dict keyed by id(param) → fp32 CPU tensor of H_diag estimates.
    """
    h_diag = {id(p): torch.zeros_like(p, dtype=torch.float32) for p in params}

    # First backward with create_graph=True to enable second-order autograd.
    grads = torch.autograd.grad(
        loss, params, create_graph=True, retain_graph=True,
    )

    for _ in range(k_samples):
        zs = [
            (torch.randint(0, 2, p.shape, generator=generator,
                           device=p.device, dtype=torch.float32) * 2 - 1)
            for p in params
        ]
        # dot = Σ g·z (scalar). Its gradient w.r.t. params is Hz.
        dot = sum((g * z).sum() for g, z in zip(grads, zs))
        hz_list = torch.autograd.grad(
            dot, params, retain_graph=True,
        )
        for p, z, hz in zip(params, zs, hz_list):
            h_diag[id(p)] += (z * hz).to(torch.float32) / k_samples

    return h_diag


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().to(torch.float64)
    b = b.flatten().to(torch.float64)
    a = a - a.mean()
    b = b - b.mean()
    num = (a * b).sum().item()
    den = (a.norm().item() * b.norm().item())
    return num / den if den > 0 else 0.0


def spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    if not _HAS_NP:
        return float("nan")
    a = a.flatten().cpu().numpy()
    b = b.flatten().cpu().numpy()
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def topk_jaccard(a: torch.Tensor, b: torch.Tensor, k_ratio: float) -> float:
    n = a.numel()
    k = max(1, int(k_ratio * n))
    a_top = torch.topk(a.flatten(), k, largest=True).indices
    b_top = torch.topk(b.flatten(), k, largest=True).indices
    inter = len(set(a_top.tolist()) & set(b_top.tolist()))
    return inter / (2 * k - inter) if (2 * k - inter) > 0 else 0.0


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--experts", type=int, default=4)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--seq", type=int, default=32)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--vocab", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--beta2", type=float, default=0.999)
    ap.add_argument("--k-samples", type=int, default=4,
                    help="Hutchinson samples (more → lower variance, K× cost)")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--snapshot-every", type=int, default=0,
                    help="If >0, also run the Hessian/v_t comparison every N iters "
                         "during training and dump JSONL history.")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "plots").mkdir(exist_ok=True)
    log_f = open(out / "smoke.jsonl", "w")

    torch.manual_seed(args.seed)
    gen = torch.Generator(device=args.device).manual_seed(args.seed)

    model = ToyModel(args.vocab, args.hidden, args.experts, args.layers).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            betas=(0.9, args.beta2), eps=1e-8, weight_decay=0.0)

    expert_params = [p for p in model.parameters() if getattr(p, "is_expert", False)]
    total_exp = sum(p.numel() for p in expert_params)
    print(f"[smoke] model: {sum(p.numel() for p in model.parameters())/1e6:.2f} M params | "
          f"expert: {total_exp/1e6:.2f} M across {len(expert_params)} tensors | "
          f"device={args.device} | K={args.k_samples}")

    def new_batch():
        return torch.randint(0, args.vocab, (args.batch, args.seq), device=args.device)

    # ----- train -----
    t0 = time.perf_counter()
    losses = []
    snapshot_records = []
    for it in range(1, args.iters + 1):
        x = new_batch()
        y = torch.randint(0, args.vocab, (args.batch, args.seq), device=args.device)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, args.vocab), y.view(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())

        if args.snapshot_every > 0 and it % args.snapshot_every == 0 and it < args.iters:
            snap = run_comparison(
                model, expert_params, opt, args, gen,
                label=f"iter{it}",
            )
            snap["iter"] = it
            snap["train_loss"] = loss.item()
            snapshot_records.append(snap)
            log_f.write(json.dumps(snap) + "\n"); log_f.flush()

        if it % max(1, args.iters // 10) == 0 or it == 1:
            print(f"[smoke] iter {it:4d}/{args.iters} loss={loss.item():.4f}")

    train_time = time.perf_counter() - t0

    # ----- final comparison -----
    print(f"\n[smoke] Training done in {train_time:.1f} s. "
          f"Running v_t vs H_diag comparison (K={args.k_samples} samples)...")
    rec = run_comparison(model, expert_params, opt, args, gen, label="final")
    rec["iter"] = args.iters
    rec["train_loss"] = losses[-1]
    log_f.write(json.dumps(rec) + "\n")
    log_f.close()

    # ----- summary table -----
    print_table(rec)

    # ----- plots -----
    try:
        plot_results(rec, snapshot_records, out / "plots")
    except Exception as e:
        print(f"[smoke] plot failed: {e}")

    print(f"[smoke] Done. Output: {out}")


def run_comparison(
    model: nn.Module,
    expert_params: List[torch.Tensor],
    opt: torch.optim.Optimizer,
    args: argparse.Namespace,
    gen: torch.Generator,
    label: str,
) -> Dict:
    """At the current model state, compute v_t (from Adam) and H_diag (Hutchinson)
    for each expert param. Return a dict with per-param metrics + aggregate stats."""
    t_start = time.perf_counter()

    # v_t from optimizer.state, bias-corrected
    step_now = None
    for g in opt.param_groups:
        for p in g["params"]:
            st = opt.state.get(p, {})
            s = st.get("step")
            if s is not None:
                step_now = int(s.item()) if torch.is_tensor(s) else int(s)
                break
        if step_now is not None:
            break
    if step_now is None:
        step_now = args.iters
    bias_corr2 = max(1 - args.beta2 ** step_now, 1e-12)

    v_hat: Dict[int, torch.Tensor] = {}
    for p in expert_params:
        st = opt.state.get(p, {})
        v = st.get("exp_avg_sq")
        if v is None:
            v_hat[id(p)] = torch.zeros_like(p, dtype=torch.float32)
        else:
            v_hat[id(p)] = (v / bias_corr2).to(torch.float32).detach()

    # H_diag via Hutchinson on a fresh batch
    x = torch.randint(0, args.vocab, (args.batch, args.seq), device=args.device)
    y = torch.randint(0, args.vocab, (args.batch, args.seq), device=args.device)
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, args.vocab), y.view(-1))
    t_h0 = time.perf_counter()
    h_diag = hutchinson_h_diag(loss, expert_params, args.k_samples, gen)
    t_h1 = time.perf_counter()

    # per-param metrics + concat for global stats
    per_param = []
    all_v: List[torch.Tensor] = []
    all_h: List[torch.Tensor] = []
    for name, p in model.named_parameters():
        if not getattr(p, "is_expert", False):
            continue
        v = v_hat[id(p)].flatten()
        h = h_diag[id(p)].flatten()
        p_abs = v.abs()   # v̂_t is always ≥ 0
        h_abs = h.abs()   # H may be signed; for importance magnitude matters
        row = {
            "name": name,
            "numel": int(p.numel()),
            "v_mean": float(v.mean().item()),
            "v_max": float(v.max().item()),
            "h_mean": float(h.mean().item()),
            "h_max": float(h.max().item()),
            "h_neg_frac": float((h < 0).float().mean().item()),
            "pearson_v_h": pearson(v, h),
            "pearson_v_habs": pearson(v, h_abs),
            "spearman_v_h": spearman(v, h),
            "spearman_v_habs": spearman(v, h_abs),
            "topk_0.1pct_jaccard_habs": topk_jaccard(v, h_abs, 0.001),
            "topk_1pct_jaccard_habs": topk_jaccard(v, h_abs, 0.01),
            "topk_10pct_jaccard_habs": topk_jaccard(v, h_abs, 0.1),
        }
        per_param.append(row)
        all_v.append(v.cpu())
        all_h.append(h_abs.cpu())

    cat_v = torch.cat(all_v)
    cat_h = torch.cat(all_h)
    agg = {
        "pearson_v_habs_all": pearson(cat_v, cat_h),
        "spearman_v_habs_all": spearman(cat_v, cat_h),
        "topk_0.1pct_jaccard_all": topk_jaccard(cat_v, cat_h, 0.001),
        "topk_1pct_jaccard_all": topk_jaccard(cat_v, cat_h, 0.01),
        "topk_10pct_jaccard_all": topk_jaccard(cat_v, cat_h, 0.1),
        "total_expert_numel": int(cat_v.numel()),
        "hutchinson_samples": args.k_samples,
        "hutchinson_seconds": t_h1 - t_h0,
        "comparison_seconds": time.perf_counter() - t_start,
        "step": step_now,
        "bias_corr2": bias_corr2,
        "label": label,
    }
    rec = {"aggregate": agg, "per_param": per_param}

    # also stash the actual flat arrays for plotting (small toy model, OK)
    rec["_cat_v"] = cat_v.tolist()
    rec["_cat_h"] = cat_h.tolist()
    return rec


def print_table(rec: Dict) -> None:
    agg = rec["aggregate"]
    print("\n" + "=" * 90)
    print(f"SCORE v_t vs |H_diag| — SUMMARY  (K={agg['hutchinson_samples']} Hutchinson samples, "
          f"step={agg['step']})")
    print("=" * 90)
    print(f"Hutchinson compute time: {agg['hutchinson_seconds']*1000:.1f} ms")
    print(f"Aggregate over {agg['total_expert_numel']:,} expert elements:")
    print(f"  Pearson(v̂_t, |H|)         = {agg['pearson_v_habs_all']:+.4f}")
    print(f"  Spearman(v̂_t, |H|)        = {agg['spearman_v_habs_all']:+.4f}")
    print(f"  Jaccard top 0.1%           = {agg['topk_0.1pct_jaccard_all']:.4f}")
    print(f"  Jaccard top 1.0%           = {agg['topk_1pct_jaccard_all']:.4f}")
    print(f"  Jaccard top 10.0%          = {agg['topk_10pct_jaccard_all']:.4f}")
    print("\nPer-param (short view):")
    print(f"  {'name':<60s} {'Pearson':>9s} {'Spearman':>9s} {'J0.1%':>7s} {'J1%':>6s}  {'h<0%':>6s}")
    for r in rec["per_param"]:
        print(f"  {r['name']:<60s} {r['pearson_v_habs']:+9.4f} {r['spearman_v_habs']:+9.4f} "
              f"{r['topk_0.1pct_jaccard_habs']:>7.3f} {r['topk_1pct_jaccard_habs']:>6.3f}  "
              f"{r['h_neg_frac']*100:>5.1f}%")
    print("=" * 90)
    print(f"Interpretation: Pearson/Spearman near 1 → v̂_t agrees with |H|.")
    print(f"                Jaccard near 1 → Top-K selection equivalent.")
    print(f"                'h<0%' = fraction of H_diag elements that are negative (saddle/ridge).")


def plot_results(rec: Dict, history: List[Dict], out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    v = rec["_cat_v"]; h = rec["_cat_h"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    ax = axes[0]
    ax.scatter(v, h, s=2, alpha=0.3)
    ax.set_xscale("symlog", linthresh=1e-10)
    ax.set_yscale("symlog", linthresh=1e-10)
    ax.set_xlabel("v̂_t (Adam)")
    ax.set_ylabel("|H_diag| (Hutchinson)")
    ax.set_title(f"Element-wise scatter "
                 f"(Pearson={rec['aggregate']['pearson_v_habs_all']:+.3f})")
    ax.grid(alpha=0.3, which="both")

    ax = axes[1]
    for arr, lbl in [(v, "v̂_t"), (h, "|H_diag|")]:
        pos = [x for x in arr if x > 0]
        if pos:
            ax.hist(pos, bins=60, alpha=0.5, label=lbl, log=True)
    ax.set_xscale("log")
    ax.set_xlabel("value")
    ax.set_ylabel("count")
    ax.set_title("Distribution")
    ax.legend(); ax.grid(alpha=0.3, which="both")

    ax = axes[2]
    if history:
        iters = [s["iter"] for s in history] + [rec["iter"]]
        per = [s["aggregate"]["pearson_v_habs_all"] for s in history] + \
              [rec["aggregate"]["pearson_v_habs_all"]]
        j1 = [s["aggregate"]["topk_1pct_jaccard_all"] for s in history] + \
             [rec["aggregate"]["topk_1pct_jaccard_all"]]
        ax.plot(iters, per, marker="o", label="Pearson")
        ax.plot(iters, j1, marker="s", label="Jaccard@1%")
        ax.set_xlabel("iter"); ax.set_ylabel("similarity")
        ax.set_title("Agreement over training")
        ax.legend(); ax.grid(alpha=0.3)
    else:
        ax.text(0.5, 0.5, "(no --snapshot-every history)",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Agreement over training")

    plt.suptitle("Option C smoke: v_t vs |H_diag| as importance score")
    plt.tight_layout()
    plt.savefig(out_dir / "v_vs_h_summary.png", dpi=120, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
