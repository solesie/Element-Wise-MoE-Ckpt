"""Standalone validation: how close is K=16 FD to a high-K Hutch "truth"?

Question this answers
---------------------
Production cross-check of K=16 FD vs K=4 Hutch saw bare Pearson ~0.45 — no
improvement over the inflated K=4/K=4 baseline. We can't tell from that
alone whether (a) K=16 FD is genuinely far from the true |H_diag|, or
(b) K=4 Hutch itself is too noisy to be a useful reference.

Toy model lets us afford K=256 Hutch (~64× the production K=4) as a
near-truth reference and ask:

    Pearson(K=16 FD, K=256 Hutch)  vs  Pearson(K=4 Hutch, K=256 Hutch)

If FD wins, K=16 FD is closer to truth than K=4 Hutch — and the production
~0.45 plateau is a Hutch-noise ceiling, not an FD bias.
If Hutch wins, FD's truncation/operator-scope bias dominates regardless
of K and we should stop pushing K up.

Setup
-----
A small dense MLP plays the role of an expert FFN (no MoE routing, no
gate freeze needed because there's no gate). Output is a CE loss against
random labels. Same fixed batch (CRN) across all estimators within a
trial. Multiple trials with different model/data seeds for noise bars.

Hutchinson is implemented inline (not via the production
``compute_hessian_hutchinson``) because the production path requires the
Megatron forward_step_func + DeepSpeed engine wrappers. The math is
identical: grad(loss, p, create_graph=True) once, then K times sample
Rademacher z, take grad(g·z, p), accumulate |z·Hz|/K.

Run
---
    python tests/standalone/fd_vs_hutch_truth.py
"""
from __future__ import annotations

import sys
import os
import math
from typing import List, Tuple, Dict

import torch

# Ensure the repo root is on the path before the proposal import.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from megatron.proposal.hessian import compute_hessian_finite_diff_core


# ---------------------------------------------------------------------------
# Toy "expert" model
# ---------------------------------------------------------------------------
class ToyExpert(torch.nn.Module):
    """Small dense network that mimics an expert FFN's structure: an
    embedding → two FFN linears (with GELU) → output projection. All
    params are FP32 to avoid BF16 rounding interference in the FD step.
    """
    def __init__(self, vocab: int = 64, hidden: int = 64, ffn: int = 256,
                 n_blocks: int = 1):
        super().__init__()
        self.emb = torch.nn.Embedding(vocab, hidden)
        self.blocks = torch.nn.ModuleList([
            torch.nn.ModuleDict({
                "ln": torch.nn.LayerNorm(hidden),
                "fc1": torch.nn.Linear(hidden, ffn, bias=False),
                "fc2": torch.nn.Linear(ffn, hidden, bias=False),
            }) for _ in range(n_blocks)
        ])
        self.out = torch.nn.Linear(hidden, vocab, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: (B, T) long
        h = self.emb(tokens)                       # (B, T, H)
        for blk in self.blocks:
            r = blk["ln"](h)
            r = torch.nn.functional.gelu(blk["fc1"](r))  # (B, T, F)
            r = blk["fc2"](r)                            # (B, T, H)
            h = h + r
        logits = self.out(h)                       # (B, T, V)
        return logits


def make_batch(batch: int, seq: int, vocab: int, gen: torch.Generator,
               device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    tokens = torch.randint(0, vocab, (batch, seq), generator=gen, device=device)
    labels = torch.randint(0, vocab, (batch, seq), generator=gen, device=device)
    return tokens, labels


def make_forward_loss_fn(model: torch.nn.Module, tokens: torch.Tensor,
                          labels: torch.Tensor):
    """Closure used by both FD and inline-Hutch. Same batch each call (CRN)."""
    def _fn():
        logits = model(tokens)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
        )
        return loss, {}
    return _fn


# ---------------------------------------------------------------------------
# Inline Hutchinson (FP32, same Rademacher RNG seed as FD for fair comparison)
# ---------------------------------------------------------------------------
def hutchinson_diag(forward_loss_fn, params: List[torch.nn.Parameter],
                    k_samples: int, rng_seed: int = 0xBADCAFE
                    ) -> List[torch.Tensor]:
    loss, _ = forward_loss_fn()
    if loss.dim() > 0:
        loss = loss.mean()
    grads = torch.autograd.grad(
        loss, params, create_graph=True, retain_graph=True, allow_unused=True,
    )
    grads = [g if g is not None else torch.zeros_like(p) for g, p in zip(grads, params)]
    gen = torch.Generator(device=params[0].device).manual_seed(rng_seed)
    h_accum = [torch.zeros_like(p, dtype=torch.float32) for p in params]
    inv_k = 1.0 / max(1, int(k_samples))
    for k in range(int(k_samples)):
        zs = [
            (torch.randint(0, 2, p.shape, generator=gen, device=p.device,
                           dtype=torch.float32) * 2 - 1)
            for p in params
        ]
        dot = sum((g * z).sum() for g, z in zip(grads, zs))
        hz = torch.autograd.grad(
            dot, params, retain_graph=(k < k_samples - 1), allow_unused=True,
        )
        hz = [h if h is not None else torch.zeros_like(p) for h, p in zip(hz, params)]
        for i, (z, h) in enumerate(zip(zs, hz)):
            h_accum[i].add_((z * h).to(torch.float32).abs_().mul_(inv_k))
        del zs, hz, dot
    return h_accum


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def pearson_two_lists(a: List[torch.Tensor], b: List[torch.Tensor]) -> float:
    af = torch.cat([t.detach().to(torch.float64).flatten() for t in a])
    bf = torch.cat([t.detach().to(torch.float64).flatten() for t in b])
    am = af - af.mean()
    bm = bf - bf.mean()
    denom = am.norm() * bm.norm()
    if denom.item() <= 0:
        return 0.0
    return float((am * bm).sum().item() / denom.item())


def jaccard_topk(a: List[torch.Tensor], b: List[torch.Tensor], pct: float) -> float:
    af = torch.cat([t.detach().to(torch.float64).flatten() for t in a])
    bf = torch.cat([t.detach().to(torch.float64).flatten() for t in b])
    n = af.numel()
    k = max(1, int(round(n * pct)))
    a_idx = set(af.topk(k).indices.tolist())
    b_idx = set(bf.topk(k).indices.tolist())
    inter = len(a_idx & b_idx)
    union = len(a_idx | b_idx)
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run_one_trial(trial_seed: int, device: torch.device, eps: float,
                   k_truth: int, k_hutch_low: int, k_fd_list: List[int],
                   vocab: int, hidden: int, ffn: int, n_blocks: int,
                   batch: int, seq: int, warmup_steps: int = 30,
                   ) -> Dict[str, Dict[str, float]]:
    torch.manual_seed(trial_seed)
    g = torch.Generator(device=device).manual_seed(trial_seed)
    model = ToyExpert(vocab=vocab, hidden=hidden, ffn=ffn,
                       n_blocks=n_blocks).to(device).to(torch.float32)
    n_param = sum(p.numel() for p in model.parameters())
    # Run a few train steps first so params aren't at init.
    opt = torch.optim.SGD(model.parameters(), lr=1e-2)
    for _ in range(warmup_steps):
        tk, lb = make_batch(batch, seq, vocab, g, device)
        opt.zero_grad()
        logits = model(tk)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, vocab), lb.reshape(-1))
        loss.backward()
        opt.step()
    # Frozen probe batch (shared across all estimators in this trial).
    tk, lb = make_batch(batch, seq, vocab, g, device)
    fwd = make_forward_loss_fn(model, tk, lb)

    params = list(model.parameters())

    # K=256 Hutch as near-truth reference.
    h_truth = hutchinson_diag(fwd, params, k_samples=k_truth, rng_seed=0xC0FFEE)

    # K=4 Hutch — what production cross-check uses.
    h_hutch_low = hutchinson_diag(fwd, params, k_samples=k_hutch_low,
                                   rng_seed=0xBADCAFE)

    out: Dict[str, Dict[str, float]] = {}
    out["hutch_low_vs_truth"] = {
        "pearson": pearson_two_lists(h_hutch_low, h_truth),
        "jacc1pct": jaccard_topk(h_hutch_low, h_truth, 0.01),
        "jacc10pct": jaccard_topk(h_hutch_low, h_truth, 0.10),
    }

    # K=8/16/32 FD vs truth.
    for k_fd in k_fd_list:
        h_fd = compute_hessian_finite_diff_core(
            forward_loss_fn=fwd,
            expert_ps=params,
            k_samples=k_fd,
            eps=eps,
            gates_for_freeze=None,
            rng_seed=0xFD15CAFE,
        )
        if h_fd is None:
            out[f"fd_k{k_fd}_vs_truth"] = {"pearson": float("nan"),
                                            "jacc1pct": float("nan"),
                                            "jacc10pct": float("nan")}
            continue
        out[f"fd_k{k_fd}_vs_truth"] = {
            "pearson": pearson_two_lists(h_fd, h_truth),
            "jacc1pct": jaccard_topk(h_fd, h_truth, 0.01),
            "jacc10pct": jaccard_topk(h_fd, h_truth, 0.10),
        }
    return out


def _run_size(label: str, model_cfg: Dict, device: torch.device, eps: float,
              k_truth: int, k_hutch_low: int, k_fd_list: List[int],
              n_trials: int):
    agg: Dict[str, Dict[str, List[float]]] = {}
    n_param_seen = None
    for trial in range(n_trials):
        seed = 1000 + trial
        print(f"# [{label}] trial {trial} (seed={seed}) ...", flush=True)
        out = run_one_trial(seed, device, eps,
                             k_truth=k_truth, k_hutch_low=k_hutch_low,
                             k_fd_list=k_fd_list, **model_cfg)
        for k, v in out.items():
            slot = agg.setdefault(k, {})
            for m, x in v.items():
                slot.setdefault(m, []).append(x)
    n_param = (
        model_cfg["vocab"] * model_cfg["hidden"]   # emb
        + model_cfg["n_blocks"] * (
            2 * model_cfg["hidden"]                 # LN gain+bias
            + model_cfg["hidden"] * model_cfg["ffn"]  # fc1
            + model_cfg["ffn"] * model_cfg["hidden"]  # fc2
        )
        + model_cfg["hidden"] * model_cfg["vocab"]    # out
    )
    print()
    print(f"# === {label} ===")
    print(f"# Model: vocab={model_cfg['vocab']} hidden={model_cfg['hidden']} "
          f"ffn={model_cfg['ffn']} n_blocks={model_cfg['n_blocks']} "
          f"batch={model_cfg['batch']} seq={model_cfg['seq']}  "
          f"(~{n_param/1e3:.1f}K params)")
    print(f"# Truth: K={k_truth} Hutch (CRN-aligned via shared batch)")
    print(f"# Trials: {n_trials}, eps={eps}, FP32 toy model")
    print(f"# Layout: mean ± std of metric over {n_trials} trials")
    print()
    header = f"{'estimator':<22} {'pearson':>14} {'jacc@1%':>14} {'jacc@10%':>14}"
    print(header)
    print("-" * len(header))
    rows = sorted(agg.keys())
    rows.sort(key=lambda r: (0 if r.startswith("hutch_low") else 1, r))
    for k in rows:
        slot = agg[k]
        line = f"{k:<22}"
        for m in ("pearson", "jacc1pct", "jacc10pct"):
            xs = [x for x in slot[m] if not math.isnan(x)]
            if not xs:
                line += f" {'nan':>14}"
            else:
                mean = sum(xs) / len(xs)
                var = sum((x - mean) ** 2 for x in xs) / max(1, len(xs) - 1)
                std = math.sqrt(max(0.0, var))
                line += f" {mean:.3f}±{std:.3f}".rjust(15)
        print(line)
    print()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"# device={device}", flush=True)
    eps = 1e-3  # FP32 model — finite-diff stable in [1e-4, 1e-2]
    k_truth = 256
    k_hutch_low = 4
    k_fd_list = [4, 8, 16, 32]
    n_trials = 5

    small = dict(vocab=64, hidden=64, ffn=256, n_blocks=1, batch=8, seq=32)
    medium = dict(vocab=512, hidden=256, ffn=1024, n_blocks=2, batch=8, seq=64)

    _run_size("small (~52K params)", small, device, eps,
              k_truth, k_hutch_low, k_fd_list, n_trials)
    _run_size("medium (~1M params)", medium, device, eps,
              k_truth, k_hutch_low, k_fd_list, n_trials)


if __name__ == "__main__":
    main()
