"""50M-scale standalone validation of K=16 FD vs K=4 Hutch using real
SlimPajama tokens. Mirrors the production cross-check pattern
(200 train iters, refresh every 40 iters, 5 measurement points), but
adds a K=N high-K Hutch reference as "truth" to anchor the comparison.

Why
---
The 1.3M-param toy verified that K=16 FD is closer to a K=256 Hutch truth
than K=4 Hutch is — but only on synthetic batches. This script lifts the
test to ~50M params on real SlimPajama tokens and a real training
trajectory, the same conditions production cross-check runs under.

What stays the same vs the toy script
-------------------------------------
* CRN: shared probe batch across all estimators in a measurement
* Inline Hutchinson (FP32, Rademacher z, identical seed convention)
* Pearson + Jaccard@1%/@10%

What's different
----------------
* Real SlimPajama .bin/.idx (mmap-loaded, no Megatron init required)
* GPT-2 vocab (50257) → real embedding/output projections
* 200-iter Adam training on real data between measurements
* K_truth chosen by env var ``K_TRUTH`` (default 64); raise to 128/256
  if memory permits

Run
---
    K_TRUTH=64  python3 tests/standalone/fd_vs_hutch_truth_50M.py
    K_TRUTH=128 python3 tests/standalone/fd_vs_hutch_truth_50M.py
"""
from __future__ import annotations

import os
import sys
import struct
import math
import time
from typing import Callable, List, Tuple, Dict, Optional

import numpy as np
import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from megatron.proposal.hessian import compute_hessian_finite_diff_core


# ---------------------------------------------------------------------------
# Minimal SlimPajama .bin/.idx reader (no Megatron init dependency).
# Compatible with Megatron's MMapIndexedDataset format.
# ---------------------------------------------------------------------------
_DTYPES = {1: np.uint8, 2: np.int8, 3: np.int16, 4: np.int32,
           5: np.int64, 6: np.float64, 7: np.float32, 8: np.uint16}


class SlimDataset:
    """Read-only mmap of a Megatron-formatted indexed dataset."""

    def __init__(self, prefix: str):
        idx_path = prefix + ".idx"
        bin_path = prefix + ".bin"
        with open(idx_path, "rb") as f:
            assert f.read(9) == b"MMIDIDX\x00\x00", "bad magic"
            (version,) = struct.unpack("<Q", f.read(8))
            assert version == 1
            (dtype_code,) = struct.unpack("<B", f.read(1))
            self.dtype = _DTYPES[dtype_code]
            (self._n,) = struct.unpack("<Q", f.read(8))
            (self._n_doc,) = struct.unpack("<Q", f.read(8))
            offset = f.tell()
        idx_mm = np.memmap(idx_path, mode="r", order="C")
        self._sizes = np.frombuffer(idx_mm, dtype=np.int32, count=self._n,
                                     offset=offset)
        self._pointers = np.frombuffer(idx_mm, dtype=np.int64, count=self._n,
                                        offset=offset + self._sizes.nbytes)
        self._bin = np.memmap(bin_path, mode="r", order="C", dtype=np.uint8)

    def n_docs(self) -> int:
        return self._n

    def get(self, doc_idx: int) -> np.ndarray:
        ptr = int(self._pointers[doc_idx])
        size = int(self._sizes[doc_idx])
        return np.frombuffer(self._bin, dtype=self.dtype, count=size, offset=ptr)


def sample_batch(ds: SlimDataset, batch: int, seq: int, vocab_size: int,
                  rng: np.random.Generator, device: torch.device
                  ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pick `batch` random docs and crop a `seq+1` window from each. Wrap
    around if a doc is too short. Returns (tokens[B,T], labels[B,T]) where
    labels are tokens shifted by 1 (next-token prediction)."""
    n_docs = ds.n_docs()
    out = np.zeros((batch, seq + 1), dtype=np.int64)
    for b in range(batch):
        # Try a few docs until one is long enough; else concatenate two.
        for _ in range(8):
            di = int(rng.integers(0, n_docs))
            arr = ds.get(di)
            if arr.size >= seq + 1:
                start = int(rng.integers(0, arr.size - seq))
                out[b] = arr[start:start + seq + 1].astype(np.int64)
                break
        else:
            # Concatenate from random start across multiple docs.
            buf = []
            need = seq + 1
            while need > 0:
                di = int(rng.integers(0, n_docs))
                arr = ds.get(di)
                buf.append(arr)
                need -= arr.size
            cat = np.concatenate(buf).astype(np.int64)[:seq + 1]
            out[b] = cat
    out = np.clip(out, 0, vocab_size - 1)
    tk = torch.from_numpy(out[:, :-1]).to(device).long()
    lb = torch.from_numpy(out[:, 1:]).to(device).long()
    return tk, lb


# ---------------------------------------------------------------------------
# 50M-ish toy expert (no MoE, no routing). vocab=50257 (GPT-2 like).
# ---------------------------------------------------------------------------
class ToyLM(torch.nn.Module):
    def __init__(self, vocab: int = 50257, hidden: int = 384, ffn: int = 1536,
                 n_blocks: int = 6, seq: int = 512):
        super().__init__()
        self.emb = torch.nn.Embedding(vocab, hidden)
        self.pos = torch.nn.Embedding(seq, hidden)
        self.blocks = torch.nn.ModuleList([
            torch.nn.ModuleDict({
                "ln": torch.nn.LayerNorm(hidden),
                "fc1": torch.nn.Linear(hidden, ffn, bias=False),
                "fc2": torch.nn.Linear(ffn, hidden, bias=False),
            }) for _ in range(n_blocks)
        ])
        self.ln_f = torch.nn.LayerNorm(hidden)
        self.out = torch.nn.Linear(hidden, vocab, bias=False)
        self._seq = seq

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        B, T = tokens.shape
        pos = torch.arange(T, device=tokens.device)
        h = self.emb(tokens) + self.pos(pos)[None, :, :]
        for blk in self.blocks:
            r = blk["ln"](h)
            r = torch.nn.functional.gelu(blk["fc1"](r))
            r = blk["fc2"](r)
            h = h + r
        h = self.ln_f(h)
        logits = self.out(h)
        return logits


# ---------------------------------------------------------------------------
# Estimators (inline; same math as production)
# ---------------------------------------------------------------------------
def hutchinson_diag(forward_loss_fn: Callable[[], Tuple[torch.Tensor, dict]],
                    params: List[torch.nn.Parameter],
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
    del loss, grads
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
    return len(a_idx & b_idx) / max(1, len(a_idx | b_idx))


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    k_truth = int(os.environ.get("K_TRUTH", "64"))
    k_hutch_low = 4
    k_fd_list = [4, 16]  # production-relevant points
    train_iters = int(os.environ.get("TRAIN_ITERS", "200"))
    refresh_every = int(os.environ.get("REFRESH_EVERY", "40"))
    eps = float(os.environ.get("FD_EPS", "1e-3"))
    batch = int(os.environ.get("BATCH", "4"))
    seq = int(os.environ.get("SEQ", "512"))
    seed = int(os.environ.get("SEED", "1234"))

    print(f"# device={device}  K_TRUTH={k_truth}  train_iters={train_iters}  "
          f"refresh_every={refresh_every}  batch={batch}  seq={seq}  eps={eps}",
          flush=True)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    ds = SlimDataset("/home/asdf1234/data/slimpajama_6b/slimpajama_6b_train_text_document")
    print(f"# slimpajama: {ds.n_docs():,} docs, dtype={ds.dtype}", flush=True)
    rng = np.random.default_rng(seed)

    torch.manual_seed(seed)
    model = ToyLM(vocab=50257, hidden=384, ffn=1536, n_blocks=6,
                   seq=seq).to(device).to(torch.float32)
    n_param = sum(p.numel() for p in model.parameters())
    print(f"# model: {n_param:,} params  (~{n_param/1e6:.1f}M)", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95),
                             weight_decay=0.1)
    params = list(model.parameters())

    # Aggregator across measurement points.
    series: List[Dict[str, Dict[str, float]]] = []

    t_start = time.perf_counter()
    for step in range(1, train_iters + 1):
        tk, lb = sample_batch(ds, batch, seq, 50257, rng, device)
        opt.zero_grad(set_to_none=True)
        logits = model(tk)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), lb.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % refresh_every == 0:
            # Probe with a fresh sampled batch (production also uses a
            # captured batch from the current micro-step).
            ptk, plb = sample_batch(ds, batch, seq, 50257, rng, device)

            def _fwd():
                logits = model(ptk)
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), plb.reshape(-1))
                return loss, {}

            t0 = time.perf_counter()
            h_truth = hutchinson_diag(_fwd, params, k_samples=k_truth,
                                       rng_seed=0xC0FFEE)
            t_truth = time.perf_counter() - t0
            t0 = time.perf_counter()
            h_h4 = hutchinson_diag(_fwd, params, k_samples=k_hutch_low,
                                    rng_seed=0xBADCAFE)
            t_h4 = time.perf_counter() - t0

            row: Dict[str, Dict[str, float]] = {}
            row["hutch_low_vs_truth"] = {
                "pearson": pearson_two_lists(h_h4, h_truth),
                "jacc1pct": jaccard_topk(h_h4, h_truth, 0.01),
                "jacc10pct": jaccard_topk(h_h4, h_truth, 0.10),
            }
            for k_fd in k_fd_list:
                t0 = time.perf_counter()
                h_fd = compute_hessian_finite_diff_core(
                    forward_loss_fn=_fwd, expert_ps=params,
                    k_samples=k_fd, eps=eps, gates_for_freeze=None,
                    rng_seed=0xFD15CAFE,
                )
                tfd = time.perf_counter() - t0
                if h_fd is None:
                    row[f"fd_k{k_fd}_vs_truth"] = {"pearson": float("nan"),
                                                     "jacc1pct": float("nan"),
                                                     "jacc10pct": float("nan")}
                    continue
                row[f"fd_k{k_fd}_vs_truth"] = {
                    "pearson": pearson_two_lists(h_fd, h_truth),
                    "jacc1pct": jaccard_topk(h_fd, h_truth, 0.01),
                    "jacc10pct": jaccard_topk(h_fd, h_truth, 0.10),
                }
                row[f"fd_k{k_fd}_vs_truth"]["t_s"] = float(tfd)
            row["hutch_low_vs_truth"]["t_s"] = float(t_h4)
            row["__truth_t_s__"] = {"t_s": float(t_truth)}

            series.append({"step": step, **row})

            tot = time.perf_counter() - t_start
            line = (f"# step {step:3d} loss={loss.item():.3f} "
                    f"truth(K={k_truth})={t_truth:.1f}s "
                    f"h{k_hutch_low}={t_h4:.1f}s "
                    f"fd16={row.get('fd_k16_vs_truth', {}).get('t_s', float('nan')):.1f}s "
                    f"|  Hk{k_hutch_low}->truth P={row['hutch_low_vs_truth']['pearson']:.3f}  "
                    f"FDk16->truth P={row['fd_k16_vs_truth']['pearson']:.3f}  "
                    f"elapsed={tot:.0f}s")
            if torch.cuda.is_available():
                peak = torch.cuda.max_memory_allocated() / 1024 / 1024
                line += f"  peak={peak:.0f}MB"
            print(line, flush=True)

            del h_truth, h_h4

    # Aggregate
    keys = ["hutch_low_vs_truth", "fd_k4_vs_truth", "fd_k16_vs_truth"]
    metrics = ["pearson", "jacc1pct", "jacc10pct"]
    print()
    print(f"# === 50M scale: K_truth={k_truth}, train_iters={train_iters}, "
          f"{len(series)} measurement points ===")
    print(f"# Model: {n_param/1e6:.1f}M params  (vocab=50257 hidden=384 "
          f"ffn=1536 6 blocks)")
    print(f"# Data: SlimPajama, batch={batch} seq={seq}")
    print()
    header = f"{'estimator':<22} {'pearson':>14} {'jacc@1%':>14} {'jacc@10%':>14}"
    print(header)
    print("-" * len(header))
    for k in keys:
        line = f"{k:<22}"
        for m in metrics:
            xs = [r[k][m] for r in series if k in r and not math.isnan(r[k][m])]
            if not xs:
                line += f" {'nan':>14}"
            else:
                mean = sum(xs) / len(xs)
                var = sum((x - mean) ** 2 for x in xs) / max(1, len(xs) - 1)
                std = math.sqrt(max(0.0, var))
                line += f" {mean:.3f}±{std:.3f}".rjust(15)
        print(line)
    print()
    print("# Per-step trajectory (Pearson vs truth):")
    print(f"# {'step':>4}  {'Hk4':>6}  {'FDk4':>6}  {'FDk16':>6}")
    for r in series:
        print(f"  {r['step']:>4}  "
              f"{r['hutch_low_vs_truth']['pearson']:>6.3f}  "
              f"{r['fd_k4_vs_truth']['pearson']:>6.3f}  "
              f"{r['fd_k16_vs_truth']['pearson']:>6.3f}")
    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated() / 1024 / 1024
        print(f"\n# peak GPU memory: {peak:.0f} MB")


if __name__ == "__main__":
    main()
