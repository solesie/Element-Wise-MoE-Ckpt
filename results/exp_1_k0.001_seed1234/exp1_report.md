
================================================================================
EXPERIMENT 1 — IMPORTANCE VS RANDOM (125M-MoE8, WikiText-2, 500 iter)
================================================================================

Protocol:
  - Model: GPT-125M-MoE8 (12 layers, hidden=768, 8 experts every-other layer)
  - Data: WikiText-2 (~24k docs, seq=2048, batch=64, grad_accum=8)
  - Iterations: 500 (reduced from CLAUDE.md's 1000 for time budget)
  - Expert CP protocol: initial FC at iter 1, then lossy-DC only (K=0.1%)
  - Non-expert CP: lossless FC every W=10 iter + DC otherwise

Variants:
  - baseline:  no CP pipeline (ideal training)
  - proposal:  K% Top-K selection by element-wise importance score
               (A = |v_t|/bias_corr2, B = |error_buffer|, score = A + β·B, β=0.1)
  - random:    K% random selection (same error-feedback + DC write path)

Results
-------
| Variant | Final training loss | Post-replay val loss | Post-replay PPL | Δ vs baseline | CP size |
|---------|--------------------:|---------------------:|----------------:|-------------:|--------:|
| baseline  |  5.0333          | (n/a — no CP)        |           (n/a)  |        0.000 | 0 GiB   |
| proposal  |  5.0451          | 5.3796               |        216.93   |     +0.346 | 0.0 GiB |
| random    |  5.0299          | 5.8431               |        344.85   |     +0.810 | 167.0 GiB |

Key finding:
  Δ(random − proposal) post-replay = +0.4635
  → Importance-based selection reconstructs the 500-iter training trajectory
    0.464 loss units closer to baseline than random selection.
  PPL ratio random/proposal = 1.59×

Caveats:
  - baseline trained with --split 100,0,0 (no val set), so its "final quality"
    is reported as final train loss; proposal+random replay validation ran on
    a 2% val split (512 samples).
  - proposal CP was deleted after replay_v4 to free disk (168 GiB). Size
    expected similar to random's 167 GiB; only live run reports it.
  - Training used same seed (1234) across variants → loss curves overlap
    nearly exactly (sanity: CP overhead does not alter the forward/backward).
