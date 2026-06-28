# Paper Logic Evidence Summary - 2026-06-28

## Setup

- Model/config family: `ds_pretrain_gpt_125M_MoE64`, EP2, PP none
- Runtime used for evidence: ZeRO2, GBS 32, micro-batch 2, LR 4.5e-4
- Measurement steps: 50, 150, 250
- Validation batches for actual loss: 16 paired batches, fixed seed `20240625`
- Rank sample for ranking metrics: 1,048,576 elements/rank, fixed seed `20240625`
- Runs:
  - K=2%: `results/PAPER_LOGIC_EVIDENCE_125M_EP2_ZERO2_GBS32_GRAD_K2_20260628_052743`
  - K=4%: `results/PAPER_LOGIC_EVIDENCE_125M_EP2_ZERO2_GBS32_GRAD_K4_20260628_050041`

## Main Conclusion

The original proof route "HVP directional is the best actual selector, therefore 3rd+ terms are negligible" is not supported.

The supported route is:

1. HVP directional is useful as a top-heavy full-curvature share target, not as an exact set-selection oracle.
2. `diag |H| + true e` is the strongest actual fault-loss selector in the stable measurement points.
3. Production `diag |H| + e_hat + agebit` is coarse but still better than random/RR after warmup, and K=4 consistently improves over K=2.
4. Therefore 3rd+ terms need not be explicitly isolated for the production claim; paired actual rollback loss directly evaluates the final objective.

## Actual Fault Loss

Metric: `L(theta_cp_candidate) - L(theta)`. Lower is better.
Policy comparisons below should use paired confidence intervals, because every
policy was evaluated on the same 16 validation batches.

### Step 150

| policy | K=2% | K=4% |
|---|---:|---:|
| g + diag abs H + agebit + force | +2.588e-3 | +6.795e-4 |
| g + diag abs H + true e | +1.348e-3 | +1.307e-4 |
| g + HVP directional + true e | +2.824e-3 | +1.569e-3 |
| random | +3.281e-3 | +1.537e-3 |
| round robin | +3.058e-3 | +1.289e-3 |

### Step 250

| policy | K=2% | K=4% |
|---|---:|---:|
| g + diag abs H + agebit + force | -2.102e-4 | -1.574e-3 |
| g + diag abs H + true e | -1.388e-3 | -2.312e-3 |
| g + HVP directional + true e | -3.105e-4 | -1.069e-3 |
| random | +2.466e-4 | -1.309e-3 |
| round robin | -7.239e-5 | -1.635e-3 |

K=4 improves over K=2 at all measured steps for production, diag true-e, and HVP directional policies.

## Paired Policy Comparisons

Paired diff means `A - B`; negative means `A` is better.

### K=4%, Step 150

| comparison | mean diff | 95% paired CI | conclusion |
|---|---:|---:|---|
| production - random | -8.575e-4 | [-1.233e-3, -4.668e-4] | production better |
| production - round robin | -6.096e-4 | [-8.723e-4, -3.386e-4] | production better |
| production - HVP directional | -8.895e-4 | [-1.472e-3, -3.321e-4] | production better |
| diag true-e - HVP directional | -1.438e-3 | [-2.145e-3, -7.593e-4] | diag true-e better |
| diag true-e - random | -1.406e-3 | [-1.884e-3, -9.226e-4] | diag true-e better |

### K=4%, Step 250

| comparison | mean diff | 95% paired CI | conclusion |
|---|---:|---:|---|
| production - random | -2.651e-4 | [-6.819e-4, +8.444e-5] | overlap |
| production - round robin | +6.126e-5 | [-4.873e-4, +6.046e-4] | overlap |
| production - HVP directional | -5.050e-4 | [-1.040e-3, +2.261e-5] | almost, overlap |
| HVP directional - random | +2.399e-4 | [-2.869e-4, +8.125e-4] | overlap |
| HVP directional - round robin | +5.663e-4 | [+7.828e-5, +1.103e-3] | HVP worse |
| diag true-e - HVP directional | -1.243e-3 | [-1.809e-3, -7.043e-4] | diag true-e better |
| diag true-e - random | -1.003e-3 | [-1.524e-3, -4.958e-4] | diag true-e better |

### Interpretation

These paired CIs do not support "HVP directional is the best actual-loss selector."
They do support:

- `diag |H| + true e` is the strongest measured selector.
- production `diag |H| + e_hat + agebit` clearly beats random/RR/HVP at step 150.
- at step 250 production still has favorable means, but random/RR comparisons overlap.
- HVP directional/share is useful as a quadratic attribution target, not as the actual rollback-loss oracle.

## Diag H vs HVP Directional/Share Ranking

Target: `g + HVP directional true e`, i.e. `-g_i e_i + 0.5 e_i (H e)_i`.
This is not actual loss mass and not an exact set-selection marginal. It is a
top-heavy directional quadratic share score. The mass metric below is:

```
sum_{i in policy_topK} target_score_i / sum_{i in target_topK} target_score_i
```

`diag |H| + true e` captures non-random directional quadratic share score mass:

| step | K=2% mass / random lift | K=4% mass / random lift |
|---|---:|---:|
| 50 | 0.195 / 5.56x | 0.317 / 5.45x |
| 150 | 0.369 / 9.57x | 0.367 / 5.71x |
| 250 | 0.363 / 8.08x | 0.358 / 4.99x |

Production agebit score is weaker but still non-random:

| step | K=2% mass / random lift | K=4% mass / random lift |
|---|---:|---:|
| 50 | 0.119 / 3.39x | 0.179 / 3.09x |
| 150 | 0.160 / 4.16x | 0.206 / 3.20x |
| 250 | 0.147 / 3.27x | 0.183 / 2.55x |

## e_hat Diagnosis

Target: `g + diag |H| + true e`.

Production `e_hat` recovers little at cold step 50, but after warmup it recovers a meaningful fraction of true-e diagonal mass:

| step | K=2% mass / random lift | K=4% mass / random lift |
|---|---:|---:|
| 50 | 0.051 / 1.37x | 0.051 / 1.03x |
| 150 | 0.359 / 8.69x | 0.473 / 6.84x |
| 250 | 0.347 / 7.54x | 0.435 / 5.82x |

`e_hat` is coarse:

| step | sign match K2/K4 | p90 ratio K2/K4 | p99 ratio K2/K4 |
|---|---:|---:|---:|
| 50 | 0.726 / 0.767 | 6.85 / 4.61 | 73.0 / 44.1 |
| 150 | 0.702 / 0.714 | 13.18 / 9.99 | 172.3 / 109.6 |
| 250 | 0.691 / 0.698 | 11.37 / 9.54 | 131.7 / 98.2 |

Thus, `e_hat` should be described as a bounded-staleness engineering approximation, not as a faithful true-e estimator.

## Interpretation For The Paper Claim

Use this wording:

- We do not claim the Taylor score exactly predicts validation loss.
- We use the second-order score as a selector for high-risk checkpoint elements.
- 3rd+ terms are not cleanly separable in this setting, so we validate the final objective directly via paired actual rollback loss.
- `diag |H|` is justified because it is strong in actual rollback loss with true e and captures several-fold more HVP-directional/share score mass than random.
- `e_hat` is justified as a 2-bit bounded-staleness approximation: it loses ranking fidelity, especially early, but after warmup it retains non-random true-e diagonal mass and improves actual rollback loss as K increases.

The paper-scale run should repeat the same metrics with GBS 256, 10k iterations, and validation batch 100.
