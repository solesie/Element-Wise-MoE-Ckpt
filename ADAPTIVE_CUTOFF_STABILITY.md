# Adaptive Cutoff Stability

## Final Path Contract

Production cutoff is:

```
AdaptiveCutoffController tau ladder
-> GPU-generated block-stratified approximate probe
-> GPU between-block variance/CI tau decision
-> device-guarded exact candidate fallback
-> direct emission cursor feedback
```

There is no host sampling table, host tau decision, pre-fold validation, per-tname
selection count, or pre-materialized selection mask in the production path.

## Correctness Vs Quality

- `capacity-HARD`: direct emission's per-expert cursor capacity cap. This is
  the only hard overflow/corruption guarantee.
- `K-band`: soft tau quality target. Approx CI and exact fallback improve tau;
  they are not overflow correctness sources.
- If tau quality is poor, direct emission may hit cursor overflow. The kernel
  must skip the overflowing write, fold the update into `dg`, increment overflow
  diagnostics, and let the next iteration's controller/probe recover from the
  realized cursor count.

## Probe Math

The probe samples contiguous entry-blocks by deterministic arithmetic from
`(range_id, sample_id)`. No sampled block table is stored or copied per iter.

For candidate tau and stratum `h`:

```
q_j       = selected_in_sampled_block_j / elements_in_block_j
mean_h    = avg(q_j)
s_h^2     = sample variance of q_j across blocks
R_hat          = sum_h W_h * mean_h
Var(Rhat)      = sum_h W_h^2 * s_h^2 / n_h
strict_cert    iff |R_hat-rho| + 1.96*sqrt(Var) <= rho*tolerance
quality_accept iff a finite positive bracket/best tau exists
```

This uses between-block variance, so cluster correlation and expert/layer
concentration are measured rather than assumed independent.

`strict_cert` controls diagnostics. It does not force a production exact full scan.
Because correctness is
capacity-hard rather than strict-hard, a finite approximate tau is allowed to
flow into direct emission; realized cursor count and overflow feedback correct
the next iteration. The guarded exact fallback is reserved for genuinely
clueless probe output: no finite tau, no positive estimated survival, NaN, or
otherwise invalid device decision.

The production sample controller starts small: roughly one sampled block per
active expert stratum, with a bounded cap for invalid-probe recovery. It is not
sized from a fixed K% magic constant. Wide CI remains a quality diagnostic, but
it does not enlarge the hot-path sample count. Sample growth is driven by
repeated mismatch between direct-emission actual cursor count and the probe's
expected selected count.

## Refresh Boundary

H refresh is treated as a non-scalar score remap. Before replacing `h_q`, the
previous direct-emission feedback is consumed so the old score basis is not
mixed into the new one. After the refresh, learned tau-scale rows are dropped,
and the first normal cutoff step after refresh uses the cold-start cutoff
bootstrap to establish a fresh tau scale. The H refresh body itself does not run
the approximate probe, direct-emission layout, or AOS materialization.

The measured `kappa_h` bridge estimate is recorded for telemetry and future
predictor analysis only. It is not applied to production tau rows: applying a
single scalar bridge to H refresh was observed to create severe over-selection
on the refresh iteration.

## Device-Only Decision

The decision kernel writes `total_th` and `needs_exact` on device. The guarded
exact fallback is always enqueued; if `needs_exact == 0`, it no-ops. Production
sets `needs_exact == 1` only when the probe cannot produce any valid tau, not
when the CI is merely wide. Production must not call `.cpu()`, `.item()`,
`.tolist()`, or stream synchronize to choose tau.

## Bit Identity

Probe, guarded exact fallback, direct emission, and controller feedback use the
same score definition:

```
s_elem(i) = H(v_i) * delta_i^2 * element_score_priority
s_grp(g)  = mean(H(v_i in g)) * (dg_g + sum(delta_i^2 in g)) / G
s'(i)     = max(s_elem(i), s_grp(g(i)))
```

Tests may compare this predicate against direct cursor counts, but production
does not materialize selection masks.
