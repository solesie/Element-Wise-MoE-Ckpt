"""Phase 3 — compare both implementations against a full FP32 reference."""
import torch
import sys
sys.path.insert(0, '/home/asdf1234/Megatron-DeepSpeed')
from megatron.proposal.hessian import _h_accum_streaming_for_tensor


def reference_f32(gp, gm, z, h_accum, g_blk, inv_step, inv_k):
    """Pure FP32 reference (no BF16 intermediates) — accumulator stays
    FP32 throughout. The 'true' value modulo accumulation order."""
    gp_f = gp.to(torch.float32)
    gm_f = gm.to(torch.float32)
    z_f = z.to(torch.float32)
    contrib = ((z_f * (gp_f - gm_f) * inv_step).abs_() * inv_k)
    n_aligned = (contrib.numel() // g_blk) * g_blk
    if n_aligned > 0:
        # accumulator is FP32 here
        h_accum.add_(contrib[:n_aligned].reshape(-1, g_blk).mean(dim=1))


def legacy_chain(gp, gm, z, h_accum, g_blk, inv_step, inv_k):
    """Production chain — has bf16 intermediates."""
    hz = (gp - gm).mul(inv_step).to(torch.float32)
    contrib = (z.to(torch.float32) * hz).abs_().mul_(inv_k)
    n_aligned = (contrib.numel() // g_blk) * g_blk
    if n_aligned > 0:
        h_accum.add_(
            contrib[:n_aligned].reshape(-1, g_blk).mean(dim=1).to(torch.bfloat16)
        )


def stream_kernel(gp, gm, z, h_accum, g_blk, inv_step, inv_k):
    _h_accum_streaming_for_tensor(
        gp=gp, gm=gm, z=z, h_accum=h_accum,
        g_blk=g_blk, inv_step=inv_step, inv_k=inv_k,
    )


def test(label, N, g_blk, k=4):
    torch.manual_seed(42 + sum(ord(c) for c in label))
    dev = torch.device("cuda")
    n_groups = N // g_blk
    N = n_groups * g_blk
    inv_step = 1.0 / (2 * 1e-3)
    inv_k = 1.0 / k

    h_ref = torch.zeros(n_groups, dtype=torch.float32, device=dev)
    h_legacy = torch.zeros(n_groups, dtype=torch.bfloat16, device=dev)
    h_stream = torch.zeros(n_groups, dtype=torch.bfloat16, device=dev)

    for kk in range(k):
        gp = torch.randn(N, device=dev, dtype=torch.bfloat16) * 0.01
        gm = torch.randn(N, device=dev, dtype=torch.bfloat16) * 0.01
        z = (torch.randint(0, 2, (N,), device=dev, dtype=torch.float32) * 2 - 1)
        reference_f32(gp, gm, z, h_ref, g_blk, inv_step, inv_k)
        legacy_chain(gp, gm, z, h_legacy, g_blk, inv_step, inv_k)
        stream_kernel(gp, gm, z, h_stream, g_blk, inv_step, inv_k)

    h_legacy_f = h_legacy.to(torch.float32)
    h_stream_f = h_stream.to(torch.float32)

    legacy_diff = (h_legacy_f - h_ref).abs().max().item()
    stream_diff = (h_stream_f - h_ref).abs().max().item()
    ref_max = h_ref.abs().max().item()

    legacy_rel = legacy_diff / max(ref_max, 1e-12)
    stream_rel = stream_diff / max(ref_max, 1e-12)

    print(f"[{label}] N={N} g_blk={g_blk} K={k}  "
          f"legacy rel_err={legacy_rel:.3e}  "
          f"stream rel_err={stream_rel:.3e}  "
          f"stream_<= legacy? {stream_rel <= legacy_rel + 1e-9}")
    # Streaming should match or be better than legacy.
    return stream_rel <= max(legacy_rel * 1.5, 1e-2)


results = []
results.append(test("small_g1",      1024,    1, k=1))
results.append(test("small_g8",      1024,    8, k=1))
results.append(test("medium_g8",     65536,   8, k=1))
results.append(test("large_g8",      1048576, 8, k=1))
results.append(test("large_g64",     1048576, 64, k=1))
results.append(test("large_g256",    1048576, 256, k=1))
results.append(test("multi_k4_g8",   65536,   8, k=4))
results.append(test("multi_k16_g8",  65536,   8, k=16))

print()
print(f"{sum(results)}/{len(results)} PASS")
