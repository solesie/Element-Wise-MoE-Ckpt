"""Per-segment Pass B equivalence test.

Goal: a single Pass B call over the full range vs. multiple calls over
disjoint sub-ranges produce the SAME final AOS buffer + dg update +
slot counts (modulo atomic-ordering of records within a name).
"""
import torch
import sys
sys.path.insert(0, '/home/asdf1234/Megatron-DeepSpeed')
from megatron.proposal.outlier_kernels import (
    _outlier_count_kernel, _aos_write_kernel,
    launch_aos_write_for_subrange,
    PREFIX_BYTES, RECORD_BYTES, SECTOR_BYTES,
    _aos_multisets_equal,
)


def _round_up(n, m):
    return ((n + m - 1) // m) * m


def setup_problem(N_NAMES, NAME_SIZE, G=8, N_Q=1024):
    torch.manual_seed(2025)
    N_TOTAL = N_NAMES * NAME_SIZE
    N_GROUPS = N_TOTAL // G
    dev = torch.device("cuda")
    m_flat = torch.randn(N_TOTAL, device=dev, dtype=torch.float32) * 0.01
    v_flat = torch.rand(N_TOTAL, device=dev, dtype=torch.float32) * 0.001 + 1e-6
    p_flat = torch.randn(N_TOTAL, device=dev, dtype=torch.float32) * 0.1
    # Build monotone v_q / h_q quantile tables (FP32, N_Q each).
    # Test uses random monotone arrays — kernel just does bsearch + lerp.
    v_q = torch.linspace(0.0, 1e-3, N_Q, device=dev, dtype=torch.float32)
    h_q = torch.linspace(0.5, 1.0, N_Q, device=dev, dtype=torch.float32)
    dg_init = torch.rand(N_GROUPS, device=dev, dtype=torch.float32) * 0.001
    ofs_elem = torch.tensor(
        [i * NAME_SIZE for i in range(N_NAMES + 1)],
        device=dev, dtype=torch.int64,
    )
    return dev, m_flat, v_flat, p_flat, v_q, h_q, dg_init, ofs_elem, N_TOTAL, N_GROUPS


def run_full_aos_write(m_flat, v_flat, p_flat, v_q, h_q, dg_init, ofs_elem,
                    N_TOTAL, N_GROUPS, name_offset_bytes, name_t_off,
                    name_count, prev_th, group_th, dev, g_blk):
    """Reference: one Pass B call over the whole range."""
    aos_words = max(1, sum(
        _round_up(PREFIX_BYTES + n * RECORD_BYTES, SECTOR_BYTES)
        for n in name_count
    ) // 4)
    aos_buf = torch.zeros(aos_words, dtype=torch.int32, device=dev)
    name_slot = torch.zeros(len(name_count), device=dev, dtype=torch.int64)
    dg_after = dg_init.clone()
    prev_th_t = torch.tensor(prev_th, device=dev, dtype=torch.float32)
    group_th_t = torch.tensor(group_th, device=dev, dtype=torch.float32)
    N_Q = int(v_q.numel())
    LOG_N_Q = N_Q.bit_length() - 1

    # Full Pass B
    GPB = 8
    n_progs = (N_GROUPS + GPB - 1) // GPB
    _aos_write_kernel[(n_progs,)](
        m_flat, v_flat, p_flat, dg_after,
        v_q, h_q,
        ofs_elem, name_offset_bytes, name_t_off, name_slot,
        aos_buf,
        prev_th_t, group_th_t,
        N_GROUPS, len(name_count),
        0,                                       # master_off_elem
        1e-3, 1.0, 1e-8,
        G=g_blk, GROUPS_PER_BLOCK=GPB,
        N_Q=N_Q, LOG_N_Q=LOG_N_Q,
    )
    return aos_buf, name_slot, dg_after


def run_split_aos_write(m_flat, v_flat, p_flat, v_q, h_q, dg_init, ofs_elem,
                     N_TOTAL, N_GROUPS, name_offset_bytes, name_t_off,
                     name_count, prev_th, group_th, dev, g_blk,
                     n_splits):
    """Split the name range into n_splits chunks. Each chunk = a 'segment'.
    Run Pass B per chunk. The aos_buf is SHARED (single buffer like single
    segment); slot counter is shared too. Verify result matches full Pass B.
    """
    aos_words = max(1, sum(
        _round_up(PREFIX_BYTES + n * RECORD_BYTES, SECTOR_BYTES)
        for n in name_count
    ) // 4)
    aos_buf = torch.zeros(aos_words, dtype=torch.int32, device=dev)
    name_slot = torch.zeros(len(name_count), device=dev, dtype=torch.int64)
    dg_after = dg_init.clone()
    prev_th_t = torch.tensor(prev_th, device=dev, dtype=torch.float32)
    group_th_t = torch.tensor(group_th, device=dev, dtype=torch.float32)
    N_Q = int(v_q.numel())

    n_names = len(name_count)
    chunk = max(1, n_names // n_splits)
    name_lo = 0
    while name_lo < n_names:
        name_hi = min(name_lo + chunk, n_names)
        # Element range
        elem_lo = int(ofs_elem[name_lo].item())
        elem_hi = int(ofs_elem[name_hi].item())
        # Group range
        grp_lo = elem_lo // g_blk
        grp_hi = elem_hi // g_blk
        # Sliced views
        m_view = m_flat.narrow(0, elem_lo, elem_hi - elem_lo)
        v_view = v_flat.narrow(0, elem_lo, elem_hi - elem_lo)
        p_view = p_flat.narrow(0, elem_lo, elem_hi - elem_lo)
        accum_view = dg_after.narrow(0, grp_lo, grp_hi - grp_lo)

        launch_aos_write_for_subrange(
            m_view=m_view, v_view=v_view, p_view=p_view,
            accum_view=accum_view,
            v_q=v_q, h_q=h_q,
            ofs_elem_full=ofs_elem,
            name_offset_bytes=name_offset_bytes, name_t_off=name_t_off,
            name_slot=name_slot,
            aos_out_i32=aos_buf,
            prev_th_t=prev_th_t, group_th_t=group_th_t,
            master_off_elem=elem_lo,
            g_blk=g_blk,
            lr_over_bc1=1e-3, sqrt_bc2_inv=1.0, eps=1e-8,
            n_names=n_names,
            n_q=N_Q,
        )
        name_lo = name_hi

    return aos_buf, name_slot, dg_after


def test_one(label, N_NAMES, NAME_SIZE, G, n_splits, prev_th=1e-7, group_th=1e-5):
    dev, m_flat, v_flat, p_flat, v_q, h_q, dg_init, ofs_elem, N_TOTAL, N_GROUPS = setup_problem(
        N_NAMES, NAME_SIZE, G,
    )
    # First run Pass A to get name_count.
    name_count = torch.zeros(N_NAMES, dtype=torch.int64, device=dev)
    prev_th_t = torch.tensor(prev_th, device=dev, dtype=torch.float32)
    floor_t = torch.tensor(1e-8, device=dev, dtype=torch.float32)
    group_th_t = torch.tensor(group_th, device=dev, dtype=torch.float32)
    GPB = 8
    n_progs = (N_GROUPS + GPB - 1) // GPB
    N_Q = int(v_q.numel())
    LOG_N_Q = N_Q.bit_length() - 1
    _outlier_count_kernel[(n_progs,)](
        m_flat, v_flat, dg_init.clone(),
        v_q, h_q,
        name_count, ofs_elem,
        prev_th_t, floor_t, group_th_t,
        N_GROUPS, N_NAMES,
        0, 1e-3, 1.0, 1e-8,
        name_count, 0, 0,  # dummy sample (stride=0 → skip)
        G=G, GROUPS_PER_BLOCK=GPB,
        N_Q=N_Q, LOG_N_Q=LOG_N_Q,
    )

    name_count_host = name_count.cpu().tolist()

    # Build AOS layout from name_count.
    name_byte_off_list = []
    cursor = 0
    for nr in name_count_host:
        name_byte_off_list.append(cursor + PREFIX_BYTES)
        cursor += _round_up(PREFIX_BYTES + nr * RECORD_BYTES, SECTOR_BYTES)
    name_offset_bytes = torch.tensor(name_byte_off_list, device=dev, dtype=torch.int64)
    name_t_off = torch.zeros(N_NAMES, device=dev, dtype=torch.int64)

    # Reference: full Pass B
    aos_full, slot_full, dg_full = run_full_aos_write(
        m_flat, v_flat, p_flat, v_q, h_q, dg_init, ofs_elem,
        N_TOTAL, N_GROUPS, name_offset_bytes, name_t_off,
        name_count_host, prev_th, group_th, dev, G,
    )

    # Split
    aos_split, slot_split, dg_split = run_split_aos_write(
        m_flat, v_flat, p_flat, v_q, h_q, dg_init, ofs_elem,
        N_TOTAL, N_GROUPS, name_offset_bytes, name_t_off,
        name_count_host, prev_th, group_th, dev, G,
        n_splits=n_splits,
    )

    # Compare
    slot_match = torch.equal(slot_full, slot_split)
    dg_match = torch.allclose(dg_full, dg_split, atol=1e-6, rtol=1e-5)
    aos_match = _aos_multisets_equal(
        aos_full, aos_split, name_byte_off_list, name_count_host,
    )
    n_sel = sum(name_count_host)
    ok = slot_match and dg_match and aos_match
    status = "PASS" if ok else "FAIL"
    print(f"[{label}] N_NAMES={N_NAMES} NAME_SIZE={NAME_SIZE} G={G} splits={n_splits} "
          f"n_sel={n_sel}  slot={slot_match} dg={dg_match} aos={aos_match}  {status}")
    return ok


results = []
results.append(test_one("split_1",     N_NAMES=8,  NAME_SIZE=256,  G=8, n_splits=1))
results.append(test_one("split_2",     N_NAMES=8,  NAME_SIZE=256,  G=8, n_splits=2))
results.append(test_one("split_4",     N_NAMES=8,  NAME_SIZE=256,  G=8, n_splits=4))
results.append(test_one("split_8_uneven", N_NAMES=12, NAME_SIZE=512, G=8, n_splits=5))
results.append(test_one("medium_split_4", N_NAMES=64, NAME_SIZE=1024, G=8, n_splits=4))
results.append(test_one("large_split_8",  N_NAMES=128, NAME_SIZE=4096, G=8, n_splits=8))
results.append(test_one("all_zero",   N_NAMES=8,  NAME_SIZE=256,  G=8, n_splits=2, prev_th=1.0))

print()
print(f"{sum(results)}/{len(results)} PASS")
