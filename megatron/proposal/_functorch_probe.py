"""Probe torch.func compatibility with Megatron + DeepSpeed + MoE.

Run inside the Hutchinson refresh path when ``HUTCHINSON_FUNCTORCH_TEST=1``
is set. Executes three tiered checks and exits with a clear verdict:

  Step 0: Can ``torch.func.functional_call`` invoke the model?
  Step 1: Can we build a pure ``grad_fn`` (functional forward + autograd.grad)?
  Step 2: Does ``torch.func.jvp`` on that grad_fn run to completion?
  Step 3: Peak memory of the JVP path (if Step 2 passed).

Each step is wrapped in try/except — any failure prints the exact traceback
so we know which specific operation (MoE routing, Flash-Attn custom CUDA
kernel, fused LayerNorm, etc.) is incompatible.

The intent is to replace speculation with empirical data per the user's
critique: before claiming "won't work", measure it.
"""
from __future__ import annotations

import os
import traceback
from typing import Any, Dict, List

import torch


def _probe(
    model,
    forward_step_func,
    data_iterator,
    expert_ps: List[torch.nn.Parameter],
) -> None:
    """Run the three-step probe and print a summary table. Exits the process
    after completion so the pilot doesn't continue training."""
    di = data_iterator[0] if isinstance(data_iterator, (list, tuple)) else data_iterator
    m0 = model[0] if isinstance(model, (list, tuple)) else model

    print("=" * 70, flush=True)
    print("[functorch-probe] torch.func compatibility probe starting", flush=True)
    print(f"[functorch-probe] PyTorch {torch.__version__}", flush=True)
    print("=" * 70, flush=True)

    # Parameter inventory. For the probe we only attempt to vary EXPERT
    # params through torch.func; non-expert params are frozen (treated as
    # constants in the functional_call). This mirrors the Hutchinson setup
    # where only expert params need HVP.
    expert_ids = set(id(p) for p in expert_ps)
    all_named = dict(m0.named_parameters())
    expert_named: Dict[str, torch.nn.Parameter] = {
        n: p for n, p in all_named.items() if id(p) in expert_ids
    }
    non_expert_named: Dict[str, torch.nn.Parameter] = {
        n: p for n, p in all_named.items() if id(p) not in expert_ids
    }
    print(f"[functorch-probe] expert params: {len(expert_named)} tensors, "
          f"{sum(p.numel() for p in expert_named.values()):_} elems", flush=True)
    print(f"[functorch-probe] non-expert params: {len(non_expert_named)} tensors", flush=True)

    # -------- STEP 0: functional_call --------
    try:
        from torch.func import functional_call
    except ImportError as exc:
        print(f"[functorch-probe] STEP 0 FAIL — torch.func unavailable: {exc}", flush=True)
        return _bail()

    # We can't easily replay forward_step_func with a custom params dict
    # because it uses the Megatron ``model`` variable captured at definition
    # time. Instead we run forward the same way Hutchinson does, then patch
    # the ``parameters`` via functional_call. First verify baseline works.
    # Run the baseline inside no_grad so its activation graph does not
    # stay pinned through Step 1 (a fresh forward via functional_call
    # would otherwise OOM on mid-scale MoE configs).
    try:
        with torch.no_grad():
            loss_baseline, _ = forward_step_func(di, m0)
        print(f"[functorch-probe] baseline forward OK (no_grad), "
              f"loss={float(loss_baseline.mean().item()):.4f}", flush=True)
    except Exception:
        print("[functorch-probe] STEP 0 FAIL — baseline forward_step_func failed:", flush=True)
        traceback.print_exc()
        return _bail()
    # Explicit release before the real probes allocate their own graphs.
    del loss_baseline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Now try functional_call with the current parameter dict (should be
    # identical to the baseline). Use all params to verify the model works
    # under functional invocation at all.
    try:
        full_params = {n: p.detach() for n, p in all_named.items()}
        # functional_call replaces module parameters via parametrization.
        # Forward requires input args; we reuse Megatron's forward_step_func
        # by temporarily installing the params via functional_call and
        # invoking it. Trick: call forward_step_func inside a reparam ctx.
        with torch.no_grad():
            # Smoke test: raw model forward with a dummy tensor of the
            # expected shape. We cannot easily reconstruct the tokens
            # batch here, so we skip actual forward and just verify
            # functional_call accepts the param dict.
            _ = functional_call(m0, full_params, ())
            print("[functorch-probe] STEP 0 OK (trivial)", flush=True)
    except TypeError as te:
        # Expected for GPTModel: forward takes tokens, attention_mask, etc.
        # We care whether functional_call accepts the param dict shape at
        # all, not whether we pass right args. TypeError here is OK for our
        # purpose; real failure would be elsewhere.
        print(f"[functorch-probe] STEP 0 (trivial call TypeError, expected): {te}", flush=True)
    except Exception:
        print("[functorch-probe] STEP 0 FAIL — functional_call path broken:", flush=True)
        traceback.print_exc()
        return _bail()

    # -------- STEP 1: grad_fn via functional_call + autograd.grad --------
    # This is the meaningful test. We need a closure that takes a params
    # dict and returns a scalar loss so we can differentiate through it.
    # Megatron's forward_step_func captures the model via closure so we
    # wrap it with functional_call explicitly.
    try:
        # Build expert-only params dict (tangent space of Hutchinson).
        expert_params_dict = {n: p.detach().clone().requires_grad_(True)
                              for n, p in expert_named.items()}

        def loss_fn(ep_dict: Dict[str, torch.nn.Parameter]) -> torch.Tensor:
            # Combine expert + non-expert params for functional_call.
            combined = dict(non_expert_named)
            for n, p in ep_dict.items():
                combined[n] = p
            # Use functional_call to run the model forward with this param
            # set. Then let Megatron's forward_step_func compute the loss.
            # NB: forward_step_func internally calls model(*args), so we
            # have to monkey-patch model's parameters for the duration.
            # That's essentially what functional_call does, but for
            # Megatron's forward_step_func we can't easily intercept the
            # model call without rewriting. Fallback: call m0 directly.
            # We also need a batch. Borrow forward_step_func's iterator.
            batch = _borrow_batch(di)
            if batch is None:
                raise RuntimeError("could not fetch a batch for probe")
            # Inline the Megatron-style forward: tokens / position_ids / mask
            tokens, labels, loss_mask, attn_mask, position_ids = batch[:5]
            output = functional_call(
                m0, combined, (tokens, position_ids, attn_mask),
            )
            logits = output[0] if isinstance(output, tuple) else output
            import torch.nn.functional as F
            V = logits.size(-1) if logits.dim() >= 2 else None
            flat = logits.reshape(-1, V) if V is not None else logits
            lbl = labels.view(-1)
            if loss_mask is not None and loss_mask.numel() == flat.size(0):
                per_tok = F.cross_entropy(flat.float(), lbl, reduction="none")
                denom = loss_mask.view(-1).float().sum().clamp_min(1.0)
                return (per_tok * loss_mask.view(-1).float()).sum() / denom
            return F.cross_entropy(flat.float(), lbl, reduction="mean")

        def grad_fn(ep_dict: Dict[str, torch.nn.Parameter]) -> tuple:
            loss = loss_fn(ep_dict)
            grads = torch.autograd.grad(
                loss, tuple(ep_dict.values()), allow_unused=True,
            )
            return tuple(g if g is not None else torch.zeros_like(p)
                         for g, p in zip(grads, ep_dict.values()))

        grads = grad_fn(expert_params_dict)
        print(f"[functorch-probe] STEP 1 OK — grad_fn produced {len(grads)} grads", flush=True)
    except Exception:
        print("[functorch-probe] STEP 1 FAIL — grad_fn broke:", flush=True)
        traceback.print_exc()
        return _bail()

    # -------- STEP 2: torch.func.jvp on grad_fn --------
    try:
        from torch.func import jvp as _jvp

        # Random tangent of same shape as expert params.
        tangents = {n: torch.randn_like(p) for n, p in expert_params_dict.items()}

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        grads_out, hvp = _jvp(grad_fn, (expert_params_dict,), (tangents,))
        n_nonzero = sum(1 for h in hvp if h.abs().sum().item() > 0)
        print(f"[functorch-probe] STEP 2 OK — jvp returned {len(hvp)} HVP tensors "
              f"({n_nonzero} non-zero)", flush=True)
        if torch.cuda.is_available():
            peak = torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024
            print(f"[functorch-probe] STEP 3 JVP peak allocated: {peak:.2f} GB", flush=True)
    except Exception:
        print("[functorch-probe] STEP 2 FAIL — torch.func.jvp broke:", flush=True)
        traceback.print_exc()
        return _bail()

    print("=" * 70, flush=True)
    print("[functorch-probe] ALL STEPS PASSED — forward-over-reverse HVP viable", flush=True)
    print("=" * 70, flush=True)
    return _bail()


def _borrow_batch(data_iterator) -> tuple:
    """Fetch one batch the same way Megatron's forward_step would.

    This depends on pretrain_gpt.get_batch; we reuse it directly.
    """
    try:
        from pretrain_gpt import get_batch as _get_batch_gpt
    except Exception:
        return None
    try:
        return _get_batch_gpt(data_iterator)
    except Exception:
        return None


def _bail() -> None:
    """Flush and exit the process after the probe completes."""
    import sys
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


def is_probe_enabled() -> bool:
    return os.environ.get("HUTCHINSON_FUNCTORCH_TEST", "") in ("1", "true", "True")
