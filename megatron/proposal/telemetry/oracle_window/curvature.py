"""T0 curvature preparation for the measurement-only oracle window.

Expert and non-expert curvature can be built from one common FD-Hutchinson
pass over all local trainable low-precision parameters.  Only mapper-owned
FP32 fragments contribute to telemetry.
The distinction is important under DP/EP replication: every replica perturbs
the same router/non-expert LP state, but each optimizer fragment is counted
once after a world-level ownership audit.

This module has no checkpoint, fault, controller, or file-I/O behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Callable, Iterable, Mapping, Sequence

import torch

from ...refresh.fp32_state import (
    discard_zero12_grad_state,
    get_engine_optimizer,
    pinned_snapshot,
    restore_training_state,
    snapshot_training_state,
)
from ...refresh.hessian_refresh import (
    collect_parameter_fd_curvature_views,
    compute_parameter_hessian_finite_diff,
    validate_parameter_fd_curvature_views,
)
from ...refresh.quantile import QuantileRefs, make_quantile_refs
from .fragments import (
    DOMAIN_EXPERT,
    PackedFragmentLayout,
    Zero12FragmentMapper,
    build_nonexpert_layout,
    stable_u64,
)
from .lut import (
    DomainLut,
    DomainLutBuilder,
    DomainLutMetadata,
    FiniteDifferenceAudit,
)


_DS_MUTABLE_ATTRS = (
    "micro_steps",
    "global_steps",
    "loss_scale",
    "cur_iter",
    "cur_scale",
    "micro_step_id",
    "_is_gradient_accumulation_boundary",
    "is_gradient_accumulation_boundary",
)
_UINT32_SPACE = 1 << 32


@dataclass(frozen=True)
class NonExpertFdMetadata:
    pp_rank: int
    perturb_parameter_count: int
    perturb_element_count: int
    measurement_fragment_count: int
    measurement_element_count: int
    measurement_owner_rank: int
    measurement_owner_count: int
    measurement_ownership: str
    router_parameter_names: tuple[str, ...]
    router_element_count: int
    hutchinson_probes: int
    data_batches: int
    epsilon: float
    rng_seed: int
    routing_freeze: bool
    dedicated_data_iterator: bool
    perturb_identity_scheme: str
    perturb_parameter_fingerprint: str
    joint_all_parameters: bool
    expert_perturb_parameter_count: int
    expert_perturb_element_count: int
    nonexpert_perturb_parameter_count: int
    nonexpert_perturb_element_count: int
    fd: FiniteDifferenceAudit


@dataclass(frozen=True, eq=False)
class NonExpertCurvature:
    lut: DomainLut
    measurement_layout: PackedFragmentLayout
    metadata: NonExpertFdMetadata
    joint_expert_lut: DomainLut | None = None


@dataclass(frozen=True, eq=False)
class OracleWindowCurvature:
    expert_lut: DomainLut
    nonexpert: NonExpertCurvature


def _unwrap_local_module(model: Any) -> torch.nn.Module:
    root = model[0] if isinstance(model, (list, tuple)) else model
    module = getattr(root, "module", root)
    if not isinstance(module, torch.nn.Module):
        raise TypeError("oracle curvature model does not expose a local nn.Module")
    return module


def _parameter_fingerprint(
    named_parameters: Sequence[tuple[str, torch.nn.Parameter]],
    *,
    pp_rank: int,
) -> str:
    digest = hashlib.sha256()
    for name, parameter in named_parameters:
        digest.update(
            f"pp={int(pp_rank)}|{name}|{tuple(parameter.shape)}|"
            f"{parameter.dtype}|{int(parameter.numel())}\n".encode("utf-8")
        )
    return digest.hexdigest()


def collect_nonexpert_perturb_parameters(
    *,
    model: Any,
    expert_parameter_names: Iterable[str],
    pp_rank: int,
    router_parameter_names: Iterable[str] | None = None,
) -> tuple[
    tuple[tuple[str, torch.nn.Parameter], ...],
    tuple[str, ...],
    tuple[int, ...],
]:
    """Return canonical perturb ownership, router names, and PCG offsets.

    The result includes every trainable local parameter not classified as an
    expert.  It is intentionally broader than mapper measurement ownership.
    Sorting by ``(pp_rank, canonical name)`` gives every DP/EP replica the same
    PCG offsets without communication.
    """

    module = _unwrap_local_module(model)
    experts = {str(name) for name in expert_parameter_names}
    local = tuple(
        sorted(
            (
                (str(name), parameter)
                for name, parameter in module.named_parameters()
                if bool(parameter.requires_grad) and str(name) not in experts
            ),
            key=lambda item: item[0],
        )
    )
    if not local:
        raise RuntimeError("oracle curvature found no local non-expert parameters")
    names = [name for name, _parameter in local]
    if len(set(names)) != len(names) or len({id(p) for _n, p in local}) != len(local):
        raise RuntimeError("local non-expert perturb parameters are not unique")

    if router_parameter_names is None:
        from ...refresh.routing_freeze import identify_gate_modules_safe

        gate_modules, _rules = identify_gate_modules_safe(model)
        gate_param_ids = {
            id(parameter)
            for _gate_name, gate in gate_modules
            for parameter in gate.parameters(recurse=True)
        }
        routers = tuple(name for name, parameter in local if id(parameter) in gate_param_ids)
    else:
        requested = {str(name) for name in router_parameter_names}
        missing = sorted(requested - set(names))
        if missing:
            raise RuntimeError(
                "router parameters are absent from non-expert perturb ownership: "
                f"{missing[:8]}"
            )
        routers = tuple(name for name in names if name in requested)
    if not routers:
        raise RuntimeError(
            "non-expert FD must include local router parameters; none were identified"
        )

    offsets: list[int] = []
    cursor = 0
    for _name, parameter in local:
        offsets.append(cursor)
        cursor += int(parameter.numel())
    if cursor >= _UINT32_SPACE:
        raise RuntimeError(
            "canonical non-expert perturb population exceeds the current "
            "32-bit PCG element-index space"
        )
    # PP participates through the seed; sorted names determine offsets.  Thus
    # DP/EP replicas of one PP stage are identical, while different PP stages
    # cannot accidentally reuse the same z stream.
    del pp_rank
    return local, routers, tuple(offsets)


def select_nonexpert_measurement_parameters(
    *,
    mapper: Zero12FragmentMapper,
    perturb_parameters: Sequence[tuple[str, torch.nn.Parameter]],
    owner_rank: int,
    owner_count: int,
) -> tuple[tuple[tuple[str, torch.nn.Parameter], ...], str]:
    """Choose unique measurement owners without changing perturb ownership.

    ZeRO/HP partition fragments already provide element ownership.  A fully
    replicated mapper view instead uses the same deterministic size-balanced
    tensor-name sharding as non-expert checkpoint storage across DP x EP.
    """

    rank = int(owner_rank)
    world = int(owner_count)
    if world <= 0 or not 0 <= rank < world:
        raise ValueError("measurement owner rank/count are invalid")
    params = tuple(perturb_parameters)
    by_name = {name: parameter for name, parameter in params}
    raw = mapper.zero12_local_fragments(params)
    if not raw:
        raise RuntimeError("mapper exposes no local non-expert measurement fragments")
    partitioned = str(getattr(mapper, "mode", getattr(mapper, "_mode", ""))) == "zero12"
    for name, parameter, _group, _local, parameter_offset, numel in raw:
        if (
            int(parameter_offset) != 0
            or int(numel) != int(parameter.numel())
        ):
            partitioned = True
        if str(name) not in by_name:
            raise RuntimeError(f"mapper returned unexpected parameter {name!r}")
    if partitioned or world == 1:
        return params, "mapper_partition" if partitioned else "single_owner"

    shard_names: list[list[str]] = [[] for _ in range(world)]
    shard_loads = [0] * world
    weighted = sorted(
        ((int(parameter.numel()), name) for name, parameter in params),
        reverse=True,
    )
    for numel, name in weighted:
        shard = min(range(world), key=lambda index: (shard_loads[index], index))
        shard_names[shard].append(name)
        shard_loads[shard] += int(numel)
    selected = set(shard_names[rank])
    owned = tuple((name, parameter) for name, parameter in params if name in selected)
    if not owned:
        raise RuntimeError(
            f"non-expert measurement name shard {rank}/{world} is empty"
        )
    return owned, "replicated_name_shard"


def freeze_expert_lut(source: DomainLut) -> DomainLut:
    """Deep-copy the small expert quantile/LUT state, never alias production."""

    if source.domain != DOMAIN_EXPERT or source.metadata.domain != DOMAIN_EXPERT:
        raise ValueError("only an expert DomainLut can seed the expert oracle LUT")
    refs = make_quantile_refs(
        device=source.refs.v.values.device,
        n_q=int(source.refs.v.values.numel()),
        h_lookup_lut_bins=int(source.refs.h_lut.values.numel()),
    )
    refs.v.values.copy_(source.refs.v.values)
    refs.h.values.copy_(source.refs.h.values)
    refs.h_lut.values.copy_(source.refs.h_lut.values)
    refs.h_lut.log_v_min.copy_(source.refs.h_lut.log_v_min)
    refs.h_lut.inv_log_v_width.copy_(source.refs.h_lut.inv_log_v_width)
    return DomainLut(
        domain=DOMAIN_EXPERT,
        refs=refs,
        metadata=source.metadata,
    )


def freeze_expert_quantile_refs(
    *,
    source: QuantileRefs,
    metadata: DomainLutMetadata,
) -> DomainLut:
    """Freeze raw production refs when their provenance metadata is supplied."""

    return freeze_expert_lut(
        DomainLut(domain=DOMAIN_EXPERT, refs=source, metadata=metadata)
    )


def _snapshot_attrs(*objects: object | None) -> tuple[tuple[object, str, Any], ...]:
    records: list[tuple[object, str, Any]] = []
    seen: set[tuple[int, str]] = set()
    for obj in objects:
        if obj is None:
            continue
        for name in _DS_MUTABLE_ATTRS:
            key = (id(obj), name)
            if key in seen or not hasattr(obj, name):
                continue
            seen.add(key)
            try:
                records.append((obj, name, getattr(obj, name)))
            except Exception:
                continue
    return tuple(records)


def _restore_attrs(records: Sequence[tuple[object, str, Any]]) -> None:
    failures: list[str] = []
    for obj, name, value in reversed(tuple(records)):
        try:
            setattr(obj, name, value)
        except Exception as exc:
            failures.append(f"{type(obj).__name__}.{name}: {exc}")
    if failures:
        raise RuntimeError(
            "failed to restore DeepSpeed mutable attributes: " + "; ".join(failures[:8])
        )


def _collect_h_by_name(
    *,
    mapper: Any,
    engine_optimizer: Any,
    named_parameters: Sequence[tuple[str, torch.nn.Parameter]],
    allow_create: bool,
) -> dict[str, torch.Tensor]:
    names, views = collect_parameter_fd_curvature_views(
        mapper=mapper,
        engine_optimizer=engine_optimizer,
        named_parameters=list(named_parameters),
        allow_create=bool(allow_create),
    )
    if len(names) != len(views) or len(set(names)) != len(names):
        raise RuntimeError("non-expert FD curvature views have ambiguous identities")
    out = {str(name): view.detach().reshape(-1) for name, view in zip(names, views)}
    for name, view in out.items():
        if view.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise RuntimeError(
                f"non-expert FD view {name!r} has unsupported dtype {view.dtype}"
            )
        if not view.is_contiguous():
            raise RuntimeError(f"non-expert FD view {name!r} is not contiguous")
    return out


def _layout_h_views(
    layout: PackedFragmentLayout,
    h_by_name: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, ...]:
    out: list[torch.Tensor] = []
    for fragment in layout.fragments:
        source = h_by_name.get(fragment.parameter_name)
        if source is None:
            raise RuntimeError(
                "measurement-owned optimizer fragment has no FD curvature: "
                f"{fragment.parameter_name!r}"
            )
        if int(source.numel()) == int(fragment.parameter_numel):
            view = source.narrow(
                0, int(fragment.parameter_offset), int(fragment.numel),
            )
        elif int(source.numel()) == int(fragment.numel):
            view = source
        else:
            raise RuntimeError(
                "FD/optimizer fragment size mismatch for "
                f"{fragment.parameter_name!r}: h={source.numel()} "
                f"fragment={fragment.numel} parameter={fragment.parameter_numel}"
            )
        out.append(view)
    if sum(int(view.numel()) for view in out) != int(layout.total_numel):
        raise RuntimeError("non-expert FD coverage does not match measurement population")
    return tuple(out)


def _validate_finite_signal(views: Sequence[torch.Tensor]) -> None:
    validate_parameter_fd_curvature_views(list(views))


def measure_nonexpert_curvature(
    *,
    model: Any,
    forward_step_func: Callable,
    data_iterator: Any,
    mapper: Zero12FragmentMapper,
    inner_optimizer: Any,
    nonexpert_lut_builder: DomainLutBuilder,
    expert_parameter_names: Iterable[str],
    expert_measurement_layout: PackedFragmentLayout | None = None,
    expert_lut_builder: DomainLutBuilder | None = None,
    joint_all_parameters: bool = False,
    pp_rank: int,
    measurement_owner_rank: int = 0,
    measurement_owner_count: int = 1,
    router_parameter_names: Iterable[str] | None = None,
    namespace: str = "model",
    epsilon: float = 1.0e-3,
    hutchinson_probes: int = 4,
    data_batches: int = 1,
    rng_seed: int = 0x4E454644,
    minimum_changed_fraction: float = 0.95,
    process_group: object | None = None,
    dedicated_data_iterator: bool = False,
    fd_runner: Callable[..., Any] = compute_parameter_hessian_finite_diff,
    stage_consensus: Callable[[str, BaseException | None], None] | None = None,
) -> NonExpertCurvature:
    """Measure curvature and build domain-private ``v -> |H|`` LUTs.

    This is intentionally a T0 blocking operation.  It consumes
    ``data_batches`` items from a *dedicated* oracle iterator, never the
    training iterator.  In joint mode one shared all-parameter perturbation
    produces both expert and non-expert diagonal estimates with identical
    K/epsilon/data.  ``hutchinson_probes >= 4`` is mandatory because router
    parameters are perturbed and routing freeze is deliberately disabled.
    """

    probes = int(hutchinson_probes)
    batches = int(data_batches)
    if probes < 4:
        raise ValueError("router-inclusive non-expert FD requires K >= 4 probes")
    if batches <= 0:
        raise ValueError("non-expert FD data_batches must be positive")
    if not bool(dedicated_data_iterator):
        raise ValueError(
            "non-expert FD requires a dedicated data iterator; consuming the "
            "training iterator would change the training trajectory"
        )
    production_fd_runner = fd_runner is compute_parameter_hessian_finite_diff
    if production_fd_runner and not torch.cuda.is_available():
        raise RuntimeError("non-expert FD curvature requires CUDA")

    nonexpert_perturb, routers, _nonexpert_offsets = collect_nonexpert_perturb_parameters(
        model=model,
        expert_parameter_names=expert_parameter_names,
        pp_rank=int(pp_rank),
        router_parameter_names=router_parameter_names,
    )
    measurement_parameters, measurement_ownership = (
        select_nonexpert_measurement_parameters(
            mapper=mapper,
            perturb_parameters=nonexpert_perturb,
            owner_rank=int(measurement_owner_rank),
            owner_count=int(measurement_owner_count),
        )
    )
    measurement_layout = build_nonexpert_layout(
        mapper=mapper,
        named_params=measurement_parameters,
        pp_rank=int(pp_rank),
        namespace=str(namespace),
    )
    if measurement_layout.total_numel <= 0:
        raise RuntimeError("current rank owns no non-expert FP32 measurement fragment")

    if bool(joint_all_parameters) != (
        expert_measurement_layout is not None and expert_lut_builder is not None
    ):
        raise ValueError(
            "joint all-parameter FD requires both expert layout and LUT builder"
        )
    expert_perturb: tuple[tuple[str, torch.nn.Parameter], ...] = ()
    if joint_all_parameters:
        expert_names = {str(name) for name in expert_parameter_names}
        module_named = dict(_unwrap_local_module(model).named_parameters())
        missing = sorted(expert_names - set(module_named))
        if missing:
            raise RuntimeError(
                f"joint FD is missing expert parameters: {missing[:8]}"
            )
        expert_perturb = tuple(
            sorted(
                (
                    (name, module_named[name])
                    for name in expert_names
                    if bool(module_named[name].requires_grad)
                ),
                key=lambda item: item[0],
            )
        )
        if not expert_perturb:
            raise RuntimeError("joint FD found no trainable expert parameters")
        perturb = tuple(
            sorted(expert_perturb + nonexpert_perturb, key=lambda item: item[0])
        )
        offsets: list[int] = []
        cursor = 0
        for _name, parameter in perturb:
            offsets.append(cursor)
            cursor += int(parameter.numel())
        if cursor >= _UINT32_SPACE:
            raise RuntimeError("joint FD perturb population exceeds 32-bit PCG space")
        grad_offsets = tuple(offsets)
    else:
        perturb = nonexpert_perturb
        grad_offsets = _nonexpert_offsets

    parameters = [parameter for _name, parameter in perturb]
    if production_fd_runner and any(
        parameter.device.type != "cuda" for parameter in parameters
    ):
        raise RuntimeError("all non-expert LP perturb parameters must be CUDA tensors")
    dtypes = {parameter.dtype for parameter in parameters}
    if dtypes != {torch.bfloat16}:
        raise RuntimeError(
            "audited shared FD perturb currently requires BF16 LP parameters; "
            f"got {sorted(str(dtype) for dtype in dtypes)}"
        )

    root = model[0] if isinstance(model, (list, tuple)) else model
    module = _unwrap_local_module(model)
    engine_optimizer = get_engine_optimizer(model)
    train_snapshot = snapshot_training_state(inner_optimizer, model=model)
    attr_snapshot = _snapshot_attrs(root, engine_optimizer)
    was_training = bool(module.training)
    lp_snapshots = [
        pinned_snapshot(
            parameter.data.detach(),
            key=(
                "oracle_window_joint_theta" if joint_all_parameters else "oracle_window_ne_theta",
                str(namespace),
                int(pp_rank),
                name,
                tuple(parameter.shape),
                parameter.dtype,
            ),
        )
        for name, parameter in perturb
    ]
    device = parameters[0].device
    changed = torch.zeros(2, dtype=torch.int64, device=device)
    nonexpert_lut: DomainLut | None = None
    joint_expert_lut: DomainLut | None = None
    fd_audit: FiniteDifferenceAudit | None = None
    primary_error: BaseException | None = None

    pp_seed = int(rng_seed) ^ stable_u64(
        f"{str(namespace)}/pp={int(pp_rank)}/"
        f"{'joint-all-parameter-fd' if joint_all_parameters else 'nonexpert-fd'}"
    )
    pp_seed &= (1 << 64) - 1
    try:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        module.train(True)
        mapper_mode = str(getattr(mapper, "mode", getattr(mapper, "_mode", "")))
        if mapper_mode in ("zero12", "unfused_fp16"):
            discard_zero12_grad_state(engine_optimizer)
            grad_views: list[torch.Tensor] = []
        else:
            before = _collect_h_by_name(
                mapper=mapper,
                engine_optimizer=engine_optimizer,
                named_parameters=perturb,
                allow_create=False,
            )
            grad_views = list(before.values())
            for view in grad_views:
                view.zero_()
        for parameter in module.parameters():
            parameter.grad = None
            if hasattr(parameter, "grad_accum"):
                parameter.grad_accum = None

        total_backward_passes = batches * probes * 2
        succeeded = 0
        for batch_index in range(batches):
            result = fd_runner(
                model,
                forward_step_func,
                data_iterator,
                parameters,
                probes,
                eps=float(epsilon),
                rng_seed=int(pp_seed),
                theta_snap_ref=lp_snapshots,
                grad_offsets=list(grad_offsets),
                grad_accum_views=grad_views,
                perturb_changed_counts=(changed[0], changed[1]),
                ds_pass_base=batch_index * probes * 2,
                ds_total_backward_passes=total_backward_passes,
                final_grad_accum_boundary=True,
                phase_rng_state=None,
            )
            if result is None:
                raise RuntimeError(
                    f"oracle FD failed at data batch {batch_index + 1}/{batches}"
                )
            succeeded += 1
        if succeeded != batches:
            raise RuntimeError("oracle FD did not complete every requested data batch")

        h_views: tuple[torch.Tensor, ...] = ()
        v_views: tuple[torch.Tensor, ...] = ()
        expert_h_views: tuple[torch.Tensor, ...] = ()
        expert_v_views: tuple[torch.Tensor, ...] = ()
        result_error: BaseException | None = None
        try:
            after = _collect_h_by_name(
                mapper=mapper,
                engine_optimizer=engine_optimizer,
                named_parameters=perturb,
                allow_create=False,
            )
            for view in after.values():
                view.mul_(1.0 / float(batches)).abs_()
            h_views = _layout_h_views(measurement_layout, after)
            _validate_finite_signal(h_views)
            if joint_all_parameters:
                assert expert_measurement_layout is not None
                expert_h_views = _layout_h_views(expert_measurement_layout, after)
                _validate_finite_signal(expert_h_views)

            compared = (
                sum(int(parameter.numel()) for parameter in parameters)
                * probes * batches
            )
            fd_audit = FiniteDifferenceAudit(
                parameter_dtype="bfloat16",
                epsilon=float(epsilon),
                compared_elements=int(compared),
                plus_changed_elements=int(changed[0].item()),
                minus_changed_elements=int(changed[1].item()),
                minimum_changed_fraction=float(minimum_changed_fraction),
                estimator=(
                    "fd_hutchinson_diagonal_joint_all_parameters"
                    if joint_all_parameters
                    else "fd_hutchinson_diagonal_router_inclusive"
                ),
            )
            fd_audit.validate()
            v_views = tuple(
                fragment.v for fragment in measurement_layout.fragments
            )
            if joint_all_parameters:
                assert expert_measurement_layout is not None
                expert_v_views = tuple(
                    fragment.v for fragment in expert_measurement_layout.fragments
                )
            expected_device = getattr(
                nonexpert_lut_builder,
                "device",
                h_views[0].device if h_views else None,
            )
            for label, views in (("H", h_views), ("v", v_views)):
                if not views:
                    raise RuntimeError(
                        f"non-expert {label} population is empty before LUT build"
                    )
                for view in views:
                    allowed_dtypes = (
                        (torch.float32, torch.float16, torch.bfloat16)
                        if label == "H"
                        else (torch.float32,)
                    )
                    if (
                        view.device != expected_device
                        or view.dtype not in allowed_dtypes
                        or not view.is_contiguous()
                    ):
                        raise RuntimeError(
                            f"non-expert {label} view is incompatible with its LUT builder"
                        )
            if joint_all_parameters:
                assert expert_lut_builder is not None
                expert_device = getattr(expert_lut_builder, "device", expected_device)
                for label, views in (("H", expert_h_views), ("v", expert_v_views)):
                    if not views:
                        raise RuntimeError(
                            f"expert {label} population is empty before LUT build"
                        )
                    for view in views:
                        allowed_dtypes = (
                            (torch.float32, torch.float16, torch.bfloat16)
                            if label == "H"
                            else (torch.float32,)
                        )
                        if (
                            view.device != expert_device
                            or view.dtype not in allowed_dtypes
                            or not view.is_contiguous()
                        ):
                            raise RuntimeError(
                                f"expert {label} view is incompatible with its LUT builder"
                            )
        except BaseException as exc:
            result_error = exc
        if stage_consensus is not None:
            stage_consensus("joint_lut_inputs", result_error)
        if result_error is not None:
            raise result_error
        if fd_audit is None:
            raise AssertionError("successful non-expert LUT preflight has no FD audit")
        if joint_all_parameters:
            assert expert_lut_builder is not None
            joint_expert_lut = expert_lut_builder.build(
                h_views=expert_h_views,
                v_views=expert_v_views,
                fd_audit=fd_audit,
                h_values_are_absolute=True,
                process_group=process_group,
            )
        nonexpert_lut = nonexpert_lut_builder.build(
            h_views=h_views,
            v_views=v_views,
            fd_audit=fd_audit,
            h_values_are_absolute=True,
            process_group=process_group,
        )
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        restore_errors: list[str] = []
        try:
            with torch.no_grad():
                for parameter, snapshot in zip(parameters, lp_snapshots):
                    parameter.data.copy_(snapshot, non_blocking=True)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
        except BaseException as exc:
            restore_errors.append(f"LP theta: {type(exc).__name__}: {exc}")
        try:
            discard_zero12_grad_state(engine_optimizer)
            for parameter in module.parameters():
                parameter.grad = None
                if hasattr(parameter, "grad_accum"):
                    parameter.grad_accum = None
        except BaseException as exc:
            restore_errors.append(f"grad cleanup: {type(exc).__name__}: {exc}")
        try:
            _restore_attrs(attr_snapshot)
        except BaseException as exc:
            restore_errors.append(f"DS attrs: {type(exc).__name__}: {exc}")
        try:
            restore_training_state(inner_optimizer, train_snapshot)
        except BaseException as exc:
            restore_errors.append(f"training state: {type(exc).__name__}: {exc}")
        try:
            module.train(was_training)
        except BaseException as exc:
            restore_errors.append(f"model mode: {type(exc).__name__}: {exc}")
        if restore_errors:
            message = "oracle non-expert FD restoration failed: " + "; ".join(restore_errors)
            if primary_error is None:
                raise RuntimeError(message)
            try:
                primary_error.add_note(message)
            except Exception:
                pass

    if nonexpert_lut is None or fd_audit is None:
        raise AssertionError("non-expert curvature completed without a LUT")
    router_set = set(routers)
    router_elements = sum(
        int(parameter.numel())
        for name, parameter in nonexpert_perturb
        if name in router_set
    )
    metadata = NonExpertFdMetadata(
        pp_rank=int(pp_rank),
        perturb_parameter_count=len(perturb),
        perturb_element_count=sum(int(parameter.numel()) for _name, parameter in perturb),
        measurement_fragment_count=len(measurement_layout.fragments),
        measurement_element_count=int(measurement_layout.total_numel),
        measurement_owner_rank=int(measurement_owner_rank),
        measurement_owner_count=int(measurement_owner_count),
        measurement_ownership=str(measurement_ownership),
        router_parameter_names=tuple(routers),
        router_element_count=int(router_elements),
        hutchinson_probes=int(probes),
        data_batches=int(batches),
        epsilon=float(epsilon),
        rng_seed=int(pp_seed),
        routing_freeze=False,
        dedicated_data_iterator=True,
        perturb_identity_scheme="canonical(pp_rank, sorted_parameter_name, element_offset)",
        perturb_parameter_fingerprint=_parameter_fingerprint(
            perturb, pp_rank=int(pp_rank),
        ),
        joint_all_parameters=bool(joint_all_parameters),
        expert_perturb_parameter_count=len(expert_perturb),
        expert_perturb_element_count=sum(
            int(parameter.numel()) for _name, parameter in expert_perturb
        ),
        nonexpert_perturb_parameter_count=len(nonexpert_perturb),
        nonexpert_perturb_element_count=sum(
            int(parameter.numel()) for _name, parameter in nonexpert_perturb
        ),
        fd=fd_audit,
    )
    return NonExpertCurvature(
        lut=nonexpert_lut,
        measurement_layout=measurement_layout,
        metadata=metadata,
        joint_expert_lut=joint_expert_lut,
    )


def prepare_oracle_window_curvature(
    *,
    expert_lut: DomainLut | None = None,
    **nonexpert_kwargs: Any,
) -> OracleWindowCurvature:
    """Build both domains from either a shared or legacy separate FD path."""

    nonexpert = measure_nonexpert_curvature(**nonexpert_kwargs)
    if expert_lut is None:
        frozen_expert = nonexpert.joint_expert_lut
        if frozen_expert is None:
            raise RuntimeError("joint FD did not produce an expert LUT")
    else:
        frozen_expert = freeze_expert_lut(expert_lut)
        if frozen_expert.refs.h_lut.values.data_ptr() == expert_lut.refs.h_lut.values.data_ptr():
            raise AssertionError("frozen expert LUT unexpectedly aliases production storage")
    return OracleWindowCurvature(
        expert_lut=frozen_expert,
        nonexpert=nonexpert,
    )


__all__ = [
    "NonExpertCurvature",
    "NonExpertFdMetadata",
    "OracleWindowCurvature",
    "collect_nonexpert_perturb_parameters",
    "freeze_expert_lut",
    "freeze_expert_quantile_refs",
    "measure_nonexpert_curvature",
    "prepare_oracle_window_curvature",
    "select_nonexpert_measurement_parameters",
]
