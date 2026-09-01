"""Rank-owned optimizer fragments for the oracle-window experiment.

The experiment must observe the FP32 master and Adam slots that the active
DeepSpeed optimizer actually owns.  It must not reconstruct full parameters
with the DeepSpeed safe API: doing so duplicates DP data and loses the exact
partition ownership used by recovery.  This module is deliberately limited to
describing those local views.  It does not start an oracle, communicate, or
write telemetry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Iterable, Mapping, Protocol, Sequence

import torch

from ...storage.expert_partition import ActiveTrackedFragmentSpec


DOMAIN_EXPERT = "expert"
DOMAIN_NONEXPERT = "nonexpert"
_VALID_DOMAINS = frozenset((DOMAIN_EXPERT, DOMAIN_NONEXPERT))


class Zero12FragmentMapper(Protocol):
    """Small mapper surface consumed by this experiment."""

    def zero12_local_fragments(
        self,
        named_params: Iterable[tuple[str, torch.nn.Parameter]],
    ) -> list[tuple[str, torch.nn.Parameter, int, int, int, int]]:
        ...

    def zero12_partition_view(
        self,
        group_idx: int,
        local_offset: int,
        numel: int,
        *,
        state_key: str | None = None,
    ) -> torch.Tensor | None:
        ...


def stable_u64(text: str) -> int:
    """Return a process-independent 64-bit identity for ``text``."""

    digest = hashlib.blake2b(
        str(text).encode("utf-8"),
        digest_size=8,
        person=b"oracle-w",
    ).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)


@dataclass(frozen=True, eq=False)
class OracleFragment:
    """One canonical parameter interval owned by the current optimizer rank."""

    domain: str
    canonical_key: str
    canonical_parameter_id: int
    runtime_name: str
    parameter_name: str
    parameter_offset: int
    parameter_numel: int
    numel: int
    optimizer_group_idx: int
    optimizer_local_offset: int
    packed_offset: int
    p: torch.Tensor = field(repr=False, compare=False)
    m: torch.Tensor = field(repr=False, compare=False)
    v: torch.Tensor = field(repr=False, compare=False)
    expert_id: tuple[int, int] | None = None
    pp_rank: int | None = None

    @property
    def parameter_end(self) -> int:
        return int(self.parameter_offset) + int(self.numel)

    @property
    def packed_end(self) -> int:
        return int(self.packed_offset) + int(self.numel)

    @property
    def canonical_id_words(self) -> tuple[int, int]:
        value = int(self.canonical_parameter_id)
        return value & 0xFFFFFFFF, (value >> 32) & 0xFFFFFFFF


@dataclass(frozen=True)
class FragmentCoverageRecord:
    """Tensor-free record suitable for cross-rank duplicate/coverage checks."""

    domain: str
    canonical_key: str
    canonical_parameter_id: int
    parameter_offset: int
    numel: int
    parameter_numel: int
    owner_rank: int

    @property
    def end(self) -> int:
        return int(self.parameter_offset) + int(self.numel)


@dataclass(frozen=True, eq=False)
class PackedFragmentLayout:
    """Stable packed-shadow layout over local fragment views."""

    domain: str
    fragments: tuple[OracleFragment, ...]
    total_numel: int

    def coverage_records(self, *, owner_rank: int) -> tuple[FragmentCoverageRecord, ...]:
        return tuple(
            FragmentCoverageRecord(
                domain=fragment.domain,
                canonical_key=fragment.canonical_key,
                canonical_parameter_id=fragment.canonical_parameter_id,
                parameter_offset=fragment.parameter_offset,
                numel=fragment.numel,
                parameter_numel=fragment.parameter_numel,
                owner_rank=int(owner_rank),
            )
            for fragment in self.fragments
        )


def _validate_domain(domain: str) -> str:
    value = str(domain)
    if value not in _VALID_DOMAINS:
        raise ValueError(f"domain must be one of {sorted(_VALID_DOMAINS)}; got {value!r}")
    return value


def _state_views(
    mapper: Zero12FragmentMapper,
    *,
    group_idx: int,
    local_offset: int,
    numel: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    p = mapper.zero12_partition_view(
        group_idx, local_offset, numel, state_key=None,
    )
    m = mapper.zero12_partition_view(
        group_idx, local_offset, numel, state_key="exp_avg",
    )
    v = mapper.zero12_partition_view(
        group_idx, local_offset, numel, state_key="exp_avg_sq",
    )
    if p is None or m is None or v is None:
        raise RuntimeError(
            "oracle-window fragment cannot resolve FP32 p/m/v views: "
            f"group={group_idx} local_offset={local_offset} numel={numel}"
        )
    views = (p.detach().reshape(-1), m.detach().reshape(-1), v.detach().reshape(-1))
    for label, view in zip(("p", "m", "v"), views):
        if int(view.numel()) != int(numel):
            raise RuntimeError(
                f"oracle-window {label} view has {view.numel()} elements; expected {numel}"
            )
        if view.dtype != torch.float32:
            raise RuntimeError(
                f"oracle-window {label} view must be the active FP32 optimizer slot; "
                f"got {view.dtype}"
            )
        if not view.is_contiguous():
            raise RuntimeError(f"oracle-window {label} view must be contiguous")
    if not (views[0].device == views[1].device == views[2].device):
        raise RuntimeError("oracle-window p/m/v fragment devices do not match")
    return views


def _finalize_layout(
    domain: str,
    rows: Sequence[dict[str, object]],
) -> PackedFragmentLayout:
    domain = _validate_domain(domain)
    ordered = sorted(
        rows,
        key=lambda row: (
            str(row["canonical_key"]),
            int(row["parameter_offset"]),
            int(row["optimizer_group_idx"]),
            int(row["optimizer_local_offset"]),
        ),
    )
    fragments: list[OracleFragment] = []
    packed_offset = 0
    seen_optimizer_intervals: set[tuple[int, int, int]] = set()
    for row in ordered:
        optimizer_interval = (
            int(row["optimizer_group_idx"]),
            int(row["optimizer_local_offset"]),
            int(row["numel"]),
        )
        if optimizer_interval in seen_optimizer_intervals:
            raise RuntimeError(
                "duplicate rank-owned optimizer fragment: "
                f"group={optimizer_interval[0]} offset={optimizer_interval[1]} "
                f"numel={optimizer_interval[2]}"
            )
        seen_optimizer_intervals.add(optimizer_interval)
        fragment = OracleFragment(
            domain=domain,
            canonical_key=str(row["canonical_key"]),
            canonical_parameter_id=int(row["canonical_parameter_id"]),
            runtime_name=str(row["runtime_name"]),
            parameter_name=str(row["parameter_name"]),
            parameter_offset=int(row["parameter_offset"]),
            parameter_numel=int(row["parameter_numel"]),
            numel=int(row["numel"]),
            optimizer_group_idx=int(row["optimizer_group_idx"]),
            optimizer_local_offset=int(row["optimizer_local_offset"]),
            packed_offset=int(packed_offset),
            p=row["p"],  # type: ignore[arg-type]
            m=row["m"],  # type: ignore[arg-type]
            v=row["v"],  # type: ignore[arg-type]
            expert_id=row.get("expert_id"),  # type: ignore[arg-type]
            pp_rank=row.get("pp_rank"),  # type: ignore[arg-type]
        )
        if fragment.numel <= 0:
            raise RuntimeError("oracle-window fragment must be non-empty")
        if (
            fragment.parameter_offset < 0
            or fragment.parameter_end > fragment.parameter_numel
        ):
            raise RuntimeError(
                "oracle-window canonical parameter interval is invalid: "
                f"{fragment.canonical_key} [{fragment.parameter_offset}, "
                f"{fragment.parameter_end}) / {fragment.parameter_numel}"
            )
        fragments.append(fragment)
        packed_offset += int(fragment.numel)

    validate_fragment_coverage(
        tuple(
            FragmentCoverageRecord(
                domain=f.domain,
                canonical_key=f.canonical_key,
                canonical_parameter_id=f.canonical_parameter_id,
                parameter_offset=f.parameter_offset,
                numel=f.numel,
                parameter_numel=f.parameter_numel,
                owner_rank=0,
            )
            for f in fragments
        ),
        require_complete=False,
    )
    return PackedFragmentLayout(
        domain=domain,
        fragments=tuple(fragments),
        total_numel=int(packed_offset),
    )


def build_nonexpert_layout(
    *,
    mapper: Zero12FragmentMapper,
    named_params: Iterable[tuple[str, torch.nn.Parameter]],
    pp_rank: int,
    namespace: str = "model",
) -> PackedFragmentLayout:
    """Build local non-expert measurement ownership from mapper fragments."""

    params = tuple((str(name), param) for name, param in named_params)
    raw = mapper.zero12_local_fragments(params)
    rows: list[dict[str, object]] = []
    for name, param, group_idx, local_offset, param_offset, numel in raw:
        n = int(numel)
        if n <= 0:
            continue
        canonical_key = (
            f"{str(namespace)}/pp={int(pp_rank)}/nonexpert/{str(name)}"
        )
        p, m, v = _state_views(
            mapper,
            group_idx=int(group_idx),
            local_offset=int(local_offset),
            numel=n,
        )
        rows.append({
            "canonical_key": canonical_key,
            "canonical_parameter_id": stable_u64(canonical_key),
            "runtime_name": str(name),
            "parameter_name": str(name),
            "parameter_offset": int(param_offset),
            "parameter_numel": int(param.numel()),
            "numel": n,
            "optimizer_group_idx": int(group_idx),
            "optimizer_local_offset": int(local_offset),
            "p": p,
            "m": m,
            "v": v,
            "expert_id": None,
            "pp_rank": int(pp_rank),
        })
    return _finalize_layout(DOMAIN_NONEXPERT, rows)


def build_expert_layout(
    *,
    mapper: Zero12FragmentMapper,
    specs: Iterable[ActiveTrackedFragmentSpec],
    name_to_param: Mapping[str, torch.nn.Parameter],
    namespace: str = "model",
    global_expert_base: int = 0,
) -> PackedFragmentLayout:
    """Build expert ownership using stable global expert IDs from storage.

    ``ActiveTrackedFragmentSpec`` is the storage layer's canonical mapping
    from rank-local runtime names to ``(layer, global expert)``.  We still
    re-read mapper fragments and require an exact interval match, preventing a
    stale storage descriptor from silently observing a different optimizer
    shard.
    """

    spec_rows = tuple(specs)
    unique_params = {
        spec.parameter_name: name_to_param[spec.parameter_name]
        for spec in spec_rows
        if spec.parameter_name in name_to_param
    }
    missing = sorted(
        {spec.parameter_name for spec in spec_rows} - set(unique_params)
    )
    if missing:
        raise RuntimeError(
            f"oracle-window expert parameters cannot be resolved: {missing[:8]}"
        )
    mapper_rows = mapper.zero12_local_fragments(unique_params.items())
    raw_by_interval: dict[tuple[str, int, int, int, int], tuple[object, ...]] = {}
    for row in mapper_rows:
        name, _param, group_idx, local_offset, param_offset, numel = row
        key = (
            str(name), int(param_offset), int(numel),
            int(group_idx), int(local_offset),
        )
        if key in raw_by_interval:
            raise RuntimeError(f"duplicate mapper expert fragment interval: {key}")
        raw_by_interval[key] = tuple(row)

    rows: list[dict[str, object]] = []
    for spec in spec_rows:
        if spec.optimizer_group_idx is None or spec.optimizer_local_offset is None:
            candidates = [
                key for key in raw_by_interval
                if key[0] == spec.parameter_name
                and key[1] == int(spec.parameter_offset)
                and key[2] == int(spec.numel)
            ]
            if len(candidates) != 1:
                raise RuntimeError(
                    "oracle-window expert spec has no unique mapper fragment: "
                    f"runtime={spec.runtime_name!r} candidates={len(candidates)}"
                )
            key = candidates[0]
        else:
            key = (
                spec.parameter_name,
                int(spec.parameter_offset),
                int(spec.numel),
                int(spec.optimizer_group_idx),
                int(spec.optimizer_local_offset),
            )
        raw = raw_by_interval.pop(key, None)
        if raw is None:
            raise RuntimeError(
                "oracle-window expert spec does not match active mapper ownership: "
                f"runtime={spec.runtime_name!r} key={key}"
            )
        _name, param, group_idx, local_offset, param_offset, numel = raw
        layer = int(spec.expert_id[0])
        local_expert = int(spec.expert_id[1])
        expert = int(global_expert_base) + local_expert
        if expert < 0:
            raise RuntimeError(
                "oracle-window global expert id must be non-negative: "
                f"base={global_expert_base} local={local_expert}"
            )
        canonical_key = (
            f"{str(namespace)}/expert/layer={layer}/expert={expert}/"
            f"{spec.parameter_name}"
        )
        p, m, v = _state_views(
            mapper,
            group_idx=int(group_idx),
            local_offset=int(local_offset),
            numel=int(numel),
        )
        rows.append({
            "canonical_key": canonical_key,
            "canonical_parameter_id": stable_u64(canonical_key),
            "runtime_name": spec.runtime_name,
            "parameter_name": spec.parameter_name,
            "parameter_offset": int(param_offset),
            "parameter_numel": int(param.numel()),
            "numel": int(numel),
            "optimizer_group_idx": int(group_idx),
            "optimizer_local_offset": int(local_offset),
            "p": p,
            "m": m,
            "v": v,
            "expert_id": (layer, expert),
            "pp_rank": None,
        })
    if raw_by_interval:
        raise RuntimeError(
            "active expert mapper ownership contains fragments absent from the "
            f"storage descriptor: {list(raw_by_interval)[:8]}"
        )
    return _finalize_layout(DOMAIN_EXPERT, rows)


def validate_fragment_coverage(
    records: Iterable[FragmentCoverageRecord],
    *,
    require_complete: bool,
) -> None:
    """Reject duplicate/overlapping ownership and optionally require coverage.

    Call this locally for construction checks and again on gathered tensor-free
    records to detect DP duplication or missing ZeRO shards.  Complete
    coverage means exactly ``[0, parameter_numel)`` once per canonical key.
    """

    grouped: dict[str, list[FragmentCoverageRecord]] = {}
    ids: dict[int, str] = {}
    for record in records:
        if record.domain not in _VALID_DOMAINS:
            raise RuntimeError(f"invalid fragment domain {record.domain!r}")
        if record.numel <= 0 or record.parameter_numel <= 0:
            raise RuntimeError("fragment coverage records must be non-empty")
        if record.parameter_offset < 0 or record.end > record.parameter_numel:
            raise RuntimeError(
                f"invalid fragment coverage interval for {record.canonical_key}"
            )
        previous_key = ids.setdefault(
            int(record.canonical_parameter_id), record.canonical_key,
        )
        if previous_key != record.canonical_key:
            raise RuntimeError(
                "canonical 64-bit parameter ID collision: "
                f"{previous_key!r} and {record.canonical_key!r}"
            )
        grouped.setdefault(record.canonical_key, []).append(record)

    for key, group in grouped.items():
        parameter_sizes = {int(record.parameter_numel) for record in group}
        if len(parameter_sizes) != 1:
            raise RuntimeError(f"inconsistent parameter size for {key}")
        ordered = sorted(
            group,
            key=lambda record: (
                int(record.parameter_offset), int(record.end), int(record.owner_rank),
            ),
        )
        cursor = 0
        for record in ordered:
            if int(record.parameter_offset) < cursor:
                raise RuntimeError(
                    "duplicate or overlapping optimizer ownership for "
                    f"{key}: offset={record.parameter_offset} cursor={cursor}"
                )
            if require_complete and int(record.parameter_offset) != cursor:
                raise RuntimeError(
                    f"missing optimizer ownership for {key}: [{cursor}, "
                    f"{record.parameter_offset})"
                )
            cursor = int(record.end)
        if require_complete and cursor != next(iter(parameter_sizes)):
            raise RuntimeError(
                f"incomplete optimizer ownership for {key}: {cursor}/"
                f"{next(iter(parameter_sizes))}"
            )


__all__ = [
    "DOMAIN_EXPERT",
    "DOMAIN_NONEXPERT",
    "FragmentCoverageRecord",
    "OracleFragment",
    "PackedFragmentLayout",
    "Zero12FragmentMapper",
    "build_expert_layout",
    "build_nonexpert_layout",
    "stable_u64",
    "validate_fragment_coverage",
]
