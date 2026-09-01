"""High-level collector for the experiment-only saliency window.

The collector coordinates bounded score sampling, a CPU/Gloo global cutoff,
exact full-population GPU counts, deterministic tie handling, shadow updates,
and rank-sharded telemetry.  Curvature acquisition and layout construction are
injected dependencies: this module never imports production checkpoint or
fault-recovery code.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

import numpy as np
import torch

from .fragments import (
    DOMAIN_EXPERT,
    DOMAIN_NONEXPERT,
    PackedFragmentLayout,
    Zero12FragmentMapper,
    build_expert_layout,
    build_nonexpert_layout,
    validate_fragment_coverage,
)
from .gpu import (
    UINT32_MAX,
    FullCountWorkspace,
    PackedFp32Shadow,
    SampleBuffers,
    SelectionCountWorkspace,
    build_block_sampling_plan,
    launch_full_count,
    launch_sample_scores,
    launch_update_selected,
)
from .io import build_rank_arrays, write_oracle_window_shard
from .stats import population_weights, weighted_cutoff


DOMAIN_ORDER = (DOMAIN_EXPERT, DOMAIN_NONEXPERT)


@dataclass(frozen=True)
class OracleWindowTopology:
    rank: int
    world: int
    dp_rank: int = 0
    dp_size: int = 1
    pp_rank: int = 0
    pp_size: int = 1
    ep_rank: int = 0
    ep_size: int = 1

    def validated(self) -> "OracleWindowTopology":
        if self.world < 1 or not 0 <= self.rank < self.world:
            raise ValueError("oracle-window rank/world is invalid")
        for name in ("dp", "pp", "ep"):
            rank = int(getattr(self, f"{name}_rank"))
            size = int(getattr(self, f"{name}_size"))
            if size < 1 or not 0 <= rank < size:
                raise ValueError(f"oracle-window {name} rank/size is invalid")
        return self

    @classmethod
    def from_object(cls, value: Any) -> "OracleWindowTopology":
        return cls(
            rank=int(getattr(value, "global_rank", getattr(value, "rank", 0))),
            world=int(getattr(value, "world_size", getattr(value, "world", 1))),
            dp_rank=int(getattr(value, "dp_rank", 0)),
            dp_size=int(getattr(value, "dp_size", 1)),
            pp_rank=int(getattr(value, "pp_rank", 0)),
            pp_size=int(getattr(value, "pp_size", 1)),
            ep_rank=int(getattr(value, "ep_rank", 0)),
            ep_size=int(getattr(value, "ep_size", 1)),
        ).validated()


@dataclass(frozen=True)
class OracleWindowSettings:
    output_dir: Path
    topk_ratio: float
    window_updates: int = 100
    snapshot_offsets: tuple[int, ...] = tuple(range(10, 101, 10))
    sample_rounds: int = 4
    max_sample_rounds: int = 8
    sample_blocks: int = 4096
    block_elements: int = 256
    sampling_seed: int = 0
    cutoff_tolerance: float = 0.04
    min_effective_tail_count: float = 8.0
    fd_epsilon: float = 1.0e-2
    fd_probes: int = 4
    fd_data_batches: int = 1
    fd_min_changed_fraction: float = 0.95

    def validated(self) -> "OracleWindowSettings":
        offsets = tuple(int(value) for value in self.snapshot_offsets)
        if not (0.0 < float(self.topk_ratio) <= 1.0):
            raise ValueError("oracle-window topk_ratio must lie in (0, 1]")
        if int(self.window_updates) < 1:
            raise ValueError("oracle-window window_updates must be positive")
        if (
            not offsets
            or tuple(sorted(set(offsets))) != offsets
            or offsets[0] < 1
            or offsets[-1] > int(self.window_updates)
        ):
            raise ValueError("oracle-window snapshot offsets are invalid")
        if int(self.sample_rounds) < 4:
            raise ValueError("oracle-window requires at least four sample rounds")
        if int(self.max_sample_rounds) < int(self.sample_rounds):
            raise ValueError("max_sample_rounds must be >= sample_rounds")
        if int(self.sample_blocks) < 1 or int(self.block_elements) < 1:
            raise ValueError("oracle-window sampling sizes must be positive")
        if not (0.0 <= float(self.cutoff_tolerance) < 1.0):
            raise ValueError("oracle-window cutoff_tolerance must lie in [0, 1)")
        if float(self.min_effective_tail_count) < 0.0:
            raise ValueError("min_effective_tail_count must be non-negative")
        if not math.isfinite(float(self.fd_epsilon)) or float(self.fd_epsilon) <= 0.0:
            raise ValueError("oracle-window fd_epsilon must be finite and positive")
        if int(self.fd_probes) < 4 or int(self.fd_data_batches) < 1:
            raise ValueError("oracle-window FD requires >=4 probes and >=1 batch")
        if not 0.0 <= float(self.fd_min_changed_fraction) <= 1.0:
            raise ValueError("fd_min_changed_fraction must lie in [0, 1]")
        return self

    @classmethod
    def from_config(cls, config: Any) -> "OracleWindowSettings":
        output = getattr(config, "oracle_window_log_dir", None)
        if output is None:
            base = getattr(config, "independent_oracle_log_dir", None)
            output = (
                Path(base) / "oracle_window"
                if base is not None
                else Path(config.cp_dir) / "independent_oracle" / "oracle_window"
            )
        return cls(
            output_dir=Path(output),
            topk_ratio=float(config.topk_ratio),
            window_updates=int(getattr(config, "oracle_window_updates", 100)),
            snapshot_offsets=tuple(
                int(value)
                for value in getattr(
                    config,
                    "oracle_window_snapshot_offsets",
                    tuple(range(10, 101, 10)),
                )
            ),
            sample_rounds=int(
                getattr(config, "oracle_window_sample_rounds", 4)
            ),
            max_sample_rounds=int(
                getattr(config, "oracle_window_max_sample_rounds", 8)
            ),
            sample_blocks=int(
                getattr(config, "oracle_window_sample_blocks", 4096)
            ),
            block_elements=int(
                getattr(config, "oracle_window_block_elements", 256)
            ),
            sampling_seed=int(
                getattr(config, "oracle_window_sampling_seed", 0)
            ),
            cutoff_tolerance=float(
                getattr(config, "cutoff_K_tolerance", 0.04)
            ),
            min_effective_tail_count=float(
                getattr(
                    config,
                    "oracle_window_min_effective_tail_count",
                    8.0,
                )
            ),
            fd_epsilon=float(
                getattr(config, "oracle_window_fd_epsilon", 1.0e-2)
            ),
            fd_probes=int(getattr(config, "oracle_window_fd_probes", 4)),
            fd_data_batches=int(
                getattr(config, "oracle_window_fd_data_batches", 1)
            ),
            fd_min_changed_fraction=float(
                getattr(
                    config,
                    "oracle_window_fd_min_changed_fraction",
                    0.95,
                )
            ),
        ).validated()


@dataclass(frozen=True)
class SampleBatch:
    raw: np.ndarray
    clipped: np.ndarray
    canonical_ids: np.ndarray
    weights: np.ndarray

    def validated(self) -> "SampleBatch":
        raw = np.asarray(self.raw, dtype=np.float32).reshape(-1)
        clipped = np.asarray(self.clipped, dtype=np.float32).reshape(-1)
        ids = np.asarray(self.canonical_ids, dtype=np.int64).reshape(-1)
        weights = np.asarray(self.weights, dtype=np.float64).reshape(-1)
        if raw.size <= 0 or not (
            raw.shape == clipped.shape == ids.shape == weights.shape
        ):
            raise ValueError("oracle-window sample batch is empty or unaligned")
        if np.any(~np.isfinite(weights)) or np.any(weights <= 0.0):
            raise ValueError("oracle-window sample weights must be finite and positive")
        if not np.all(np.isfinite(raw)):
            raise RuntimeError("oracle-window sampled non-finite raw saliency")
        if not np.allclose(clipped, np.maximum(raw, 0.0), rtol=1e-6, atol=0.0):
            raise RuntimeError("oracle-window clipped sample does not match raw")
        return SampleBatch(
            raw=raw,
            clipped=clipped,
            canonical_ids=ids,
            weights=weights,
        )


@dataclass(frozen=True)
class RankSamplePacket:
    rank: int
    population_count: np.ndarray
    sample_domain: np.ndarray
    sample_round: np.ndarray
    sample_global_id: np.ndarray
    sample_raw: np.ndarray
    sample_clipped: np.ndarray
    sample_weight: np.ndarray


@dataclass(frozen=True)
class FullCount:
    greater: int
    equal: int
    raw_negative: int = 0
    raw_zero: int = 0
    raw_positive: int = 0
    nonfinite: int = 0

    def as_array(self) -> np.ndarray:
        return np.asarray(
            (
                self.greater,
                self.equal,
                self.raw_negative,
                self.raw_zero,
                self.raw_positive,
                self.nonfinite,
            ),
            dtype=np.int64,
        )


@dataclass(frozen=True)
class CutoffDecision:
    tau: float
    target_count: int
    band_low: int
    band_high: int
    greater_count: int
    equal_count: int
    tie_hash_limit: int
    tie_salt: int
    sample_rounds: int


class OracleWindowLayoutProvider(Protocol):
    def build_layouts(self, *, model: Any) -> Mapping[str, PackedFragmentLayout]:
        ...


def _local_module(model: Any) -> torch.nn.Module:
    root = model[0] if isinstance(model, (list, tuple)) else model
    module = getattr(root, "module", root)
    if not isinstance(module, torch.nn.Module):
        raise TypeError("oracle-window model does not expose a local nn.Module")
    return module


def _fdp_global_expert_base() -> int:
    ids_are_global = os.environ.get(
        "PROPOSAL_FDP_EXPERT_IDS_GLOBAL", "",
    ).strip().lower() in {"1", "true", "yes", "on"}
    if ids_are_global:
        return 0
    base = int(os.environ.get("PROPOSAL_FDP_EXPERT_ID_BASE", "0") or "0")
    if base < 0:
        raise RuntimeError("PROPOSAL_FDP_EXPERT_ID_BASE must be non-negative")
    return base


class RankOwnedOracleWindowLayoutProvider:
    """Build both domains from the active optimizer ownership descriptors."""

    def __init__(
        self,
        *,
        mapper: Zero12FragmentMapper,
        expert_specs_provider: Callable[[], Iterable[Any]],
        nonexpert_parameters_provider: Callable[
            [], Iterable[tuple[str, torch.nn.Parameter]]
        ],
        pp_rank: int,
        namespace: str = "model",
    ) -> None:
        self._mapper = mapper
        self._expert_specs_provider = expert_specs_provider
        self._nonexpert_parameters_provider = nonexpert_parameters_provider
        self._pp_rank = int(pp_rank)
        self._namespace = str(namespace)

    def build_layouts(self, *, model: Any) -> Mapping[str, PackedFragmentLayout]:
        module = _local_module(model)
        named = dict(module.named_parameters())
        specs = tuple(self._expert_specs_provider())
        nonexperts = tuple(self._nonexpert_parameters_provider())
        if not specs or not nonexperts:
            raise RuntimeError(
                "oracle-window layout requires active expert and nonexpert ownership"
            )
        return {
            DOMAIN_EXPERT: build_expert_layout(
                mapper=self._mapper,
                specs=specs,
                name_to_param=named,
                namespace=self._namespace,
                global_expert_base=_fdp_global_expert_base(),
            ),
            DOMAIN_NONEXPERT: build_nonexpert_layout(
                mapper=self._mapper,
                named_params=nonexperts,
                pp_rank=self._pp_rank,
                namespace=self._namespace,
            ),
        }


class OracleWindowCurvatureProvider(Protocol):
    def prepare_window(
        self,
        *,
        model: Any,
        forward_step_func: Any,
        data_iterator: Any,
        start_iteration: int,
        layouts: Mapping[str, PackedFragmentLayout],
        device_process_group: Any,
        stage_consensus: Callable[[str, BaseException | None], None],
    ) -> Mapping[str, Any]:
        """Return independent expert/non-expert domain LUTs."""

    def provenance(self) -> Mapping[str, Any]:
        """Return immutable per-domain FD/LUT provenance for rank shards."""


class LiveOracleWindowCurvatureProvider:
    """Measure expert and non-expert LUTs in one common all-parameter FD."""

    def __init__(
        self,
        *,
        lossy_cp: Any,
        mapper: Zero12FragmentMapper,
        inner_optimizer: Any,
        expert_parameter_names: Iterable[str],
        topology: OracleWindowTopology,
        finite_difference_epsilon: float,
        hutchinson_probes: int,
        data_batches: int,
        rng_seed: int,
        minimum_changed_fraction: float = 0.95,
    ) -> None:
        del lossy_cp  # retained in the constructor for call-site compatibility
        self._mapper = mapper
        self._inner_optimizer = inner_optimizer
        self._expert_parameter_names = tuple(
            sorted(str(name) for name in expert_parameter_names)
        )
        self._topology = topology.validated()
        self._epsilon = float(finite_difference_epsilon)
        self._probes = max(4, int(hutchinson_probes))
        self._data_batches = max(1, int(data_batches))
        self._rng_seed = int(rng_seed)
        self._minimum_changed_fraction = float(minimum_changed_fraction)
        self._provenance: dict[str, Any] | None = None

    def provenance(self) -> Mapping[str, Any]:
        if self._provenance is None:
            raise RuntimeError("oracle curvature provenance requested before T0")
        return dict(self._provenance)

    def prepare_window(
        self,
        *,
        model: Any,
        forward_step_func: Any,
        data_iterator: Any,
        start_iteration: int,
        layouts: Mapping[str, PackedFragmentLayout],
        device_process_group: Any,
        stage_consensus: Callable[[str, BaseException | None], None],
    ) -> Mapping[str, Any]:
        from .curvature import (
            collect_nonexpert_perturb_parameters,
            prepare_oracle_window_curvature,
            select_nonexpert_measurement_parameters,
        )
        from .lut import DomainLutBuilder

        expert_layout = layouts[DOMAIN_EXPERT]
        if not expert_layout.fragments:
            raise RuntimeError("oracle joint FD has an empty expert layout")
        device = expert_layout.fragments[0].p.device
        owner_count = max(
            1, int(self._topology.dp_size) * int(self._topology.ep_size),
        )
        owner_rank = (
            int(self._topology.dp_rank) * int(self._topology.ep_size)
            + int(self._topology.ep_rank)
        )
        nonexpert_error: BaseException | None = None
        try:
            perturb, _routers, _offsets = collect_nonexpert_perturb_parameters(
                model=model,
                expert_parameter_names=self._expert_parameter_names,
                pp_rank=int(self._topology.pp_rank),
            )
            measurement_parameters, _ownership = (
                select_nonexpert_measurement_parameters(
                    mapper=self._mapper,
                    perturb_parameters=perturb,
                    owner_rank=int(owner_rank),
                    owner_count=int(owner_count),
                )
            )
            measured = build_nonexpert_layout(
                mapper=self._mapper,
                named_params=measurement_parameters,
                pp_rank=int(self._topology.pp_rank),
            )
            expected = layouts[DOMAIN_NONEXPERT]
            measured_identity = tuple(
                (
                    fragment.canonical_key,
                    int(fragment.parameter_offset),
                    int(fragment.numel),
                )
                for fragment in measured.fragments
            )
            expected_identity = tuple(
                (
                    fragment.canonical_key,
                    int(fragment.parameter_offset),
                    int(fragment.numel),
                )
                for fragment in expected.fragments
            )
            if (
                int(measured.total_numel) != int(expected.total_numel)
                or measured_identity != expected_identity
            ):
                raise RuntimeError(
                    "nonexpert curvature ownership disagrees with oracle shadow layout"
                )
            parameters = tuple(parameter for _name, parameter in perturb)
            if not parameters or any(
                parameter.device.type != "cuda" for parameter in parameters
            ):
                raise RuntimeError(
                    "all non-expert LP perturb parameters must be CUDA tensors"
                )
            dtypes = {parameter.dtype for parameter in parameters}
            if dtypes != {torch.bfloat16}:
                raise RuntimeError(
                    "audited shared FD perturb currently requires BF16 LP parameters; "
                    f"got {sorted(str(dtype) for dtype in dtypes)}"
                )
            if data_iterator is None:
                raise RuntimeError("non-expert FD requires a dedicated data iterator")
        except BaseException as exc:
            nonexpert_error = exc
        stage_consensus("nonexpert_fd_inputs", nonexpert_error)
        result = prepare_oracle_window_curvature(
            model=model,
            forward_step_func=forward_step_func,
            data_iterator=data_iterator,
            mapper=self._mapper,
            inner_optimizer=self._inner_optimizer,
            nonexpert_lut_builder=DomainLutBuilder(
                domain=DOMAIN_NONEXPERT,
                device=device,
            ),
            expert_measurement_layout=expert_layout,
            expert_lut_builder=DomainLutBuilder(
                domain=DOMAIN_EXPERT,
                device=device,
            ),
            joint_all_parameters=True,
            expert_parameter_names=self._expert_parameter_names,
            pp_rank=int(self._topology.pp_rank),
            measurement_owner_rank=int(owner_rank),
            measurement_owner_count=int(owner_count),
            epsilon=self._epsilon,
            hutchinson_probes=self._probes,
            data_batches=self._data_batches,
            process_group=device_process_group,
            rng_seed=self._rng_seed,
            minimum_changed_fraction=self._minimum_changed_fraction,
            dedicated_data_iterator=True,
            stage_consensus=stage_consensus,
        )
        expert_meta = result.expert_lut.metadata
        ne_meta = result.nonexpert.lut.metadata
        ne_fd_meta = result.nonexpert.metadata
        self._provenance = {
            "expert": {
                "estimator": str(expert_meta.fd.estimator),
                "parameter_dtype": str(expert_meta.fd.parameter_dtype),
                "epsilon": float(expert_meta.fd.epsilon),
                "hutchinson_probes": int(ne_fd_meta.hutchinson_probes),
                "data_batches": int(ne_fd_meta.data_batches),
                "compared_elements": int(expert_meta.fd.compared_elements),
                "plus_changed_elements": int(expert_meta.fd.plus_changed_elements),
                "minus_changed_elements": int(expert_meta.fd.minus_changed_elements),
                "minimum_changed_fraction": float(
                    expert_meta.fd.minimum_changed_fraction
                ),
                "local_population": int(expert_meta.local_population),
                "global_population": int(expert_meta.global_population),
                "global_h_positive": int(expert_meta.global_h_positive),
                "global_v_positive": int(expert_meta.global_v_positive),
                "h_min": float(expert_meta.h_min),
                "h_max": float(expert_meta.h_max),
                "v_min": float(expert_meta.v_min),
                "v_max": float(expert_meta.v_max),
                "n_quantiles": int(expert_meta.n_quantiles),
                "n_histogram_bins": int(expert_meta.n_histogram_bins),
                "n_lut_bins": int(expert_meta.n_lut_bins),
                "joint_all_parameters": bool(ne_fd_meta.joint_all_parameters),
                "perturb_parameter_fingerprint": str(
                    ne_fd_meta.perturb_parameter_fingerprint
                ),
            },
            "nonexpert": {
                "estimator": str(ne_meta.fd.estimator),
                "parameter_dtype": str(ne_meta.fd.parameter_dtype),
                "epsilon": float(ne_meta.fd.epsilon),
                "hutchinson_probes": int(ne_fd_meta.hutchinson_probes),
                "data_batches": int(ne_fd_meta.data_batches),
                "compared_elements": int(ne_meta.fd.compared_elements),
                "plus_changed_elements": int(ne_meta.fd.plus_changed_elements),
                "minus_changed_elements": int(ne_meta.fd.minus_changed_elements),
                "minimum_changed_fraction": float(
                    ne_meta.fd.minimum_changed_fraction
                ),
                "local_population": int(ne_meta.local_population),
                "global_population": int(ne_meta.global_population),
                "global_h_positive": int(ne_meta.global_h_positive),
                "global_v_positive": int(ne_meta.global_v_positive),
                "h_min": float(ne_meta.h_min),
                "h_max": float(ne_meta.h_max),
                "v_min": float(ne_meta.v_min),
                "v_max": float(ne_meta.v_max),
                "n_quantiles": int(ne_meta.n_quantiles),
                "n_histogram_bins": int(ne_meta.n_histogram_bins),
                "n_lut_bins": int(ne_meta.n_lut_bins),
                "pp_rank": int(ne_fd_meta.pp_rank),
                "measurement_owner_rank": int(
                    ne_fd_meta.measurement_owner_rank
                ),
                "measurement_owner_count": int(
                    ne_fd_meta.measurement_owner_count
                ),
                "measurement_ownership": str(ne_fd_meta.measurement_ownership),
                "router_parameter_count": int(
                    len(ne_fd_meta.router_parameter_names)
                ),
                "router_element_count": int(ne_fd_meta.router_element_count),
                "perturb_parameter_fingerprint": str(
                    ne_fd_meta.perturb_parameter_fingerprint
                ),
                "routing_freeze": bool(ne_fd_meta.routing_freeze),
                "dedicated_data_iterator": bool(
                    ne_fd_meta.dedicated_data_iterator
                ),
                "joint_all_parameters": bool(ne_fd_meta.joint_all_parameters),
                "expert_perturb_parameter_count": int(
                    ne_fd_meta.expert_perturb_parameter_count
                ),
                "expert_perturb_element_count": int(
                    ne_fd_meta.expert_perturb_element_count
                ),
                "nonexpert_perturb_parameter_count": int(
                    ne_fd_meta.nonexpert_perturb_parameter_count
                ),
                "nonexpert_perturb_element_count": int(
                    ne_fd_meta.nonexpert_perturb_element_count
                ),
            },
        }
        measured_layout = result.nonexpert.measurement_layout
        expected_layout = layouts[DOMAIN_NONEXPERT]
        if (
            int(measured_layout.total_numel) != int(expected_layout.total_numel)
            or tuple(record.canonical_key for record in measured_layout.fragments)
            != tuple(record.canonical_key for record in expected_layout.fragments)
        ):
            raise RuntimeError(
                "nonexpert curvature ownership disagrees with oracle shadow layout"
            )
        return {
            DOMAIN_EXPERT: result.expert_lut,
            DOMAIN_NONEXPERT: result.nonexpert.lut,
        }


class OracleWindowCompute(Protocol):
    def initialize(
        self,
        *,
        layouts: Mapping[str, PackedFragmentLayout],
        settings: OracleWindowSettings,
    ) -> None:
        ...

    def begin(
        self,
        *,
        layouts: Mapping[str, PackedFragmentLayout],
    ) -> None:
        ...

    def sample_round(
        self,
        *,
        domain: str,
        lut: Any,
        round_index: int,
        base_salt: int,
    ) -> SampleBatch:
        ...

    def full_count(self, *, domain: str, lut: Any, tau: float) -> FullCount:
        ...

    def update_selected(
        self,
        *,
        domain: str,
        lut: Any,
        tau: float,
        tie_hash_limit: int,
        tie_salt: int,
    ) -> int:
        ...


class OracleWindowCoordinator(Protocol):
    rank: int
    world: int
    cpu_process_group: Any

    def gather_samples(self, packet: RankSamplePacket) -> Sequence[RankSamplePacket]:
        ...

    def gather_objects(self, value: Any) -> Sequence[Any]:
        ...

    def sum_ints(self, values: np.ndarray) -> np.ndarray:
        ...


class OracleWindowShardWriter(Protocol):
    def write(
        self,
        *,
        settings: OracleWindowSettings,
        topology: OracleWindowTopology,
        window_start: int,
        iteration: int,
        offset: int,
        arrays: Mapping[str, np.ndarray],
        manifest: Mapping[str, Any],
    ) -> None:
        ...


class LocalOracleWindowCoordinator:
    rank = 0
    world = 1
    cpu_process_group = None

    def gather_samples(self, packet: RankSamplePacket) -> Sequence[RankSamplePacket]:
        if int(packet.rank) != 0:
            raise ValueError("local coordinator requires rank zero")
        return (packet,)

    def gather_objects(self, value: Any) -> Sequence[Any]:
        return (value,)

    def sum_ints(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=np.int64).copy()


class TorchGlooOracleWindowCoordinator:
    """Small CPU-only collectives over the proposal Gloo process group."""

    def __init__(self, *, rank: int, world: int, process_group: Any) -> None:
        self.rank = int(rank)
        self.world = int(world)
        self.cpu_process_group = process_group
        if self.world < 2 or not 0 <= self.rank < self.world:
            raise ValueError("Gloo oracle coordinator requires a multi-rank world")
        if process_group is None:
            raise ValueError("Gloo oracle coordinator requires a process group")

    def gather_samples(self, packet: RankSamplePacket) -> Sequence[RankSamplePacket]:
        gathered: list[Any] = [None] * self.world
        torch.distributed.all_gather_object(
            gathered,
            packet,
            group=self.cpu_process_group,
        )
        if any(not isinstance(item, RankSamplePacket) for item in gathered):
            raise RuntimeError("oracle-window Gloo sample gather is incomplete")
        ranks = [int(item.rank) for item in gathered]
        if set(ranks) != set(range(self.world)):
            raise RuntimeError("oracle-window Gloo sample ranks are incomplete")
        return tuple(sorted(gathered, key=lambda item: int(item.rank)))

    def gather_objects(self, value: Any) -> Sequence[Any]:
        gathered: list[Any] = [None] * self.world
        torch.distributed.all_gather_object(
            gathered,
            value,
            group=self.cpu_process_group,
        )
        return tuple(gathered)

    def sum_ints(self, values: np.ndarray) -> np.ndarray:
        tensor = torch.from_numpy(
            np.asarray(values, dtype=np.int64).copy()
        )
        torch.distributed.all_reduce(
            tensor,
            op=torch.distributed.ReduceOp.SUM,
            group=self.cpu_process_group,
        )
        return tensor.numpy().copy()


class AtomicOracleWindowShardWriter:
    def write(
        self,
        *,
        settings: OracleWindowSettings,
        topology: OracleWindowTopology,
        window_start: int,
        iteration: int,
        offset: int,
        arrays: Mapping[str, np.ndarray],
        manifest: Mapping[str, Any],
    ) -> None:
        record = dict(manifest)
        record.update({
            "world": int(topology.world),
            "topk_ratio": float(settings.topk_ratio),
            "window_length": int(settings.window_updates),
            "sampling_seed": int(settings.sampling_seed),
            "min_effective_tail_count": float(
                settings.min_effective_tail_count
            ),
            "fd_epsilon": float(settings.fd_epsilon),
            "fd_probes": int(settings.fd_probes),
            "fd_data_batches": int(settings.fd_data_batches),
            "fd_min_changed_fraction": float(
                settings.fd_min_changed_fraction
            ),
            "optimizer_iteration": int(iteration),
            "dp_rank": int(topology.dp_rank),
            "dp_size": int(topology.dp_size),
            "pp_rank": int(topology.pp_rank),
            "pp_size": int(topology.pp_size),
            "ep_rank": int(topology.ep_rank),
            "ep_size": int(topology.ep_size),
            "state_phase": "pre_shadow_reset",
        })
        write_oracle_window_shard(
            output_dir=Path(settings.output_dir),
            window_start=int(window_start),
            offset=int(offset),
            step=int(iteration),
            rank=int(topology.rank),
            arrays=arrays,
            manifest=record,
        )


class CudaOracleWindowCompute:
    """Allocation-owning adapter around the bounded oracle GPU primitives."""

    def __init__(self, *, device_process_group: Any = None) -> None:
        self.device_process_group = device_process_group
        self.full_counts_are_global = device_process_group is not None
        self._layouts: dict[str, PackedFragmentLayout] = {}
        self._shadows: dict[str, PackedFp32Shadow] = {}
        self._plans: dict[str, Any] = {}
        self._samples: dict[str, SampleBuffers] = {}
        self._counts: dict[str, FullCountWorkspace] = {}
        self._selections: dict[str, SelectionCountWorkspace] = {}

    def initialize(
        self,
        *,
        layouts: Mapping[str, PackedFragmentLayout],
        settings: OracleWindowSettings,
    ) -> None:
        if self._layouts:
            raise RuntimeError("oracle CUDA compute initialized twice")
        for domain in DOMAIN_ORDER:
            layout = layouts[domain]
            plan = build_block_sampling_plan(
                layout,
                block_elements=int(settings.block_elements),
                sample_blocks=int(settings.sample_blocks),
            )
            device = layout.fragments[0].p.device
            self._layouts[domain] = layout
            self._shadows[domain] = PackedFp32Shadow.allocate(layout)
            self._plans[domain] = plan
            self._samples[domain] = SampleBuffers.allocate(
                plan.output_elements,
                device=device,
            )
            self._counts[domain] = FullCountWorkspace.allocate(device=device)
            self._selections[domain] = SelectionCountWorkspace.allocate(
                device=device
            )

    def begin(
        self,
        *,
        layouts: Mapping[str, PackedFragmentLayout],
    ) -> None:
        for domain in DOMAIN_ORDER:
            self._shadows[domain].begin(layouts[domain])

    def sample_round(
        self,
        *,
        domain: str,
        lut: Any,
        round_index: int,
        base_salt: int,
    ) -> SampleBatch:
        plan = self._plans[domain]
        output = self._samples[domain]
        # The public engine contract carries an explicit round index.  The
        # current primitive consumes the equivalent consecutive salt; its GPU
        # API can adopt round_index/base_salt without changing this collector.
        launch_sample_scores(
            layout=self._layouts[domain],
            shadow=self._shadows[domain],
            lut=lut,
            plan=plan,
            round_salt=(int(base_salt) + int(round_index)) & UINT32_MAX,
            output=output,
        )
        count = int(plan.output_elements)
        return SampleBatch(
            raw=output.raw.narrow(0, 0, count).detach().cpu().numpy().copy(),
            clipped=(
                output.clipped.narrow(0, 0, count).detach().cpu().numpy().copy()
            ),
            canonical_ids=(
                output.canonical_element_id.narrow(0, 0, count)
                .detach().cpu().numpy().copy()
            ),
            weights=np.asarray(plan.population_weights(), dtype=np.float64),
        ).validated()

    def full_count(self, *, domain: str, lut: Any, tau: float) -> FullCount:
        workspace = self._counts[domain]
        workspace.reset()
        launch_full_count(
            layout=self._layouts[domain],
            shadow=self._shadows[domain],
            lut=lut,
            tau=float(tau),
            workspace=workspace,
        )
        if self.device_process_group is not None:
            torch.distributed.all_reduce(
                workspace.counters,
                op=torch.distributed.ReduceOp.SUM,
                group=self.device_process_group,
            )
        values = workspace.counters.detach().cpu().numpy().astype(np.int64)
        return FullCount(*(int(value) for value in values))

    def update_selected(
        self,
        *,
        domain: str,
        lut: Any,
        tau: float,
        tie_hash_limit: int,
        tie_salt: int,
    ) -> int:
        workspace = self._selections[domain]
        workspace.reset()
        launch_update_selected(
            layout=self._layouts[domain],
            shadow=self._shadows[domain],
            lut=lut,
            tau=float(tau),
            tie_hash_limit=int(tie_hash_limit),
            tie_salt=int(tie_salt),
            workspace=workspace,
        )
        return int(workspace.count.detach().cpu().item())


def _deduplicate_packet(packet: RankSamplePacket) -> RankSamplePacket:
    domains = np.asarray(packet.sample_domain, dtype=np.int16).reshape(-1)
    rounds = np.asarray(packet.sample_round, dtype=np.int16).reshape(-1)
    ids = np.asarray(packet.sample_global_id, dtype=np.int64).reshape(-1)
    raw = np.asarray(packet.sample_raw, dtype=np.float32).reshape(-1)
    clipped = np.asarray(packet.sample_clipped, dtype=np.float32).reshape(-1)
    weights = np.asarray(packet.sample_weight, dtype=np.float64).reshape(-1)
    if not (
        domains.shape == rounds.shape == ids.shape == raw.shape
        == clipped.shape == weights.shape
    ):
        raise ValueError("oracle-window rank packet arrays are unaligned")
    keep: list[int] = []
    seen: dict[tuple[int, int], int] = {}
    aggregated_weights = weights.copy()
    for index, key in enumerate(zip(domains.tolist(), ids.tolist())):
        previous = seen.get(key)
        if previous is None:
            seen[key] = index
            keep.append(index)
            continue
        if not (
            math.isclose(float(raw[previous]), float(raw[index]), rel_tol=1e-5, abs_tol=1e-7)
            and math.isclose(
                float(clipped[previous]),
                float(clipped[index]),
                rel_tol=1e-5,
                abs_tol=1e-7,
            )
        ):
            raise RuntimeError(
                "duplicate canonical sample changed within one optimizer state"
            )
        # A tiny stratum can be exhausted before max_sample_rounds. Its later
        # rounds are the same observation, not independent tail evidence.
        # Preserve its full population mass on the first observation while
        # letting Kish effective-N count it once.
        aggregated_weights[previous] += float(weights[index])
    take = np.asarray(keep, dtype=np.int64)
    kept_domains = domains[take]
    kept_rounds = rounds[take]
    kept_weights = aggregated_weights[take]
    # Every original round carried 1/R of its fragment population. Summing
    # duplicates therefore preserves exact domain mass without pretending the
    # exhausted stratum supplied additional independent samples.
    for domain in range(len(DOMAIN_ORDER)):
        domain_mask = kept_domains == domain
        distinct_rounds = np.unique(kept_rounds[domain_mask])
        if distinct_rounds.size == 0:
            continue
        actual_mass = float(kept_weights[domain_mask].sum())
        expected_mass = float(packet.population_count[domain])
        if not math.isclose(actual_mass, expected_mass, rel_tol=1.0e-10, abs_tol=1.0e-8):
            raise RuntimeError(
                "deduplicated oracle samples do not preserve domain population: "
                f"domain={DOMAIN_ORDER[domain]} actual={actual_mass} "
                f"expected={expected_mass}"
            )
    result = RankSamplePacket(
        rank=int(packet.rank),
        population_count=np.asarray(packet.population_count, dtype=np.int64).copy(),
        sample_domain=kept_domains,
        sample_round=kept_rounds,
        sample_global_id=ids[take],
        sample_raw=raw[take],
        sample_clipped=clipped[take],
        sample_weight=kept_weights,
    )
    for domain in range(len(DOMAIN_ORDER)):
        mask = result.sample_domain == domain
        if int(result.population_count[domain]) > 0:
            distinct = np.unique(result.sample_round[mask])
            if distinct.size < 4:
                raise RuntimeError(
                    f"oracle-window domain {DOMAIN_ORDER[domain]} has fewer "
                    "than four distinct post-dedup sample rounds"
                )
    return result


def _global_cutoff(
    packets: Sequence[RankSamplePacket],
    *,
    topk_ratio: float,
) -> tuple[dict[str, float | int], int]:
    if not packets:
        raise RuntimeError("oracle-window global cutoff has no rank samples")
    ranks = [int(packet.rank) for packet in packets]
    if len(set(ranks)) != len(ranks):
        raise RuntimeError("oracle-window global sample gather duplicated a rank")
    values: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    identities: set[tuple[int, int]] = set()
    total_population = 0
    for packet in packets:
        populations = np.asarray(packet.population_count, dtype=np.int64).reshape(-1)
        domains = np.asarray(packet.sample_domain, dtype=np.int16).reshape(-1)
        rounds = np.asarray(packet.sample_round, dtype=np.int16).reshape(-1)
        ids = np.asarray(packet.sample_global_id, dtype=np.int64).reshape(-1)
        clipped = np.asarray(packet.sample_clipped, dtype=np.float32).reshape(-1)
        if populations.shape != (2,) or np.any(populations <= 0):
            raise RuntimeError("each oracle rank must own both measured domains")
        for key in zip(domains.tolist(), ids.tolist()):
            if key in identities:
                raise RuntimeError(
                    "canonical oracle sample is duplicated across optimizer ranks"
                )
            identities.add(key)
        values.append(clipped)
        packet_weights = np.asarray(
            packet.sample_weight,
            dtype=np.float64,
        ).reshape(-1)
        if packet_weights.shape != clipped.shape:
            raise RuntimeError("oracle global packet weights are unaligned")
        weights.append(packet_weights)
        total_population += int(populations.sum())
    cutoff = weighted_cutoff(
        np.concatenate(values),
        np.concatenate(weights),
        topk_ratio=float(topk_ratio),
    )
    return cutoff, int(total_population)


def _tie_hash_limit(ties_needed: int, equal_count: int) -> int:
    need, equal = int(ties_needed), int(equal_count)
    if need <= 0 or equal <= 0:
        return -1
    if need >= equal:
        return UINT32_MAX
    # Hash predicate is inclusive, hence subtract one from the desired number
    # of 2^32 buckets.
    return max(-1, min(UINT32_MAX, (need * (UINT32_MAX + 1)) // equal - 1))


class OracleWindowEngine:
    """Collector attached to :class:`OracleWindowSession`."""

    def __init__(
        self,
        *,
        settings: OracleWindowSettings,
        topology: OracleWindowTopology,
        layout_provider: OracleWindowLayoutProvider,
        curvature_provider: OracleWindowCurvatureProvider,
        compute: OracleWindowCompute,
        coordinator: OracleWindowCoordinator,
        device_process_group: Any = None,
        writer: OracleWindowShardWriter | None = None,
    ) -> None:
        self.settings = settings.validated()
        self.topology = topology.validated()
        if int(coordinator.rank) != self.topology.rank:
            raise ValueError("oracle coordinator rank disagrees with topology")
        if int(coordinator.world) != self.topology.world:
            raise ValueError("oracle coordinator world disagrees with topology")
        self.layout_provider = layout_provider
        self.curvature_provider = curvature_provider
        self.compute = compute
        self.coordinator = coordinator
        self.device_process_group = device_process_group
        if (
            self.topology.world > 1
            and isinstance(compute, CudaOracleWindowCompute)
            and device_process_group is None
        ):
            raise ValueError(
                "multi-rank CUDA oracle requires a device/NCCL process group"
            )
        self.writer = writer or AtomicOracleWindowShardWriter()
        self._layouts: Mapping[str, PackedFragmentLayout] | None = None
        self._luts: Mapping[str, Any] | None = None
        self._active_start: int | None = None
        self._last_offset = 0
        self._last_iteration: int | None = None
        self._written_offsets: set[int] = set()
        self._curvature_provenance: Mapping[str, Any] | None = None

    def _ensure_layouts(self, model: Any) -> Mapping[str, PackedFragmentLayout]:
        if self._layouts is None:
            layouts: dict[str, PackedFragmentLayout] | None = None
            local_error = ""
            try:
                layouts = dict(self.layout_provider.build_layouts(model=model))
                if set(layouts) != set(DOMAIN_ORDER):
                    raise RuntimeError(
                        "oracle layout provider must return expert and nonexpert"
                    )
                if any(int(layouts[name].total_numel) <= 0 for name in DOMAIN_ORDER):
                    raise RuntimeError("oracle measured domains must be non-empty")
            except BaseException as exc:
                local_error = f"{type(exc).__name__}: {exc}"
            statuses = self.coordinator.gather_objects(
                (int(self.topology.rank), not bool(local_error), local_error)
            )
            failures = [item for item in statuses if not bool(item[1])]
            if failures:
                raise RuntimeError(
                    "oracle layout preflight failed on distributed rank(s): "
                    + "; ".join(f"rank{item[0]}={item[2]}" for item in failures)
                )
            if layouts is None:
                raise AssertionError("successful oracle layout preflight has no layout")
            if set(layouts) != set(DOMAIN_ORDER):
                raise RuntimeError(
                    "oracle layout provider must return expert and nonexpert"
                )
            local_records = tuple(
                record
                for domain in DOMAIN_ORDER
                for record in layouts[domain].coverage_records(
                    owner_rank=int(self.topology.rank)
                )
            )
            gathered_records = self.coordinator.gather_objects(local_records)
            validate_fragment_coverage(
                (
                    record
                    for rank_records in gathered_records
                    for record in rank_records
                ),
                require_complete=True,
            )
            init_error = ""
            try:
                self.compute.initialize(layouts=layouts, settings=self.settings)
            except BaseException as exc:
                init_error = f"{type(exc).__name__}: {exc}"
            init_statuses = self.coordinator.gather_objects(
                (int(self.topology.rank), not bool(init_error), init_error)
            )
            init_failures = [item for item in init_statuses if not bool(item[1])]
            if init_failures:
                raise RuntimeError(
                    "oracle compute initialization failed on distributed rank(s): "
                    + "; ".join(
                        f"rank{item[0]}={item[2]}" for item in init_failures
                    )
                )
            self._layouts = layouts
        return self._layouts

    def consensus_applied(self, *, applied: bool) -> bool:
        """Require every measurement rank to agree on optimizer-step success."""

        totals = self.coordinator.sum_ints(
            np.asarray([int(bool(applied)), 1], dtype=np.int64)
        )
        if totals.shape != (2,) or int(totals[1]) != int(self.topology.world):
            raise RuntimeError("oracle applied-step consensus world mismatch")
        applied_count = int(totals[0])
        if applied_count not in (0, int(self.topology.world)):
            raise RuntimeError(
                "optimizer skip diverged across oracle measurement ranks: "
                f"applied={applied_count}/{self.topology.world}"
            )
        return applied_count == int(self.topology.world)

    def curvature_stage_consensus(
        self,
        stage: str,
        local_error: BaseException | None,
    ) -> None:
        """Fail every rank before the next curvature device collective.

        Curvature preparation contains rank-local discovery/validation followed
        by NCCL histogram reductions.  A local exception must therefore be
        exchanged over the independent CPU group before any rank is allowed to
        enter the next device collective.
        """

        stage_name = str(stage).strip()
        if not stage_name:
            raise ValueError("oracle curvature stage name must be non-empty")
        detail = (
            ""
            if local_error is None
            else f"{type(local_error).__name__}: {local_error}"
        )
        gathered = self.coordinator.gather_objects((
            int(self.topology.rank),
            stage_name,
            local_error is None,
            detail,
        ))
        if len(gathered) != int(self.topology.world):
            raise RuntimeError(
                "oracle curvature preflight consensus world mismatch: "
                f"{len(gathered)}/{self.topology.world}"
            )
        ranks = [int(item[0]) for item in gathered]
        if set(ranks) != set(range(int(self.topology.world))):
            raise RuntimeError("oracle curvature preflight ranks are incomplete")
        mismatched = [
            item for item in gathered if str(item[1]) != stage_name
        ]
        if mismatched:
            raise RuntimeError(
                "oracle curvature ranks entered different preflight stages: "
                + "; ".join(
                    f"rank{item[0]}={item[1]}" for item in mismatched
                )
            )
        failures = [item for item in gathered if not bool(item[2])]
        if failures:
            raise RuntimeError(
                f"oracle curvature stage {stage_name!r} failed before device "
                "collectives: "
                + "; ".join(
                    f"rank{item[0]}={item[3]}" for item in failures
                )
            )

    def prepare_window(
        self,
        *,
        model: Any,
        forward_step_func: Any,
        data_iterator: Any,
        start_iteration: int,
    ) -> None:
        """Acquire curvature then initialize e=0 at T0.

        The session calls this after the normal T0 H refresh. No score is
        sampled and no selection occurs at this offset-zero boundary.
        """

        if self._active_start is not None:
            raise RuntimeError("oracle engine window overlap")
        layouts = self._ensure_layouts(model)
        luts: dict[str, Any] | None = None
        curvature_error = ""
        try:
            luts = dict(self.curvature_provider.prepare_window(
                model=model,
                forward_step_func=forward_step_func,
                data_iterator=data_iterator,
                start_iteration=int(start_iteration),
                layouts=layouts,
                device_process_group=self.device_process_group,
                stage_consensus=self.curvature_stage_consensus,
            ))
        except BaseException as exc:
            curvature_error = f"{type(exc).__name__}: {exc}"
        statuses = self.coordinator.gather_objects(
            (int(self.topology.rank), not bool(curvature_error), curvature_error)
        )
        failures = [item for item in statuses if not bool(item[1])]
        if failures:
            raise RuntimeError(
                "oracle curvature preparation failed on distributed rank(s): "
                + "; ".join(f"rank{item[0]}={item[2]}" for item in failures)
            )
        if luts is None:
            raise AssertionError("successful oracle curvature preparation has no LUTs")
        if set(luts) != set(DOMAIN_ORDER):
            raise RuntimeError(
                "oracle curvature provider must return expert and nonexpert LUTs"
            )
        provenance_reader = getattr(self.curvature_provider, "provenance", None)
        if callable(provenance_reader):
            provenance = dict(provenance_reader())
        else:
            provenance = {
                domain: {
                    "estimator": "injected_test_provider",
                    "epsilon": float("nan"),
                }
                for domain in DOMAIN_ORDER
            }
        if set(provenance) != set(DOMAIN_ORDER):
            raise RuntimeError(
                "oracle curvature provenance must contain expert and nonexpert"
            )
        self._curvature_provenance = provenance
        begin_error = ""
        try:
            self.compute.begin(layouts=layouts)
        except BaseException as exc:
            begin_error = f"{type(exc).__name__}: {exc}"
        begin_statuses = self.coordinator.gather_objects(
            (int(self.topology.rank), not bool(begin_error), begin_error)
        )
        begin_failures = [item for item in begin_statuses if not bool(item[1])]
        if begin_failures:
            raise RuntimeError(
                "oracle shadow initialization failed on distributed rank(s): "
                + "; ".join(
                    f"rank{item[0]}={item[2]}" for item in begin_failures
                )
            )
        self._luts = luts
        self._active_start = int(start_iteration)
        self._last_offset = 0
        self._last_iteration = None
        self._written_offsets.clear()

    def _sample_packet(self, *, offset: int, rounds: int) -> RankSamplePacket:
        assert self._layouts is not None and self._luts is not None
        raw_parts: list[np.ndarray] = []
        clipped_parts: list[np.ndarray] = []
        id_parts: list[np.ndarray] = []
        weight_parts: list[np.ndarray] = []
        domain_parts: list[np.ndarray] = []
        round_parts: list[np.ndarray] = []
        base_salt = (
            int(self.settings.sampling_seed)
            ^ (int(self._active_start or 0) * 0x9E3779B1)
            ^ (int(offset) * 0x85EBCA77)
        ) & UINT32_MAX
        for round_index in range(int(rounds)):
            for domain_index, domain in enumerate(DOMAIN_ORDER):
                batch = self.compute.sample_round(
                    domain=domain,
                    lut=self._luts[domain],
                    round_index=int(round_index),
                    base_salt=int(base_salt),
                ).validated()
                n = int(batch.raw.size)
                raw_parts.append(batch.raw)
                clipped_parts.append(batch.clipped)
                id_parts.append(batch.canonical_ids)
                # The plan's fragment-aware population mass is divided equally
                # across distinct rounds, preserving both fragment and domain
                # population weighting.
                weight_parts.append(batch.weights / float(rounds))
                domain_parts.append(
                    np.full(n, domain_index, dtype=np.int16)
                )
                round_parts.append(
                    np.full(n, round_index, dtype=np.int16)
                )
        return _deduplicate_packet(RankSamplePacket(
            rank=int(self.topology.rank),
            population_count=np.asarray(
                [self._layouts[name].total_numel for name in DOMAIN_ORDER],
                dtype=np.int64,
            ),
            sample_domain=np.concatenate(domain_parts),
            sample_round=np.concatenate(round_parts),
            sample_global_id=np.concatenate(id_parts),
            sample_raw=np.concatenate(raw_parts),
            sample_clipped=np.concatenate(clipped_parts),
            sample_weight=np.concatenate(weight_parts),
        ))

    def _solve_cutoff(
        self,
        *,
        offset: int,
    ) -> tuple[CutoffDecision, RankSamplePacket, np.ndarray]:
        assert self._luts is not None
        last_detail = ""
        for rounds in range(
            int(self.settings.sample_rounds),
            int(self.settings.max_sample_rounds) + 1,
        ):
            local_packet = self._sample_packet(offset=offset, rounds=rounds)
            packets = self.coordinator.gather_samples(local_packet)
            cutoff, population = _global_cutoff(
                packets,
                topk_ratio=float(self.settings.topk_ratio),
            )
            target = max(1, min(
                int(population),
                int(round(float(self.settings.topk_ratio) * population)),
            ))
            tolerance = max(
                1,
                int(math.ceil(target * float(self.settings.cutoff_tolerance))),
            )
            band_low = max(0, target - tolerance)
            band_high = min(population, target + tolerance)
            local_counts = np.stack([
                self.compute.full_count(
                    domain=domain,
                    lut=self._luts[domain],
                    tau=float(cutoff["tau"]),
                ).as_array()
                for domain in DOMAIN_ORDER
            ])
            global_counts = (
                local_counts
                if bool(getattr(self.compute, "full_counts_are_global", False))
                else self.coordinator.sum_ints(local_counts)
            )
            if global_counts.shape != (2, 6):
                raise RuntimeError("oracle global count reduction shape mismatch")
            if int(global_counts[:, FullCountWorkspace.NONFINITE].sum()) != 0:
                raise RuntimeError("oracle full scan found non-finite saliency")
            greater = int(global_counts[:, FullCountWorkspace.GREATER].sum())
            equal = int(global_counts[:, FullCountWorkspace.EQUAL].sum())
            if greater <= band_high and greater + equal >= band_low:
                desired = min(max(target, greater), greater + equal)
                tie_salt = (
                    int(self.settings.sampling_seed)
                    ^ (int(self._active_start or 0) * 0xC2B2AE3D)
                    ^ (int(offset) * 0x27D4EB2F)
                ) & UINT32_MAX
                return (
                    CutoffDecision(
                        tau=float(cutoff["tau"]),
                        target_count=int(target),
                        band_low=int(band_low),
                        band_high=int(band_high),
                        greater_count=int(greater),
                        equal_count=int(equal),
                        tie_hash_limit=_tie_hash_limit(desired - greater, equal),
                        tie_salt=int(tie_salt),
                        sample_rounds=int(rounds),
                    ),
                    local_packet,
                    global_counts,
                )
            last_detail = (
                f"rounds={rounds} tau={float(cutoff['tau']):.9g} "
                f"greater={greater} equal={equal} band=[{band_low},{band_high}]"
            )
        raise RuntimeError(
            "oracle cutoff refinement exhausted without entering K-band: "
            + last_detail
        )

    def observe_successful_update(
        self,
        *,
        model: Any,
        start_iteration: int,
        iteration: int,
        offset: int,
        snapshot_due: bool,
    ) -> None:
        del model
        if self._active_start != int(start_iteration) or self._luts is None:
            raise RuntimeError("oracle update observed without a prepared window")
        if int(offset) != self._last_offset + 1:
            raise RuntimeError("oracle successful-update offsets are not contiguous")
        if self._last_iteration is not None and int(iteration) <= self._last_iteration:
            raise RuntimeError("oracle optimizer iterations are not increasing")

        decision, packet, exact_counts = self._solve_cutoff(offset=int(offset))
        # packet owns CPU copies of PRE-reset raw/clipped samples. Shadow state
        # may now advance without changing the snapshot payload.
        local_selected = np.asarray([
            self.compute.update_selected(
                domain=domain,
                lut=self._luts[domain],
                tau=decision.tau,
                tie_hash_limit=decision.tie_hash_limit,
                tie_salt=decision.tie_salt,
            )
            for domain in DOMAIN_ORDER
        ], dtype=np.int64)
        global_selected_by_domain = self.coordinator.sum_ints(local_selected)
        actual = int(global_selected_by_domain.sum())
        if not decision.band_low <= actual <= decision.band_high:
            raise RuntimeError(
                "oracle deterministic tie selection missed K-band after reset: "
                f"actual={actual} band=[{decision.band_low},{decision.band_high}]"
            )
        if np.any(local_selected < 0) or np.any(
            local_selected > packet.population_count
        ):
            raise RuntimeError("oracle local selected counts are invalid")

        if bool(snapshot_due):
            arrays = build_rank_arrays(
                population_count=packet.population_count,
                sample_domain=packet.sample_domain,
                sample_round=packet.sample_round,
                sample_global_id=packet.sample_global_id,
                sample_raw=packet.sample_raw,
                sample_clipped=packet.sample_clipped,
                sample_weight=packet.sample_weight,
                selected_count=local_selected,
                tau=decision.tau,
                target_count=decision.target_count,
                min_effective_tail_count=(
                    self.settings.min_effective_tail_count
                ),
            )
            self.writer.write(
                settings=self.settings,
                topology=self.topology,
                window_start=int(start_iteration),
                iteration=int(iteration),
                offset=int(offset),
                arrays=arrays,
                manifest={
                    "sample_rounds": int(decision.sample_rounds),
                    "cutoff_band_low": int(decision.band_low),
                    "cutoff_band_high": int(decision.band_high),
                    "global_greater_count": int(decision.greater_count),
                    "global_equal_count": int(decision.equal_count),
                    "global_actual_selected_count": int(actual),
                    "tie_hash_limit": int(decision.tie_hash_limit),
                    "tie_salt": int(decision.tie_salt),
                    "global_domain_full_counts": exact_counts.tolist(),
                    "curvature_provenance": self._curvature_provenance,
                },
            )
            self._written_offsets.add(int(offset))

        self._last_offset = int(offset)
        self._last_iteration = int(iteration)

    def finish_window(
        self,
        *,
        model: Any,
        start_iteration: int,
        end_iteration: int,
    ) -> None:
        del model
        if self._active_start != int(start_iteration):
            raise RuntimeError("oracle finish does not match the active window")
        if self._last_offset != int(self.settings.window_updates):
            raise RuntimeError(
                "oracle window ended before all successful updates were observed"
            )
        expected = set(int(value) for value in self.settings.snapshot_offsets)
        if self._written_offsets != expected:
            raise RuntimeError(
                "oracle window snapshot set is incomplete: "
                f"written={sorted(self._written_offsets)} expected={sorted(expected)}"
            )
        if self._last_iteration != int(end_iteration):
            raise RuntimeError("oracle finish iteration disagrees with final update")
        self._active_start = None
        self._luts = None
        self._curvature_provenance = None
        self._last_offset = 0
        self._last_iteration = None
        self._written_offsets.clear()


def build_rank_owned_oracle_window_engine(
    *,
    config: Any,
    topology: Any,
    mapper: Zero12FragmentMapper,
    expert_specs_provider: Callable[[], Iterable[Any]],
    nonexpert_parameters_provider: Callable[
        [], Iterable[tuple[str, torch.nn.Parameter]]
    ],
    curvature_provider: OracleWindowCurvatureProvider,
    cpu_process_group: Any = None,
    device_process_group: Any = None,
    writer: OracleWindowShardWriter | None = None,
) -> OracleWindowEngine:
    """Construct the live collector after optimizer ownership is initialized.

    CPU object/sample collectives use the Gloo group. CUDA LUT/count
    collectives use the separate device group; the two are never
    interchangeable.
    """

    oracle_topology = OracleWindowTopology.from_object(topology)
    if oracle_topology.world == 1:
        coordinator: OracleWindowCoordinator = LocalOracleWindowCoordinator()
    else:
        coordinator = TorchGlooOracleWindowCoordinator(
            rank=int(oracle_topology.rank),
            world=int(oracle_topology.world),
            process_group=cpu_process_group,
        )
        if device_process_group is None:
            raise ValueError(
                "multi-rank oracle factory requires a CUDA/device process group"
            )
    layout_provider = RankOwnedOracleWindowLayoutProvider(
        mapper=mapper,
        expert_specs_provider=expert_specs_provider,
        nonexpert_parameters_provider=nonexpert_parameters_provider,
        pp_rank=int(oracle_topology.pp_rank),
    )
    compute = CudaOracleWindowCompute(
        device_process_group=device_process_group,
    )
    return OracleWindowEngine(
        settings=OracleWindowSettings.from_config(config),
        topology=oracle_topology,
        layout_provider=layout_provider,
        curvature_provider=curvature_provider,
        compute=compute,
        coordinator=coordinator,
        device_process_group=device_process_group,
        writer=writer,
    )


__all__ = [
    "AtomicOracleWindowShardWriter",
    "CudaOracleWindowCompute",
    "CutoffDecision",
    "DOMAIN_ORDER",
    "FullCount",
    "LocalOracleWindowCoordinator",
    "LiveOracleWindowCurvatureProvider",
    "OracleWindowCompute",
    "OracleWindowCoordinator",
    "OracleWindowCurvatureProvider",
    "OracleWindowEngine",
    "OracleWindowLayoutProvider",
    "OracleWindowSettings",
    "OracleWindowShardWriter",
    "OracleWindowTopology",
    "RankSamplePacket",
    "RankOwnedOracleWindowLayoutProvider",
    "SampleBatch",
    "TorchGlooOracleWindowCoordinator",
    "build_rank_owned_oracle_window_engine",
]
