"""CPU contracts for the experiment-only oracle saliency window."""

from .stats import (
    PERCENTILE_LABELS,
    PERCENTILE_PROBABILITIES,
    aggregate_sample_rounds,
    compute_saliency,
    deterministic_round_indices,
    population_weights,
    summarize_by_domain,
    weighted_cutoff,
    weighted_percentiles,
)
from .session import OracleWindowCollector, OracleWindowSession
from .engine import (
    AtomicOracleWindowShardWriter,
    CudaOracleWindowCompute,
    LocalOracleWindowCoordinator,
    OracleWindowEngine,
    OracleWindowSettings,
    OracleWindowTopology,
    RankOwnedOracleWindowLayoutProvider,
    TorchGlooOracleWindowCoordinator,
    build_rank_owned_oracle_window_engine,
)

__all__ = [
    "PERCENTILE_LABELS",
    "PERCENTILE_PROBABILITIES",
    "aggregate_sample_rounds",
    "compute_saliency",
    "deterministic_round_indices",
    "population_weights",
    "summarize_by_domain",
    "weighted_cutoff",
    "weighted_percentiles",
    "OracleWindowCollector",
    "OracleWindowSession",
    "AtomicOracleWindowShardWriter",
    "CudaOracleWindowCompute",
    "LocalOracleWindowCoordinator",
    "OracleWindowEngine",
    "OracleWindowSettings",
    "OracleWindowTopology",
    "RankOwnedOracleWindowLayoutProvider",
    "TorchGlooOracleWindowCoordinator",
    "build_rank_owned_oracle_window_engine",
]
