"""Lifecycle-only orchestration for the measurement oracle window.

This module intentionally knows nothing about CUDA, Hessian estimation, LUTs,
or persistence.  A collector owns those responsibilities and is injected by
the measurement engine at proposal initialization.
"""

from __future__ import annotations

from typing import Any, Optional, Protocol, Sequence, Tuple, runtime_checkable


@runtime_checkable
class OracleWindowCollector(Protocol):
    """Backend invoked at the three lifecycle boundaries.

    ``observe_successful_update`` must take any requested snapshot before it
    resets its shadow checkpoint state for the next update.
    """

    def prepare_window(
        self,
        *,
        model: Any,
        forward_step_func: Any,
        data_iterator: Any,
        start_iteration: int,
    ) -> None:
        """Prepare curvature and initialize both shadows at offset zero."""

    def observe_successful_update(
        self,
        *,
        model: Any,
        start_iteration: int,
        iteration: int,
        offset: int,
        snapshot_due: bool,
    ) -> None:
        """Observe one applied optimizer update, snapshotting before reset."""

    def finish_window(
        self,
        *,
        model: Any,
        start_iteration: int,
        end_iteration: int,
    ) -> None:
        """Finish a complete window after its final successful update."""


class OracleWindowSession:
    """Fail-closed state machine for independent-oracle measurement windows."""

    def __init__(
        self,
        *,
        start_iterations: Sequence[int],
        window_updates: int = 100,
        snapshot_offsets: Sequence[int] = tuple(range(10, 101, 10)),
    ) -> None:
        starts = tuple(int(value) for value in start_iterations)
        updates = int(window_updates)
        offsets = tuple(int(value) for value in snapshot_offsets)
        if updates < 1:
            raise ValueError("oracle window_updates must be >= 1")
        if not starts:
            raise ValueError("oracle window requires explicit start iterations")
        if any(value < 0 for value in starts):
            raise ValueError("oracle window start iterations must be >= 0")
        if tuple(sorted(set(starts))) != starts:
            raise ValueError(
                "oracle window start iterations must be sorted and unique"
            )
        if not offsets:
            raise ValueError("oracle window requires at least one snapshot offset")
        if tuple(sorted(set(offsets))) != offsets:
            raise ValueError("oracle window snapshot offsets must be sorted and unique")
        if offsets[0] < 1 or offsets[-1] > updates:
            raise ValueError(
                "oracle window snapshot offsets must lie in [1, window_updates]"
            )
        for previous, current in zip(starts, starts[1:]):
            if current - previous <= updates:
                raise ValueError(
                    "oracle measurement windows overlap: "
                    f"start {previous} covers {previous + 1}..{previous + updates}, "
                    f"next start is {current}"
                )

        self._start_iterations: Tuple[int, ...] = starts
        self._start_set = frozenset(starts)
        self._window_updates = updates
        self._snapshot_offsets = frozenset(offsets)
        self._collector: Optional[OracleWindowCollector] = None
        self._active_start: Optional[int] = None
        self._updates_observed = 0
        self._last_observed_iteration: Optional[int] = None
        self._started = set()

    @property
    def active(self) -> bool:
        return self._active_start is not None

    @property
    def active_start(self) -> Optional[int]:
        return self._active_start

    @property
    def updates_observed(self) -> int:
        return self._updates_observed

    @property
    def start_iterations(self) -> Tuple[int, ...]:
        return self._start_iterations

    def attach_collector(self, collector: OracleWindowCollector) -> None:
        """Attach exactly one backend before any configured window starts."""

        if self.active or self._started:
            raise RuntimeError("cannot attach oracle collector after a window starts")
        if self._collector is not None and self._collector is not collector:
            raise RuntimeError("oracle window collector is already attached")
        if not isinstance(collector, OracleWindowCollector):
            raise TypeError("collector does not implement OracleWindowCollector")
        self._collector = collector

    def assert_complete(self) -> None:
        """Fail closed if training ends with a partial or never-started window."""

        if self.active:
            raise RuntimeError(
                "training ended with an active oracle measurement window: "
                f"start={self._active_start} successful_updates={self._updates_observed}/"
                f"{self._window_updates}"
            )
        missing = sorted(set(self._start_iterations) - self._started)
        if missing:
            raise RuntimeError(
                "training ended before configured oracle window(s) started: "
                f"{missing}"
            )

    def maybe_start(
        self,
        *,
        model: Any,
        forward_step_func: Any,
        data_iterator: Any,
        iteration: int,
    ) -> bool:
        """Start a scheduled window at offset zero without selecting anything."""

        iteration = int(iteration)
        if iteration not in self._start_set:
            return False
        if iteration in self._started:
            raise RuntimeError(f"oracle window at iteration {iteration} started twice")
        if self.active:
            raise RuntimeError(
                "oracle measurement windows overlap at runtime: "
                f"active_start={self._active_start} new_start={iteration}"
            )
        if self._collector is None:
            raise RuntimeError(
                "oracle window measurement mode has no attached collector"
            )

        self._collector.prepare_window(
            model=model,
            forward_step_func=forward_step_func,
            data_iterator=data_iterator,
            start_iteration=iteration,
        )
        self._active_start = iteration
        self._updates_observed = 0
        self._last_observed_iteration = None
        self._started.add(iteration)
        return True

    def observe_post_optimizer(
        self,
        *,
        model: Any,
        iteration: int,
        applied: bool,
    ) -> bool:
        """Observe an applied post-T0 optimizer update.

        Skipped/overflowed optimizer steps do not advance the window offset.
        The training loop must call this before ``maybe_start`` for the same
        iteration so offset zero remains a pure reset point.
        """

        if not self.active:
            return False
        if self._collector is None:  # Defensive; start already enforces this.
            raise RuntimeError("oracle window collector detached while active")
        consensus = getattr(self._collector, "consensus_applied", None)
        if callable(consensus):
            applied = bool(consensus(applied=bool(applied)))
        if not bool(applied):
            return False
        assert self._active_start is not None
        iteration = int(iteration)
        if iteration <= self._active_start:
            raise RuntimeError(
                "oracle window cannot observe its offset-zero iteration: "
                f"start={self._active_start} iteration={iteration}"
            )
        if (
            self._last_observed_iteration is not None
            and iteration <= self._last_observed_iteration
        ):
            raise RuntimeError(
                "oracle window optimizer observations must be strictly increasing"
            )
        offset = self._updates_observed + 1
        snapshot_due = offset in self._snapshot_offsets
        self._collector.observe_successful_update(
            model=model,
            start_iteration=self._active_start,
            iteration=iteration,
            offset=offset,
            snapshot_due=snapshot_due,
        )
        self._updates_observed = offset
        self._last_observed_iteration = iteration

        if offset == self._window_updates:
            start_iteration = self._active_start
            self._collector.finish_window(
                model=model,
                start_iteration=start_iteration,
                end_iteration=iteration,
            )
            self._active_start = None
            self._updates_observed = 0
            self._last_observed_iteration = None
        return True


__all__ = ["OracleWindowCollector", "OracleWindowSession"]
