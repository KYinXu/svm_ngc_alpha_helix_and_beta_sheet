"""Sliding-window residue profiles for one fixed SVM."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

Scorer = Callable[[Sequence[str]], tuple[np.ndarray, np.ndarray]]


@dataclass(frozen=True)
class WindowSettings:
    min_len: int = 10
    max_len: int = 35
    stride: int = 1
    precision: int = 6
    batch_starts: int = 64


@dataclass(frozen=True)
class Window:
    start: int
    end: int
    sequence: str


METRIC_COLUMNS = (
    "sequence_length",
    "window_min_len",
    "window_max_len",
    "stride",
    "window_count",
    "profile_aggregation",
    "profile_index_base",
    "profile_delimiter",
    "p_amp_mean",
    "p_amp_max_residue_index",
    "hyperplane_distance_mean",
    "hyperplane_distance_max_residue_index",
    "best_window_p_amp",
    "best_window_hyperplane_distance",
    "best_window_start_0based",
    "best_window_end_0based_exclusive",
    "best_window_start_index",
    "best_window_end_index_inclusive",
    "best_window_length",
    "best_window_sequence",
    "p_amp_mean_profile",
    "hyperplane_distance_mean_profile",
    "p_amp_max_profile",
    "hyperplane_distance_max_profile",
)


def validate_settings(settings: WindowSettings) -> None:
    if settings.min_len < 1:
        raise ValueError("--window-min-len must be at least 1.")
    if settings.max_len < settings.min_len:
        raise ValueError("--window-max-len must be greater than or equal to --window-min-len.")
    if settings.stride < 1:
        raise ValueError("--stride must be at least 1.")


def profile_fieldnames(extra_columns: Sequence[str]) -> list[str]:
    known = {"id", *METRIC_COLUMNS}
    extras = [name if name not in known else f"label_{name}" for name in extra_columns]
    return ["id", *extras, *METRIC_COLUMNS]


def profile_sequence(
    record_id: str,
    sequence: str,
    extras: dict[str, str],
    score_windows: Scorer,
    settings: WindowSettings,
) -> dict[str, object]:
    length = len(sequence)
    accumulator = _Accumulator(length)
    best = _BestWindow()
    window_count = 0
    for windows in iter_window_batches(sequence, settings):
        p_amp, sigma = score_windows([window.sequence for window in windows])
        accumulator.update(windows, p_amp, sigma)
        _update_best(windows, p_amp, sigma, best)
        window_count += len(windows)
    return _output_row(record_id, sequence, extras, accumulator, best, window_count, settings)


def iter_window_batches(sequence: str, settings: WindowSettings) -> list[list[Window]]:
    length = len(sequence)
    if length < settings.min_len:
        return []
    starts = range(0, length - settings.min_len + 1, settings.stride)
    batches: list[list[Window]] = []
    pending: list[Window] = []
    for index, start in enumerate(starts, start=1):
        pending.extend(_windows_at_start(sequence, start, settings.min_len, settings.max_len))
        if index % settings.batch_starts == 0:
            batches.append(pending)
            pending = []
    if pending:
        batches.append(pending)
    return batches


def _windows_at_start(sequence: str, start: int, min_len: int, max_len: int) -> list[Window]:
    windows = []
    for window_len in range(min_len, max_len + 1):
        end = start + window_len
        if end <= len(sequence):
            windows.append(Window(start=start, end=end, sequence=sequence[start:end]))
    return windows


class _Accumulator:
    def __init__(self, length: int) -> None:
        self.sum_p = np.zeros(length, dtype=np.float64)
        self.sum_d = np.zeros(length, dtype=np.float64)
        self.max_p = np.full(length, -np.inf, dtype=np.float64)
        self.max_d = np.full(length, -np.inf, dtype=np.float64)
        self.coverage = np.zeros(length, dtype=np.int32)

    def update(self, windows: Sequence[Window], p_amp: np.ndarray, sigma: np.ndarray) -> None:
        if len(windows) != len(p_amp) or len(windows) != len(sigma):
            raise ValueError("Window and score counts do not match.")
        for window, probability, distance in zip(windows, p_amp, sigma):
            if not np.isfinite(probability) or not np.isfinite(distance):
                continue
            span = slice(window.start, window.end)
            self.sum_p[span] += float(probability)
            self.sum_d[span] += float(distance)
            np.maximum(self.max_p[span], float(probability), out=self.max_p[span])
            np.maximum(self.max_d[span], float(distance), out=self.max_d[span])
            self.coverage[span] += 1

    def profiles(self) -> dict[str, np.ndarray]:
        covered = self.coverage > 0
        mean_p = np.full(self.coverage.shape, np.nan, dtype=np.float64)
        mean_d = np.full(self.coverage.shape, np.nan, dtype=np.float64)
        max_p = np.full(self.coverage.shape, np.nan, dtype=np.float64)
        max_d = np.full(self.coverage.shape, np.nan, dtype=np.float64)
        mean_p[covered] = self.sum_p[covered] / self.coverage[covered]
        mean_d[covered] = self.sum_d[covered] / self.coverage[covered]
        max_p[covered] = self.max_p[covered]
        max_d[covered] = self.max_d[covered]
        return {
            "p_amp_mean": mean_p,
            "hyperplane_distance_mean": mean_d,
            "p_amp_max": max_p,
            "hyperplane_distance_max": max_d,
        }


@dataclass
class _BestWindow:
    p_amp: float = float("-inf")
    distance: float = float("nan")
    start: int = -1
    end: int = -1
    sequence: str = ""


def _update_best(windows: Sequence[Window], p_amp: np.ndarray, sigma: np.ndarray, best: _BestWindow) -> None:
    for window, probability, distance in zip(windows, p_amp, sigma):
        if float(probability) > best.p_amp:
            best.p_amp = float(probability)
            best.distance = float(distance)
            best.start = window.start
            best.end = window.end
            best.sequence = window.sequence


def _output_row(
    record_id: str,
    sequence: str,
    extras: dict[str, str],
    accumulator: _Accumulator,
    best: _BestWindow,
    window_count: int,
    settings: WindowSettings,
) -> dict[str, object]:
    profiles = accumulator.profiles()
    p_amp_max_index = _finite_argmax(profiles["p_amp_max"])
    distance_max_index = _finite_argmax(profiles["hyperplane_distance_max"])
    known = {"id", *METRIC_COLUMNS}
    row: dict[str, object] = {
        "id": record_id,
        "sequence_length": len(sequence),
        "window_min_len": settings.min_len,
        "window_max_len": settings.max_len,
        "stride": settings.stride,
        "window_count": window_count,
        "profile_aggregation": "both_over_covering_windows",
        "profile_index_base": 1,
        "profile_delimiter": ";",
        "p_amp_mean": _finite_mean(profiles["p_amp_mean"]),
        "p_amp_max_residue_index": None if p_amp_max_index is None else p_amp_max_index + 1,
        "hyperplane_distance_mean": _finite_mean(profiles["hyperplane_distance_mean"]),
        "hyperplane_distance_max_residue_index": None if distance_max_index is None else distance_max_index + 1,
        "best_window_p_amp": _finite_or_none(best.p_amp),
        "best_window_hyperplane_distance": _finite_or_none(best.distance),
        "best_window_start_0based": None if best.start < 0 else best.start,
        "best_window_end_0based_exclusive": None if best.end < 0 else best.end,
        "best_window_start_index": None if best.start < 0 else best.start + 1,
        "best_window_end_index_inclusive": None if best.end < 0 else best.end,
        "best_window_length": None if best.end < 0 else best.end - best.start,
        "best_window_sequence": best.sequence,
        "p_amp_mean_profile": _format_profile(profiles["p_amp_mean"], settings.precision),
        "hyperplane_distance_mean_profile": _format_profile(profiles["hyperplane_distance_mean"], settings.precision),
        "p_amp_max_profile": _format_profile(profiles["p_amp_max"], settings.precision),
        "hyperplane_distance_max_profile": _format_profile(profiles["hyperplane_distance_max"], settings.precision),
    }
    for name, value in extras.items():
        row[name if name not in known else f"label_{name}"] = value
    return row


def _finite_argmax(values: np.ndarray) -> int | None:
    finite = np.flatnonzero(np.isfinite(values))
    if finite.size == 0:
        return None
    return int(finite[np.argmax(values[finite])])


def _finite_mean(values: np.ndarray) -> float | None:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    return float(np.mean(finite))


def _finite_or_none(value: float) -> float | None:
    return value if np.isfinite(value) else None


def _format_profile(values: np.ndarray, precision: int) -> str:
    return ";".join("nan" if not np.isfinite(value) else f"{float(value):.{precision}g}" for value in values)
