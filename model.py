"""Load the svm_no_class_weight checkpoint and score raw QSAR rows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from descriptors import QSAR_COLUMNS, descriptor_matrix

CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoint"
MODEL_PATH = CHECKPOINT_DIR / "svm_qsar12_model.pkl"
ZSCORE_PATH = CHECKPOINT_DIR / "svm_qsar12_zscores.csv"


@dataclass(frozen=True)
class Checkpoint:
    svm: object
    names: tuple[str, ...]
    means: np.ndarray
    stds: np.ndarray


def load_model(
    model_path: Path = MODEL_PATH,
    zscore_path: Path = ZSCORE_PATH,
) -> Checkpoint:
    svm = _load_svm(model_path)
    names, means, stds = _load_zscores(zscore_path)
    n_features = int(getattr(svm, "n_features_in_", len(names)))
    if n_features != len(names):
        raise ValueError(f"Checkpoint expects {n_features} features; z-score file has {len(names)}.")
    if not hasattr(svm, "predict_proba") or not hasattr(svm, "decision_function"):
        raise TypeError("Checkpoint SVM must provide predict_proba and decision_function.")
    classes = np.asarray(getattr(svm, "classes_", []))
    if 1 not in classes:
        raise ValueError(f"Checkpoint classes must include positive label 1; found {classes.tolist()}.")
    return Checkpoint(svm=svm, names=names, means=means, stds=stds)


def score_sequences(
    model: Checkpoint,
    sequences: list[str],
    distance_matrix: dict[str, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not sequences:
        empty_int = np.asarray([], dtype=np.int64)
        empty_float = np.asarray([], dtype=np.float64)
        return empty_int, empty_float, empty_float
    return score_matrix(model, descriptor_matrix(sequences, distance_matrix))


def score_matrix(model: Checkpoint, features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    column_index = [QSAR_COLUMNS.index(name) for name in model.names]
    ordered = features[:, column_index]
    scaled = (ordered - model.means) / model.stds
    svm = model.svm
    classes = np.asarray(svm.classes_)
    positive_index = int(np.where(classes == 1)[0][0])
    raw = np.asarray(svm.predict(scaled)).ravel()
    prediction = np.where(raw == 1, 1, -1).astype(np.int64)
    sigma = np.asarray(svm.decision_function(scaled)).ravel().astype(np.float64)
    p_amp = np.asarray(svm.predict_proba(scaled))[:, positive_index].ravel().astype(np.float64)
    return prediction, sigma, p_amp


def _load_svm(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"SVM pickle not found: {path}")
    import joblib

    return joblib.load(path)


def _load_zscores(path: Path) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Z-score file not found: {path}")
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) < 3:
        raise ValueError("Z-score file must contain descriptor names, means, and standard deviations.")
    names = tuple(name.strip() for name in lines[0].split(",") if name.strip())
    means = np.asarray([float(value) for value in lines[1].split(",")], dtype=np.float64)
    stds = np.asarray([float(value) for value in lines[2].split(",")], dtype=np.float64)
    if len(names) != len(means) or len(names) != len(stds):
        raise ValueError("Z-score descriptor, mean, and standard deviation counts do not match.")
    missing = [name for name in names if name not in QSAR_COLUMNS]
    if missing:
        raise ValueError(f"Unsupported descriptor(s) in z-score file: {missing}")
    return names, means, np.where(stds > 0, stds, 1.0)
