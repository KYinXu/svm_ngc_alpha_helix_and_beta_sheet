"""QSAR-12 descriptors for the expanded-dataset SVM."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

QSAR_COLUMNS = (
    "netCharge",
    "FC",
    "LW",
    "DP",
    "NK",
    "AE",
    "pcMK",
    "_SolventAccessibilityD1025",
    "tau2_GRAR740104",
    "tau4_GRAR740104",
    "QSO50_GRAR740104",
    "QSO29_GRAR740104",
)

CHARGE = {
    "A": 0,
    "C": 0,
    "D": -1,
    "E": -1,
    "F": 0,
    "G": 0,
    "H": 1,
    "I": 0,
    "K": 1,
    "L": 0,
    "M": 0,
    "N": 0,
    "P": 0,
    "Q": 0,
    "R": 1,
    "S": 0,
    "T": 0,
    "V": 0,
    "W": 0,
    "Y": 0,
}
STANDARD_AA = set(CHARGE)
DISTANCE_MATRIX_PATH = Path(__file__).resolve().parent / "checkpoint" / "grar740104.json"


def clean_sequence(sequence: str, record_id: str) -> str:
    cleaned = "".join(sequence.split()).upper()
    if not cleaned:
        raise ValueError(f"{record_id}: empty sequence.")
    invalid = sorted(set(cleaned) - STANDARD_AA)
    if invalid:
        raise ValueError(f"{record_id}: unsupported residue(s) {''.join(invalid)}.")
    return cleaned


def load_distance_matrix(path: Path = DISTANCE_MATRIX_PATH) -> dict[str, float]:
    if not path.is_file():
        raise FileNotFoundError(f"GRAR740104 matrix not found: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or len(raw) != 400:
        raise ValueError(f"GRAR740104 matrix must contain 400 amino-acid pairs: {path}")
    return {str(pair): float(value) for pair, value in raw.items()}


def descriptor_matrix(sequences: list[str], distance_matrix: dict[str, float]) -> np.ndarray:
    rows = [qsar12(sequence, distance_matrix) for sequence in sequences]
    return np.asarray([[row[name] for name in QSAR_COLUMNS] for row in rows], dtype=np.float64)


def qsar12(sequence: str, distance_matrix: dict[str, float]) -> dict[str, float]:
    from propy import ProCheck
    from propy.PyPro import GetProDes

    if ProCheck.ProteinCheck(sequence) == 0:
        raise ValueError(f"ProPy rejected sequence: {sequence}")
    descriptor = GetProDes(sequence)
    dipeptides = descriptor.GetDPComp()
    composition = descriptor.GetCTD()
    sequence_order = sequence_order_descriptors(descriptor, sequence, distance_matrix)
    methionine = sequence.count("M")
    lysine = sequence.count("K")
    features = {
        "netCharge": float(sum(CHARGE[residue] for residue in sequence)),
        **{name: float(f"{dipeptides[name]:.2f}") for name in ("FC", "LW", "DP", "NK", "AE")},
        "pcMK": 0.0 if methionine == 0 else methionine / (methionine + lysine),
        "_SolventAccessibilityD1025": float(composition["_SolventAccessibilityD1025"]),
        **sequence_order,
    }
    invalid = [name for name, value in features.items() if not np.isfinite(value)]
    if invalid:
        raise ValueError(f"Non-finite descriptors for {sequence!r}: {', '.join(invalid)}.")
    return features


def sequence_order_descriptors(descriptor, sequence: str, distance_matrix: dict[str, float]) -> dict[str, float]:
    try:
        socn = descriptor.GetSOCNp(maxlag=30, distancematrix=distance_matrix)
        qso = descriptor.GetQSOp(maxlag=30, weight=0.05, distancematrix=distance_matrix)
    except Exception as error:
        raise RuntimeError(
            f"Could not compute GRAR740104 descriptors for {sequence!r} (length {len(sequence)})."
        ) from error
    missing = [name for name, values in (("tau2", socn), ("tau4", socn), ("QSO50", qso), ("QSO29", qso)) if name not in values]
    if missing:
        raise RuntimeError(f"ProPy omitted descriptors for {sequence!r}: {', '.join(missing)}.")
    length = len(sequence)
    return {
        "tau2_GRAR740104": float(socn["tau2"] / (length - 2)) if length > 2 else 0.0,
        "tau4_GRAR740104": float(socn["tau4"] / (length - 4)) if length > 4 else 0.0,
        "QSO50_GRAR740104": float(qso["QSO50"]),
        "QSO29_GRAR740104": float(qso["QSO29"]),
    }
