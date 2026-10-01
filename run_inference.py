#!/usr/bin/env python3
"""Score peptides with the expanded-dataset SVM checkpoint."""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from descriptors import clean_sequence, load_distance_matrix
from model import load_model, score_sequences
from windows import WindowSettings, profile_fieldnames, profile_sequence, validate_settings

ROOT = Path(__file__).resolve().parent
SEQUENCE_OUTPUT = "sequence_predictions.csv"
WINDOW_OUTPUT = "window_sequence_profiles.csv"
ID_COLUMNS = ("id", "peptide_id", "sequence_id", "name")


@dataclass(frozen=True)
class Record:
    id: str
    sequence: str
    extras: dict[str, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score peptide sequences with the expanded alpha-helix and beta-sheet SVM."
    )
    parser.add_argument("input", type=Path, help="CSV (id, sequence), FASTA, or one sequence per line.")
    parser.add_argument("--windows", action="store_true", help="Write per-residue window profiles instead of one score per sequence.")
    parser.add_argument("-o", "--output", type=Path, help="Output CSV. Default: <input directory>/results/<mode>.csv")
    parser.add_argument("--window-min-len", type=int, default=10, help="Minimum window length (default: 10).")
    parser.add_argument("--window-max-len", type=int, default=35, help="Maximum window length (default: 35).")
    parser.add_argument("--stride", type=int, default=1, help="Window start stride (default: 1).")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        settings = WindowSettings(min_len=args.window_min_len, max_len=args.window_max_len, stride=args.stride)
        if args.windows:
            validate_settings(settings)
        extra_columns, records = read_records(args.input)
        model = load_model()
        distance_matrix = load_distance_matrix()
        output = args.output or default_output(args.input, args.windows)
        if args.windows:
            rows = score_windows(records, model, distance_matrix, settings)
            write_csv(output, rows, profile_fieldnames(extra_columns))
        else:
            rows = score_full_sequences(records, model, distance_matrix)
            write_csv(output, rows, sequence_fieldnames(extra_columns))
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    print(f"Saved {len(records)} sequence(s) to {output.resolve()}")
    return 0


def default_output(path: Path, windows: bool) -> Path:
    name = WINDOW_OUTPUT if windows else SEQUENCE_OUTPUT
    return path.parent / "results" / name


def score_full_sequences(records: list[Record], model, distance_matrix) -> list[dict[str, object]]:
    prediction, sigma, p_amp = score_sequences(model, [record.sequence for record in records], distance_matrix)
    reserved = {"id", "sequence", "prediction", "sigma", "p_amp"}
    rows = []
    for record, label, score, probability in zip(records, prediction, sigma, p_amp):
        extras = {name if name not in reserved else f"label_{name}": value for name, value in record.extras.items()}
        rows.append({
            "id": record.id,
            "sequence": record.sequence,
            **extras,
            "prediction": int(label),
            "sigma": float(score),
            "p_amp": float(probability),
        })
    return rows


def sequence_fieldnames(extra_columns: list[str]) -> list[str]:
    reserved = {"id", "sequence", "prediction", "sigma", "p_amp"}
    extras = [name if name not in reserved else f"label_{name}" for name in extra_columns]
    return ["id", "sequence", *extras, "prediction", "sigma", "p_amp"]


def score_windows(records: list[Record], model, distance_matrix, settings: WindowSettings) -> list[dict[str, object]]:
    def score_batch(sequences: list[str]) -> tuple[np.ndarray, np.ndarray]:
        _prediction, sigma, p_amp = score_sequences(model, list(sequences), distance_matrix)
        return p_amp, sigma

    rows = []
    total = len(records)
    for index, record in enumerate(records, start=1):
        if index == 1 or index % 20 == 0 or index == total:
            print(f"Windows {index}/{total}: {record.id}", flush=True)
        rows.append(profile_sequence(record.id, record.sequence, record.extras, score_batch, settings))
    return rows


def read_records(path: Path) -> tuple[list[str], list[Record]]:
    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path}")
    text = path.read_text(encoding="utf-8-sig")
    if not text.strip():
        raise ValueError("Input contains no sequences.")
    if path.suffix.lower() == ".csv" or _looks_like_sequence_csv(text):
        extra_columns, raw_rows = read_csv(path)
    elif text.lstrip().startswith(">"):
        extra_columns, raw_rows = [], read_fasta(text)
    else:
        extra_columns, raw_rows = [], read_lines(text)
    records = [
        Record(record_id, clean_sequence(sequence, record_id), extras)
        for record_id, sequence, extras in raw_rows
    ]
    if not records:
        raise ValueError("Input contains no sequences.")
    _reject_duplicate_ids(records)
    return extra_columns, records


def read_csv(path: Path) -> tuple[list[str], list[tuple[str, str, dict[str, str]]]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    if "sequence" not in fieldnames:
        raise ValueError("CSV input requires a sequence column.")
    id_column = next((name for name in ID_COLUMNS if name in fieldnames), None)
    extra_columns = [name for name in fieldnames if name not in {id_column, "sequence"}]
    parsed = []
    for index, row in enumerate(rows, start=1):
        record_id = (row.get(id_column) or "").strip() if id_column else ""
        record_id = record_id or f"seq_{index}"
        extras = {name: row.get(name, "") for name in extra_columns}
        parsed.append((record_id, row.get("sequence", ""), extras))
    return extra_columns, parsed


def read_fasta(text: str) -> list[tuple[str, str, dict[str, str]]]:
    records: list[tuple[str, str, dict[str, str]]] = []
    header: str | None = None
    chunks: list[str] = []

    def flush() -> None:
        if header is not None:
            records.append((header, "".join(chunks), {}))

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            flush()
            header = line[1:].split()[0] or f"seq_{len(records) + 1}"
            chunks = []
        elif header is None:
            raise ValueError("FASTA records must start with a '>' header.")
        else:
            chunks.append(line)
    flush()
    return records


def read_lines(text: str) -> list[tuple[str, str, dict[str, str]]]:
    records = []
    for index, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) == 1:
            records.append((f"seq_{index}", parts[0], {}))
        else:
            records.append((parts[0], "".join(parts[1:]), {}))
    return records


def _looks_like_sequence_csv(text: str) -> bool:
    header = text.lstrip().splitlines()[0]
    columns = {name.strip() for name in header.split(",")}
    return "sequence" in columns


def _reject_duplicate_ids(records: list[Record]) -> None:
    seen: set[str] = set()
    duplicates: list[str] = []
    for record in records:
        if record.id in seen and record.id not in duplicates:
            duplicates.append(record.id)
        seen.add(record.id)
    if duplicates:
        preview = ", ".join(duplicates[:5])
        raise ValueError(f"Sequence ids must be unique; duplicates include: {preview}")


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: "" if row.get(name) is None else row.get(name) for name in fieldnames})


if __name__ == "__main__":
    raise SystemExit(main())
