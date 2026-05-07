from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class SplitManifestConfig:
    input_csv: Path
    output_train_csv: Path
    output_eval_csv: Path
    eval_ratio: float = 0.2
    seed: int = 42


def split_manifest_csv(config: SplitManifestConfig) -> dict[str, int | Path]:
    if not config.input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {config.input_csv}")
    if not 0.0 < config.eval_ratio < 1.0:
        raise ValueError("eval_ratio must be between 0 and 1")

    with config.input_csv.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    if not rows:
        raise ValueError(f"Input CSV is empty: {config.input_csv}")

    group_key = None
    for candidate in ("merged_pdf_id", "pdf_id"):
        if candidate in fieldnames:
            group_key = candidate
            break

    rng = random.Random(config.seed)

    if group_key is None:
        rng.shuffle(rows)
        eval_count = max(1, int(len(rows) * config.eval_ratio))
        eval_rows = rows[:eval_count]
        train_rows = rows[eval_count:]
    else:
        grouped: dict[str, list[dict[str, str]]] = {}
        for row in rows:
            grouped.setdefault(row[group_key], []).append(row)

        group_ids = list(grouped)
        rng.shuffle(group_ids)
        eval_group_count = max(1, int(len(group_ids) * config.eval_ratio))
        eval_group_ids = set(group_ids[:eval_group_count])
        train_rows = [row for group_id, group_rows in grouped.items() if group_id not in eval_group_ids for row in group_rows]
        eval_rows = [row for group_id, group_rows in grouped.items() if group_id in eval_group_ids for row in group_rows]

    if not train_rows or not eval_rows:
        raise ValueError("Split would produce an empty train or eval set; use more data or adjust eval_ratio")

    config.output_train_csv.parent.mkdir(parents=True, exist_ok=True)
    config.output_eval_csv.parent.mkdir(parents=True, exist_ok=True)

    for output_path, output_rows in ((config.output_train_csv, train_rows), (config.output_eval_csv, eval_rows)):
        with output_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(output_rows)

    return {
        "train_csv": config.output_train_csv,
        "eval_csv": config.output_eval_csv,
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
    }
