from __future__ import annotations

import csv
import random
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
import time

import fitz

from .dataset_split import SplitManifestConfig, split_manifest_csv
from .labels import PageLabel, PairLabel
from .pdf import pdf_to_images


@dataclass(slots=True)
class SyntheticDatasetConfig:
    input_dir: Path
    output_dir: Path
    num_merged_pdfs: int
    min_docs_per_merge: int = 2
    max_docs_per_merge: int = 5
    dpi: int = 200
    seed: int = 42
    auto_create_splits: bool = True
    eval_ratio: float = 0.2


def _discover_pdf_paths(input_dir: Path) -> list[Path]:
    return sorted(path for path in input_dir.rglob("*.pdf") if path.is_file())


def _page_labels_for_document(page_count: int) -> list[str]:
    if page_count <= 1:
        return [PageLabel.SINGLE_PAGE_DOC.value]
    if page_count == 2:
        return [PageLabel.START_DOC.value, PageLabel.END_DOC.value]
    return [
        PageLabel.START_DOC.value,
        *([PageLabel.MIDDLE_DOC.value] * (page_count - 2)),
        PageLabel.END_DOC.value,
    ]


def create_synthetic_merged_dataset(
    config: SyntheticDatasetConfig,
    progress_callback: Callable[[dict[str, object]], None] | None = None,
) -> dict[str, Path]:
    return create_synthetic_merged_dataset_with_progress(config, progress_callback=progress_callback)


def create_synthetic_merged_dataset_with_progress(
    config: SyntheticDatasetConfig,
    progress_callback: Callable[[dict[str, object]], None] | None = None,
) -> dict[str, Path]:
    rng = random.Random(config.seed)
    source_pdfs = _discover_pdf_paths(config.input_dir)
    if not source_pdfs:
        raise ValueError(f"No PDF files found under {config.input_dir}")

    if config.min_docs_per_merge < 1 or config.max_docs_per_merge < config.min_docs_per_merge:
        raise ValueError("Invalid min/max docs per merge configuration")
    start_time = time.time()

    merged_dir = config.output_dir / "merged_pdfs"
    images_dir = config.output_dir / "page_images"
    merged_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    page_manifest_path = config.output_dir / "page_labels.csv"
    pair_manifest_path = config.output_dir / "pair_labels.csv"

    with page_manifest_path.open("w", newline="", encoding="utf-8") as page_file, pair_manifest_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as pair_file:
        page_writer = csv.DictWriter(
            page_file,
            fieldnames=[
                "merged_pdf_id",
                "merged_pdf_path",
                "page",
                "image_path",
                "label",
                "source_doc_path",
                "source_doc_index",
                "source_page_in_doc",
            ],
        )
        pair_writer = csv.DictWriter(
            pair_file,
            fieldnames=[
                "merged_pdf_id",
                "merged_pdf_path",
                "left_page",
                "right_page",
                "left_image_path",
                "right_image_path",
                "label",
                "same_document",
                "left_source_doc_index",
                "right_source_doc_index",
            ],
        )
        page_writer.writeheader()
        pair_writer.writeheader()

        for merged_index in range(1, config.num_merged_pdfs + 1):
            doc_count = rng.randint(config.min_docs_per_merge, config.max_docs_per_merge)
            chosen_docs = rng.choices(source_pdfs, k=doc_count)
            merged_pdf_id = f"merged_{merged_index:05d}"
            merged_pdf_path = merged_dir / f"{merged_pdf_id}.pdf"

            merged_doc = fitz.open()
            page_records: list[dict[str, str | int]] = []
            global_page = 1

            for source_doc_index, source_doc_path in enumerate(chosen_docs, start=1):
                source_doc = fitz.open(source_doc_path)
                merged_doc.insert_pdf(source_doc)
                labels = _page_labels_for_document(source_doc.page_count)

                for source_page_in_doc, label in enumerate(labels, start=1):
                    page_records.append(
                        {
                            "merged_pdf_id": merged_pdf_id,
                            "merged_pdf_path": str(merged_pdf_path),
                            "page": global_page,
                            "label": label,
                            "source_doc_path": str(source_doc_path),
                            "source_doc_index": source_doc_index,
                            "source_page_in_doc": source_page_in_doc,
                        }
                    )
                    global_page += 1

                source_doc.close()

            merged_doc.save(merged_pdf_path)
            merged_doc.close()

            page_image_dir = images_dir / merged_pdf_id
            image_paths = pdf_to_images(merged_pdf_path, page_image_dir, dpi=config.dpi)

            for page_record, image_path in zip(page_records, image_paths, strict=True):
                page_record["image_path"] = str(image_path)
                page_writer.writerow(page_record)

            for left_record, right_record in zip(page_records, page_records[1:], strict=False):
                same_document = int(left_record["source_doc_index"] == right_record["source_doc_index"])
                pair_writer.writerow(
                    {
                        "merged_pdf_id": merged_pdf_id,
                        "merged_pdf_path": str(merged_pdf_path),
                        "left_page": left_record["page"],
                        "right_page": right_record["page"],
                        "left_image_path": left_record["image_path"],
                        "right_image_path": right_record["image_path"],
                        "label": PairLabel.SAME_DOCUMENT.value if same_document else PairLabel.NEW_DOCUMENT.value,
                        "same_document": same_document,
                        "left_source_doc_index": left_record["source_doc_index"],
                        "right_source_doc_index": right_record["source_doc_index"],
                    }
                )

            if progress_callback is not None:
                elapsed = max(time.time() - start_time, 1e-6)
                rate = merged_index / elapsed if merged_index else 0.0
                eta_seconds = int((config.num_merged_pdfs - merged_index) / rate) if rate > 0 else None
                progress_callback(
                    {
                        "phase": "generate_synthetic",
                        "current": merged_index,
                        "total": config.num_merged_pdfs,
                        "fraction": merged_index / config.num_merged_pdfs,
                        "message": f"Generated merged PDF {merged_index}/{config.num_merged_pdfs}: {merged_pdf_id}",
                        "eta_seconds": eta_seconds,
                    }
                )

    outputs: dict[str, Path] = {
        "page_manifest": page_manifest_path,
        "pair_manifest": pair_manifest_path,
        "merged_dir": merged_dir,
        "images_dir": images_dir,
    }

    if config.auto_create_splits:
        page_split = split_manifest_csv(
            SplitManifestConfig(
                input_csv=page_manifest_path,
                output_train_csv=config.output_dir / "page_labels_train.csv",
                output_eval_csv=config.output_dir / "page_labels_eval.csv",
                eval_ratio=config.eval_ratio,
                seed=config.seed,
            )
        )
        pair_split = split_manifest_csv(
            SplitManifestConfig(
                input_csv=pair_manifest_path,
                output_train_csv=config.output_dir / "pair_labels_train.csv",
                output_eval_csv=config.output_dir / "pair_labels_eval.csv",
                eval_ratio=config.eval_ratio,
                seed=config.seed,
            )
        )
        outputs.update(
            {
                "page_train_manifest": Path(page_split["train_csv"]),
                "page_eval_manifest": Path(page_split["eval_csv"]),
                "pair_train_manifest": Path(pair_split["train_csv"]),
                "pair_eval_manifest": Path(pair_split["eval_csv"]),
            }
        )

    return outputs
