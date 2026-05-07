from __future__ import annotations

import csv
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
import time

import fitz

from .dataset_split import SplitManifestConfig, split_manifest_csv
from .labels import PageLabel, PairLabel
from .mupdf_warnings import suppress_mupdf_stderr
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
    num_workers: int = 1


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
    source_pdfs = _discover_pdf_paths(config.input_dir)
    if not source_pdfs:
        raise ValueError(f"No PDF files found under {config.input_dir}")

    if config.num_merged_pdfs < 1:
        raise ValueError("num_merged_pdfs must be at least 1")
    if config.min_docs_per_merge < 1 or config.max_docs_per_merge < config.min_docs_per_merge:
        raise ValueError("Invalid min/max docs per merge configuration")
    if config.num_workers < 1:
        raise ValueError("num_workers must be at least 1")
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

        jobs = _build_synthetic_jobs(config, source_pdfs, merged_dir, images_dir)
        if config.num_workers == 1:
            _write_synthetic_jobs_serial(
                jobs,
                page_writer=page_writer,
                pair_writer=pair_writer,
                progress_callback=progress_callback,
                start_time=start_time,
                total=config.num_merged_pdfs,
            )
        else:
            _write_synthetic_jobs_parallel(
                jobs,
                page_writer=page_writer,
                pair_writer=pair_writer,
                progress_callback=progress_callback,
                start_time=start_time,
                total=config.num_merged_pdfs,
                num_workers=config.num_workers,
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


def _build_synthetic_jobs(
    config: SyntheticDatasetConfig,
    source_pdfs: list[Path],
    merged_dir: Path,
    images_dir: Path,
) -> list[dict[str, object]]:
    rng = random.Random(config.seed)
    jobs: list[dict[str, object]] = []

    for merged_index in range(1, config.num_merged_pdfs + 1):
        doc_count = rng.randint(config.min_docs_per_merge, config.max_docs_per_merge)
        chosen_docs = rng.choices(source_pdfs, k=doc_count)
        merged_pdf_id = f"merged_{merged_index:05d}"
        jobs.append(
            {
                "merged_index": merged_index,
                "merged_pdf_id": merged_pdf_id,
                "merged_pdf_path": merged_dir / f"{merged_pdf_id}.pdf",
                "page_image_dir": images_dir / merged_pdf_id,
                "source_doc_paths": chosen_docs,
                "dpi": config.dpi,
            }
        )

    return jobs


def _write_synthetic_jobs_serial(
    jobs: list[dict[str, object]],
    *,
    page_writer: csv.DictWriter,
    pair_writer: csv.DictWriter,
    progress_callback: Callable[[dict[str, object]], None] | None,
    start_time: float,
    total: int,
) -> None:
    completed = 0
    for job in jobs:
        result = _run_synthetic_job(job, progress_callback=progress_callback, start_time=start_time, total=total)
        _write_synthetic_job_result(result, page_writer=page_writer, pair_writer=pair_writer)
        completed += 1
        _emit_generation_progress(
            progress_callback,
            start_time=start_time,
            current=completed,
            total=total,
            message=f"Generated merged PDF {completed}/{total}: {result['merged_pdf_id']}",
        )


def _write_synthetic_jobs_parallel(
    jobs: list[dict[str, object]],
    *,
    page_writer: csv.DictWriter,
    pair_writer: csv.DictWriter,
    progress_callback: Callable[[dict[str, object]], None] | None,
    start_time: float,
    total: int,
    num_workers: int,
) -> None:
    worker_count = min(num_workers, len(jobs))
    _emit_generation_progress(
        progress_callback,
        start_time=start_time,
        current=0,
        total=total,
        message=f"Starting {worker_count} synthetic generation workers",
    )
    completed = 0
    results: list[dict[str, object]] = []

    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        future_to_job = {executor.submit(_run_synthetic_job, job): job for job in jobs}
        for future in as_completed(future_to_job):
            job = future_to_job[future]
            merged_pdf_id = str(job["merged_pdf_id"])
            try:
                result = future.result()
            except Exception as exc:
                source_names = ", ".join(path.name for path in job["source_doc_paths"])
                raise RuntimeError(f"Failed generating {merged_pdf_id} from sources: {source_names}") from exc

            results.append(result)
            completed += 1
            _emit_generation_progress(
                progress_callback,
                start_time=start_time,
                current=completed,
                total=total,
                message=f"Completed merged PDF {completed}/{total}: {merged_pdf_id}",
            )

    for result in sorted(results, key=lambda item: int(item["merged_index"])):
        _write_synthetic_job_result(result, page_writer=page_writer, pair_writer=pair_writer)


def _run_synthetic_job(
    job: dict[str, object],
    progress_callback: Callable[[dict[str, object]], None] | None = None,
    start_time: float | None = None,
    total: int | None = None,
) -> dict[str, object]:
    merged_index = int(job["merged_index"])
    merged_pdf_id = str(job["merged_pdf_id"])
    merged_pdf_path = Path(job["merged_pdf_path"])
    page_image_dir = Path(job["page_image_dir"])
    source_doc_paths = [Path(path) for path in job["source_doc_paths"]]
    dpi = int(job["dpi"])

    merged_doc = fitz.open()
    page_records: list[dict[str, str | int]] = []
    global_page = 1
    doc_count = len(source_doc_paths)
    if start_time is not None and total is not None:
        _emit_generation_progress(
            progress_callback,
            start_time=start_time,
            current=merged_index - 1,
            total=total,
            message=f"Preparing {merged_pdf_id} with {doc_count} source PDFs",
        )

    try:
        for source_doc_index, source_doc_path in enumerate(source_doc_paths, start=1):
            if start_time is not None and total is not None:
                _emit_generation_progress(
                    progress_callback,
                    start_time=start_time,
                    current=merged_index - 1,
                    total=total,
                    message=(
                        f"Merging {merged_pdf_id}: source {source_doc_index}/{doc_count} "
                        f"{source_doc_path.name}"
                    ),
                )
            with suppress_mupdf_stderr():
                source_doc = fitz.open(source_doc_path)
            try:
                with suppress_mupdf_stderr():
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
            finally:
                source_doc.close()

        if start_time is not None and total is not None:
            _emit_generation_progress(
                progress_callback,
                start_time=start_time,
                current=merged_index - 1,
                total=total,
                message=f"Saving {merged_pdf_id}",
            )
        with suppress_mupdf_stderr():
            merged_doc.save(merged_pdf_path)
    finally:
        merged_doc.close()

    if start_time is not None and total is not None:
        _emit_generation_progress(
            progress_callback,
            start_time=start_time,
            current=merged_index - 1,
            total=total,
            message=f"Rendering page images for {merged_pdf_id}",
        )
    image_paths = pdf_to_images(merged_pdf_path, page_image_dir, dpi=dpi)

    for page_record, image_path in zip(page_records, image_paths, strict=True):
        page_record["image_path"] = str(image_path)

    pair_records: list[dict[str, str | int]] = []
    for left_record, right_record in zip(page_records, page_records[1:], strict=False):
        same_document = int(left_record["source_doc_index"] == right_record["source_doc_index"])
        pair_records.append(
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

    return {
        "merged_index": merged_index,
        "merged_pdf_id": merged_pdf_id,
        "page_records": page_records,
        "pair_records": pair_records,
    }


def _write_synthetic_job_result(
    result: dict[str, object],
    *,
    page_writer: csv.DictWriter,
    pair_writer: csv.DictWriter,
) -> None:
    for page_record in result["page_records"]:
        page_writer.writerow(page_record)

    for pair_record in result["pair_records"]:
        pair_writer.writerow(pair_record)


def _emit_generation_progress(
    progress_callback: Callable[[dict[str, object]], None] | None,
    *,
    start_time: float,
    current: int,
    total: int,
    message: str,
) -> None:
    if progress_callback is None:
        return

    elapsed = max(time.time() - start_time, 1e-6)
    rate = current / elapsed if current else 0.0
    eta_seconds = int((total - current) / rate) if rate > 0 else None
    progress_callback(
        {
            "phase": "generate_synthetic",
            "current": current,
            "total": total,
            "fraction": current / total,
            "message": message,
            "eta_seconds": eta_seconds,
        }
    )
