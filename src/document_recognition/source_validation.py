from __future__ import annotations

import csv
import hashlib
import shutil
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
import time

import fitz


@dataclass(slots=True)
class SourcePdfRecord:
    path: str
    filename: str
    file_size_bytes: int
    page_count: int | None
    readable: bool
    duplicate_group: str
    keep: bool
    reason: str


@dataclass(slots=True)
class SourceValidationConfig:
    input_dir: Path
    output_dir: Path
    min_pages: int = 1
    max_pages: int = 10
    copy_filtered_pdfs: bool = True
    deduplicate: bool = True


def _discover_pdf_paths(input_dir: Path) -> list[Path]:
    return sorted(path for path in input_dir.rglob("*.pdf") if path.is_file())


def _file_sha1(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _inspect_pdf(path: Path) -> tuple[int | None, bool, str]:
    try:
        doc = fitz.open(path)
        page_count = doc.page_count
        doc.close()
        return page_count, True, ""
    except Exception as exc:
        return None, False, str(exc)


def validate_source_pdfs(
    config: SourceValidationConfig,
    progress_callback: Callable[[dict[str, object]], None] | None = None,
) -> dict[str, Path | int]:
    return validate_source_pdfs_with_progress(config, progress_callback=progress_callback)


def validate_source_pdfs_with_progress(
    config: SourceValidationConfig,
    progress_callback: Callable[[dict[str, object]], None] | None = None,
) -> dict[str, Path | int]:
    pdf_paths = _discover_pdf_paths(config.input_dir)
    if not pdf_paths:
        raise ValueError(f"No PDF files found under {config.input_dir}")

    config.output_dir.mkdir(parents=True, exist_ok=True)
    filtered_dir = config.output_dir / "filtered_pdfs"
    if config.copy_filtered_pdfs:
        filtered_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = config.output_dir / "source_pdf_report.csv"
    summary_path = config.output_dir / "source_pdf_summary.txt"

    seen_hashes: set[str] = set()
    records: list[SourcePdfRecord] = []
    kept_count = 0
    start_time = time.time()

    total_pdfs = len(pdf_paths)

    for index, path in enumerate(pdf_paths, start=1):
        file_hash = _file_sha1(path)
        duplicate_group = file_hash[:12]
        page_count, readable, error = _inspect_pdf(path)

        keep = True
        reasons: list[str] = []

        if not readable:
            keep = False
            reasons.append(f"unreadable:{error}")

        if page_count is not None and page_count < config.min_pages:
            keep = False
            reasons.append(f"below_min_pages:{page_count}")

        if page_count is not None and page_count > config.max_pages:
            keep = False
            reasons.append(f"above_max_pages:{page_count}")

        if config.deduplicate and file_hash in seen_hashes:
            keep = False
            reasons.append("duplicate_content")

        if not reasons:
            reasons.append("accepted")

        if keep:
            kept_count += 1
            seen_hashes.add(file_hash)
            if config.copy_filtered_pdfs:
                shutil.copy2(path, filtered_dir / path.name)

        records.append(
            SourcePdfRecord(
                path=str(path),
                filename=path.name,
                file_size_bytes=path.stat().st_size,
                page_count=page_count,
                readable=readable,
                duplicate_group=duplicate_group,
                keep=keep,
                reason=";".join(reasons),
            )
        )

        if progress_callback is not None:
            elapsed = max(time.time() - start_time, 1e-6)
            rate = index / elapsed if index else 0.0
            eta_seconds = int((total_pdfs - index) / rate) if rate > 0 else None
            progress_callback(
                {
                    "phase": "validate_source_pdfs",
                    "current": index,
                    "total": total_pdfs,
                    "fraction": index / total_pdfs,
                    "message": f"Validated {index}/{total_pdfs}: {path.name}",
                    "eta_seconds": eta_seconds,
                }
            )

    with manifest_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(asdict(records[0]).keys()))
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))

    unreadable_count = sum(1 for record in records if not record.readable)
    duplicate_count = sum(1 for record in records if "duplicate_content" in record.reason)
    over_limit_count = sum(1 for record in records if "above_max_pages" in record.reason)

    summary_lines = [
        f"input_dir={config.input_dir}",
        f"total_pdfs={len(records)}",
        f"kept_pdfs={kept_count}",
        f"unreadable_pdfs={unreadable_count}",
        f"duplicate_pdfs={duplicate_count}",
        f"over_max_pages_pdfs={over_limit_count}",
        f"filtered_dir={filtered_dir if config.copy_filtered_pdfs else ''}",
        f"manifest_path={manifest_path}",
    ]
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    return {
        "manifest_path": manifest_path,
        "summary_path": summary_path,
        "filtered_dir": filtered_dir if config.copy_filtered_pdfs else config.output_dir,
        "total_pdfs": len(records),
        "kept_pdfs": kept_count,
    }
