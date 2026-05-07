from __future__ import annotations

from pathlib import Path

import fitz

from .mupdf_warnings import suppress_mupdf_stderr


def pdf_to_images(pdf_path: str | Path, out_dir: str | Path, dpi: int = 200) -> list[Path]:
    pdf_path = Path(pdf_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with suppress_mupdf_stderr():
        doc = fitz.open(pdf_path)
    image_paths: list[Path] = []

    try:
        for index, page in enumerate(doc, start=1):
            with suppress_mupdf_stderr():
                pixmap = page.get_pixmap(dpi=dpi)
                image_path = out_dir / f"page_{index:04d}.png"
                pixmap.save(image_path)
            image_paths.append(image_path)
    finally:
        doc.close()

    return image_paths


def split_pdf_ranges(
    pdf_path: str | Path,
    ranges: list[tuple[int, int]],
    out_dir: str | Path,
    prefix: str = "doc",
) -> list[Path]:
    pdf_path = Path(pdf_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with suppress_mupdf_stderr():
        source = fitz.open(pdf_path)
    outputs: list[Path] = []

    try:
        for index, (start_page, end_page) in enumerate(ranges, start=1):
            target = fitz.open()
            output_path = out_dir / f"{prefix}_{index:03d}_{start_page}-{end_page}.pdf"
            try:
                with suppress_mupdf_stderr():
                    target.insert_pdf(source, from_page=start_page - 1, to_page=end_page - 1)
                    target.save(output_path)
            finally:
                target.close()
            outputs.append(output_path)
    finally:
        source.close()

    return outputs
