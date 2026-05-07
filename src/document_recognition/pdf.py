from __future__ import annotations

from pathlib import Path

import fitz


def pdf_to_images(pdf_path: str | Path, out_dir: str | Path, dpi: int = 200) -> list[Path]:
    pdf_path = Path(pdf_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf_path)
    image_paths: list[Path] = []

    for index, page in enumerate(doc, start=1):
        pixmap = page.get_pixmap(dpi=dpi)
        image_path = out_dir / f"page_{index:04d}.png"
        pixmap.save(image_path)
        image_paths.append(image_path)

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

    source = fitz.open(pdf_path)
    outputs: list[Path] = []

    for index, (start_page, end_page) in enumerate(ranges, start=1):
        target = fitz.open()
        target.insert_pdf(source, from_page=start_page - 1, to_page=end_page - 1)
        output_path = out_dir / f"{prefix}_{index:03d}_{start_page}-{end_page}.pdf"
        target.save(output_path)
        target.close()
        outputs.append(output_path)

    return outputs
