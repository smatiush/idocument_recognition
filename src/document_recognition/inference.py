from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import LayoutLMv3ForSequenceClassification, LayoutLMv3Processor

from .labels import ID_TO_LABEL
from .ocr import ensure_tesseract_available, ocr_page
from .pdf import pdf_to_images, split_pdf_ranges
from .postprocess import build_documents, enforce_valid_sequence


@dataclass(slots=True)
class PagePrediction:
    page_number: int
    label: str
    confidence: float


@dataclass(slots=True)
class PredictionResult:
    predictions: list[PagePrediction]
    ranges: list[tuple[int, int]]
    output_pdfs: list[Path]


def _sanitize_boxes(boxes: list[list[int]]) -> list[list[int]]:
    sanitized: list[list[int]] = []
    for box in boxes:
        if len(box) != 4:
            sanitized.append([0, 0, 0, 0])
            continue

        left = max(0, min(int(box[0]), 1000))
        top = max(0, min(int(box[1]), 1000))
        right = max(0, min(int(box[2]), 1000))
        bottom = max(0, min(int(box[3]), 1000))
        if right < left:
            right = left
        if bottom < top:
            bottom = top
        sanitized.append([left, top, right, bottom])
    return sanitized


def _predict_page(
    image_path: Path,
    processor: LayoutLMv3Processor,
    model: LayoutLMv3ForSequenceClassification,
    max_length: int,
    tesseract_lang: str,
) -> tuple[str, float]:
    page = ocr_page(image_path, tesseract_lang=tesseract_lang)
    encoding = processor(
        page.image,
        page.words,
        boxes=_sanitize_boxes(page.boxes),
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )

    with torch.no_grad():
        outputs = model(**encoding)
        probabilities = torch.softmax(outputs.logits, dim=-1).squeeze(0)
        label_id = int(torch.argmax(probabilities).item())
        confidence = float(probabilities[label_id].item())

    return ID_TO_LABEL[label_id], confidence


def predict_pdf_boundaries(
    pdf_path: str | Path,
    model_dir: str | Path,
    work_dir: str | Path,
    dpi: int = 200,
    max_length: int = 512,
    tesseract_lang: str = "eng",
    split_output: bool = True,
) -> PredictionResult:
    ensure_tesseract_available(tesseract_lang=tesseract_lang)

    pdf_path = Path(pdf_path)
    model_dir = Path(model_dir)
    work_dir = Path(work_dir)
    image_dir = work_dir / "pages"
    split_dir = work_dir / "split_docs"

    image_paths = pdf_to_images(pdf_path, image_dir, dpi=dpi)
    processor = LayoutLMv3Processor.from_pretrained(str(model_dir), apply_ocr=False)
    model = LayoutLMv3ForSequenceClassification.from_pretrained(str(model_dir))
    model.eval()

    raw_predictions: list[PagePrediction] = []
    for page_number, image_path in enumerate(image_paths, start=1):
        label, confidence = _predict_page(
            image_path=image_path,
            processor=processor,
            model=model,
            max_length=max_length,
            tesseract_lang=tesseract_lang,
        )
        raw_predictions.append(PagePrediction(page_number, label, confidence))

    normalized_labels = enforce_valid_sequence([prediction.label for prediction in raw_predictions])
    predictions = [
        PagePrediction(
            page_number=prediction.page_number,
            label=normalized_labels[index],
            confidence=prediction.confidence,
        )
        for index, prediction in enumerate(raw_predictions)
    ]

    ranges = build_documents(normalized_labels)
    output_pdfs = split_pdf_ranges(pdf_path, ranges, split_dir) if split_output else []
    return PredictionResult(predictions=predictions, ranges=ranges, output_pdfs=output_pdfs)
