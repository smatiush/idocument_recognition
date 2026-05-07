from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import LayoutLMv3Processor

from .labels import PairLabel
from .ocr import ensure_tesseract_available, ocr_page_cached
from .pairwise_model import PairwiseLayoutLMv3Classifier
from .pdf import pdf_to_images, split_pdf_ranges
from .postprocess import build_documents_from_same_doc_probs, ranges_to_page_labels


@dataclass(slots=True)
class PairBoundaryPrediction:
    left_page: int
    right_page: int
    same_document_probability: float
    label: str


@dataclass(slots=True)
class PairwisePredictionResult:
    pair_predictions: list[PairBoundaryPrediction]
    page_labels: list[str]
    ranges: list[tuple[int, int]]
    output_pdfs: list[Path]


def _encode_page(image_path: Path, processor: LayoutLMv3Processor, max_length: int, tesseract_lang: str) -> dict[str, torch.Tensor]:
    page = ocr_page_cached(str(image_path), tesseract_lang=tesseract_lang)
    return processor(
        page.image,
        page.words,
        boxes=page.boxes,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )


def predict_pdf_boundaries_pairwise(
    pdf_path: str | Path,
    model_dir: str | Path,
    work_dir: str | Path,
    dpi: int = 200,
    max_length: int = 512,
    tesseract_lang: str = "eng",
    threshold: float = 0.5,
    split_output: bool = True,
) -> PairwisePredictionResult:
    ensure_tesseract_available(tesseract_lang=tesseract_lang)

    pdf_path = Path(pdf_path)
    model_dir = Path(model_dir)
    work_dir = Path(work_dir)
    image_dir = work_dir / "pages"
    split_dir = work_dir / "split_docs"

    image_paths = pdf_to_images(pdf_path, image_dir, dpi=dpi)
    if len(image_paths) == 1:
        ranges = [(1, 1)]
        return PairwisePredictionResult(
            pair_predictions=[],
            page_labels=ranges_to_page_labels(ranges, total_pages=1),
            ranges=ranges,
            output_pdfs=split_pdf_ranges(pdf_path, ranges, split_dir) if split_output else [],
        )

    processor = LayoutLMv3Processor.from_pretrained(str(model_dir), apply_ocr=False)
    model = PairwiseLayoutLMv3Classifier.from_saved(model_dir)
    model.eval()

    same_document_probabilities: list[float] = []
    pair_predictions: list[PairBoundaryPrediction] = []

    for left_page_number, (left_image_path, right_image_path) in enumerate(zip(image_paths, image_paths[1:], strict=False), start=1):
        left_encoding = _encode_page(left_image_path, processor, max_length, tesseract_lang)
        right_encoding = _encode_page(right_image_path, processor, max_length, tesseract_lang)

        with torch.no_grad():
            outputs = model(
                left_input_ids=left_encoding["input_ids"],
                left_attention_mask=left_encoding["attention_mask"],
                left_bbox=left_encoding["bbox"],
                left_pixel_values=left_encoding["pixel_values"],
                right_input_ids=right_encoding["input_ids"],
                right_attention_mask=right_encoding["attention_mask"],
                right_bbox=right_encoding["bbox"],
                right_pixel_values=right_encoding["pixel_values"],
            )
            probabilities = torch.softmax(outputs["logits"], dim=-1).squeeze(0)
            same_document_probability = float(probabilities[0].item())

        same_document_probabilities.append(same_document_probability)
        pair_predictions.append(
            PairBoundaryPrediction(
                left_page=left_page_number,
                right_page=left_page_number + 1,
                same_document_probability=same_document_probability,
                label=PairLabel.SAME_DOCUMENT.value if same_document_probability >= threshold else PairLabel.NEW_DOCUMENT.value,
            )
        )

    ranges = build_documents_from_same_doc_probs(same_document_probabilities, threshold=threshold)
    page_labels = ranges_to_page_labels(ranges, total_pages=len(image_paths))
    output_pdfs = split_pdf_ranges(pdf_path, ranges, split_dir) if split_output else []
    return PairwisePredictionResult(
        pair_predictions=pair_predictions,
        page_labels=page_labels,
        ranges=ranges,
        output_pdfs=output_pdfs,
    )
