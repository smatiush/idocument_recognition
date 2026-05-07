from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from datasets import Dataset
from transformers import LayoutLMv3Processor

from .labels import LABEL_TO_ID
from .ocr import OCREngine, ocr_page
from .training_control import check_training_control


REQUIRED_PAGE_COLUMNS = {"image_path", "label"}


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


def _single_example_encoding(encoding: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in encoding.items():
        if isinstance(value, np.ndarray):
            normalized[key] = value[0] if value.ndim > 0 and value.shape[0] == 1 else value
        elif isinstance(value, list) and len(value) == 1:
            normalized[key] = value[0]
        else:
            normalized[key] = value
    return normalized


def load_csv_dataset(csv_path: str | Path) -> Dataset:
    dataset = Dataset.from_csv(str(csv_path))
    missing_columns = REQUIRED_PAGE_COLUMNS - set(dataset.column_names)
    if missing_columns:
        raise ValueError(
            f"Baseline training requires a page-label manifest with columns "
            f"{sorted(REQUIRED_PAGE_COLUMNS)}. The CSV at {csv_path} has columns "
            f"{dataset.column_names} and is missing {sorted(missing_columns)}. "
            "Use `data/synthetic/page_labels_train.csv` and `data/synthetic/page_labels_eval.csv`, "
            "or generate them from the Synthetic Data tab."
        )
    return dataset


def _validate_label(label: str) -> None:
    if label not in LABEL_TO_ID:
        raise ValueError(
            f"Unsupported page label: {label!r}. Expected one of {sorted(LABEL_TO_ID)}."
        )


def encode_example(
    example: dict[str, Any],
    processor: LayoutLMv3Processor,
    max_length: int = 512,
    tesseract_lang: str = "eng",
    ocr_engine: OCREngine = "tesseract",
    ocr_gpu: bool = False,
) -> dict[str, Any]:
    _validate_label(str(example["label"]))
    page = ocr_page(example["image_path"], tesseract_lang=tesseract_lang, ocr_engine=ocr_engine, ocr_gpu=ocr_gpu)

    encoding = processor(
        page.image,
        page.words,
        boxes=_sanitize_boxes(page.boxes),
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )
    encoding = _single_example_encoding(dict(encoding))
    encoding["labels"] = LABEL_TO_ID[example["label"]]
    return encoding


def encode_dataset(
    dataset: Dataset,
    processor: LayoutLMv3Processor,
    max_length: int = 512,
    tesseract_lang: str = "eng",
    num_proc: int | None = None,
    control_path: Path | None = None,
    ocr_engine: OCREngine = "tesseract",
    ocr_gpu: bool = False,
) -> Dataset:
    columns_to_remove = dataset.column_names

    def mapper(example: dict[str, Any]) -> dict[str, Any]:
        check_training_control(control_path)
        return encode_example(
            example,
            processor=processor,
            max_length=max_length,
            tesseract_lang=tesseract_lang,
            ocr_engine=ocr_engine,
            ocr_gpu=ocr_gpu,
        )

    map_num_proc = num_proc if num_proc is not None and num_proc > 1 else None
    encoded = dataset.map(mapper, remove_columns=columns_to_remove, num_proc=map_num_proc)
    encoded.set_format("torch")
    return encoded
