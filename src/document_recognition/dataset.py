from __future__ import annotations

from pathlib import Path
from typing import Any

from datasets import Dataset
from transformers import LayoutLMv3Processor

from .labels import LABEL_TO_ID
from .ocr import ocr_page
from .training_control import check_training_control


REQUIRED_PAGE_COLUMNS = {"image_path", "label"}


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
) -> dict[str, Any]:
    _validate_label(str(example["label"]))
    page = ocr_page(example["image_path"], tesseract_lang=tesseract_lang)

    encoding = processor(
        page.image,
        page.words,
        boxes=page.boxes,
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )
    encoding["labels"] = LABEL_TO_ID[example["label"]]
    return encoding


def encode_dataset(
    dataset: Dataset,
    processor: LayoutLMv3Processor,
    max_length: int = 512,
    tesseract_lang: str = "eng",
    num_proc: int | None = None,
    control_path: Path | None = None,
) -> Dataset:
    columns_to_remove = dataset.column_names

    def mapper(example: dict[str, Any]) -> dict[str, Any]:
        check_training_control(control_path)
        return encode_example(
            example,
            processor=processor,
            max_length=max_length,
            tesseract_lang=tesseract_lang,
        )

    map_num_proc = num_proc if num_proc is not None and num_proc > 1 else None
    encoded = dataset.map(mapper, remove_columns=columns_to_remove, num_proc=map_num_proc)
    encoded.set_format("torch")
    return encoded
