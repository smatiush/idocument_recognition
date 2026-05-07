from __future__ import annotations

from pathlib import Path
from typing import Any

from datasets import Dataset
from transformers import LayoutLMv3Processor

from .labels import PAIR_LABEL_TO_ID
from .ocr import ocr_page_cached
from .processor_encoding import single_example_encoding_value
from .training_control import check_training_control


REQUIRED_PAIR_COLUMNS = {"left_image_path", "right_image_path", "label"}


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


def load_pair_csv_dataset(csv_path: str | Path) -> Dataset:
    dataset = Dataset.from_csv(str(csv_path))
    missing_columns = REQUIRED_PAIR_COLUMNS - set(dataset.column_names)
    if missing_columns:
        raise ValueError(
            f"Pairwise training requires a pair-label manifest with columns "
            f"{sorted(REQUIRED_PAIR_COLUMNS)}. The CSV at {csv_path} has columns "
            f"{dataset.column_names} and is missing {sorted(missing_columns)}. "
            "Use `data/synthetic/pair_labels_train.csv` and `data/synthetic/pair_labels_eval.csv`, "
            "or generate them from the Synthetic Data tab."
        )
    return dataset


def _validate_pair_label(label: str) -> None:
    if label not in PAIR_LABEL_TO_ID:
        raise ValueError(
            f"Unsupported pair label: {label!r}. Expected one of {sorted(PAIR_LABEL_TO_ID)}."
        )


def encode_pair_example(
    example: dict[str, Any],
    processor: LayoutLMv3Processor,
    max_length: int = 512,
    tesseract_lang: str = "eng",
) -> dict[str, Any]:
    _validate_pair_label(str(example["label"]))
    left_page = ocr_page_cached(str(example["left_image_path"]), tesseract_lang=tesseract_lang)
    right_page = ocr_page_cached(str(example["right_image_path"]), tesseract_lang=tesseract_lang)

    left_encoding = processor(
        left_page.image,
        left_page.words,
        boxes=_sanitize_boxes(left_page.boxes),
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )
    right_encoding = processor(
        right_page.image,
        right_page.words,
        boxes=_sanitize_boxes(right_page.boxes),
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )

    return {
        "left_input_ids": single_example_encoding_value(left_encoding["input_ids"]),
        "left_attention_mask": single_example_encoding_value(left_encoding["attention_mask"]),
        "left_bbox": single_example_encoding_value(left_encoding["bbox"]),
        "left_pixel_values": single_example_encoding_value(left_encoding["pixel_values"]),
        "right_input_ids": single_example_encoding_value(right_encoding["input_ids"]),
        "right_attention_mask": single_example_encoding_value(right_encoding["attention_mask"]),
        "right_bbox": single_example_encoding_value(right_encoding["bbox"]),
        "right_pixel_values": single_example_encoding_value(right_encoding["pixel_values"]),
        "labels": PAIR_LABEL_TO_ID[example["label"]],
    }


def encode_pair_dataset(
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
        return encode_pair_example(
            example,
            processor=processor,
            max_length=max_length,
            tesseract_lang=tesseract_lang,
        )

    map_num_proc = num_proc if num_proc is not None and num_proc > 1 else None
    encoded = dataset.map(mapper, remove_columns=columns_to_remove, num_proc=map_num_proc)
    encoded.set_format("torch")
    return encoded
