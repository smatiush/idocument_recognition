from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import pytesseract
from PIL import Image


@dataclass(slots=True)
class OCRPage:
    image: Image.Image
    words: list[str]
    boxes: list[list[int]]
    text: str


class OCREnvironmentError(RuntimeError):
    pass


def _ocr_cache_enabled() -> bool:
    return os.environ.get("DOCUMENT_RECOGNITION_DISABLE_OCR_CACHE", "0") != "1"


def _ocr_cache_dir() -> Path:
    return Path(os.environ.get("DOCUMENT_RECOGNITION_OCR_CACHE_DIR", ".cache/document_recognition/ocr")).expanduser()


def _ocr_cache_key(image_path: Path, tesseract_lang: str) -> str:
    resolved_path = image_path.resolve()
    stat = resolved_path.stat()
    payload = {
        "path": str(resolved_path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "lang": tesseract_lang,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _read_ocr_cache(cache_path: Path, image: Image.Image) -> OCRPage | None:
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    words = payload.get("words")
    boxes = payload.get("boxes")
    text = payload.get("text")
    if not isinstance(words, list) or not isinstance(boxes, list) or not isinstance(text, str):
        return None

    return OCRPage(
        image=image,
        words=[str(word) for word in words],
        boxes=[_sanitize_box(box) for box in boxes],
        text=text,
    )


def _write_ocr_cache(cache_path: Path, words: list[str], boxes: list[list[int]], text: str) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"words": words, "boxes": boxes, "text": text}

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=cache_path.parent,
        prefix=f"{cache_path.name}.",
        suffix=".tmp",
        delete=False,
    ) as file:
        json.dump(payload, file)
        temp_path = Path(file.name)

    temp_path.replace(cache_path)


@lru_cache(maxsize=16)
def ensure_tesseract_available(tesseract_lang: str = "eng") -> None:
    tesseract_cmd = pytesseract.pytesseract.tesseract_cmd
    if Path(tesseract_cmd).is_absolute():
        tesseract_path = tesseract_cmd if Path(tesseract_cmd).exists() else None
    else:
        tesseract_path = shutil.which(tesseract_cmd)

    if tesseract_path is None:
        raise OCREnvironmentError(
            "Tesseract OCR is required but was not found on PATH. "
            "Install it with `sudo apt-get install tesseract-ocr tesseract-ocr-eng` "
            "on Ubuntu/Debian, or set `pytesseract.pytesseract.tesseract_cmd` "
            "to the full tesseract binary path before running OCR."
        )

    try:
        languages = set(pytesseract.get_languages(config=""))
    except pytesseract.TesseractNotFoundError as exc:
        raise OCREnvironmentError(
            "Tesseract OCR is required but pytesseract could not execute it. "
            "Verify the tesseract binary is installed and available on PATH."
        ) from exc

    requested_languages = {language for language in tesseract_lang.split("+") if language}
    missing_languages = sorted(requested_languages - languages)
    if missing_languages:
        raise OCREnvironmentError(
            "Tesseract OCR is installed, but the requested language data is missing: "
            f"{', '.join(missing_languages)}. Install the matching language package, "
            "for example `sudo apt-get install tesseract-ocr-eng` for English."
        )


def _clamp_layout_coordinate(value: int) -> int:
    return max(0, min(int(value), 1000))


def _sanitize_box(box: list[int]) -> list[int]:
    if len(box) != 4:
        return [0, 0, 0, 0]

    left = _clamp_layout_coordinate(box[0])
    top = _clamp_layout_coordinate(box[1])
    right = _clamp_layout_coordinate(box[2])
    bottom = _clamp_layout_coordinate(box[3])

    if right < left:
        right = left
    if bottom < top:
        bottom = top

    return [left, top, right, bottom]


def _normalize_box(x: int, y: int, w: int, h: int, width: int, height: int) -> list[int]:
    if width <= 0 or height <= 0:
        return [0, 0, 0, 0]

    left = int(1000 * x / width)
    top = int(1000 * y / height)
    right = int(1000 * (x + w) / width)
    bottom = int(1000 * (y + h) / height)
    return _sanitize_box([left, top, right, bottom])


def ocr_page(image_path: str | Path, tesseract_lang: str = "eng") -> OCRPage:
    ensure_tesseract_available(tesseract_lang=tesseract_lang)
    image_path = Path(image_path)
    image = Image.open(image_path).convert("RGB")

    cache_path: Path | None = None
    if _ocr_cache_enabled():
        cache_path = _ocr_cache_dir() / f"{_ocr_cache_key(image_path, tesseract_lang)}.json"
        cached_page = _read_ocr_cache(cache_path, image) if cache_path.exists() else None
        if cached_page is not None:
            return cached_page

    data = pytesseract.image_to_data(
        image,
        lang=tesseract_lang,
        output_type=pytesseract.Output.DICT,
    )

    width, height = image.size
    words: list[str] = []
    boxes: list[list[int]] = []

    for index, raw_text in enumerate(data["text"]):
        text = raw_text.strip()
        if not text:
            continue

        words.append(text)
        boxes.append(
            _normalize_box(
                data["left"][index],
                data["top"][index],
                data["width"][index],
                data["height"][index],
                width,
                height,
            )
        )

    text = " ".join(words)
    if cache_path is not None:
        _write_ocr_cache(cache_path, words=words, boxes=boxes, text=text)

    return OCRPage(image=image, words=words, boxes=boxes, text=text)


@lru_cache(maxsize=512)
def ocr_page_cached(image_path: str, tesseract_lang: str = "eng") -> OCRPage:
    return ocr_page(image_path, tesseract_lang=tesseract_lang)
