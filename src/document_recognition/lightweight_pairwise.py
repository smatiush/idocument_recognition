from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Callable

import numpy as np

from .labels import PairLabel
from .ocr import ensure_tesseract_available, ocr_page_cached
from .pairwise_dataset import REQUIRED_PAIR_COLUMNS
from .pdf import pdf_to_images, split_pdf_ranges
from .postprocess import ranges_to_page_labels
from .training_control import check_training_control


ProgressCallback = Callable[[dict[str, object]], None]


@dataclass(slots=True)
class LightweightPairwiseTrainConfig:
    train_csv: Path
    eval_csv: Path
    output_dir: Path
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    classifier_type: str = "logistic_regression"
    tesseract_lang: str = "eng"
    max_pairs: int | None = None
    random_state: int = 42
    control_path: Path | None = None


@dataclass(slots=True)
class LightweightPairBoundaryPrediction:
    left_page: int
    right_page: int
    same_document_probability: float
    new_document_probability: float
    label: str


@dataclass(slots=True)
class LightweightPairwisePredictionResult:
    pair_predictions: list[LightweightPairBoundaryPrediction]
    page_labels: list[str]
    ranges: list[tuple[int, int]]
    output_pdfs: list[Path]


def _load_sentence_transformer(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "The lightweight pairwise pipeline requires `sentence-transformers`. "
            "Install the project again with `pip install -e .` or run "
            "`pip install sentence-transformers scikit-learn joblib`."
        ) from exc

    return SentenceTransformer(model_name)


def _load_classifier(classifier_type: str, random_state: int):
    if classifier_type == "logistic_regression":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state),
        )

    if classifier_type == "random_forest":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )

    raise ValueError("classifier_type must be either `logistic_regression` or `random_forest`.")


def _read_pair_manifest(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        fieldnames = set(reader.fieldnames or [])
        missing_columns = REQUIRED_PAIR_COLUMNS - fieldnames
        if missing_columns:
            raise ValueError(
                f"Pairwise training requires columns {sorted(REQUIRED_PAIR_COLUMNS)}. "
                f"The CSV at {csv_path} is missing {sorted(missing_columns)}."
            )
        rows = [dict(row) for row in reader]

    if not rows:
        raise ValueError(f"No rows found in pairwise manifest: {csv_path}")
    return rows


def _page_text(image_path: str | Path, tesseract_lang: str) -> str:
    page = ocr_page_cached(str(image_path), tesseract_lang=tesseract_lang)
    return page.text


def _text_features(left_text: str, right_text: str) -> list[float]:
    left_words = left_text.split()
    right_words = right_text.split()
    left_set = {word.lower() for word in left_words}
    right_set = {word.lower() for word in right_words}
    union = left_set | right_set
    intersection = left_set & right_set
    jaccard = len(intersection) / len(union) if union else 0.0

    return [
        float(len(left_words)),
        float(len(right_words)),
        abs(float(len(left_words) - len(right_words))),
        float(len(left_text)),
        float(len(right_text)),
        abs(float(len(left_text) - len(right_text))),
        jaccard,
        1.0 if "page 1" in right_text.lower() or "page: 1" in right_text.lower() else 0.0,
        1.0 if "invoice" in right_text.lower() else 0.0,
        1.0 if "delivery note" in right_text.lower() else 0.0,
    ]


def _pair_text(left_text: str, right_text: str) -> str:
    return f"LEFT PAGE:\n{left_text}\n\nRIGHT PAGE:\n{right_text}"


def _build_examples(
    rows: list[dict[str, str]],
    tesseract_lang: str,
    progress_callback: ProgressCallback | None = None,
    phase: str = "ocr",
    control_path: Path | None = None,
) -> tuple[list[str], list[list[float]], list[str]]:
    pair_texts: list[str] = []
    numeric_features: list[list[float]] = []
    labels: list[str] = []
    total = len(rows)
    start_time = time.time()

    for index, row in enumerate(rows, start=1):
        check_training_control(control_path)
        label = str(row["label"])
        if label not in {PairLabel.SAME_DOCUMENT.value, PairLabel.NEW_DOCUMENT.value}:
            raise ValueError(f"Unsupported pair label {label!r} in manifest.")

        left_text = _page_text(row["left_image_path"], tesseract_lang=tesseract_lang)
        right_text = _page_text(row["right_image_path"], tesseract_lang=tesseract_lang)
        pair_texts.append(_pair_text(left_text, right_text))
        numeric_features.append(_text_features(left_text, right_text))
        labels.append(label)

        if progress_callback is not None:
            elapsed = max(time.time() - start_time, 1e-6)
            rate = index / elapsed
            eta_seconds = int((total - index) / rate) if rate > 0 else None
            progress_callback(
                {
                    "phase": phase,
                    "current": index,
                    "total": total,
                    "fraction": index / max(total, 1),
                    "message": f"OCR pair {index}/{total}",
                    "eta_seconds": eta_seconds,
                }
            )

    return pair_texts, numeric_features, labels


def _encode_features(model, pair_texts: list[str], numeric_features: list[list[float]], batch_size: int = 16) -> np.ndarray:
    embeddings = model.encode(
        pair_texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    numeric = np.asarray(numeric_features, dtype=np.float32)
    return np.hstack([embeddings.astype(np.float32), numeric])


def _positive_probability(classifier, features: np.ndarray, positive_label: str) -> np.ndarray:
    probabilities = classifier.predict_proba(features)
    classes = list(classifier.classes_)
    if positive_label not in classes:
        raise ValueError(f"Classifier was not trained with label {positive_label!r}. Classes: {classes}")
    return probabilities[:, classes.index(positive_label)]


def _compute_metrics(labels: list[str], new_document_probabilities: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    actual = np.asarray([label == PairLabel.NEW_DOCUMENT.value for label in labels], dtype=bool)
    predicted = new_document_probabilities >= threshold
    accuracy = float((predicted == actual).mean())
    true_positive = float(np.logical_and(predicted, actual).sum())
    precision = true_positive / max(float(predicted.sum()), 1.0)
    recall = true_positive / max(float(actual.sum()), 1.0)
    return {
        "accuracy": accuracy,
        "boundary_precision": precision,
        "boundary_recall": recall,
    }


def train_lightweight_pairwise_model(
    config: LightweightPairwiseTrainConfig,
    progress_callback: ProgressCallback | None = None,
) -> Path:
    ensure_tesseract_available(tesseract_lang=config.tesseract_lang)

    train_rows = _read_pair_manifest(config.train_csv)
    eval_rows = _read_pair_manifest(config.eval_csv)
    if config.max_pairs is not None:
        train_rows = train_rows[: config.max_pairs]
        eval_rows = eval_rows[: config.max_pairs]

    train_texts, train_numeric, train_labels = _build_examples(
        train_rows,
        tesseract_lang=config.tesseract_lang,
        progress_callback=progress_callback,
        phase="train_ocr",
        control_path=config.control_path,
    )
    eval_texts, eval_numeric, eval_labels = _build_examples(
        eval_rows,
        tesseract_lang=config.tesseract_lang,
        progress_callback=progress_callback,
        phase="eval_ocr",
        control_path=config.control_path,
    )

    check_training_control(config.control_path)
    if progress_callback is not None:
        progress_callback(
            {
                "phase": "embedding",
                "current": 0,
                "total": 1,
                "fraction": 0.0,
                "message": "Encoding text embeddings...",
                "eta_seconds": None,
            }
        )

    embedding_model = _load_sentence_transformer(config.embedding_model_name)
    train_features = _encode_features(embedding_model, train_texts, train_numeric)
    eval_features = _encode_features(embedding_model, eval_texts, eval_numeric)

    check_training_control(config.control_path)
    classifier = _load_classifier(config.classifier_type, random_state=config.random_state)
    classifier.fit(train_features, train_labels)

    eval_new_probabilities = _positive_probability(classifier, eval_features, PairLabel.NEW_DOCUMENT.value)
    metrics = _compute_metrics(eval_labels, eval_new_probabilities)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    import joblib

    joblib.dump(classifier, config.output_dir / "classifier.joblib")
    (config.output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "embedding_model_name": config.embedding_model_name,
                "classifier_type": config.classifier_type,
                "tesseract_lang": config.tesseract_lang,
                "feature_version": 1,
                "metrics": metrics,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    if progress_callback is not None:
        progress_callback(
            {
                "phase": "complete",
                "current": 1,
                "total": 1,
                "fraction": 1.0,
                "message": "Lightweight pairwise training complete.",
                "eta_seconds": 0,
            }
        )

    return config.output_dir


def _load_saved_model(model_dir: Path):
    metadata_path = model_dir / "metadata.json"
    classifier_path = model_dir / "classifier.joblib"
    if not metadata_path.exists() or not classifier_path.exists():
        raise ValueError(
            f"Lightweight model not found in {model_dir}. Expected `metadata.json` and `classifier.joblib`."
        )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    embedding_model = _load_sentence_transformer(str(metadata["embedding_model_name"]))
    import joblib

    classifier = joblib.load(classifier_path)
    return metadata, embedding_model, classifier


def _build_ranges_from_new_document_probs(
    new_document_probabilities: list[float],
    threshold: float,
) -> list[tuple[int, int]]:
    if not new_document_probabilities:
        return [(1, 1)]

    ranges: list[tuple[int, int]] = []
    start_page = 1
    for left_page, new_document_probability in enumerate(new_document_probabilities, start=1):
        if new_document_probability >= threshold:
            ranges.append((start_page, left_page))
            start_page = left_page + 1
    ranges.append((start_page, len(new_document_probabilities) + 1))
    return ranges


def predict_pdf_boundaries_lightweight_pairwise(
    pdf_path: str | Path,
    model_dir: str | Path,
    work_dir: str | Path,
    dpi: int = 200,
    tesseract_lang: str | None = None,
    threshold: float = 0.7,
    split_output: bool = True,
) -> LightweightPairwisePredictionResult:
    pdf_path = Path(pdf_path)
    model_dir = Path(model_dir)
    work_dir = Path(work_dir)
    image_dir = work_dir / "pages"
    split_dir = work_dir / "split_docs"

    metadata, embedding_model, classifier = _load_saved_model(model_dir)
    ocr_lang = tesseract_lang or str(metadata.get("tesseract_lang", "eng"))
    ensure_tesseract_available(tesseract_lang=ocr_lang)

    image_paths = pdf_to_images(pdf_path, image_dir, dpi=dpi)
    if len(image_paths) == 1:
        ranges = [(1, 1)]
        return LightweightPairwisePredictionResult(
            pair_predictions=[],
            page_labels=ranges_to_page_labels(ranges, total_pages=1),
            ranges=ranges,
            output_pdfs=split_pdf_ranges(pdf_path, ranges, split_dir) if split_output else [],
        )

    pair_texts: list[str] = []
    numeric_features: list[list[float]] = []
    for left_image_path, right_image_path in zip(image_paths, image_paths[1:], strict=False):
        left_text = _page_text(left_image_path, tesseract_lang=ocr_lang)
        right_text = _page_text(right_image_path, tesseract_lang=ocr_lang)
        pair_texts.append(_pair_text(left_text, right_text))
        numeric_features.append(_text_features(left_text, right_text))

    features = _encode_features(embedding_model, pair_texts, numeric_features)
    new_document_probabilities = _positive_probability(classifier, features, PairLabel.NEW_DOCUMENT.value)

    pair_predictions: list[LightweightPairBoundaryPrediction] = []
    for left_page_number, new_document_probability in enumerate(new_document_probabilities, start=1):
        same_document_probability = 1.0 - float(new_document_probability)
        pair_predictions.append(
            LightweightPairBoundaryPrediction(
                left_page=left_page_number,
                right_page=left_page_number + 1,
                same_document_probability=same_document_probability,
                new_document_probability=float(new_document_probability),
                label=(
                    PairLabel.NEW_DOCUMENT.value
                    if float(new_document_probability) >= threshold
                    else PairLabel.SAME_DOCUMENT.value
                ),
            )
        )

    ranges = _build_ranges_from_new_document_probs(
        [float(probability) for probability in new_document_probabilities],
        threshold=threshold,
    )
    page_labels = ranges_to_page_labels(ranges, total_pages=len(image_paths))
    output_pdfs = split_pdf_ranges(pdf_path, ranges, split_dir) if split_output else []
    return LightweightPairwisePredictionResult(
        pair_predictions=pair_predictions,
        page_labels=page_labels,
        ranges=ranges,
        output_pdfs=output_pdfs,
    )
