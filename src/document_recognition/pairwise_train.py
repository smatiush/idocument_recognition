from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
from transformers import DefaultDataCollator, LayoutLMv3Processor, Trainer, TrainerCallback, TrainingArguments

from .labels import PAIR_LABEL_TO_ID
from .ocr import ensure_tesseract_available
from .pairwise_dataset import encode_pair_dataset, load_pair_csv_dataset
from .pairwise_model import PairwiseLayoutLMv3Classifier
from .training_control import TrainingStoppedError, check_training_control


@dataclass(slots=True)
class PairwiseTrainConfig:
    train_csv: Path
    eval_csv: Path
    output_dir: Path
    pretrained_model_name: str = "microsoft/layoutlmv3-base"
    learning_rate: float = 2e-5
    train_batch_size: int = 2
    eval_batch_size: int = 2
    num_train_epochs: int = 5
    max_length: int = 512
    logging_steps: int = 20
    tesseract_lang: str = "eng"
    ocr_num_proc: int = 1
    control_path: Path | None = None
    classifier_dropout: float = 0.1


def compute_pairwise_metrics(eval_pred: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = float((predictions == labels).mean())

    positive_id = PAIR_LABEL_TO_ID["NEW_DOCUMENT"]
    predicted_positive = predictions == positive_id
    actual_positive = labels == positive_id
    true_positive = float(np.logical_and(predicted_positive, actual_positive).sum())
    precision = true_positive / max(float(predicted_positive.sum()), 1.0)
    recall = true_positive / max(float(actual_positive.sum()), 1.0)

    return {
        "accuracy": accuracy,
        "boundary_precision": precision,
        "boundary_recall": recall,
    }


class PairwiseProgressTrainerCallback(TrainerCallback):
    def __init__(self, progress_callback=None, control_path: Path | None = None) -> None:
        self.progress_callback = progress_callback
        self.control_path = control_path
        self.start_time = 0.0

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        if self.progress_callback is not None:
            self.progress_callback(
                {
                    "phase": "train_pairwise",
                    "current": 0,
                    "total": max(int(state.max_steps), 1),
                    "fraction": 0.0,
                    "message": "Starting pairwise training...",
                    "eta_seconds": None,
                }
            )

    def on_step_end(self, args, state, control, **kwargs):
        try:
            check_training_control(self.control_path)
        except TrainingStoppedError:
            control.should_training_stop = True
            raise

        if self.progress_callback is None or state.max_steps <= 0:
            return
        elapsed = max(time.time() - self.start_time, 1e-6)
        steps_done = int(state.global_step)
        steps_total = int(state.max_steps)
        rate = steps_done / elapsed if steps_done else 0.0
        eta_seconds = int((steps_total - steps_done) / rate) if rate > 0 else None
        self.progress_callback(
            {
                "phase": "train_pairwise",
                "current": steps_done,
                "total": steps_total,
                "fraction": min(steps_done / steps_total, 1.0),
                "message": f"Pairwise training step {steps_done}/{steps_total}",
                "eta_seconds": eta_seconds,
            }
        )


def train_pairwise_model(config: PairwiseTrainConfig, progress_callback=None) -> Path:
    ensure_tesseract_available(tesseract_lang=config.tesseract_lang)

    processor = LayoutLMv3Processor.from_pretrained(
        config.pretrained_model_name,
        apply_ocr=False,
    )
    model = PairwiseLayoutLMv3Classifier.from_pretrained_backbone(
        config.pretrained_model_name,
        classifier_dropout=config.classifier_dropout,
    )

    train_dataset = encode_pair_dataset(
        load_pair_csv_dataset(config.train_csv),
        processor=processor,
        max_length=config.max_length,
        tesseract_lang=config.tesseract_lang,
        num_proc=config.ocr_num_proc,
        control_path=config.control_path,
    )
    eval_dataset = encode_pair_dataset(
        load_pair_csv_dataset(config.eval_csv),
        processor=processor,
        max_length=config.max_length,
        tesseract_lang=config.tesseract_lang,
        num_proc=config.ocr_num_proc,
        control_path=config.control_path,
    )

    args = TrainingArguments(
        output_dir=str(config.output_dir),
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.train_batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        num_train_epochs=config.num_train_epochs,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=config.logging_steps,
        remove_unused_columns=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DefaultDataCollator(),
        compute_metrics=compute_pairwise_metrics,
        callbacks=[PairwiseProgressTrainerCallback(progress_callback, control_path=config.control_path)],
    )
    trainer.train()

    config.output_dir.mkdir(parents=True, exist_ok=True)
    model.save(config.output_dir, processor)
    return config.output_dir
