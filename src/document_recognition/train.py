from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
from transformers import (
    DefaultDataCollator,
    LayoutLMv3ForSequenceClassification,
    LayoutLMv3Processor,
    TrainerCallback,
    Trainer,
    TrainingArguments,
)

from .dataset import encode_dataset, load_csv_dataset
from .labels import ID_TO_LABEL, LABELS, LABEL_TO_ID
from .ocr import ensure_tesseract_available
from .training_control import TrainingStoppedError, check_training_control


@dataclass(slots=True)
class TrainConfig:
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


def compute_metrics(eval_pred: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = float((predictions == labels).mean())
    return {"accuracy": accuracy}


class ProgressTrainerCallback(TrainerCallback):
    def __init__(self, progress_callback=None, control_path: Path | None = None) -> None:
        self.progress_callback = progress_callback
        self.control_path = control_path
        self.start_time = 0.0

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        if self.progress_callback is not None:
            self.progress_callback(
                {
                    "phase": "train_baseline",
                    "current": 0,
                    "total": max(int(state.max_steps), 1),
                    "fraction": 0.0,
                    "message": "Starting baseline training...",
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
                "phase": "train_baseline",
                "current": steps_done,
                "total": steps_total,
                "fraction": min(steps_done / steps_total, 1.0),
                "message": f"Baseline training step {steps_done}/{steps_total}",
                "eta_seconds": eta_seconds,
            }
        )


def train_model(config: TrainConfig, progress_callback=None) -> Path:
    ensure_tesseract_available(tesseract_lang=config.tesseract_lang)

    processor = LayoutLMv3Processor.from_pretrained(
        config.pretrained_model_name,
        apply_ocr=False,
    )
    model = LayoutLMv3ForSequenceClassification.from_pretrained(
        config.pretrained_model_name,
        num_labels=len(LABELS),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )

    train_dataset = encode_dataset(
        load_csv_dataset(config.train_csv),
        processor=processor,
        max_length=config.max_length,
        tesseract_lang=config.tesseract_lang,
        num_proc=config.ocr_num_proc,
        control_path=config.control_path,
    )
    eval_dataset = encode_dataset(
        load_csv_dataset(config.eval_csv),
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
        save_strategy="epoch",
        logging_steps=config.logging_steps,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        remove_unused_columns=False,
        save_total_limit=2,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DefaultDataCollator(),
        compute_metrics=compute_metrics,
        callbacks=[ProgressTrainerCallback(progress_callback, control_path=config.control_path)],
    )
    trainer.train()

    config.output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(config.output_dir))
    processor.save_pretrained(str(config.output_dir))
    return config.output_dir
