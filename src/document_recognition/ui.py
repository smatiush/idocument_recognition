from __future__ import annotations

import csv
from pathlib import Path
import sys
import threading
import uuid

import streamlit as st

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from document_recognition.inference import predict_pdf_boundaries
from document_recognition.dataset_split import SplitManifestConfig, split_manifest_csv
from document_recognition.lightweight_pairwise import (
    LightweightPairwiseTrainConfig,
    predict_pdf_boundaries_lightweight_pairwise,
    train_lightweight_pairwise_model,
)
from document_recognition.ocr import OCREnvironmentError
from document_recognition.pairwise_inference import predict_pdf_boundaries_pairwise
from document_recognition.pairwise_train import (
    PairwiseEvalConfig,
    PairwiseTrainConfig,
    evaluate_pairwise_model,
    train_pairwise_model,
)
from document_recognition.source_validation import SourceValidationConfig, validate_source_pdfs
from document_recognition.synthetic import SyntheticDatasetConfig, create_synthetic_merged_dataset
from document_recognition.train import TrainConfig, train_model
from document_recognition.training_control import TrainingStoppedError, write_training_control


_RESULT_PREFIX = "step_result_"
_TRAINING_CONTROL_DIR = Path("/tmp/document_recognition_training_controls")


class TrainingJob:
    def __init__(self, name: str, control_path: Path, output_dir: Path) -> None:
        self.name = name
        self.control_path = control_path
        self.output_dir = output_dir
        self.lock = threading.Lock()
        self.status = "starting"
        self.message = "Starting..."
        self.fraction = 0.0
        self.eta_seconds = None
        self.error = ""
        self.result = None
        self.logs: list[str] = ["Starting job..."]
        self.thread: threading.Thread | None = None

    def _append_log_locked(self, message: str) -> None:
        self.logs.append(message)
        self.logs = self.logs[-200:]

    def append_log(self, message: str) -> None:
        with self.lock:
            self._append_log_locked(message)

    def update_progress(self, event: dict[str, object]) -> None:
        with self.lock:
            self.status = "running"
            self.message = str(event.get("message", "Working..."))
            self.fraction = max(0.0, min(float(event.get("fraction", 0.0)), 1.0))
            self.eta_seconds = event.get("eta_seconds")
            current = event.get("current")
            total = event.get("total")
            phase = event.get("phase")
            prefix = f"[{phase}] " if phase else ""
            suffix = f" ({current}/{total})" if current is not None and total is not None else ""
            self._append_log_locked(f"{prefix}{self.message}{suffix}")

    def set_status(self, status: str, message: str, fraction: float | None = None) -> None:
        with self.lock:
            self.status = status
            self.message = message
            if fraction is not None:
                self.fraction = max(0.0, min(fraction, 1.0))
            self._append_log_locked(f"[{status}] {message}")

    def set_error(self, message: str) -> None:
        with self.lock:
            self.status = "failed"
            self.error = message
            self.message = message
            self._append_log_locked(f"[failed] {message}")

    def set_result(self, result) -> None:
        with self.lock:
            self.result = _stringify_paths(result)

    def snapshot(self) -> dict[str, object]:
        with self.lock:
            return {
                "name": self.name,
                "control_path": self.control_path,
                "output_dir": self.output_dir,
                "status": self.status,
                "message": self.message,
                "fraction": self.fraction,
                "eta_seconds": self.eta_seconds,
                "error": self.error,
                "result": self.result,
                "logs": list(self.logs),
                "alive": self.thread.is_alive() if self.thread is not None else False,
            }


def _path_input(label: str, key: str, value: str = "") -> Path:
    return Path(st.text_input(label, value=value, key=key)).expanduser()


def _stringify_paths(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _stringify_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_stringify_paths(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_stringify_paths(item) for item in value)
    return value


def _save_step_result(step_key: str, status: str, data) -> None:
    st.session_state[f"{_RESULT_PREFIX}{step_key}"] = {
        "status": status,
        "data": _stringify_paths(data),
    }


def _get_step_result(step_key: str):
    return st.session_state.get(f"{_RESULT_PREFIX}{step_key}")


def _render_saved_result(step_key: str, renderer) -> None:
    result = _get_step_result(step_key)
    if result is None:
        return
    st.success(result["status"])
    renderer(result["data"])


def _render_ranges(ranges: list[tuple[int, int]]) -> None:
    for index, (start_page, end_page) in enumerate(ranges, start=1):
        st.write(f"doc_{index}: pages {start_page}-{end_page}")


def _render_split_paths(paths: list[Path]) -> None:
    if not paths:
        return
    st.write("Split PDFs")
    for path in paths:
        st.code(str(path))


def _render_source_validation_result(data: dict[str, object]) -> None:
    st.write(data)
    st.info("Use the filtered directory as the input for synthetic dataset generation.")


def _render_baseline_prediction_result(data: dict[str, object]) -> None:
    st.write("Page predictions")
    st.write(data["predictions"])
    st.write("Document ranges")
    _render_ranges(data["ranges"])
    _render_split_paths(data["output_pdfs"])


def _render_pairwise_prediction_result(data: dict[str, object]) -> None:
    st.write("Pair predictions")
    st.write(data["pair_predictions"])
    st.write("Page labels")
    st.write(data["page_labels"])
    st.write("Document ranges")
    _render_ranges(data["ranges"])
    _render_split_paths(data["output_pdfs"])


def _render_pairwise_eval_result(data: dict[str, object]) -> None:
    st.write("Metrics")
    st.write(data)


def _format_eta(eta_seconds: object) -> str:
    if eta_seconds is None:
        return "estimating..."
    total_seconds = max(int(eta_seconds), 0)
    minutes, seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def _make_progress_callback(progress_bar, status_placeholder):
    def callback(event: dict[str, object]) -> None:
        fraction = float(event.get("fraction", 0.0))
        message = str(event.get("message", "Working..."))
        eta = _format_eta(event.get("eta_seconds"))
        progress_bar.progress(max(0.0, min(fraction, 1.0)))
        status_placeholder.info(f"{message}  ETA: {eta}")

    return callback


def _make_job_progress_callback(job: TrainingJob):
    def callback(event: dict[str, object]) -> None:
        job.update_progress(event)

    return callback


def _get_training_job(job_key: str) -> TrainingJob | None:
    return st.session_state.get(job_key)


def _set_training_control(job: TrainingJob, state: str) -> None:
    write_training_control(job.control_path, state)
    if state == "paused":
        job.set_status("paused", "Paused. Click Play to continue.")
    elif state == "running":
        job.set_status("running", "Running...")
    elif state == "stopped":
        job.set_status("stopping", "Stopping after the current OCR item or training step...")


def _render_training_job_controls(job_key: str, start_job) -> None:
    job = _get_training_job(job_key)
    if job is None:
        return

    snapshot = job.snapshot()
    status = str(snapshot["status"])
    st.write(f"Status: {status}")
    st.progress(float(snapshot["fraction"]))
    st.info(f"{snapshot['message']}  ETA: {_format_eta(snapshot['eta_seconds'])}")

    if snapshot["error"]:
        st.error(str(snapshot["error"]))

    with st.expander("Training log", expanded=True):
        logs = snapshot.get("logs", [])
        st.text_area(
            "Recent events",
            value="\n".join(str(line) for line in logs),
            height=220,
            key=f"{job_key}_logs",
            disabled=True,
            label_visibility="collapsed",
        )

    cols = st.columns(4)
    with cols[0]:
        if st.button("Play", key=f"{job_key}_play", disabled=status not in {"paused", "stopping"}):
            _set_training_control(job, "running")
            st.rerun()
    with cols[1]:
        if st.button("Pause", key=f"{job_key}_pause", disabled=status not in {"running", "starting"}):
            _set_training_control(job, "paused")
            st.rerun()
    with cols[2]:
        if st.button("Stop", key=f"{job_key}_stop", disabled=status in {"completed", "failed", "stopped"}):
            _set_training_control(job, "stopped")
            st.rerun()
    with cols[3]:
        if st.button("Restart", key=f"{job_key}_restart"):
            _set_training_control(job, "stopped")
            start_job()
            st.rerun()


def _new_control_path(step_key: str) -> Path:
    return _TRAINING_CONTROL_DIR / f"{step_key}_{uuid.uuid4().hex}.json"


def _start_baseline_training_job(config: TrainConfig, output_dir: Path) -> TrainingJob:
    control_path = _new_control_path("baseline")
    write_training_control(control_path, "running")
    config.control_path = control_path
    job = TrainingJob("Baseline training", control_path=control_path, output_dir=output_dir)

    def target() -> None:
        try:
            job.append_log("Loading baseline model and preparing OCR dataset...")
            train_model(config, progress_callback=_make_job_progress_callback(job))
        except TrainingStoppedError as exc:
            job.set_status("stopped", str(exc))
            return
        except (OCREnvironmentError, ValueError) as exc:
            job.set_error(str(exc))
            return
        except Exception as exc:
            job.set_error(f"{type(exc).__name__}: {exc}")
            return

        job.set_status("completed", f"Baseline model saved to {output_dir}", fraction=1.0)

    thread = threading.Thread(target=target, daemon=True)
    job.thread = thread
    st.session_state["baseline_train_job"] = job
    thread.start()
    return job


def _start_pairwise_training_job(config: PairwiseTrainConfig, output_dir: Path) -> TrainingJob:
    control_path = _new_control_path("pairwise")
    write_training_control(control_path, "running")
    config.control_path = control_path
    job = TrainingJob("Pairwise training", control_path=control_path, output_dir=output_dir)

    def target() -> None:
        try:
            job.append_log("Loading LayoutLMv3 processor and pairwise model...")
            job.append_log("If this is the first run, Hugging Face model download can take several minutes.")
            job.append_log("After model loading, OCR and dataset encoding will start.")
            train_pairwise_model(config, progress_callback=_make_job_progress_callback(job))
        except TrainingStoppedError as exc:
            job.set_status("stopped", str(exc))
            return
        except (OCREnvironmentError, ValueError) as exc:
            job.set_error(str(exc))
            return
        except Exception as exc:
            job.set_error(f"{type(exc).__name__}: {exc}")
            return

        job.set_status("completed", f"Pairwise model saved to {output_dir}", fraction=1.0)

    thread = threading.Thread(target=target, daemon=True)
    job.thread = thread
    st.session_state["pairwise_train_job"] = job
    thread.start()
    return job


def _start_pairwise_evaluation_job(config: PairwiseEvalConfig, model_dir: Path) -> TrainingJob:
    control_path = _new_control_path("pairwise_eval")
    write_training_control(control_path, "running")
    config.control_path = control_path
    job = TrainingJob("Pairwise evaluation", control_path=control_path, output_dir=model_dir)

    def target() -> None:
        try:
            job.append_log("Loading saved LayoutLMv3 pairwise model...")
            job.append_log("OCR and dataset encoding will run before metric calculation.")
            metrics = evaluate_pairwise_model(config, progress_callback=_make_job_progress_callback(job))
            job.set_result(metrics)
        except TrainingStoppedError as exc:
            job.set_status("stopped", str(exc))
            return
        except (OCREnvironmentError, ValueError) as exc:
            job.set_error(str(exc))
            return
        except Exception as exc:
            job.set_error(f"{type(exc).__name__}: {exc}")
            return

        job.set_status("completed", "Pairwise evaluation complete.", fraction=1.0)

    thread = threading.Thread(target=target, daemon=True)
    job.thread = thread
    st.session_state["pairwise_eval_job"] = job
    thread.start()
    return job


def _start_lightweight_pairwise_training_job(config: LightweightPairwiseTrainConfig, output_dir: Path) -> TrainingJob:
    control_path = _new_control_path("lightweight_pairwise")
    write_training_control(control_path, "running")
    config.control_path = control_path
    job = TrainingJob("Lightweight pairwise training", control_path=control_path, output_dir=output_dir)

    def target() -> None:
        try:
            job.append_log("Preparing OCR text pairs for lightweight training...")
            job.append_log("If this is the first run, the sentence-transformer model will be downloaded.")
            train_lightweight_pairwise_model(config, progress_callback=_make_job_progress_callback(job))
        except TrainingStoppedError as exc:
            job.set_status("stopped", str(exc))
            return
        except (OCREnvironmentError, RuntimeError, ValueError) as exc:
            job.set_error(str(exc))
            return
        except Exception as exc:
            job.set_error(f"{type(exc).__name__}: {exc}")
            return

        job.set_status("completed", f"Lightweight pairwise model saved to {output_dir}", fraction=1.0)

    thread = threading.Thread(target=target, daemon=True)
    job.thread = thread
    st.session_state["lightweight_pairwise_train_job"] = job
    thread.start()
    return job


def _require_existing_file(path: Path, label: str) -> bool:
    if path.exists() and path.is_file():
        return True
    st.error(f"{label} not found: {path}")
    return False


def _csv_columns(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.reader(file)
        return next(reader, [])


def _validate_manifest_split_input(input_csv: Path) -> bool:
    columns = set(_csv_columns(input_csv))
    if {"image_path", "label"}.issubset(columns) or {"left_image_path", "right_image_path", "label"}.issubset(columns):
        return True

    st.error(
        "This CSV is not a training manifest. Use `data/synthetic/page_labels.csv` "
        "or `data/synthetic/pair_labels.csv`. The source validation report cannot be used directly for training."
    )
    return False


def _manifest_split_tab() -> None:
    st.subheader("Prepare Train/Eval Splits")
    st.caption("Split a generated manifest into train and eval CSVs before running model training.")
    st.info("If you keep auto-splitting enabled in `Synthetic Data`, you usually do not need this tab.")

    with st.form("manifest_split_form"):
        input_csv = _path_input("Input manifest CSV", "manifest_split_input_csv", "data/synthetic/page_labels.csv")
        output_train_csv = _path_input(
            "Output train CSV",
            "manifest_split_output_train_csv",
            "data/synthetic/page_labels_train.csv",
        )
        output_eval_csv = _path_input(
            "Output eval CSV",
            "manifest_split_output_eval_csv",
            "data/synthetic/page_labels_eval.csv",
        )
        eval_ratio = st.number_input("Eval ratio", min_value=0.05, max_value=0.5, value=0.2)
        seed = st.number_input("Random seed", min_value=0, value=42, key="manifest_split_seed")
        submitted = st.form_submit_button("Create train/eval split")

    if submitted:
        if not _require_existing_file(input_csv, "Input manifest CSV"):
            return
        if not _validate_manifest_split_input(input_csv):
            return

        outputs = split_manifest_csv(
            SplitManifestConfig(
                input_csv=input_csv,
                output_train_csv=output_train_csv,
                output_eval_csv=output_eval_csv,
                eval_ratio=float(eval_ratio),
                seed=int(seed),
            )
        )
        st.success("Split created.")
        result_data = {key: str(value) if isinstance(value, Path) else value for key, value in outputs.items()}
        _save_step_result("manifest_split", "Split created.", result_data)
        st.write(result_data)
    else:
        _render_saved_result("manifest_split", st.write)


def _synthetic_tab() -> None:
    st.subheader("Synthetic Dataset Generation")
    st.caption("Create merged PDFs and auto-labeled manifests from already separated single-document PDFs.")

    with st.form("generate_synthetic_form"):
        input_dir = _path_input("Input directory with single-document PDFs", "synthetic_input_dir", "data/source_docs")
        output_dir = _path_input("Output directory", "synthetic_output_dir", "data/synthetic")
        num_merged_pdfs = st.number_input("Number of merged PDFs", min_value=1, value=100)
        min_docs_per_merge = st.number_input("Minimum source documents per merged PDF", min_value=1, value=2)
        max_docs_per_merge = st.number_input("Maximum source documents per merged PDF", min_value=1, value=5)
        dpi = st.number_input("Render DPI", min_value=72, value=200)
        seed = st.number_input("Random seed", min_value=0, value=42)
        auto_create_splits = st.checkbox("Automatically create train/eval split CSVs", value=True)
        eval_ratio = st.number_input("Eval ratio for auto-split", min_value=0.05, max_value=0.5, value=0.2)
        submitted = st.form_submit_button("Generate synthetic dataset")

    if submitted:
        progress_bar = st.progress(0.0)
        status_placeholder = st.empty()
        with st.spinner("Generating synthetic merged dataset..."):
            outputs = create_synthetic_merged_dataset(
                SyntheticDatasetConfig(
                    input_dir=input_dir,
                    output_dir=output_dir,
                    num_merged_pdfs=int(num_merged_pdfs),
                    min_docs_per_merge=int(min_docs_per_merge),
                    max_docs_per_merge=int(max_docs_per_merge),
                    dpi=int(dpi),
                    seed=int(seed),
                    auto_create_splits=auto_create_splits,
                    eval_ratio=float(eval_ratio),
                ),
                progress_callback=_make_progress_callback(progress_bar, status_placeholder),
            )

        progress_bar.progress(1.0)
        status_placeholder.success("Synthetic generation finished.")
        st.success("Synthetic dataset generated.")
        result_data = {name: str(path) for name, path in outputs.items()}
        _save_step_result("synthetic", "Synthetic dataset generated.", result_data)
        st.json(result_data)
    else:
        _render_saved_result("synthetic", st.json)


def _source_validation_tab() -> None:
    st.subheader("Source PDF Validation")
    st.caption("Scan a folder of source PDFs, reject unreadable or oversized files, and write a cleaned subset.")

    with st.form("source_validation_form"):
        input_dir = _path_input("Input PDF directory", "source_validation_input_dir", "data/Pdf")
        output_dir = _path_input("Validation output directory", "source_validation_output_dir", "artifacts/source_validation")
        min_pages = st.number_input("Minimum allowed pages", min_value=1, value=1)
        max_pages = st.number_input("Maximum allowed pages", min_value=1, value=10)
        copy_filtered_pdfs = st.checkbox("Copy accepted PDFs into filtered output folder", value=True)
        deduplicate = st.checkbox("Drop duplicate PDFs by file content hash", value=True)
        submitted = st.form_submit_button("Validate and filter source PDFs")

    if submitted:
        progress_bar = st.progress(0.0)
        status_placeholder = st.empty()
        with st.spinner("Validating source PDFs..."):
            outputs = validate_source_pdfs(
                SourceValidationConfig(
                    input_dir=input_dir,
                    output_dir=output_dir,
                    min_pages=int(min_pages),
                    max_pages=int(max_pages),
                    copy_filtered_pdfs=copy_filtered_pdfs,
                    deduplicate=deduplicate,
                ),
                progress_callback=_make_progress_callback(progress_bar, status_placeholder),
            )

        progress_bar.progress(1.0)
        status_placeholder.success("Source validation finished.")
        st.success("Validation complete.")
        result_data = {
            "total_pdfs": outputs["total_pdfs"],
            "kept_pdfs": outputs["kept_pdfs"],
            "manifest_path": str(outputs["manifest_path"]),
            "summary_path": str(outputs["summary_path"]),
            "filtered_dir": str(outputs["filtered_dir"]),
        }
        _save_step_result("source_validation", "Validation complete.", result_data)
        _render_source_validation_result(result_data)
    else:
        _render_saved_result("source_validation", _render_source_validation_result)


def _baseline_train_tab() -> None:
    st.subheader("Baseline Training")
    st.caption("Train the page-level LayoutLMv3 classifier with START/MIDDLE/END/SINGLE labels.")
    st.info("Use a split manifest such as `data/synthetic/page_labels_train.csv` and `data/synthetic/page_labels_eval.csv`.")

    with st.form("baseline_train_form"):
        train_csv = _path_input("Train CSV", "baseline_train_csv", "data/synthetic/page_labels_train.csv")
        st.caption("Page-label training CSV with `image_path` and `label` columns.")
        eval_csv = _path_input("Eval CSV", "baseline_eval_csv", "data/synthetic/page_labels_eval.csv")
        st.caption("Held-out page-label CSV used to evaluate accuracy after each epoch.")
        output_dir = _path_input("Output model directory", "baseline_output_dir", "artifacts/layoutlmv3-boundary")
        st.caption("Directory where the trained model and processor are saved.")
        pretrained_model_name = st.text_input(
            "Pretrained model name",
            value="microsoft/layoutlmv3-base",
            help="Hugging Face model checkpoint used as the starting point.",
        )
        learning_rate = st.number_input(
            "Learning rate",
            min_value=0.0,
            value=2e-5,
            format="%.8f",
            help="Optimizer step size. Lower is safer; higher trains faster but can destabilize training.",
        )
        train_batch_size = st.number_input(
            "Train batch size",
            min_value=1,
            value=2,
            help="Number of pages per training batch. Increase only if memory allows.",
        )
        eval_batch_size = st.number_input(
            "Eval batch size",
            min_value=1,
            value=2,
            help="Number of pages per evaluation batch. Can be higher than train batch size if memory allows.",
        )
        num_train_epochs = st.number_input(
            "Epochs",
            min_value=1,
            value=5,
            help="Number of full passes over the training CSV.",
        )
        max_length = st.number_input(
            "Max token length",
            min_value=64,
            value=512,
            help="Maximum OCR tokens sent to LayoutLMv3 for each page.",
        )
        logging_steps = st.number_input(
            "Logging steps",
            min_value=1,
            value=20,
            help="How often Trainer emits progress during model training.",
        )
        tesseract_lang = st.text_input(
            "Tesseract language",
            value="eng",
            help="OCR language code installed in Tesseract, for example `eng` or `eng+ita`.",
        )
        ocr_num_proc = st.number_input(
            "OCR workers",
            min_value=1,
            max_value=12,
            value=4,
            help="Parallel OCR preprocessing workers. Start with 4; raise to 6 or 8 if CPU and RAM remain stable.",
        )
        submitted = st.form_submit_button("Start baseline training")

    def start_job() -> None:
        if not _require_existing_file(train_csv, "Train CSV") or not _require_existing_file(eval_csv, "Eval CSV"):
            st.warning("If you only have `page_labels.csv`, first use the `Prepare Train/Eval Splits` tab.")
            return

        _start_baseline_training_job(
            TrainConfig(
                train_csv=train_csv,
                eval_csv=eval_csv,
                output_dir=output_dir,
                pretrained_model_name=pretrained_model_name,
                learning_rate=float(learning_rate),
                train_batch_size=int(train_batch_size),
                eval_batch_size=int(eval_batch_size),
                num_train_epochs=int(num_train_epochs),
                max_length=int(max_length),
                logging_steps=int(logging_steps),
                tesseract_lang=tesseract_lang,
                ocr_num_proc=int(ocr_num_proc),
            ),
            output_dir=output_dir,
        )

    if submitted:
        start_job()
    else:
        _render_saved_result("baseline_train", st.write)

    _render_training_job_controls("baseline_train_job", start_job)
    baseline_job = _get_training_job("baseline_train_job")
    if baseline_job is not None and baseline_job.snapshot()["status"] == "completed":
        _save_step_result("baseline_train", "Baseline training finished.", {"output_dir": baseline_job.output_dir})


def _pairwise_train_tab() -> None:
    st.subheader("Pairwise Training")
    st.caption("Train the stronger same-document vs new-document classifier on adjacent page pairs.")
    st.info("Use a split manifest such as `data/synthetic/pair_labels_train.csv` and `data/synthetic/pair_labels_eval.csv`.")

    with st.form("pairwise_train_form"):
        train_csv = _path_input("Train CSV", "pairwise_train_csv", "data/synthetic/pair_labels_train.csv")
        st.caption("Pair-label training CSV with adjacent page image paths and same/new-document labels.")
        eval_csv = _path_input("Eval CSV", "pairwise_eval_csv", "data/synthetic/pair_labels_eval.csv")
        st.caption("Held-out pair-label CSV used to evaluate boundary precision and recall.")
        output_dir = _path_input("Output model directory", "pairwise_output_dir", "artifacts/layoutlmv3-pairwise")
        st.caption("Directory where the pairwise model and processor are saved.")
        pretrained_model_name = st.text_input(
            "Pretrained model name",
            value="microsoft/layoutlmv3-base",
            key="pair_pretrained_model_name",
            help="Hugging Face model checkpoint used as the LayoutLMv3 backbone.",
        )
        learning_rate = st.number_input(
            "Learning rate",
            min_value=0.0,
            value=2e-5,
            format="%.8f",
            key="pair_learning_rate",
            help="Optimizer step size. Lower is safer; higher trains faster but can destabilize training.",
        )
        train_batch_size = st.number_input(
            "Train batch size",
            min_value=1,
            value=2,
            key="pair_train_batch_size",
            help="Number of adjacent page pairs per training batch.",
        )
        eval_batch_size = st.number_input(
            "Eval batch size",
            min_value=1,
            value=2,
            key="pair_eval_batch_size",
            help="Number of adjacent page pairs per evaluation batch.",
        )
        num_train_epochs = st.number_input(
            "Epochs",
            min_value=1,
            value=5,
            key="pair_num_train_epochs",
            help="Number of full passes over the pairwise training CSV.",
        )
        max_length = st.number_input(
            "Max token length",
            min_value=64,
            value=512,
            key="pair_max_length",
            help="Maximum OCR tokens sent to LayoutLMv3 for each page in a pair.",
        )
        logging_steps = st.number_input(
            "Logging steps",
            min_value=1,
            value=20,
            key="pair_logging_steps",
            help="How often Trainer emits progress during model training.",
        )
        tesseract_lang = st.text_input(
            "Tesseract language",
            value="eng",
            key="pair_tesseract_lang",
            help="OCR language code installed in Tesseract, for example `eng` or `eng+ita`.",
        )
        ocr_num_proc = st.number_input(
            "OCR workers",
            min_value=1,
            max_value=12,
            value=4,
            key="pair_ocr_num_proc",
            help="Parallel OCR preprocessing workers. Start with 4; raise to 6 or 8 if CPU and RAM remain stable.",
        )
        classifier_dropout = st.number_input(
            "Classifier dropout",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            key="pair_classifier_dropout",
            help="Regularization applied to the pairwise classifier head.",
        )
        submitted = st.form_submit_button("Start pairwise training")

    def start_job() -> None:
        if not _require_existing_file(train_csv, "Train CSV") or not _require_existing_file(eval_csv, "Eval CSV"):
            st.warning("If you only have `pair_labels.csv`, first use the `Prepare Train/Eval Splits` tab.")
            return

        _start_pairwise_training_job(
            PairwiseTrainConfig(
                train_csv=train_csv,
                eval_csv=eval_csv,
                output_dir=output_dir,
                pretrained_model_name=pretrained_model_name,
                learning_rate=float(learning_rate),
                train_batch_size=int(train_batch_size),
                eval_batch_size=int(eval_batch_size),
                num_train_epochs=int(num_train_epochs),
                max_length=int(max_length),
                logging_steps=int(logging_steps),
                tesseract_lang=tesseract_lang,
                ocr_num_proc=int(ocr_num_proc),
                classifier_dropout=float(classifier_dropout),
            ),
            output_dir=output_dir,
        )

    if submitted:
        start_job()
    else:
        _render_saved_result("pairwise_train", st.write)

    _render_training_job_controls("pairwise_train_job", start_job)
    pairwise_job = _get_training_job("pairwise_train_job")
    if pairwise_job is not None and pairwise_job.snapshot()["status"] == "completed":
        _save_step_result("pairwise_train", "Pairwise training finished.", {"output_dir": pairwise_job.output_dir})


def _lightweight_pairwise_train_tab() -> None:
    st.subheader("Lightweight Pairwise Training")
    st.caption("Train a CPU-friendly boundary classifier from OCR text and sentence embeddings.")
    st.info("Use the pair-label split manifests generated by `Synthetic Data`.")

    with st.form("lightweight_pairwise_train_form"):
        train_csv = _path_input(
            "Train CSV",
            "lightweight_pairwise_train_csv",
            "data/synthetic/pair_labels_train.csv",
        )
        eval_csv = _path_input(
            "Eval CSV",
            "lightweight_pairwise_eval_csv",
            "data/synthetic/pair_labels_eval.csv",
        )
        output_dir = _path_input(
            "Output model directory",
            "lightweight_pairwise_output_dir",
            "artifacts/lightweight-pairwise",
        )
        embedding_model_name = st.text_input(
            "Embedding model name",
            value="sentence-transformers/all-MiniLM-L6-v2",
            key="lightweight_pairwise_embedding_model",
            help="Small sentence-transformer used to embed adjacent page text.",
        )
        classifier_type = st.selectbox(
            "Classifier",
            options=["logistic_regression", "random_forest"],
            index=0,
            key="lightweight_pairwise_classifier",
            help="Logistic regression is fast and usually a good first choice. Random forest can capture more rules.",
        )
        tesseract_lang = st.text_input(
            "Tesseract language",
            value="eng",
            key="lightweight_pairwise_tesseract_lang",
        )
        max_pairs_enabled = st.checkbox(
            "Limit training rows",
            value=False,
            key="lightweight_pairwise_limit_rows_enabled",
            help="Useful for a quick smoke test before training on the full manifest.",
        )
        max_pairs = st.number_input(
            "Maximum rows per split",
            min_value=10,
            value=500,
            key="lightweight_pairwise_max_rows",
            disabled=not max_pairs_enabled,
        )
        random_state = st.number_input(
            "Random seed",
            min_value=0,
            value=42,
            key="lightweight_pairwise_random_state",
        )
        submitted = st.form_submit_button("Start lightweight training")

    def start_job() -> None:
        if not _require_existing_file(train_csv, "Train CSV") or not _require_existing_file(eval_csv, "Eval CSV"):
            st.warning("If you only have `pair_labels.csv`, first use the `Prepare Train/Eval Splits` tab.")
            return

        _start_lightweight_pairwise_training_job(
            LightweightPairwiseTrainConfig(
                train_csv=train_csv,
                eval_csv=eval_csv,
                output_dir=output_dir,
                embedding_model_name=embedding_model_name,
                classifier_type=str(classifier_type),
                tesseract_lang=tesseract_lang,
                max_pairs=int(max_pairs) if max_pairs_enabled else None,
                random_state=int(random_state),
            ),
            output_dir=output_dir,
        )

    if submitted:
        start_job()
    else:
        _render_saved_result("lightweight_pairwise_train", st.write)

    _render_training_job_controls("lightweight_pairwise_train_job", start_job)
    job = _get_training_job("lightweight_pairwise_train_job")
    if job is not None and job.snapshot()["status"] == "completed":
        _save_step_result(
            "lightweight_pairwise_train",
            "Lightweight pairwise training finished.",
            {"output_dir": job.output_dir},
        )


def _pairwise_eval_tab() -> None:
    st.subheader("Pairwise Evaluation")
    st.caption("Evaluate a saved LayoutLMv3 pairwise model on a held-out pair-label CSV.")

    with st.form("pairwise_eval_form"):
        eval_csv = _path_input("Eval CSV", "pairwise_eval_only_csv", "data/synthetic/pair_labels_eval.csv")
        model_dir = _path_input("Model directory", "pairwise_eval_model_dir", "artifacts/layoutlmv3-pairwise")
        output_dir = _path_input("Output directory", "pairwise_eval_output_dir", "artifacts/layoutlmv3-pairwise/eval")
        eval_batch_size = st.number_input(
            "Eval batch size",
            min_value=1,
            value=2,
            key="pairwise_eval_only_batch_size",
            help="Number of adjacent page pairs per evaluation batch.",
        )
        max_length = st.number_input(
            "Max token length",
            min_value=64,
            value=512,
            key="pairwise_eval_only_max_length",
            help="Maximum OCR tokens sent to LayoutLMv3 for each page in a pair.",
        )
        tesseract_lang = st.text_input(
            "Tesseract language",
            value="eng",
            key="pairwise_eval_only_tesseract_lang",
        )
        ocr_num_proc = 1
        st.info("Pairwise evaluation uses serial OCR with row-by-row progress to avoid Tesseract multiprocessing stalls.")
        max_eval_rows_enabled = st.checkbox(
            "Limit eval rows",
            value=False,
            key="pairwise_eval_limit_rows_enabled",
            help="Use this for a fast smoke test before running the full eval CSV.",
        )
        max_eval_rows = st.number_input(
            "Maximum eval rows",
            min_value=1,
            value=50,
            key="pairwise_eval_max_rows",
            disabled=not max_eval_rows_enabled,
        )
        encoded_cache_dir = _path_input(
            "Encoded cache directory",
            "pairwise_eval_encoded_cache_dir",
            "artifacts/layoutlmv3-pairwise/eval/encoded_eval_cache",
        )
        submitted = st.form_submit_button("Run pairwise evaluation")

    def start_job() -> None:
        if not _require_existing_file(eval_csv, "Eval CSV"):
            return
        if not model_dir.exists() or not model_dir.is_dir():
            st.error(f"Model directory not found: {model_dir}")
            return

        _start_pairwise_evaluation_job(
            PairwiseEvalConfig(
                eval_csv=eval_csv,
                model_dir=model_dir,
                output_dir=output_dir,
                eval_batch_size=int(eval_batch_size),
                max_length=int(max_length),
                tesseract_lang=tesseract_lang,
                ocr_num_proc=ocr_num_proc,
                max_eval_rows=int(max_eval_rows) if max_eval_rows_enabled else None,
                encoded_cache_dir=encoded_cache_dir,
            ),
            model_dir=model_dir,
        )

    if submitted:
        start_job()
    else:
        _render_saved_result("pairwise_eval", _render_pairwise_eval_result)

    _render_training_job_controls("pairwise_eval_job", start_job)
    job = _get_training_job("pairwise_eval_job")
    if job is not None:
        snapshot = job.snapshot()
        if snapshot["status"] == "completed" and snapshot.get("result") is not None:
            _save_step_result("pairwise_eval", "Pairwise evaluation complete.", snapshot["result"])
            _render_pairwise_eval_result(snapshot["result"])


def _baseline_predict_tab() -> None:
    st.subheader("Baseline Prediction")
    st.caption("Run page-level START/MIDDLE/END/SINGLE prediction on a merged PDF.")

    with st.form("baseline_predict_form"):
        pdf_path = _path_input("Merged PDF path", "baseline_predict_pdf_path", "samples/merged.pdf")
        model_dir = _path_input("Model directory", "baseline_predict_model_dir", "artifacts/layoutlmv3-boundary")
        work_dir = _path_input("Work directory", "baseline_predict_work_dir", "artifacts/inference/baseline")
        dpi = st.number_input("Render DPI", min_value=72, value=200, key="baseline_predict_dpi")
        max_length = st.number_input("Max token length", min_value=64, value=512, key="baseline_predict_max_length")
        tesseract_lang = st.text_input("Tesseract language", value="eng", key="baseline_predict_tesseract_lang")
        split_output = st.checkbox("Write split PDFs", value=True)
        submitted = st.form_submit_button("Run baseline prediction")

    if submitted:
        try:
            with st.spinner("Running baseline prediction..."):
                result = predict_pdf_boundaries(
                    pdf_path=pdf_path,
                    model_dir=model_dir,
                    work_dir=work_dir,
                    dpi=int(dpi),
                    max_length=int(max_length),
                    tesseract_lang=tesseract_lang,
                    split_output=split_output,
                )
        except OCREnvironmentError as exc:
            st.error(str(exc))
            return

        st.success("Prediction complete.")
        st.write("Page predictions")
        st.write(
            [
                {
                    "page": prediction.page_number,
                    "label": prediction.label,
                    "confidence": round(prediction.confidence, 4),
                }
                for prediction in result.predictions
            ]
        )
        st.write("Document ranges")
        _render_ranges(result.ranges)
        _render_split_paths(result.output_pdfs)
        _save_step_result(
            "baseline_predict",
            "Prediction complete.",
            {
                "predictions": [
                    {
                        "page": prediction.page_number,
                        "label": prediction.label,
                        "confidence": round(prediction.confidence, 4),
                    }
                    for prediction in result.predictions
                ],
                "ranges": result.ranges,
                "output_pdfs": result.output_pdfs,
            },
        )
    else:
        _render_saved_result("baseline_predict", _render_baseline_prediction_result)


def _pairwise_predict_tab() -> None:
    st.subheader("Pairwise Prediction")
    st.caption("Run adjacent-page same-document prediction and reconstruct document ranges.")

    with st.form("pairwise_predict_form"):
        pdf_path = _path_input("Merged PDF path", "pairwise_predict_pdf_path", "samples/merged.pdf")
        model_dir = _path_input("Model directory", "pairwise_predict_model_dir", "artifacts/layoutlmv3-pairwise")
        work_dir = _path_input("Work directory", "pairwise_predict_work_dir", "artifacts/inference/pairwise")
        dpi = st.number_input("Render DPI", min_value=72, value=200, key="pairwise_predict_dpi")
        max_length = st.number_input("Max token length", min_value=64, value=512, key="pairwise_predict_max_length")
        threshold = st.number_input("New-document threshold", min_value=0.0, max_value=1.0, value=0.5, key="pairwise_predict_threshold")
        tesseract_lang = st.text_input("Tesseract language", value="eng", key="pairwise_predict_tesseract_lang")
        split_output = st.checkbox("Write split PDFs", value=True, key="pairwise_predict_split_output")
        submitted = st.form_submit_button("Run pairwise prediction")

    if submitted:
        try:
            with st.spinner("Running pairwise prediction..."):
                result = predict_pdf_boundaries_pairwise(
                    pdf_path=pdf_path,
                    model_dir=model_dir,
                    work_dir=work_dir,
                    dpi=int(dpi),
                    max_length=int(max_length),
                    tesseract_lang=tesseract_lang,
                    threshold=float(threshold),
                    split_output=split_output,
                )
        except OCREnvironmentError as exc:
            st.error(str(exc))
            return

        st.success("Prediction complete.")
        st.write("Pair predictions")
        st.write(
            [
                {
                    "left_page": prediction.left_page,
                    "right_page": prediction.right_page,
                    "label": prediction.label,
                    "same_document_probability": round(prediction.same_document_probability, 4),
                }
                for prediction in result.pair_predictions
            ]
        )
        st.write("Page labels")
        st.write(
            [{"page": page_number, "label": label} for page_number, label in enumerate(result.page_labels, start=1)]
        )
        st.write("Document ranges")
        _render_ranges(result.ranges)
        _render_split_paths(result.output_pdfs)
        _save_step_result(
            "pairwise_predict",
            "Prediction complete.",
            {
                "pair_predictions": [
                    {
                        "left_page": prediction.left_page,
                        "right_page": prediction.right_page,
                        "label": prediction.label,
                        "same_document_probability": round(prediction.same_document_probability, 4),
                    }
                    for prediction in result.pair_predictions
                ],
                "page_labels": [
                    {"page": page_number, "label": label} for page_number, label in enumerate(result.page_labels, start=1)
                ],
                "ranges": result.ranges,
                "output_pdfs": result.output_pdfs,
            },
        )
    else:
        _render_saved_result("pairwise_predict", _render_pairwise_prediction_result)


def _lightweight_pairwise_predict_tab() -> None:
    st.subheader("Lightweight Pairwise Prediction")
    st.caption("Run the CPU-friendly adjacent-page boundary model and reconstruct document ranges.")

    with st.form("lightweight_pairwise_predict_form"):
        pdf_path = _path_input(
            "Merged PDF path",
            "lightweight_pairwise_predict_pdf_path",
            "samples/merged.pdf",
        )
        model_dir = _path_input(
            "Model directory",
            "lightweight_pairwise_predict_model_dir",
            "artifacts/lightweight-pairwise",
        )
        work_dir = _path_input(
            "Work directory",
            "lightweight_pairwise_predict_work_dir",
            "artifacts/inference/lightweight-pairwise",
        )
        dpi = st.number_input("Render DPI", min_value=72, value=200, key="lightweight_pairwise_predict_dpi")
        threshold = st.number_input(
            "New-document threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            key="lightweight_pairwise_predict_threshold",
        )
        tesseract_lang = st.text_input(
            "Tesseract language override",
            value="",
            key="lightweight_pairwise_predict_tesseract_lang",
            help="Leave empty to use the language saved with the model.",
        )
        split_output = st.checkbox(
            "Write split PDFs",
            value=True,
            key="lightweight_pairwise_predict_split_output",
        )
        submitted = st.form_submit_button("Run lightweight prediction")

    if submitted:
        try:
            with st.spinner("Running lightweight pairwise prediction..."):
                result = predict_pdf_boundaries_lightweight_pairwise(
                    pdf_path=pdf_path,
                    model_dir=model_dir,
                    work_dir=work_dir,
                    dpi=int(dpi),
                    tesseract_lang=tesseract_lang.strip() or None,
                    threshold=float(threshold),
                    split_output=split_output,
                )
        except (OCREnvironmentError, RuntimeError, ValueError) as exc:
            st.error(str(exc))
            return

        st.success("Prediction complete.")
        st.write("Pair predictions")
        st.write(
            [
                {
                    "left_page": prediction.left_page,
                    "right_page": prediction.right_page,
                    "label": prediction.label,
                    "same_document_probability": round(prediction.same_document_probability, 4),
                    "new_document_probability": round(prediction.new_document_probability, 4),
                }
                for prediction in result.pair_predictions
            ]
        )
        st.write("Page labels")
        st.write(
            [{"page": page_number, "label": label} for page_number, label in enumerate(result.page_labels, start=1)]
        )
        st.write("Document ranges")
        _render_ranges(result.ranges)
        _render_split_paths(result.output_pdfs)
        _save_step_result(
            "lightweight_pairwise_predict",
            "Prediction complete.",
            {
                "pair_predictions": [
                    {
                        "left_page": prediction.left_page,
                        "right_page": prediction.right_page,
                        "label": prediction.label,
                        "same_document_probability": round(prediction.same_document_probability, 4),
                        "new_document_probability": round(prediction.new_document_probability, 4),
                    }
                    for prediction in result.pair_predictions
                ],
                "page_labels": [
                    {"page": page_number, "label": label} for page_number, label in enumerate(result.page_labels, start=1)
                ],
                "ranges": result.ranges,
                "output_pdfs": result.output_pdfs,
            },
        )
    else:
        _render_saved_result("lightweight_pairwise_predict", _render_pairwise_prediction_result)


def main() -> None:
    st.set_page_config(page_title="Document Recognition", layout="wide")
    st.title("Document Boundary Detection")
    st.caption("Run synthetic generation, training, and inference pipelines for merged-PDF document splitting.")

    tabs = st.tabs(
        [
            "Validate Source PDFs",
            "Synthetic Data",
            "Prepare Train/Eval Splits",
            "Train Baseline",
            "Train Pairwise",
            "Train Lightweight",
            "Evaluate Pairwise",
            "Predict Baseline",
            "Predict Pairwise",
            "Predict Lightweight",
        ]
    )

    with tabs[0]:
        _source_validation_tab()
    with tabs[1]:
        _synthetic_tab()
    with tabs[2]:
        _manifest_split_tab()
    with tabs[3]:
        _baseline_train_tab()
    with tabs[4]:
        _pairwise_train_tab()
    with tabs[5]:
        _lightweight_pairwise_train_tab()
    with tabs[6]:
        _pairwise_eval_tab()
    with tabs[7]:
        _baseline_predict_tab()
    with tabs[8]:
        _pairwise_predict_tab()
    with tabs[9]:
        _lightweight_pairwise_predict_tab()


if __name__ == "__main__":
    main()
