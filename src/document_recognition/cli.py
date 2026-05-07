from __future__ import annotations

import argparse
from pathlib import Path
import sys


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LayoutLMv3 document boundary toolkit")
    subparsers = parser.add_subparsers(dest="command", required=True)

    synthetic_parser = subparsers.add_parser(
        "generate-synthetic",
        help="Create synthetic merged PDFs and training manifests from single-document PDFs",
    )
    synthetic_parser.add_argument("--input-dir", type=Path, required=True)
    synthetic_parser.add_argument("--output-dir", type=Path, required=True)
    synthetic_parser.add_argument("--num-merged-pdfs", type=int, required=True)
    synthetic_parser.add_argument("--min-docs-per-merge", type=int, default=2)
    synthetic_parser.add_argument("--max-docs-per-merge", type=int, default=5)
    synthetic_parser.add_argument("--dpi", type=int, default=200)
    synthetic_parser.add_argument("--seed", type=int, default=42)
    synthetic_parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of worker processes for synthetic PDF generation.",
    )
    synthetic_parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Do not print generation progress to stderr.",
    )

    train_parser = subparsers.add_parser("train", help="Train the page-label baseline")
    train_parser.add_argument("--train-csv", type=Path, required=True)
    train_parser.add_argument("--eval-csv", type=Path, required=True)
    train_parser.add_argument("--output-dir", type=Path, required=True)
    train_parser.add_argument("--pretrained-model-name", default="microsoft/layoutlmv3-base")
    train_parser.add_argument("--learning-rate", type=float, default=2e-5)
    train_parser.add_argument("--train-batch-size", type=int, default=2)
    train_parser.add_argument("--eval-batch-size", type=int, default=2)
    train_parser.add_argument("--num-train-epochs", type=int, default=5)
    train_parser.add_argument("--max-length", type=int, default=512)
    train_parser.add_argument("--logging-steps", type=int, default=20)
    train_parser.add_argument("--tesseract-lang", default="eng")
    train_parser.add_argument("--ocr-num-proc", type=int, default=1)

    pairwise_train_parser = subparsers.add_parser("train-pairwise", help="Train the pairwise same-document model")
    pairwise_train_parser.add_argument("--train-csv", type=Path, required=True)
    pairwise_train_parser.add_argument("--eval-csv", type=Path, required=True)
    pairwise_train_parser.add_argument("--output-dir", type=Path, required=True)
    pairwise_train_parser.add_argument("--pretrained-model-name", default="microsoft/layoutlmv3-base")
    pairwise_train_parser.add_argument("--learning-rate", type=float, default=2e-5)
    pairwise_train_parser.add_argument("--train-batch-size", type=int, default=2)
    pairwise_train_parser.add_argument("--eval-batch-size", type=int, default=2)
    pairwise_train_parser.add_argument("--num-train-epochs", type=int, default=5)
    pairwise_train_parser.add_argument("--max-length", type=int, default=512)
    pairwise_train_parser.add_argument("--logging-steps", type=int, default=20)
    pairwise_train_parser.add_argument("--tesseract-lang", default="eng")
    pairwise_train_parser.add_argument("--ocr-num-proc", type=int, default=1)
    pairwise_train_parser.add_argument("--ocr-engine", choices=["tesseract", "easyocr"], default="tesseract")
    pairwise_train_parser.add_argument("--ocr-gpu", action="store_true")
    pairwise_train_parser.add_argument("--classifier-dropout", type=float, default=0.1)
    pairwise_train_parser.add_argument("--fp16", action="store_true")
    pairwise_train_parser.add_argument("--dataloader-num-workers", type=int, default=0)
    pairwise_train_parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    pairwise_train_parser.add_argument("--encoded-cache-dir", type=Path)

    pairwise_eval_parser = subparsers.add_parser(
        "eval-pairwise",
        help="Evaluate a saved pairwise same-document model on a pair-label CSV",
    )
    pairwise_eval_parser.add_argument("--eval-csv", type=Path, required=True)
    pairwise_eval_parser.add_argument("--model-dir", type=Path, required=True)
    pairwise_eval_parser.add_argument("--output-dir", type=Path)
    pairwise_eval_parser.add_argument("--eval-batch-size", type=int, default=2)
    pairwise_eval_parser.add_argument("--max-length", type=int, default=512)
    pairwise_eval_parser.add_argument("--tesseract-lang", default="eng")
    pairwise_eval_parser.add_argument(
        "--ocr-num-proc",
        type=int,
        default=1,
        help="Deprecated for eval; saved-model evaluation uses serial OCR to avoid Tesseract multiprocessing stalls.",
    )
    pairwise_eval_parser.add_argument("--max-eval-rows", type=int)
    pairwise_eval_parser.add_argument("--encoded-cache-dir", type=Path)
    pairwise_eval_parser.add_argument("--ocr-engine", choices=["tesseract", "easyocr"], default="tesseract")
    pairwise_eval_parser.add_argument("--ocr-gpu", action="store_true")

    lightweight_pairwise_train_parser = subparsers.add_parser(
        "train-lightweight-pairwise",
        help="Train a CPU-friendly sentence-embedding pairwise boundary model",
    )
    lightweight_pairwise_train_parser.add_argument("--train-csv", type=Path, required=True)
    lightweight_pairwise_train_parser.add_argument("--eval-csv", type=Path, required=True)
    lightweight_pairwise_train_parser.add_argument("--output-dir", type=Path, required=True)
    lightweight_pairwise_train_parser.add_argument(
        "--embedding-model-name",
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    lightweight_pairwise_train_parser.add_argument(
        "--classifier-type",
        choices=["logistic_regression", "random_forest"],
        default="logistic_regression",
    )
    lightweight_pairwise_train_parser.add_argument("--tesseract-lang", default="eng")
    lightweight_pairwise_train_parser.add_argument("--max-pairs", type=int)
    lightweight_pairwise_train_parser.add_argument("--random-state", type=int, default=42)

    predict_parser = subparsers.add_parser("predict", help="Predict boundaries with the page-label baseline")
    predict_parser.add_argument("--pdf-path", type=Path, required=True)
    predict_parser.add_argument("--model-dir", type=Path, required=True)
    predict_parser.add_argument("--work-dir", type=Path, required=True)
    predict_parser.add_argument("--dpi", type=int, default=200)
    predict_parser.add_argument("--max-length", type=int, default=512)
    predict_parser.add_argument("--tesseract-lang", default="eng")
    predict_parser.add_argument("--no-split-output", action="store_true")

    pairwise_predict_parser = subparsers.add_parser(
        "predict-pairwise",
        help="Predict boundaries with the pairwise same-document model",
    )
    pairwise_predict_parser.add_argument("--pdf-path", type=Path, required=True)
    pairwise_predict_parser.add_argument("--model-dir", type=Path, required=True)
    pairwise_predict_parser.add_argument("--work-dir", type=Path, required=True)
    pairwise_predict_parser.add_argument("--dpi", type=int, default=200)
    pairwise_predict_parser.add_argument("--max-length", type=int, default=512)
    pairwise_predict_parser.add_argument("--tesseract-lang", default="eng")
    pairwise_predict_parser.add_argument("--threshold", type=float, default=0.5)
    pairwise_predict_parser.add_argument("--no-split-output", action="store_true")

    lightweight_pairwise_predict_parser = subparsers.add_parser(
        "predict-lightweight-pairwise",
        help="Predict boundaries with the CPU-friendly sentence-embedding pairwise model",
    )
    lightweight_pairwise_predict_parser.add_argument("--pdf-path", type=Path, required=True)
    lightweight_pairwise_predict_parser.add_argument("--model-dir", type=Path, required=True)
    lightweight_pairwise_predict_parser.add_argument("--work-dir", type=Path, required=True)
    lightweight_pairwise_predict_parser.add_argument("--dpi", type=int, default=200)
    lightweight_pairwise_predict_parser.add_argument("--tesseract-lang", default=None)
    lightweight_pairwise_predict_parser.add_argument("--threshold", type=float, default=0.7)
    lightweight_pairwise_predict_parser.add_argument("--no-split-output", action="store_true")

    return parser


def _run_generate_synthetic(args: argparse.Namespace) -> None:
    from .synthetic import SyntheticDatasetConfig, create_synthetic_merged_dataset

    progress_callback = None if args.no_progress else _make_cli_progress_callback()
    outputs = create_synthetic_merged_dataset(
        SyntheticDatasetConfig(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            num_merged_pdfs=args.num_merged_pdfs,
            min_docs_per_merge=args.min_docs_per_merge,
            max_docs_per_merge=args.max_docs_per_merge,
            dpi=args.dpi,
            seed=args.seed,
            num_workers=args.num_workers,
        ),
        progress_callback=progress_callback,
    )

    for name, path in outputs.items():
        print(f"{name}={path}")


def _format_eta(seconds: object) -> str:
    if seconds is None:
        return "eta unknown"

    total_seconds = int(seconds)
    minutes, remaining_seconds = divmod(total_seconds, 60)
    hours, remaining_minutes = divmod(minutes, 60)
    if hours:
        return f"eta {hours}h {remaining_minutes}m"
    if remaining_minutes:
        return f"eta {remaining_minutes}m {remaining_seconds}s"
    return f"eta {remaining_seconds}s"


def _make_cli_progress_callback():
    last_message = ""

    def callback(event: dict[str, object]) -> None:
        nonlocal last_message

        current = int(event.get("current", 0))
        total = int(event.get("total", 0))
        message = str(event.get("message", "Working..."))
        if message == last_message:
            return

        fraction = float(event.get("fraction", 0.0))
        percent = max(0.0, min(100.0, fraction * 100.0))
        eta = _format_eta(event.get("eta_seconds"))
        print(f"[{current}/{total} {percent:5.1f}%] {message} ({eta})", file=sys.stderr, flush=True)
        last_message = message

    return callback


def _run_train(args: argparse.Namespace) -> None:
    from .train import TrainConfig, train_model

    config = TrainConfig(
        train_csv=args.train_csv,
        eval_csv=args.eval_csv,
        output_dir=args.output_dir,
        pretrained_model_name=args.pretrained_model_name,
        learning_rate=args.learning_rate,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        max_length=args.max_length,
        logging_steps=args.logging_steps,
        tesseract_lang=args.tesseract_lang,
        ocr_num_proc=args.ocr_num_proc,
    )
    train_model(config)


def _run_train_pairwise(args: argparse.Namespace) -> None:
    from .pairwise_train import PairwiseTrainConfig, train_pairwise_model

    config = PairwiseTrainConfig(
        train_csv=args.train_csv,
        eval_csv=args.eval_csv,
        output_dir=args.output_dir,
        pretrained_model_name=args.pretrained_model_name,
        learning_rate=args.learning_rate,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        max_length=args.max_length,
        logging_steps=args.logging_steps,
        tesseract_lang=args.tesseract_lang,
        ocr_num_proc=args.ocr_num_proc,
        classifier_dropout=args.classifier_dropout,
        fp16=args.fp16,
        dataloader_num_workers=args.dataloader_num_workers,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        encoded_cache_dir=args.encoded_cache_dir,
        ocr_engine=args.ocr_engine,
        ocr_gpu=args.ocr_gpu,
    )
    train_pairwise_model(config)


def _run_eval_pairwise(args: argparse.Namespace) -> None:
    from .pairwise_train import PairwiseEvalConfig, evaluate_pairwise_model

    metrics = evaluate_pairwise_model(
        PairwiseEvalConfig(
            eval_csv=args.eval_csv,
            model_dir=args.model_dir,
            output_dir=args.output_dir,
            eval_batch_size=args.eval_batch_size,
            max_length=args.max_length,
            tesseract_lang=args.tesseract_lang,
            ocr_num_proc=args.ocr_num_proc,
            max_eval_rows=args.max_eval_rows,
            encoded_cache_dir=args.encoded_cache_dir,
            ocr_engine=args.ocr_engine,
            ocr_gpu=args.ocr_gpu,
        )
    )
    print("Pairwise evaluation metrics:")
    for key in sorted(metrics):
        print(f"{key}={metrics[key]:.6f}")


def _run_train_lightweight_pairwise(args: argparse.Namespace) -> None:
    from .lightweight_pairwise import LightweightPairwiseTrainConfig, train_lightweight_pairwise_model

    output_dir = train_lightweight_pairwise_model(
        LightweightPairwiseTrainConfig(
            train_csv=args.train_csv,
            eval_csv=args.eval_csv,
            output_dir=args.output_dir,
            embedding_model_name=args.embedding_model_name,
            classifier_type=args.classifier_type,
            tesseract_lang=args.tesseract_lang,
            max_pairs=args.max_pairs,
            random_state=args.random_state,
        )
    )
    print(f"lightweight_pairwise_model={output_dir}")


def _run_predict(args: argparse.Namespace) -> None:
    from .inference import predict_pdf_boundaries

    result = predict_pdf_boundaries(
        pdf_path=args.pdf_path,
        model_dir=args.model_dir,
        work_dir=args.work_dir,
        dpi=args.dpi,
        max_length=args.max_length,
        tesseract_lang=args.tesseract_lang,
        split_output=not args.no_split_output,
    )

    print("Predictions:")
    for prediction in result.predictions:
        print(f"page={prediction.page_number} label={prediction.label} confidence={prediction.confidence:.4f}")

    print("\nRanges:")
    for index, (start, end) in enumerate(result.ranges, start=1):
        print(f"doc_{index} = pages {start}-{end}")

    if result.output_pdfs:
        print("\nSplit PDFs:")
        for path in result.output_pdfs:
            print(path)


def _run_predict_pairwise(args: argparse.Namespace) -> None:
    from .pairwise_inference import predict_pdf_boundaries_pairwise

    result = predict_pdf_boundaries_pairwise(
        pdf_path=args.pdf_path,
        model_dir=args.model_dir,
        work_dir=args.work_dir,
        dpi=args.dpi,
        max_length=args.max_length,
        tesseract_lang=args.tesseract_lang,
        threshold=args.threshold,
        split_output=not args.no_split_output,
    )

    print("Pair predictions:")
    for prediction in result.pair_predictions:
        print(
            "left_page="
            f"{prediction.left_page} right_page={prediction.right_page} "
            f"label={prediction.label} same_document_probability={prediction.same_document_probability:.4f}"
        )

    print("\nPage labels:")
    for page_number, label in enumerate(result.page_labels, start=1):
        print(f"page={page_number} label={label}")

    print("\nRanges:")
    for index, (start, end) in enumerate(result.ranges, start=1):
        print(f"doc_{index} = pages {start}-{end}")

    if result.output_pdfs:
        print("\nSplit PDFs:")
        for path in result.output_pdfs:
            print(path)


def _run_predict_lightweight_pairwise(args: argparse.Namespace) -> None:
    from .lightweight_pairwise import predict_pdf_boundaries_lightweight_pairwise

    result = predict_pdf_boundaries_lightweight_pairwise(
        pdf_path=args.pdf_path,
        model_dir=args.model_dir,
        work_dir=args.work_dir,
        dpi=args.dpi,
        tesseract_lang=args.tesseract_lang,
        threshold=args.threshold,
        split_output=not args.no_split_output,
    )

    print("Pair predictions:")
    for prediction in result.pair_predictions:
        print(
            "left_page="
            f"{prediction.left_page} right_page={prediction.right_page} "
            f"label={prediction.label} "
            f"same_document_probability={prediction.same_document_probability:.4f} "
            f"new_document_probability={prediction.new_document_probability:.4f}"
        )

    print("\nPage labels:")
    for page_number, label in enumerate(result.page_labels, start=1):
        print(f"page={page_number} label={label}")

    print("\nRanges:")
    for index, (start, end) in enumerate(result.ranges, start=1):
        print(f"doc_{index} = pages {start}-{end}")

    if result.output_pdfs:
        print("\nSplit PDFs:")
        for path in result.output_pdfs:
            print(path)


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "generate-synthetic":
        _run_generate_synthetic(args)
        return

    if args.command == "train":
        _run_train(args)
        return

    if args.command == "train-pairwise":
        _run_train_pairwise(args)
        return

    if args.command == "eval-pairwise":
        _run_eval_pairwise(args)
        return

    if args.command == "train-lightweight-pairwise":
        _run_train_lightweight_pairwise(args)
        return

    if args.command == "predict":
        _run_predict(args)
        return

    if args.command == "predict-pairwise":
        _run_predict_pairwise(args)
        return

    if args.command == "predict-lightweight-pairwise":
        _run_predict_lightweight_pairwise(args)
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
