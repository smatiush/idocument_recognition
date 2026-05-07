# Document Recognition

Production-oriented LayoutLMv3 toolkit for document boundary detection in merged PDFs.

## Labels

- `START_DOC`
- `MIDDLE_DOC`
- `END_DOC`
- `SINGLE_PAGE_DOC`

## Install

```bash
pip install -e .
```

You also need a working Tesseract installation available on `PATH`.

To install system prerequisites, create a local virtual environment, install the
project, and verify OCR in one step:

```bash
./scripts/install_prereqs.sh
```

Useful options:

```bash
INSTALL_DEV=1 ./scripts/install_prereqs.sh
VENV_DIR=.venv PYTHON_BIN=python3.12 ./scripts/install_prereqs.sh
TESSERACT_LANGS=eng+ita ./scripts/install_prereqs.sh
```

On Ubuntu/Debian:

```bash
sudo apt-get install tesseract-ocr tesseract-ocr-eng
```

On macOS:

```bash
brew install tesseract
```

## UI

Run the local UI with:

```bash
streamlit run src/document_recognition/ui.py
```

The UI includes tabs for:

- source PDF validation and filtering
- synthetic dataset generation
- train/eval split preparation
- page-level baseline training
- pairwise model training
- baseline prediction
- pairwise prediction

The recommended flow for raw PDF collections is:

1. Validate source PDFs and filter out unreadable, duplicate, or oversized files.
2. Use the filtered output folder as input to synthetic dataset generation.
3. Let synthetic generation create train/eval CSVs automatically.
4. Train the pairwise or baseline model from the generated split manifests.

The UI shows progress bars during:

- source validation
- synthetic dataset generation
- baseline training
- pairwise training

When the pipeline can measure it, the UI also shows a rough ETA.

Training tabs include an `OCR workers` setting for parallel Tesseract preprocessing.
On a 12-core machine, start with `4`; increase to `6` or `8` only if CPU and memory
headroom remain stable.

OCR results are cached on disk after the first pass so repeated training runs can
reuse Tesseract words and bounding boxes. By default, the cache lives at:

```text
.cache/document_recognition/ocr
```

Override or disable it with:

```bash
DOCUMENT_RECOGNITION_OCR_CACHE_DIR=/fast/disk/ocr-cache streamlit run src/document_recognition/ui.py
DOCUMENT_RECOGNITION_DISABLE_OCR_CACHE=1 streamlit run src/document_recognition/ui.py
```

Baseline and pairwise training run as managed UI jobs. After starting a job, use:

- `Play` to resume a paused job
- `Pause` to pause after the current OCR item or training step
- `Stop` to stop after the current OCR item or training step
- `Restart` to stop the existing job and start a new one with the current form values

## Recommended Approach

For production, prefer the pairwise formulation:

- input: page `N` and page `N+1`
- output: `SAME_DOCUMENT` or `NEW_DOCUMENT`

The page-label model is still included as a baseline:

- `START_DOC`
- `MIDDLE_DOC`
- `END_DOC`
- `SINGLE_PAGE_DOC`

## Generate Synthetic Data

Start from a directory of already separated single-document PDFs:

```bash
document-recognition generate-synthetic \
  --input-dir data/source_docs \
  --output-dir data/synthetic \
  --num-merged-pdfs 1000
```

Outputs:

- `page_labels.csv` for page-level training
- `pair_labels.csv` for pairwise training
- `page_labels_train.csv` and `page_labels_eval.csv`
- `pair_labels_train.csv` and `pair_labels_eval.csv`
- merged PDFs and rendered page images

## Train Page Baseline

Prepare a CSV with:

```text
pdf_id,page,image_path,label
```

Then run:

```bash
document-recognition train \
  --train-csv data/synthetic/page_labels_train.csv \
  --eval-csv data/synthetic/page_labels_eval.csv \
  --output-dir artifacts/layoutlmv3-boundary
```

## Train Pairwise Model

```bash
document-recognition train-pairwise \
  --train-csv data/synthetic/pair_labels_train.csv \
  --eval-csv data/synthetic/pair_labels_eval.csv \
  --output-dir artifacts/layoutlmv3-pairwise
```

## Predict With Page Baseline

```bash
document-recognition predict \
  --pdf-path samples/merged.pdf \
  --model-dir artifacts/layoutlmv3-boundary \
  --work-dir artifacts/inference
```

## Predict With Pairwise Model

```bash
document-recognition predict-pairwise \
  --pdf-path samples/merged.pdf \
  --model-dir artifacts/layoutlmv3-pairwise \
  --work-dir artifacts/inference \
  --threshold 0.5
```

The pairwise command prints adjacency predictions, reconstructed page labels, and final document ranges.
