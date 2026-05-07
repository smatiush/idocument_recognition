# Transfer Project via SSH

This guide transfers the `document_recognition` project to another machine over SSH.

## Prerequisites

On your local machine:

```bash
ssh user@remote-host
```

must work before running the transfer commands.

Replace these values in the examples:

```bash
REMOTE_USER=user
REMOTE_HOST=remote-host
REMOTE_DIR=/home/user/document_recognition
LOCAL_DIR=/home/cg/Desktop/pmsr/document_recognition
```

## Recommended Transfer

This copies the project code, configuration, scripts, README, and current artifacts/data, while skipping Python caches and local virtual environments.

```bash
rsync -avz --progress \
  --exclude '.git/' \
  --exclude '.venv/' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude '.pytest_cache/' \
  /home/cg/Desktop/pmsr/document_recognition/ \
  user@remote-host:/home/user/document_recognition/
```

## Full Transfer Including Git Metadata

Use this if you want the remote copy to keep the local Git repository history and branches.

```bash
rsync -avz --progress \
  --exclude '.venv/' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude '.pytest_cache/' \
  /home/cg/Desktop/pmsr/document_recognition/ \
  user@remote-host:/home/user/document_recognition/
```

## Minimal Code-Only Transfer

Use this if you do not want to copy generated datasets, model artifacts, or source PDFs.

```bash
rsync -avz --progress \
  --exclude '.git/' \
  --exclude '.venv/' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude '.pytest_cache/' \
  --exclude 'data/' \
  --exclude 'artifacts/' \
  /home/cg/Desktop/pmsr/document_recognition/ \
  user@remote-host:/home/user/document_recognition/
```

## Resume an Interrupted Transfer

`rsync` can safely be run again with the same command. It will only copy missing or changed files.

For very large generated files, use:

```bash
rsync -avz --partial --append-verify --progress \
  --exclude '.venv/' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  /home/cg/Desktop/pmsr/document_recognition/ \
  user@remote-host:/home/user/document_recognition/
```

## Verify Remote Files

```bash
ssh user@remote-host
cd /home/user/document_recognition
ls -la
find src/document_recognition -maxdepth 1 -type f | sort
```

## Install Prerequisites on the Remote Machine

After transfer:

```bash
ssh user@remote-host
cd /home/user/document_recognition
./scripts/install_prereqs.sh
```

Activate the environment:

```bash
source .venv/bin/activate
```

Run the UI:

```bash
streamlit run src/document_recognition/ui.py
```

## Optional: Run Streamlit Remotely and Open It Locally

On the remote machine:

```bash
cd /home/user/document_recognition
source .venv/bin/activate
streamlit run src/document_recognition/ui.py --server.address 0.0.0.0 --server.port 8501
```

From your local machine, create an SSH tunnel:

```bash
ssh -L 8501:localhost:8501 user@remote-host
```

Then open:

```text
http://localhost:8501
```

## Notes

- Do not transfer `.venv/`; recreate it on the remote machine with `./scripts/install_prereqs.sh`.
- Use the recommended transfer if you want current generated datasets and artifacts.
- Use the minimal transfer if you only want source code and project setup files.
- If the remote server has no `sudo`, install system packages manually or ask the administrator to install Tesseract OCR and Python venv support.
