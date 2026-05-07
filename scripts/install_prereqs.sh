#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-${PROJECT_ROOT}/.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
INSTALL_DEV="${INSTALL_DEV:-0}"
TESSERACT_LANGS="${TESSERACT_LANGS:-eng}"

log() {
  printf '[install] %s\n' "$*"
}

die() {
  printf '[install] ERROR: %s\n' "$*" >&2
  exit 1
}

command_exists() {
  command -v "$1" >/dev/null 2>&1
}

run_with_sudo() {
  if [[ "${EUID}" -eq 0 ]]; then
    "$@"
  elif command_exists sudo; then
    sudo "$@"
  else
    die "This step requires root privileges and sudo is not installed: $*"
  fi
}

install_system_prereqs_debian() {
  local packages=(
    build-essential
    python3
    python3-pip
    python3-venv
    tesseract-ocr
  )

  IFS='+' read -r -a langs <<< "${TESSERACT_LANGS}"
  for lang in "${langs[@]}"; do
    [[ -n "${lang}" ]] && packages+=("tesseract-ocr-${lang}")
  done

  log "Installing Debian/Ubuntu packages: ${packages[*]}"
  run_with_sudo apt-get update
  run_with_sudo apt-get install -y "${packages[@]}"
}

install_system_prereqs_macos() {
  command_exists brew || die "Homebrew is required on macOS. Install it from https://brew.sh/ and rerun this script."

  log "Installing macOS packages with Homebrew"
  brew install python tesseract
}

install_system_prereqs() {
  case "$(uname -s)" in
    Linux)
      if command_exists apt-get; then
        install_system_prereqs_debian
      else
        die "Unsupported Linux distribution. Install Python 3, pip, venv, build tools, and Tesseract OCR manually."
      fi
      ;;
    Darwin)
      install_system_prereqs_macos
      ;;
    *)
      die "Unsupported operating system: $(uname -s)"
      ;;
  esac
}

create_venv() {
  log "Creating/updating virtual environment: ${VENV_DIR}"
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
  # shellcheck source=/dev/null
  source "${VENV_DIR}/bin/activate"
  python -m pip install --upgrade pip setuptools wheel
}

install_python_package() {
  cd "${PROJECT_ROOT}"

  if [[ "${INSTALL_DEV}" == "1" ]]; then
    log "Installing project with development dependencies"
    python -m pip install -e '.[dev]'
  else
    log "Installing project"
    python -m pip install -e .
  fi
}

verify_installation() {
  log "Verifying Python imports and Tesseract availability"
  python - <<'PY'
import os

from document_recognition.ocr import ensure_tesseract_available

ensure_tesseract_available(os.environ.get("TESSERACT_LANGS", "eng"))
print("Python package and Tesseract OCR are available.")
PY

  log "Installed versions"
  python --version
  python -m pip --version
  tesseract --version | head -n 1
}

main() {
  log "Project root: ${PROJECT_ROOT}"
  install_system_prereqs
  create_venv
  install_python_package
  verify_installation

  log "Done."
  log "Activate the environment with: source ${VENV_DIR}/bin/activate"
  log "Run the UI with: streamlit run src/document_recognition/ui.py"
}

main "$@"
