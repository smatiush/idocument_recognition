from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import os
import sys


@contextmanager
def suppress_mupdf_stderr() -> Iterator[None]:
    """Suppress MuPDF's C-level stderr warnings around known-noisy PDF operations."""
    if os.environ.get("DOCUMENT_RECOGNITION_SHOW_MUPDF_WARNINGS"):
        yield
        return

    sys.stderr.flush()
    original_stderr_fd = os.dup(2)
    try:
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            os.dup2(devnull.fileno(), 2)
            yield
    finally:
        sys.stderr.flush()
        os.dup2(original_stderr_fd, 2)
        os.close(original_stderr_fd)
