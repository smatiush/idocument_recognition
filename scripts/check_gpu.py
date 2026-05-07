from __future__ import annotations

import platform
import shutil
import subprocess


def main() -> None:
    print(f"Python: {platform.python_version()}")

    try:
        import torch
    except ImportError:
        print("torch: not installed")
        return

    print(f"torch: {torch.__version__}")
    print(f"torch cuda build: {torch.version.cuda}")
    print(f"cuda available: {torch.cuda.is_available()}")
    print(f"cuda device count: {torch.cuda.device_count()}")

    if torch.cuda.is_available():
        current_device = torch.cuda.current_device()
        print(f"current cuda device: {current_device}")
        print(f"device name: {torch.cuda.get_device_name(current_device)}")
        capability = torch.cuda.get_device_capability(current_device)
        print(f"device capability: {capability[0]}.{capability[1]}")

    nvidia_smi = shutil.which("nvidia-smi")
    print(f"nvidia-smi: {nvidia_smi or 'not found'}")
    if nvidia_smi:
        result = subprocess.run(
            [nvidia_smi],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        print(result.stdout)


if __name__ == "__main__":
    main()
