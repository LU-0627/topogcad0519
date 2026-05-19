"""Environment helpers for Topo-GCAD."""

from __future__ import annotations

import torch


def get_device() -> torch.device:
    """Return the preferred PyTorch compute device.

    CUDA is selected when an NVIDIA GPU is available; otherwise the project
    falls back to CPU so the code can run on machines without GPU support.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
