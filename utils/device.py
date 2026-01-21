"""Device detection utilities."""

import torch


def get_device():
    """
    Get the best available device.

    Returns
    -------
    torch.device
        MPS if available (macOS), CUDA if available, otherwise CPU.
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")
