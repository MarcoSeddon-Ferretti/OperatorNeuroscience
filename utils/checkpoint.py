"""Checkpoint save/load utilities."""

import os
import torch


def save_checkpoint(path, model, optimizer=None, epoch=None, best_val=None, extra=None):
    """
    Save a training checkpoint.

    Parameters
    ----------
    path : str
        File path for the checkpoint
    model : nn.Module
        Model to save
    optimizer : Optimizer or None
        Optimizer state to save
    epoch : int or None
        Current epoch number
    best_val : float or None
        Best validation metric
    extra : dict or None
        Additional metadata
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    ckpt = {
        "model_state": model.state_dict(),
        "epoch": epoch,
        "best_val": best_val,
        "extra": extra or {},
    }
    if optimizer is not None:
        ckpt["optim_state"] = optimizer.state_dict()
    torch.save(ckpt, path)
    print(f"Saved checkpoint to: {path}")


def load_checkpoint(path, model, optimizer=None, map_location="cpu"):
    """
    Load a training checkpoint.

    Parameters
    ----------
    path : str
        File path of the checkpoint
    model : nn.Module
        Model to load weights into
    optimizer : Optimizer or None
        Optimizer to load state into
    map_location : str or torch.device
        Device to map tensors to

    Returns
    -------
    dict
        The loaded checkpoint dictionary
    """
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and "optim_state" in ckpt:
        optimizer.load_state_dict(ckpt["optim_state"])
    print(f"Loaded checkpoint from: {path}")
    return ckpt
