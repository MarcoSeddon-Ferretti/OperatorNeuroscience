"""Base class for operator learning models."""

import torch
from torch import nn
from tqdm import tqdm

from utils.device import get_device


class OperatorModel(nn.Module):
    """
    Base class for neural operator models.

    Provides common training, saving, and loading functionality.
    Subclasses should implement forward().
    """

    def fit(
        self,
        x,
        y,
        *,
        epochs=50,
        batch_size=32,
        lr=1e-3,
        multistep=1,
        weight_decay=1e-4,
        clip_grad=1.0,
        verbose=False,
        device=None,
    ):
        """
        Train the model.

        Parameters
        ----------
        x : Tensor
            Input data
        y : Tensor
            Target data
        epochs : int
            Number of training epochs
        batch_size : int
            Batch size
        lr : float
            Learning rate
        multistep : int
            Number of autoregressive steps for loss computation
        weight_decay : float
            L2 regularization
        clip_grad : float or None
            Gradient clipping threshold
        verbose : bool
            Print epoch losses
        device : torch.device or None
            Device to train on (auto-detected if None)
        """
        if device is None:
            device = get_device()

        self.to(device)
        x, y = x.to(device), y.to(device)

        optimizer = torch.optim.Adam(
            self.parameters(), lr=lr, weight_decay=weight_decay
        )
        loss_fn = nn.MSELoss()

        n = x.shape[0]

        for epoch in range(epochs):
            perm = torch.randperm(n, device=device)
            total_loss = 0.0
            num_batches = 0

            pbar = tqdm(
                range(0, n, batch_size),
                desc=f"Epoch {epoch + 1}/{epochs}",
                leave=False,
            )

            for i in pbar:
                idx = perm[i : i + batch_size]
                optimizer.zero_grad()

                pred = x[idx]
                loss = 0.0
                for _ in range(multistep):
                    pred = self(pred)
                    loss += loss_fn(pred, y[idx])
                loss /= multistep

                loss.backward()
                if clip_grad is not None:
                    nn.utils.clip_grad_norm_(self.parameters(), clip_grad)
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1
                pbar.set_postfix(loss=f"{loss.item():.2e}")

            if verbose:
                print(f"Epoch {epoch:03d}: loss={total_loss / num_batches:.4e}")

    def save(self, path):
        """Save model weights to disk."""
        torch.save(self.state_dict(), path)
        print(f"Model saved to: {path}")

    def load(self, path, device=None):
        """Load model weights from disk."""
        if device is None:
            device = next(self.parameters()).device

        state = torch.load(path, map_location=device, weights_only=True)
        self.load_state_dict(state)
        print(f"Model loaded from: {path}")
