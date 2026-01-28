"""Graph FNO block implementation."""

import torch
from torch import nn


class GraphFNOBlock(nn.Module):
    """
    Graph Fourier Neural Operator block.

    Performs spectral convolution using truncated graph Laplacian eigenbasis.
    Eigenbasis is passed in during forward() to avoid redundant storage.
    """

    def __init__(self, modes, width):
        """
        Parameters
        ----------
        modes : int
            Number of spectral modes (truncated basis size)
        width : int
            Number of channels
        """
        super().__init__()

        self.modes = modes
        self.width = width

        # Learnable spectral weights
        self.weight = nn.Parameter(torch.randn(modes, width, width) * 0.02)

        # Feedforward mixing in node space
        self.ff = nn.Sequential(
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, width),
        )

    def forward(self, x, U_k):
        """
        Parameters
        ----------
        x : Tensor
            Input of shape (B, N, width)
        U_k : Tensor
            Truncated eigenbasis of shape (N, modes)

        Returns
        -------
        Tensor
            Output of shape (B, N, width)
        """
        # Graph Fourier Transform (truncated)
        # U_k.T @ x: (modes, N) @ (B, N, width) -> (B, modes, width)
        x_hat = torch.einsum("kn,bnc->bkc", U_k.T, x)  # (B, modes, width)

        # Spectral mixing
        x_hat_mix = torch.einsum("bkc,kcw->bkw", x_hat, self.weight)

        # Inverse Graph Fourier Transform
        # U_k @ x_hat_mix: (N, modes) @ (B, modes, width) -> (B, N, width)
        x_out = torch.einsum("nk,bkc->bnc", U_k, x_hat_mix)  # (B, N, width)

        # Skip connection + spectral output + feedforward
        x_out = x + x_out + self.ff(x)

        return x_out
