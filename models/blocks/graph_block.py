"""Graph FNO block implementation."""

import torch
from torch import nn


class GraphFNOBlock(nn.Module):
    """
    Graph Fourier Neural Operator block.

    Performs spectral convolution using graph Laplacian eigenvectors.
    """

    def __init__(self, graph_U, width, modes=None, device=None):
        """
        Parameters
        ----------
        graph_U : Tensor
            Laplacian eigenvectors of shape (N, N)
        width : int
            Number of channels
        modes : int or None
            Number of spectral modes to use (all if None)
        device : torch.device or None
            Device to place tensors on
        """
        super().__init__()
        if device is None:
            device = graph_U.device

        self.U = graph_U.to(device)
        self.U_t = graph_U.T.to(device)

        self.N = graph_U.shape[0]
        self.width = width

        # Truncate number of spectral modes if needed
        if modes is None:
            modes = self.N
        self.modes = modes

        # Learnable spectral weights
        self.weight = nn.Parameter(torch.randn(modes, width, width) * 0.02)

        # Feedforward mixing in node space
        self.ff = nn.Sequential(
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, width),
        )

    def forward(self, x):
        """
        Parameters
        ----------
        x : Tensor
            Input of shape (B, N, width)

        Returns
        -------
        Tensor
            Output of shape (B, N, width)
        """
        B, N, C = x.shape
        assert N == self.N

        # Graph Fourier Transform
        x_hat = torch.matmul(self.U_t, x)  # (B, N, C)

        # Truncate to low modes
        x_hat_low = x_hat[:, : self.modes, :]  # (B, modes, C)

        # Spectral mixing
        x_hat_mix = torch.einsum("bmc,mcw->bmw", x_hat_low, self.weight)

        # Pad back to full spectral size
        x_hat_new = torch.zeros_like(x_hat)
        x_hat_new[:, : self.modes, :] = x_hat_mix

        # Inverse Graph Fourier Transform
        x_out = torch.matmul(self.U, x_hat_new)  # (B, N, C)

        # Skip connection + spectral output + feedforward
        x_out = x + x_out + self.ff(x)

        return x_out
