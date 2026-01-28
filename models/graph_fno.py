"""Graph Fourier Neural Operator."""

import torch
from torch import nn

from models.base import OperatorModel
from models.blocks.graph_block import GraphFNOBlock


class GraphFNO(OperatorModel):
    """
    Graph Fourier Neural Operator.

    Operates on graph-structured data using spectral convolutions
    defined by the graph Laplacian eigenvectors. Uses truncated
    eigenbasis for efficiency: O(N·k) instead of O(N²).

    Input: (B, N) node features
    Output: (B, N) predicted node features
    """

    def __init__(self, graph, width=32, depth=4, modes=32, device=None):
        """
        Parameters
        ----------
        graph : Tensor
            Laplacian eigenvectors of shape (N, N)
        width : int
            Hidden channel dimension
        depth : int
            Number of GraphFNO blocks
        modes : int
            Number of spectral modes to use (truncated basis size)
        device : torch.device or None
            Device to place model on
        """
        super().__init__()

        if device is None:
            device = graph.device

        # Store truncated eigenbasis: first `modes` eigenvectors
        # U_k has shape (N, modes)
        N = graph.shape[0]
        modes = min(modes, N)
        self.modes = modes
        self.N = N

        U_k = graph[:, :modes].to(device)
        self.register_buffer("U_k", U_k)  # (N, modes)

        self.input_proj = nn.Linear(1, width)
        self.blocks = nn.ModuleList(
            [GraphFNOBlock(modes=modes, width=width) for _ in range(depth)]
        )
        self.output_proj = nn.Linear(width, 1)

    def forward(self, x):
        """
        Parameters
        ----------
        x : Tensor
            Input node features of shape (B, N)

        Returns
        -------
        Tensor
            Output node features of shape (B, N)
        """
        x = x.unsqueeze(-1)  # (B, N, 1)
        x = self.input_proj(x)  # (B, N, width)

        for block in self.blocks:
            x = block(x, self.U_k)

        x = self.output_proj(x)  # (B, N, 1)
        return x.squeeze(-1)  # (B, N)
