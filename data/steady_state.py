"""Steady-state neural network data generation for graph-based models."""

import math
from pathlib import Path

import torch
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Graph construction
# -----------------------------------------------------------------------------


def ring_graph(N, frac_inhib=0.2, scale=0.8):
    """
    Create a ring graph with excitatory and inhibitory connections.

    Parameters
    ----------
    N : int
        Number of nodes
    frac_inhib : float
        Fraction of inhibitory nodes
    scale : float
        Connection strength scale

    Returns
    -------
    Tensor
        Weight matrix of shape (N, N)
    """
    idx = torch.arange(N)
    theta = 2 * math.pi * idx / N
    dtheta = theta.unsqueeze(1) - theta.unsqueeze(0)

    W = scale * torch.cos(dtheta)

    n_inhib = int(N * frac_inhib)
    inhib_idx = torch.randperm(N)[:n_inhib]
    W[inhib_idx, :] *= -1

    W.fill_diagonal_(0.0)

    return W


# -----------------------------------------------------------------------------
# Stability
# -----------------------------------------------------------------------------


def scale_spectral_radius(W, safety=0.05):
    """
    Scale weight matrix to ensure spectral radius < 1.

    Parameters
    ----------
    W : Tensor
        Weight matrix
    safety : float
        Safety margin below 1

    Returns
    -------
    Tensor
        Scaled weight matrix
    """
    eigvals = torch.linalg.eigvals(W)
    rho = eigvals.abs().max().item()
    if rho >= 1:
        W = W / (rho + safety)
    return W


# -----------------------------------------------------------------------------
# Steady-state solver
# -----------------------------------------------------------------------------


def steady_state(W, I, iters=500):
    """
    Compute steady state of nonlinear recurrent network.

    Iterates: r_{t+1} = ReLU(W @ r_t + I)

    Parameters
    ----------
    W : Tensor
        Weight matrix (N, N)
    I : Tensor
        External input (N,)
    iters : int
        Number of iterations

    Returns
    -------
    Tensor
        Steady-state activity (N,)
    """
    r = torch.zeros_like(I)
    for _ in range(iters):
        r = F.relu(W @ r + I)
    return r


# -----------------------------------------------------------------------------
# Input generation
# -----------------------------------------------------------------------------


def random_sinusoid(N, input_scale=1.0, n_modes=3):
    """
    Generate random sinusoidal input pattern.

    Parameters
    ----------
    N : int
        Number of nodes
    input_scale : float
        Amplitude scale
    n_modes : int
        Number of sinusoidal components

    Returns
    -------
    Tensor
        Input pattern of shape (N,)
    """
    x = torch.linspace(0, 2 * math.pi, N)
    I = torch.zeros_like(x)

    for _ in range(n_modes):
        freq = torch.randint(1, 6, (1,)).item()
        phase = 2 * math.pi * torch.rand(1).item()
        amp = input_scale * (0.5 + torch.rand(1).item())
        I += amp * torch.sin(freq * x + phase)

    return I


# -----------------------------------------------------------------------------
# Laplacian eigenbasis
# -----------------------------------------------------------------------------


def laplacian_basis(W):
    """
    Compute graph Laplacian and its eigenbasis.

    Parameters
    ----------
    W : Tensor
        Weight matrix (N, N)

    Returns
    -------
    L : Tensor
        Laplacian matrix
    evals : Tensor
        Eigenvalues
    evecs : Tensor
        Eigenvectors
    """
    # Symmetrize connectivity
    A = (W + W.T) / 2
    D = torch.diag(A.sum(dim=1))
    L = D - A
    evals, evecs = torch.linalg.eigh(L)
    return L, evals, evecs


# -----------------------------------------------------------------------------
# Dataset generation
# -----------------------------------------------------------------------------


def generate_dataset(
    N,
    n_samples,
    frac_inhib=0.2,
    scale=0.8,
    input_scale=1.0,
):
    """
    Generate steady-state dataset for graph neural operator.

    Parameters
    ----------
    N : int
        Number of nodes
    n_samples : int
        Number of samples
    frac_inhib : float
        Fraction of inhibitory nodes
    scale : float
        Connection strength scale
    input_scale : float
        Input amplitude scale

    Returns
    -------
    dict
        Dataset with inputs, outputs, graph, and spectral information
    """
    W = ring_graph(N, frac_inhib, scale)
    W = scale_spectral_radius(W)

    inputs = []
    outputs = []

    for _ in range(n_samples):
        I = random_sinusoid(N, input_scale=input_scale)
        r_inf = steady_state(W, I)
        inputs.append(I)
        outputs.append(r_inf)

    inputs = torch.stack(inputs)  # (B, N)
    outputs = torch.stack(outputs)  # (B, N)

    # Compute Laplacian eigenbasis
    L, evals, evecs = laplacian_basis(W) #Graph Fourier Transform

    # Project into spectral domain
    inputs_spec = inputs @ evecs  # (B, N)
    outputs_spec = outputs @ evecs  # (B, N)

    return {
        "inputs_node": inputs,
        "outputs_node": outputs,
        "inputs_spec": inputs_spec,
        "outputs_spec": outputs_spec,
        "W": W,
        "L": L,
        "evals": evals,
        "evecs": evecs,
    }


# -----------------------------------------------------------------------------
# Save/Load
# -----------------------------------------------------------------------------


def save_dataset(path, data):
    """Save dataset to disk."""
    torch.save(data, path)


def load_dataset(path):
    """Load dataset from disk."""
    return torch.load(path, weights_only=False)


if __name__ == "__main__":
    OUT = Path("steady_ring_dataset_with_basis.pt")

    data = generate_dataset(
        N=200,
        n_samples=5000,
        frac_inhib=0.2,
        scale=0.8,
        input_scale=1.0,
    )

    save_dataset(OUT, data)
    print(f"Saved: {OUT}")
