"""
Train GraphFNO on steady-state neural network data.

This example demonstrates:
- Loading graph-structured steady-state data
- Training a Graph Fourier Neural Operator
- Evaluating predictions on test data
- Visualizing correlation between predictions and ground truth
"""

import numpy as np
import torch
import matplotlib.pyplot as plt

from models import GraphFNO
from data import load_steady_state_dataset, generate_steady_state_dataset, save_steady_state_dataset
from utils import get_device


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

LOAD_DATA = True
DATA_PATH = "datasets/steady_ring_dataset_with_basis.pt"

# Model architecture
WIDTH = 16
DEPTH = 8
MODES = 16

# Training hyperparameters
EPOCHS = 5
LR = 1e-6
BATCH_SIZE = 64

TRAIN = True
MODEL_PATH = "checkpoints/gfno_ring.pt"


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    device = get_device()
    print(f"Using device: {device}")

    # Load or generate data
    if LOAD_DATA:
        try:
            data = load_steady_state_dataset(DATA_PATH)
            print(f"Loaded data from {DATA_PATH}")
        except FileNotFoundError:
            print(f"Data not found at {DATA_PATH}, generating...")
            data = generate_steady_state_dataset(
                N=200,
                n_samples=5000,
                frac_inhib=0.2,
                scale=0.8,
                input_scale=1.0,
            )
            import os
            os.makedirs("datasets", exist_ok=True)
            save_steady_state_dataset(DATA_PATH, data)
            print(f"Saved data to {DATA_PATH}")
    else:
        data = generate_steady_state_dataset(
            N=200,
            n_samples=5000,
            frac_inhib=0.2,
            scale=0.8,
            input_scale=1.0,
        )

    # Extract data
    u0 = data["inputs_spec"].to(device)  # Spectral domain inputs
    u_inf = data["outputs_spec"].to(device)  # Spectral domain outputs
    U = data["evecs"].to(device)  # Laplacian eigenvectors

    # Train/test split
    tr_len = 4000
    train_u0, test_u0 = u0[:tr_len], u0[tr_len:]
    train_u_inf, test_u_inf = u_inf[:tr_len], u_inf[tr_len:]

    print(f"Training samples: {len(train_u0)}, Test samples: {len(test_u0)}")

    # Create model
    model = GraphFNO(
        graph=U,
        width=WIDTH,
        depth=DEPTH,
        modes=MODES,
        device=device,
    )

    # Training or loading
    if TRAIN:
        model.fit(
            train_u0,
            train_u_inf,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            lr=LR,
            multistep=1,  # Steady-state: single step
            verbose=True,
        )
        import os
        os.makedirs("checkpoints", exist_ok=True)
        model.save(MODEL_PATH)
        print(f"Saved model to {MODEL_PATH}")
    else:
        model.load(MODEL_PATH, device=device)
        model.to(device)

    # Evaluation
    model.eval()
    with torch.no_grad():
        pred = model(test_u0)  # (B, N)

    loss = torch.mean((pred - test_u_inf) ** 2)
    print(f"Test MSE loss: {loss.item():.4e}")

    # Visualization: sample predictions
    fig, axs = plt.subplots(2, 2, figsize=(12, 6))
    for b, ax in enumerate(axs.flatten()):
        ax.plot(test_u0[b].cpu(), label="Input u0", linewidth=2)
        ax.plot(test_u_inf[b].cpu(), label="True u∞")
        ax.plot(pred[b].cpu(), label="Pred u∞", linestyle="--")
        ax.set_title(f"Sample {b}")
        ax.legend()
    plt.tight_layout()
    plt.savefig("graph_fno_samples.png")
    plt.show()

    # Correlation analysis
    true_np = test_u_inf.detach().cpu().numpy()
    pred_np = pred.detach().cpu().numpy()

    correlations = []
    for t, p in zip(true_np, pred_np):
        c = np.corrcoef(t, p)[0, 1]
        correlations.append(c)

    correlations = np.array(correlations)

    plt.figure(figsize=(5, 4))
    plt.hist(correlations, bins=20, color="steelblue")
    plt.xlabel("Correlation")
    plt.ylabel("# Samples")
    plt.title("Distribution of correlations (true vs pred)")
    plt.tight_layout()
    plt.savefig("graph_fno_correlations.png")
    plt.show()

    print(f"Mean correlation: {correlations.mean():.4f}")
    print(f"Min correlation: {correlations.min():.4f}")
    print(f"Max correlation: {correlations.max():.4f}")


if __name__ == "__main__":
    main()
