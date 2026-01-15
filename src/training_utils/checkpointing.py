"""
Checkpoint management utilities for saving and loading model training state.

This module provides utilities for creating and restoring checkpoints during
neural network training, enabling training resumption and model recovery.
"""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch.amp import GradScaler


def save_checkpoint(
    output_dir: Path,
    epoch: int,
    model: torch.nn.Module,
    optimiser: torch.optim.Optimizer,
    scaler: GradScaler,
    initial_v: np.ndarray,
    initial_g: np.ndarray,
    initial_g_FF: np.ndarray,
    input_spikes: np.ndarray,
    best_loss: float,
    **losses: float,
) -> bool:
    """Save model checkpoint to disk.

    Args:
        output_dir (Path): Directory where checkpoint will be saved
        epoch (int): Current epoch number
        model (torch.nn.Module): Model to checkpoint
        optimiser (torch.optim.Optimizer): Optimizer state to save
        scaler (GradScaler): Mixed precision scaler state to save
        initial_v (np.ndarray): Current membrane potentials
        initial_g (np.ndarray): Current recurrent conductances
        initial_g_FF (np.ndarray): Current feedforward conductances
        input_spikes (np.ndarray): Input spike trains
        best_loss (float): Best loss seen so far
        **losses: Arbitrary loss values (must include 'total' for comparison)

    Returns:
        bool: True if this is the best model so far, False otherwise
    """
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    # Convert numpy arrays to torch tensors for safe loading with weights_only=True
    def to_tensor(arr):
        """Convert numpy array to torch tensor, handling empty arrays."""
        if isinstance(arr, np.ndarray):
            if arr.size == 0:
                # Return empty tensor with original dtype
                return torch.empty(0, dtype=torch.float32)
            return torch.from_numpy(arr)
        return arr

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimiser_state_dict": optimiser.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "initial_v": to_tensor(initial_v),
        "initial_g": to_tensor(initial_g),
        "initial_g_FF": to_tensor(initial_g_FF),
        "input_spikes": to_tensor(input_spikes),
        "best_loss": best_loss,
        "rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
        **losses,  # Include all provided losses
    }

    # Save as latest checkpoint (for resumption)
    latest_path = checkpoint_dir / "checkpoint_latest.pt"
    torch.save(checkpoint, latest_path)

    # Save as best if this is the best model (requires 'total' loss)
    total_loss = losses.get("total")
    if total_loss is None:
        raise ValueError(
            "save_checkpoint requires 'total' loss for best model comparison"
        )

    is_best = total_loss <= best_loss
    if is_best:
        best_path = checkpoint_dir / "checkpoint_best.pt"
        torch.save(checkpoint, best_path)
        print(f"  ✓ New best model saved (loss: {total_loss:.6f})")

    return is_best


def load_checkpoint(
    checkpoint_path: Path,
    model: torch.nn.Module,
    optimiser: torch.optim.Optimizer,
    scaler: GradScaler,
    device: str,
) -> Tuple[
    int, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], float
]:
    """Load model checkpoint from disk.

    Args:
        checkpoint_path (Path): Path to checkpoint file
        model (torch.nn.Module): Model to load state into
        optimiser (torch.optim.Optimizer): Optimizer to load state into
        scaler (GradScaler): Mixed precision scaler to load state into
        device (str): Device to load tensors onto

    Returns:
        tuple: (epoch, initial_v, initial_g, initial_g_FF, best_loss)
            - epoch (int): Epoch number where training was checkpointed
            - initial_v (torch.Tensor | None): Initial membrane potentials
            - initial_g (torch.Tensor | None): Initial recurrent conductances
            - initial_g_FF (torch.Tensor | None): Initial feedforward conductances
            - best_loss (float): Best loss achieved so far
    """
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimiser.load_state_dict(checkpoint["optimiser_state_dict"])

    # Load scaler state if available (for backward compatibility)
    if "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    epoch = checkpoint["epoch"]

    # All data should now be tensors, just move to device
    # Handle None or empty tensors gracefully
    def process_tensor(tensor, target_device):
        """Process tensor, handling None and empty cases."""
        if tensor is None:
            return None
        if isinstance(tensor, torch.Tensor):
            # Check if empty tensor
            if tensor.numel() == 0:
                return None
            return tensor.to(target_device)
        # Backward compatibility: convert numpy if present
        if isinstance(tensor, np.ndarray):
            if tensor.size == 0:
                return None
            return torch.from_numpy(tensor).to(target_device)
        return None

    initial_v = process_tensor(checkpoint["initial_v"], device)
    initial_g = process_tensor(checkpoint["initial_g"], device)
    initial_g_FF = process_tensor(checkpoint["initial_g_FF"], device)

    best_loss = checkpoint.get("best_loss", float("inf"))

    # Restore random states
    torch.set_rng_state(checkpoint["rng_state"])
    np.random.set_state(checkpoint["numpy_rng_state"])

    print(f"  ✓ Resumed from epoch {epoch}, best loss: {best_loss:.6f}")
    return epoch, initial_v, initial_g, initial_g_FF, best_loss
