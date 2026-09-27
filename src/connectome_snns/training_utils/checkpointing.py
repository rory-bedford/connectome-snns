"""
Checkpoint management utilities for saving and loading model training state.

Models define their own simulation state via get_checkpoint_state() /
load_checkpoint_state(). The checkpoint file stores model parameters,
optimizer state, and the model's simulation state dict.
"""

from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
import torch
from torch.amp import GradScaler


def save_checkpoint(
    output_dir: Path,
    epoch: int,
    model: torch.nn.Module,
    optimiser: torch.optim.Optimizer,
    scaler: GradScaler,
    best_loss: float,
    scheduler: Optional[Any] = None,
    **losses: float,
) -> bool:
    """Save model checkpoint to disk.

    Args:
        output_dir: Directory where checkpoint will be saved.
        epoch: Current epoch number.
        model: Model to checkpoint. Must implement get_checkpoint_state().
        optimiser: Optimizer state to save.
        scaler: Mixed precision scaler state to save.
        best_loss: Best loss seen so far.
        scheduler: Optional LR scheduler whose state is saved alongside the rest. Without
            it a resumed run restarts its learning-rate schedule from the beginning.
        **losses: Arbitrary loss values (must include 'total' for comparison).

    Returns:
        True if this is the best model so far, False otherwise.
    """
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimiser_state_dict": optimiser.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "simulation_state": model.get_checkpoint_state(),
        "best_loss": best_loss,
        "rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
        **losses,
    }
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

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
    scheduler: Optional[Any] = None,
) -> Tuple[int, float]:
    """Load model checkpoint from disk.

    Restores model parameters, optimizer state, scaler state, RNG states,
    and simulation state (v, g, etc.) via model.load_checkpoint_state().

    Args:
        checkpoint_path: Path to checkpoint file, or to an output directory
            containing a ``checkpoints/`` subfolder.
        model: Model to load state into. Must implement load_checkpoint_state().
        optimiser: Optimizer to load state into.
        scaler: Mixed precision scaler to load state into.
        device: Device to load tensors onto.
        scheduler: Optional LR scheduler to restore, so the learning rate continues from
            where it stopped rather than from the start of the schedule. Checkpoints
            written before schedulers were saved carry no state; resuming from one warns
            and leaves the scheduler at its initial position.

    Returns:
        (epoch, best_loss)
    """
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.is_dir():
        ckpt_dir = checkpoint_path / "checkpoints"
        pts = sorted(ckpt_dir.glob("*.pt"))
        if not pts:
            raise FileNotFoundError(f"No .pt checkpoint files in {ckpt_dir}")
        checkpoint_path = pts[-1]
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Strip _orig_mod. prefix from torch.compile'd checkpoints
    state_dict = checkpoint["model_state_dict"]
    prefix = "_orig_mod."
    if any(k.startswith(prefix) for k in state_dict):
        state_dict = {k.removeprefix(prefix): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    optimiser.load_state_dict(checkpoint["optimiser_state_dict"])

    if "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    if scheduler is not None:
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        else:
            print(
                "  ! checkpoint has no scheduler state: the learning-rate schedule will "
                "restart from its first step"
            )

    epoch = checkpoint["epoch"]
    best_loss = checkpoint.get("best_loss", float("inf"))

    # Restore simulation state (new format)
    if "simulation_state" in checkpoint:
        model.load_checkpoint_state(checkpoint["simulation_state"])
    # Old format: initial_v/g/g_FF are ignored (model keeps reset state)

    # Restore random states
    rng_state = checkpoint["rng_state"]
    if rng_state.is_cuda:
        rng_state = rng_state.cpu()
    torch.set_rng_state(rng_state)
    np.random.set_state(checkpoint["numpy_rng_state"])

    print(f"  ✓ Resumed from epoch {epoch}, best loss: {best_loss:.6f}")
    return epoch, best_loss
