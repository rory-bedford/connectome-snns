"""Shared training configuration parameters.

Configuration for training settings that are common across all model types.
"""

from typing import Dict, Literal, Optional
from pydantic import BaseModel


class TrainingConfig(BaseModel):
    """Basic training parameters."""

    chunks_per_update: int
    log_interval: int
    checkpoint_interval: int
    mixed_precision: bool
    plot_size: int
    grad_norm_clip: Optional[float] = None
    optimisable: (
        Literal[
            "weights",
            "scaling_factors",
            "scaling_factors_recurrent",
            "scaling_factors_feedforward",
            None,
        ]
        | None
    ) = None


class StudentTrainingConfig(BaseModel):
    """Training config for student network training."""

    epochs: int
    chunks_per_update: int
    log_interval: int
    checkpoint_interval: int
    plot_size: int
    mixed_precision: bool
    weight_perturbation_variance: float
    grad_norm_clip: Optional[float] = None
    optimisable: Literal[
        "weights",
        "scaling_factors",
        "scaling_factors_recurrent",
        "scaling_factors_feedforward",
    ]

    def total_chunks(self, num_chunks_per_epoch: int) -> int:
        """Total number of chunks across all epochs.

        Args:
            num_chunks_per_epoch: Number of chunks in one epoch.

        Returns:
            Total number of training chunks.
        """
        return self.epochs * num_chunks_per_epoch

    def log_interval_s(self, chunk_duration_s: float) -> float:
        """Time interval between logging in seconds.

        Args:
            chunk_duration_s: Duration of one chunk in seconds.

        Returns:
            Log interval in seconds.
        """
        return self.log_interval * chunk_duration_s

    def checkpoint_interval_s(self, chunk_duration_s: float) -> float:
        """Time interval between checkpoints in seconds.

        Args:
            chunk_duration_s: Duration of one chunk in seconds.

        Returns:
            Checkpoint interval in seconds.
        """
        return self.checkpoint_interval * chunk_duration_s


class EMTrainingConfig(BaseModel):
    """Training config for EM-based training (no epochs needed)."""

    chunks_per_update: int
    log_interval: int
    checkpoint_interval: int
    plot_size: int
    mixed_precision: bool
    weight_perturbation_variance: float
    grad_norm_clip: Optional[float] = None
    optimisable: Literal[
        "weights",
        "scaling_factors",
        "scaling_factors_recurrent",
        "scaling_factors_feedforward",
    ]


class LossWeights(BaseModel):
    """Loss function weights for homeostatic plasticity."""

    firing_rate: float
    cv: float
    silent_penalty: float
    membrane_variance: float
    weight_ratio: float


class StudentLossWeights(BaseModel):
    """Loss function weights for student training."""

    van_rossum: float


class Hyperparameters(BaseModel):
    """Optimization hyperparameters for homeostatic plasticity."""

    surrgrad_scale: float
    learning_rate: float
    loss_weight: LossWeights


class StudentHyperparameters(BaseModel):
    """Optimization hyperparameters for student training."""

    surrgrad_scale: float
    learning_rate: float
    van_rossum_tau: float
    loss_weight: StudentLossWeights


class Targets(BaseModel):
    """Target values for homeostatic plasticity training."""

    firing_rate: Dict[str, float]
    alpha: Dict[str, float]
    threshold_ratio: Dict[str, float]
    weight_ratio: float
