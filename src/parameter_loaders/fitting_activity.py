"""Parameter loader for teacher-student fitting activity training.

For training networks to match target activity patterns without fixed connectivity.
"""

from typing import Dict
from pydantic import BaseModel
from .base_configs import (
    SimulationConfig,
    BaseRecurrentLayerConfig,
    BaseFeedforwardLayerConfig,
)


# =============================================================================
# FITTING-ACTIVITY-SPECIFIC CONFIGS
# =============================================================================


class SimulationConfigWithOdours(SimulationConfig):
    """Simulation config extended with odour stimulus parameters."""

    num_odours: int


class StudentSimulationConfig(BaseModel):
    """Minimal simulation config for student training.

    Unlike the base SimulationConfig, this doesn't have dt or duration in TOML.
    Instead, dt comes from the dataset and num_chunks is computed from
    epochs * chunks_per_epoch. Call setup() after loading the dataset.
    """

    seed: int
    chunk_size: int

    # These are set by setup() method
    dt: float = 0.0
    num_chunks: int = 0

    def setup(self, dt: float, chunks_per_epoch: int, epochs: int) -> None:
        """Initialize computed simulation parameters from dataset and training config.

        Args:
            dt: Timestep in milliseconds (from dataset).
            chunks_per_epoch: Number of chunks in one epoch (from dataset).
            epochs: Number of training epochs (from training config).
        """
        self.dt = dt
        self.num_chunks = epochs * chunks_per_epoch

    @property
    def chunk_duration_s(self) -> float:
        """Duration of a single chunk in seconds."""
        return self.chunk_size * self.dt / 1000.0

    @property
    def total_duration_s(self) -> float:
        """Total training duration in seconds."""
        return self.num_chunks * self.chunk_duration_s


class StudentTrainingConfig(BaseModel):
    """Training config for student."""

    epochs: int
    chunks_per_update: int
    log_interval: int
    checkpoint_interval: int
    mixed_precision: bool
    weight_perturbation_variance: float

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


class StudentLossWeights(BaseModel):
    """Loss function weights for student training."""

    firing_rate: float
    vanrossum: float


class StudentHyperparameters(BaseModel):
    """Optimization hyperparameters for student training."""

    surrgrad_scale: float
    learning_rate: float
    van_rossum_tau: float
    loss_weight: StudentLossWeights


class OdourInputConfig(BaseModel):
    """Configuration for modulated Poisson input activity.

    Defines baseline and modulated firing rates for input cells responding to odours.
    A fraction of cells are modulated up/down from baseline when odour is present.
    """

    baseline_rate: float
    modulation_rate: float
    modulation_fraction: float

    def get_modulated_rates(self) -> tuple[float, float]:
        """Get the up-modulated and down-modulated rates.

        Returns:
            Tuple of (up_rate, down_rate) in Hz.
        """
        up_rate = self.baseline_rate + self.modulation_rate
        down_rate = self.baseline_rate - self.modulation_rate
        return (up_rate, down_rate)

    def get_n_modulated(self, n_neurons: int) -> int:
        """Get number of neurons modulated in each direction (up or down).

        Args:
            n_neurons: Total number of neurons of this cell type.

        Returns:
            Number of neurons to modulate up (same number will be modulated down).
        """
        return int(n_neurons * self.modulation_fraction / 2.0)


# =============================================================================
# TOP-LEVEL MODELS
# =============================================================================


class TeacherActivityParams(BaseModel):
    """Parameters for generating teacher activity from a trained network.

    Loads pre-trained network structure and generates activity for use as training target.
    Does not include training or optimization parameters.
    """

    simulation: SimulationConfigWithOdours
    recurrent: BaseRecurrentLayerConfig
    feedforward: BaseFeedforwardLayerConfig
    odours: Dict[str, OdourInputConfig]


class StudentTrainingParams(BaseModel):
    """Parameters for training student network to match teacher activity.

    Includes training configuration and hyperparameters for optimizing network
    to reproduce target activity patterns. Does not include odours configuration.
    """

    simulation: StudentSimulationConfig
    training: StudentTrainingConfig
    hyperparameters: StudentHyperparameters
    recurrent: BaseRecurrentLayerConfig
    feedforward: BaseFeedforwardLayerConfig
