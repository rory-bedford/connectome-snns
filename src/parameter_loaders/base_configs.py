"""Base configuration classes shared across different parameter loaders.

Pydantic models for common configuration sections used in TOML files.
"""

import numpy as np
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


# =============================================================================
# SYNAPSE TYPE CONSTANTS
# =============================================================================

# Hardcoding synapse types for simplicity
EXCITATORY_SYNAPSE_TYPES = ["AMPA", "NMDA"]
INHIBITORY_SYNAPSE_TYPES = ["GABA_A", "GABA_B"]


# =============================================================================
# SIMULATION AND TRAINING CONFIGS
# =============================================================================


class SimulationConfig(BaseModel):
    """Simulation parameters."""

    dt: float
    duration: float
    seed: int
    chunk_size: int

    @property
    def num_chunks(self) -> int:
        """Total number of chunks"""
        return int(self.duration / (self.chunk_size * self.dt))

    @property
    def chunk_duration_s(self) -> float:
        """Duration of a single chunk in seconds."""
        return self.chunk_size * self.dt / 1000.0

    @property
    def total_duration_s(self) -> float:
        """Total simulation duration in seconds."""
        return self.num_chunks * self.chunk_duration_s


class TrainingConfig(BaseModel):
    """Training parameters."""

    chunks_per_update: int
    log_interval: int
    checkpoint_interval: int
    mixed_precision: bool
    plot_size: int
    batch_size: int


# =============================================================================
# LOSS AND OPTIMIZATION CONFIGS
# =============================================================================


class LossWeights(BaseModel):
    """Loss function weights."""

    firing_rate: float
    cv: float
    silent_penalty: float
    membrane_variance: float


class Hyperparameters(BaseModel):
    """Optimization hyperparameters."""

    surrgrad_scale: float
    learning_rate: float
    loss_weight: LossWeights


class Targets(BaseModel):
    """Target values for all cell types."""

    firing_rate: Dict[str, float]
    alpha: Dict[str, float]
    threshold_ratio: Dict[str, float]


# =============================================================================
# NETWORK STRUCTURE CONFIGS
# =============================================================================


class CellTypesConfig(BaseModel):
    """Cell types."""

    names: List[str]
    proportion_list: List[float] = Field(alias="proportion")

    @property
    def proportion(self) -> np.ndarray:
        """Cell type proportions as numpy array."""
        return np.array(self.proportion_list)

    class Config:
        populate_by_name = True


class TopologyConfig(BaseModel):
    """Network topology."""

    num_neurons: int
    num_assemblies: Optional[int] = None
    neurons_per_assembly: Optional[int] = None
    conn_within_list: Optional[List[List[float]]] = Field(
        default=None, alias="conn_within"
    )
    conn_between_list: Optional[List[List[float]]] = Field(
        default=None, alias="conn_between"
    )
    conn_inputs_list: Optional[List[List[float]]] = Field(
        default=None, alias="conn_inputs"
    )

    @property
    def conn_within(self) -> Optional[np.ndarray]:
        """Within-assembly connectivity as numpy array."""
        return (
            np.array(self.conn_within_list)
            if self.conn_within_list is not None
            else None
        )

    @property
    def conn_between(self) -> Optional[np.ndarray]:
        """Between-assembly connectivity as numpy array."""
        return (
            np.array(self.conn_between_list)
            if self.conn_between_list is not None
            else None
        )

    @property
    def conn_inputs(self) -> Optional[np.ndarray]:
        """Input connectivity as numpy array."""
        return (
            np.array(self.conn_inputs_list)
            if self.conn_inputs_list is not None
            else None
        )

    class Config:
        populate_by_name = True


class WeightsConfig(BaseModel):
    """Weight parameters."""

    w_mu_list: List[List[float]] = Field(alias="w_mu")
    w_sigma_list: List[List[float]] = Field(alias="w_sigma")

    @property
    def w_mu(self) -> np.ndarray:
        """Mean weights as numpy array."""
        return np.array(self.w_mu_list)

    @property
    def w_sigma(self) -> np.ndarray:
        """Weight standard deviations as numpy array."""
        return np.array(self.w_sigma_list)

    class Config:
        populate_by_name = True


# =============================================================================
# PHYSIOLOGY AND SYNAPSE CONFIGS
# =============================================================================


class PhysiologyConfig(BaseModel):
    """Neuronal physiology."""

    tau_mem: float
    theta: float
    U_reset: float
    E_L: float
    g_L: float
    tau_ref: float


class SynapseConfig(BaseModel):
    """Synapse parameters."""

    names: List[str]
    tau_rise_list: List[float] = Field(alias="tau_rise")
    tau_decay_list: List[float] = Field(alias="tau_decay")
    E_syn_list: List[float] = Field(alias="E_syn")
    g_bar_list: List[float] = Field(alias="g_bar")

    @property
    def tau_rise(self) -> np.ndarray:
        """Rise time constants as numpy array."""
        return np.array(self.tau_rise_list)

    @property
    def tau_decay(self) -> np.ndarray:
        """Decay time constants as numpy array."""
        return np.array(self.tau_decay_list)

    @property
    def E_syn(self) -> np.ndarray:
        """Reversal potentials as numpy array."""
        return np.array(self.E_syn_list)

    @property
    def g_bar(self) -> np.ndarray:
        """Maximal conductances as numpy array."""
        return np.array(self.g_bar_list)

    class Config:
        populate_by_name = True


class ActivityConfig(BaseModel):
    """Input activity."""

    firing_rate: float
