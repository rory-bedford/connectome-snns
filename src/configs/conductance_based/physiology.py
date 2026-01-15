"""Conductance-based model physiology configuration."""

import numpy as np
from typing import List
from pydantic import BaseModel, Field


class PhysiologyConfig(BaseModel):
    """Neuronal physiology parameters for conductance-based models."""

    tau_mem: float  # Membrane time constant (ms)
    theta: float  # Spike threshold voltage (mV)
    U_reset: float  # Reset voltage after spike (mV)
    E_L: float  # Leak reversal potential (mV)
    g_L: float  # Leak conductance (nS)
    tau_ref: float  # Refractory period (ms)


# Synapse type constants
EXCITATORY_SYNAPSE_TYPES = ["AMPA", "NMDA"]
INHIBITORY_SYNAPSE_TYPES = ["GABA_A", "GABA_B"]


class SynapseConfig(BaseModel):
    """Synapse parameters for conductance-based models.

    Supports multiple synapse types per cell type (e.g., AMPA and NMDA for excitatory cells).
    """

    names: List[str]  # Synapse type names (e.g., ["AMPA", "NMDA"])
    tau_rise_list: List[float] = Field(alias="tau_rise")  # Rise time constants (ms)
    tau_decay_list: List[float] = Field(alias="tau_decay")  # Decay time constants (ms)
    E_syn_list: List[float] = Field(alias="E_syn")  # Reversal potentials (mV)
    g_bar_list: List[float] = Field(alias="g_bar")  # Maximal conductances (nS)

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
