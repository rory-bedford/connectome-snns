"""
Utility functions for generating spike trains.

This module provides functions for creating synthetic spike trains with
various statistical properties for use in spiking neural network simulations.
"""

import torch
from typing import Dict, Union


def generate_poisson_spikes(
    n_steps: int,
    dt: float,
    num_neurons: int,
    cell_type_indices: torch.Tensor,
    cell_type_names: list,
    firing_rates: Dict[str, float],
    batch_size: int = 1,
    device: Union[str, torch.device] = "cpu",
) -> torch.Tensor:
    """
    Generate Poisson spike trains for a population of neurons.

    Each neuron generates spikes according to a homogeneous Poisson process
    with a firing rate determined by its cell type.

    Args:
        n_steps (int): Number of simulation time steps.
        dt (float): Simulation time step in milliseconds.
        num_neurons (int): Total number of neurons.
        cell_type_indices (torch.Tensor): Tensor of cell type indices for each neuron (num_neurons,).
        cell_type_names (list): List of cell type names corresponding to indices.
        firing_rates (Dict[str, float]): Dictionary mapping cell type names to firing rates in Hz.
        batch_size (int): Batch size for spike generation. Default: 1.
        device (Union[str, torch.device]): Device to generate spikes on. Default: "cpu".

    Returns:
        torch.Tensor: Boolean tensor of shape (batch_size, n_steps, num_neurons) with spike times.

    Example:
        >>> firing_rates = {"excitatory": 10.0, "inhibitory": 20.0}
        >>> cell_indices = torch.tensor([0, 0, 1, 1])
        >>> spikes = generate_poisson_spikes(
        ...     n_steps=1000,
        ...     dt=1.0,
        ...     num_neurons=4,
        ...     cell_type_indices=cell_indices,
        ...     cell_type_names=["excitatory", "inhibitory"],
        ...     firing_rates=firing_rates,
        ...     device="cuda"
        ... )
    """
    # Ensure cell_type_indices is on the correct device
    if not isinstance(cell_type_indices, torch.Tensor):
        cell_type_indices = torch.tensor(cell_type_indices, device=device)
    else:
        cell_type_indices = cell_type_indices.to(device)

    # Create a tensor of firing rates for each neuron
    neuron_rates = torch.zeros(num_neurons, device=device, dtype=torch.float32)
    for ct_idx, ct_name in enumerate(cell_type_names):
        mask = cell_type_indices == ct_idx
        neuron_rates[mask] = firing_rates[ct_name]

    # Compute spike probabilities: rate (Hz) * dt (ms) * 1e-3 (ms->s conversion)
    spike_probs = neuron_rates * dt * 1e-3  # Shape: (num_neurons,)

    # Expand to match output shape: (batch_size, n_steps, num_neurons)
    spike_probs = spike_probs[None, None, :].expand(batch_size, n_steps, num_neurons)

    # Generate random values and compare with spike probabilities
    random_vals = torch.rand(batch_size, n_steps, num_neurons, device=device)
    spikes = random_vals < spike_probs

    return spikes
