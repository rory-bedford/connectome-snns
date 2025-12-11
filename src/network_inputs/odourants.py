"""
Odour-based firing rate pattern generation.

This module provides functions for generating structured firing rate patterns
that represent odour stimuli in olfactory network models. Patterns are based on
assembly connectivity, where input neurons are modulated according to their
connection strength to specific assemblies in the recurrent network.

These firing rate patterns can be used as input to Poisson spike generators or
other stochastic input generation methods.
"""

import torch
import numpy as np
from typing import Union


def generate_odour_firing_rates(
    feedforward_weights: Union[np.ndarray, torch.Tensor],
    input_source_indices: Union[np.ndarray, torch.Tensor],
    cell_type_indices: Union[np.ndarray, torch.Tensor],
    assembly_ids: Union[np.ndarray, torch.Tensor],
    target_cell_type_idx: int,
    cell_type_names: list[str],
    odour_configs: dict[str, dict],
) -> np.ndarray:
    """
    Generate firing rate patterns for odour-modulated Poisson inputs based on assembly connectivity.

    Creates one firing rate pattern per assembly, where input neurons are modulated based on
    their summed connection strength to each assembly. Connection strengths are computed only
    for the target cell type in the recurrent network - this allows targeting specific cell
    types (e.g., only excitatory cells) when determining which input neurons to modulate.

    For each assembly pattern, input neurons with the strongest connections to the target
    cell type are modulated up, the weakest are modulated down, and the rest remain at baseline.

    Args:
        feedforward_weights: Feedforward weight matrix of shape (n_input_neurons, n_recurrent_neurons).
        input_source_indices: Array indicating cell type for each input neuron.
        cell_type_indices: Array indicating cell type for each recurrent neuron.
        assembly_ids: Array of assembly IDs for each recurrent neuron (0-indexed, -1 for unassigned).
        target_cell_type_idx: Index of the cell type in the recurrent network to target when
            computing connection strengths (e.g., 0 for excitatory cells). This determines which
            recurrent neurons' incoming weights are considered when selecting input neurons to modulate.
        cell_type_names: List of cell type names (e.g., ["mitral"]).
        odour_configs: Dict mapping cell type names to odour config dicts.
            Each config dict should have keys: 'baseline_rate', 'modulation_rate', 'modulation_fraction'.

    Returns:
        Firing rates array of shape (n_assemblies, n_input_neurons) in Hz.
        One pattern per assembly.

    Example:
        >>> odour_config = {
        ...     "baseline_rate": 6.0,
        ...     "modulation_rate": 1.0,
        ...     "modulation_fraction": 0.1
        ... }
        >>> weights = np.random.randn(1000, 5000)
        >>> cell_type_indices = np.zeros(5000, dtype=int)  # All excitatory
        >>> assembly_ids = np.random.randint(-1, 20, 5000)
        >>> firing_rates = generate_odour_firing_rates(
        ...     feedforward_weights=weights,
        ...     input_source_indices=np.zeros(1000, dtype=int),
        ...     cell_type_indices=cell_type_indices,
        ...     assembly_ids=assembly_ids,
        ...     target_cell_type_idx=0,  # Target excitatory cells
        ...     cell_type_names=["mitral"],
        ...     odour_configs={"mitral": odour_config},
        ... )
        >>> firing_rates.shape
        (20, 1000)  # 20 assemblies
    """
    # Convert to numpy if needed
    if isinstance(feedforward_weights, torch.Tensor):
        feedforward_weights = feedforward_weights.cpu().numpy()
    if isinstance(input_source_indices, torch.Tensor):
        input_source_indices = input_source_indices.cpu().numpy()
    if isinstance(cell_type_indices, torch.Tensor):
        cell_type_indices = cell_type_indices.cpu().numpy()
    if isinstance(assembly_ids, torch.Tensor):
        assembly_ids = assembly_ids.cpu().numpy()

    n_input_neurons = feedforward_weights.shape[0]

    # Get unique assemblies (excluding -1 for unassigned neurons)
    unique_assemblies = np.unique(assembly_ids[assembly_ids >= 0])
    n_patterns = len(unique_assemblies)

    if n_patterns == 0:
        raise ValueError("No assemblies found in assembly_ids")

    # Initialize firing rates: (n_patterns, n_input_neurons)
    firing_rates = np.zeros((n_patterns, n_input_neurons))

    # Create mask for target cell type
    target_cell_mask = cell_type_indices == target_cell_type_idx

    # For each assembly, compute sum of input weights from each input neuron
    # Only considering connections to the target cell type
    # Shape: (n_assemblies, n_input_neurons)
    assembly_input_weights = np.zeros((n_patterns, n_input_neurons))

    for assembly_idx, assembly_id in enumerate(unique_assemblies):
        # Get mask for neurons in this assembly AND of the target cell type
        assembly_and_target_mask = (assembly_ids == assembly_id) & target_cell_mask

        # Sum weights from each input neuron to target-type neurons in this assembly
        # feedforward_weights: (n_input, n_recurrent)
        # assembly_and_target_mask: (n_recurrent,)
        assembly_input_weights[assembly_idx, :] = feedforward_weights[
            :, assembly_and_target_mask
        ].sum(axis=1)

    # Generate modulated firing rates for each cell type
    for ct_idx, ct_name in enumerate(cell_type_names):
        # Get mask for this cell type
        mask = input_source_indices == ct_idx

        if ct_name not in odour_configs:
            raise ValueError(f"No odour configuration found for cell type '{ct_name}'")

        odour_config = odour_configs[ct_name]

        # Get modulation parameters from dict
        baseline_rate = odour_config["baseline_rate"]
        modulation_rate = odour_config["modulation_rate"]
        modulation_fraction = odour_config["modulation_fraction"]

        up_rate = baseline_rate + modulation_rate

        # Number of neurons of this cell type
        n_neurons_this_type = np.sum(mask)
        n_modulated_up = int(n_neurons_this_type * modulation_fraction)

        # Get indices of neurons of this cell type
        neuron_indices = np.where(mask)[0]

        # For each pattern (assembly), select neurons based on connection strength
        for pattern_idx in range(n_patterns):
            # Get connection strengths for this cell type to this assembly
            connection_strengths = assembly_input_weights[pattern_idx, neuron_indices]

            # Sort neuron indices by connection strength (descending)
            sorted_indices = neuron_indices[np.argsort(-connection_strengths)]

            # Assign modulated up neurons (strongest connections)
            up_indices = sorted_indices[:n_modulated_up]
            firing_rates[pattern_idx, up_indices] = up_rate

            # Compute down-modulation rate to keep population average at baseline
            # Total rate = n_up * up_rate + n_down * down_rate
            # Want: Total rate = n_total * baseline_rate
            # So: n_down * down_rate = n_total * baseline_rate - n_up * up_rate
            # down_rate = (n_total * baseline_rate - n_up * up_rate) / n_down
            n_down = n_neurons_this_type - n_modulated_up
            if n_down > 0:
                down_rate = (
                    n_neurons_this_type * baseline_rate - n_modulated_up * up_rate
                ) / n_down
            else:
                # Edge case: all neurons modulated up
                down_rate = baseline_rate

            # Assign down-modulated rate to all other neurons
            down_indices = sorted_indices[n_modulated_up:]
            firing_rates[pattern_idx, down_indices] = down_rate

    return firing_rates


def generate_baseline_firing_rates(
    n_input_neurons: int,
    input_source_indices: Union[np.ndarray, torch.Tensor],
    cell_type_names: list[str],
    odour_configs: dict[str, dict],
) -> np.ndarray:
    """
    Generate constant baseline firing rates (no modulation).

    Creates a single firing rate pattern where all cells fire at their baseline rate.
    This serves as a control condition with no odour-specific modulation.

    Args:
        n_input_neurons: Total number of input neurons.
        input_source_indices: Array indicating cell type for each input neuron.
        cell_type_names: List of cell type names (e.g., ["mitral"]).
        odour_configs: Dict mapping cell type names to odour config dicts.
            Each config dict should have key: 'baseline_rate'.

    Returns:
        Firing rates array of shape (1, n_input_neurons) in Hz.
        Single pattern with all neurons at baseline rate.

    Example:
        >>> odour_config = {
        ...     "baseline_rate": 6.0,
        ...     "modulation_rate": 1.0,
        ...     "modulation_fraction": 0.1
        ... }
        >>> firing_rates = generate_baseline_firing_rates(
        ...     n_input_neurons=1000,
        ...     input_source_indices=np.zeros(1000, dtype=int),
        ...     cell_type_names=["mitral"],
        ...     odour_configs={"mitral": odour_config}
        ... )
        >>> firing_rates.shape
        (1, 1000)
    """
    # Convert to numpy if needed
    if isinstance(input_source_indices, torch.Tensor):
        input_source_indices = input_source_indices.cpu().numpy()

    # Initialize firing rates: single pattern (1, n_input_neurons)
    firing_rates = np.zeros((1, n_input_neurons))

    for ct_idx, ct_name in enumerate(cell_type_names):
        # Get mask for this cell type
        mask = input_source_indices == ct_idx

        if ct_name not in odour_configs:
            raise ValueError(f"No odour configuration found for cell type '{ct_name}'")

        odour_config = odour_configs[ct_name]

        # Get baseline rate only (no modulation)
        baseline_rate = odour_config["baseline_rate"]

        # Set all neurons of this type to baseline
        firing_rates[0, mask] = baseline_rate

    return firing_rates
