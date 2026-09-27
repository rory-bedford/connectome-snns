"""
Script contains functions to perturb the connectome weight matrices
to test the robustness of student training.
"""

import numpy as np
from numpy.typing import NDArray

# Type aliases for clarity
IntArray = NDArray[np.int_]
FloatArray = NDArray[np.float64]


def perturb_weights_scaling_factor(
    weights: FloatArray,
    feedforward_weights: FloatArray,
    cell_type_indices: IntArray,
    feedforward_cell_type_indices: IntArray,
    variance: float,
    optimisable: str = "scaling_factors",
):
    """
    Perturb weights by cell type-specific scaling factors drawn from a log-normal distribution.

    Only perturbs the weights corresponding to scaling factors that will be optimized.

    Args:
        weights: Recurrent weight matrix of shape (N, N).
        feedforward_weights: Feedforward weight matrix of shape (N_in, N).
        cell_type_indices: Array of shape (N,) indicating cell type index for each neuron.
        feedforward_cell_type_indices: Array of shape (N_in,) indicating cell type index for each input neuron.
        variance: Variance of the log-normal distribution used for scaling.
        optimisable: Optimization mode - one of "scaling_factors", "scaling_factors_recurrent",
            or "scaling_factors_feedforward". Determines which weights to perturb.

    Returns:
        perturbed_weights: Perturbed recurrent weight matrix (or original if not optimizing recurrent).
        perturbed_feedforward_weights: Perturbed feedforward weight matrix (or original if not optimizing feedforward).
        target_scaling_factors: Reciprocal of applied scaling factors, or None if not optimizing recurrent.
        target_feedforward_scaling_factors: Reciprocal of applied scaling factors, or None if not optimizing feedforward.
    """
    # Get unique cell types
    unique_cell_types = np.unique(cell_type_indices)
    unique_ff_cell_types = np.unique(feedforward_cell_type_indices)
    n_recurrent_types = len(unique_cell_types)
    n_ff_types = len(unique_ff_cell_types)

    # Sample scaling factors per pre-post cell type pair from log-normal with mean=1
    # For log-normal: mean = exp(mu + sigma^2/2), we want mean=1, so mu = -sigma^2/2
    sigma = np.sqrt(variance)
    mu = -(sigma**2) / 2.0

    # Determine which weights to perturb based on optimization mode
    perturb_recurrent = optimisable in ["scaling_factors", "scaling_factors_recurrent"]
    perturb_feedforward = optimisable in [
        "scaling_factors",
        "scaling_factors_feedforward",
    ]

    # Initialize outputs
    target_scaling_factors = None
    target_feedforward_scaling_factors = None

    # Perturb recurrent weights if optimizing recurrent scaling factors
    if perturb_recurrent:
        # Draw scaling factors for recurrent connections: n_types x n_types matrix
        # recurrent_scalings[i, j] = scaling for connections from type i to type j
        recurrent_scalings = np.random.lognormal(
            mean=mu, sigma=sigma, size=(n_recurrent_types, n_recurrent_types)
        )

        # Apply scaling to recurrent weights
        # For each connection weights[i, j], scale by recurrent_scalings[type_i, type_j]
        perturbed_weights = weights.copy()
        for pre_idx in range(len(cell_type_indices)):
            for post_idx in range(len(cell_type_indices)):
                pre_type = cell_type_indices[pre_idx]
                post_type = cell_type_indices[post_idx]
                perturbed_weights[pre_idx, post_idx] *= recurrent_scalings[
                    pre_type, post_type
                ]

        # Target scaling factors are the reciprocals (what the student needs to learn)
        target_scaling_factors = 1.0 / recurrent_scalings
    else:
        # Don't perturb recurrent weights
        perturbed_weights = weights.copy()

    # Perturb feedforward weights if optimizing feedforward scaling factors
    if perturb_feedforward:
        # Draw scaling factors for feedforward connections: n_ff_types x n_recurrent_types matrix
        # ff_scalings[i, j] = scaling for connections from input type i to recurrent type j
        ff_scalings = np.random.lognormal(
            mean=mu, sigma=sigma, size=(n_ff_types, n_recurrent_types)
        )

        # Apply scaling to feedforward weights
        # For each connection feedforward_weights[i, j], scale by ff_scalings[type_i, type_j]
        perturbed_feedforward_weights = feedforward_weights.copy()
        for pre_idx in range(len(feedforward_cell_type_indices)):
            for post_idx in range(len(cell_type_indices)):
                pre_type = feedforward_cell_type_indices[pre_idx]
                post_type = cell_type_indices[post_idx]
                perturbed_feedforward_weights[pre_idx, post_idx] *= ff_scalings[
                    pre_type, post_type
                ]

        # Target scaling factors are the reciprocals (what the student needs to learn)
        target_feedforward_scaling_factors = 1.0 / ff_scalings
    else:
        # Don't perturb feedforward weights
        perturbed_feedforward_weights = feedforward_weights.copy()

    return (
        perturbed_weights,
        perturbed_feedforward_weights,
        target_scaling_factors,
        target_feedforward_scaling_factors,
    )
