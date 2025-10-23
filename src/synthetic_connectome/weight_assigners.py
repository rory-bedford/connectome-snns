"""Assign continuous weights to graph edges.

This module provides functions to assign continuous weight values to boolean
adjacency matrices while preserving Dale's law (excitatory/inhibitory signs).
Parameters are specified using matrices for different cell type combinations.

Typical workflow:
    >>> connectivity_graph, cell_indices = sparse_graph_generator(...)
    >>> w_mu_matrix = [[1.0, 1.2], [2.4, 1.6]]  # 2x2 for 2 cell types
    >>> weights = assign_weights_lognormal(connectivity_graph, cell_indices, cell_indices,
    ...                                   [1, -1], w_mu_matrix, w_sigma_matrix)
"""

import numpy as np
from numpy.typing import NDArray

# Type aliases for clarity
IntArray = NDArray[np.int_]
BoolArray = NDArray[np.bool_]
FloatArray = NDArray[np.float64]


def assign_weights_lognormal(
    connectivity_graph: BoolArray,
    source_cell_indices: IntArray,
    target_cell_indices: IntArray,
    cell_type_signs: list or IntArray,
    w_mu_matrix: FloatArray,
    w_sigma_matrix: FloatArray,
    parameter_space: str = "log",
) -> FloatArray:
    """
    Assign log-normal distributed weights to a boolean adjacency matrix.

    Args:
        connectivity_graph (BoolArray): Boolean adjacency matrix of shape (n_source, n_target).
        source_cell_indices (IntArray): Cell type indices for source neurons (0, 1, 2, ...).
        target_cell_indices (IntArray): Cell type indices for target neurons (0, 1, 2, ...).
        cell_type_signs (list or ndarray): Signs for each SOURCE cell type (+1/-1). This is used to
            determine if connections are excitatory or inhibitory based on Dale's law.
            Length must equal the number of unique source cell types. Can be a list or numpy array.
        w_mu_matrix (FloatArray): Matrix of shape (n_source_types, n_target_types) for parameters:
            - If parameter_space="log" (default): mu parameters of log-normal distribution (in log-space)
            - If parameter_space="linear": desired mean of resulting distribution (in linear space)
        w_sigma_matrix (FloatArray): Matrix of shape (n_source_types, n_target_types) for parameters:
            - If parameter_space="log" (default): sigma parameters of log-normal distribution (in log-space)
            - If parameter_space="linear": desired variance of resulting distribution (in linear space)
        parameter_space (str, optional): Specifies how to interpret the input parameters:
            - "log": Parameters are in log space (mu, sigma)
            - "linear": Parameters are in linear space (mean, variance)
            Defaults to "log".

    Returns:
        FloatArray: Weighted adjacency matrix with log-normal magnitudes and appropriate signs.
    """
    n_source, n_target = connectivity_graph.shape
    n_source_types = len(cell_type_signs)

    # Input validation
    assert source_cell_indices.shape[0] == n_source, (
        "Mismatch between source indices and graph rows."
    )
    assert target_cell_indices.shape[0] == n_target, (
        "Mismatch between target indices and graph columns."
    )

    # Get the number of unique target types
    n_target_types = max(target_cell_indices) + 1

    # Allow rectangular matrices for weights between different numbers of source and target types
    assert w_mu_matrix.shape == (n_source_types, n_target_types), (
        f"w_mu_matrix must be {n_source_types}x{n_target_types} for {n_source_types} source types and {n_target_types} target types."
    )
    assert w_sigma_matrix.shape == (n_source_types, n_target_types), (
        f"w_sigma_matrix must be {n_source_types}x{n_target_types} for {n_source_types} source types and {n_target_types} target types."
    )
    assert parameter_space.lower() in ["log", "linear"], (
        'parameter_space must be either "log" or "linear"'
    )

    # Convert from desired mean/variance to log-normal parameters if needed
    if parameter_space.lower() == "linear":
        # For lognormal distribution:
        # If X ~ LogNormal(μ, σ²), then:
        # E[X] = exp(μ + σ²/2)
        # Var[X] = [exp(σ²) - 1] * exp(2μ + σ²)

        # Given desired mean (m) and variance (v), we solve for μ and σ:
        # σ² = log(1 + v/m²)
        # μ = log(m) - σ²/2

        mean = w_mu_matrix.copy()
        var = w_sigma_matrix.copy()

        # Avoid division by zero or negative values
        mean_eps = np.maximum(mean, np.finfo(float).eps)
        var_ratio = np.maximum(var / (mean_eps * mean_eps), 0)

        log_sigma_squared = np.log(1 + var_ratio)
        log_mu = np.log(mean_eps) - log_sigma_squared / 2

        w_mu_matrix = log_mu
        w_sigma_matrix = np.sqrt(log_sigma_squared)

    # Extract parameters for specific connections
    # Convert cell_type_signs to numpy array if it's not already one
    cell_type_signs_arr = np.array(cell_type_signs)
    source_signs = cell_type_signs_arr[source_cell_indices]
    mu = w_mu_matrix[source_cell_indices][:, target_cell_indices]
    sigma = w_sigma_matrix[source_cell_indices][:, target_cell_indices]

    weight_magnitudes = np.random.lognormal(mu, sigma)
    weights = weight_magnitudes * source_signs[:, None]

    return weights * connectivity_graph


def assign_weights_gamma(
    connectivity_graph: BoolArray,
    source_cell_indices: IntArray,
    target_cell_indices: IntArray,
    cell_type_signs: list or IntArray,
    shape_matrix: FloatArray,
    scale_matrix: FloatArray,
) -> FloatArray:
    """
    Assign gamma-distributed weights to a boolean adjacency matrix.

    Args:
        connectivity_graph (BoolArray): Boolean adjacency matrix of shape (n_source, n_target).
        source_cell_indices (IntArray): Cell type indices for source neurons (0, 1, 2, ...).
        target_cell_indices (IntArray): Cell type indices for target neurons (0, 1, 2, ...).
        cell_type_signs (list or ndarray): Signs for each SOURCE cell type (+1/-1). This is used to
            determine if connections are excitatory or inhibitory based on Dale's law.
            Length must equal the number of unique source cell types. Can be a list or numpy array.
        shape_matrix (FloatArray): Matrix of shape (n_source_types, n_target_types) of gamma shape parameters (k).
        scale_matrix (FloatArray): Matrix of shape (n_source_types, n_target_types) of gamma scale parameters (θ).

    Returns:
        FloatArray: Weighted adjacency matrix with gamma-distributed magnitudes and appropriate signs.
    """
    n_source, n_target = connectivity_graph.shape
    n_source_types = len(cell_type_signs)

    assert source_cell_indices.shape[0] == n_source, (
        "Mismatch between source indices and graph rows."
    )
    assert target_cell_indices.shape[0] == n_target, (
        "Mismatch between target indices and graph columns."
    )

    # Get the number of unique target types
    n_target_types = max(target_cell_indices) + 1

    assert shape_matrix.shape == (n_source_types, n_target_types), (
        f"shape_matrix must be {n_source_types}x{n_target_types} for {n_source_types} source types and {n_target_types} target types."
    )
    assert scale_matrix.shape == (n_source_types, n_target_types), (
        f"scale_matrix must be {n_source_types}x{n_target_types} for {n_source_types} source types and {n_target_types} target types."
    )

    # Convert cell_type_signs to numpy array if it's not already one
    cell_type_signs_arr = np.array(cell_type_signs)
    source_signs = cell_type_signs_arr[source_cell_indices]
    shape = shape_matrix[source_cell_indices][:, target_cell_indices]
    scale = scale_matrix[source_cell_indices][:, target_cell_indices]

    weight_magnitudes = np.random.gamma(shape, scale)
    weights = weight_magnitudes * source_signs[:, None]

    return weights * connectivity_graph


def assign_weights_uniform(
    connectivity_graph: BoolArray,
    source_cell_indices: IntArray,
    target_cell_indices: IntArray,
    cell_type_signs: list or IntArray,
    low_matrix: FloatArray,
    high_matrix: FloatArray,
) -> FloatArray:
    """
    Assign uniformly distributed weights to a boolean adjacency matrix.

    Args:
        connectivity_graph (BoolArray): Boolean adjacency matrix of shape (n_source, n_target).
        source_cell_indices (IntArray): Cell type indices for source neurons (0, 1, 2, ...).
        target_cell_indices (IntArray): Cell type indices for target neurons (0, 1, 2, ...).
        cell_type_signs (list or ndarray): Signs for each SOURCE cell type (+1/-1). This is used to
            determine if connections are excitatory or inhibitory based on Dale's law.
            Length must equal the number of unique source cell types. Can be a list or numpy array.
        low_matrix (FloatArray): Matrix of shape (n_source_types, n_target_types) of uniform distribution lower bounds.
        high_matrix (FloatArray): Matrix of shape (n_source_types, n_target_types) of uniform distribution upper bounds.

    Returns:
        FloatArray: Weighted adjacency matrix with uniformly distributed magnitudes and appropriate signs.
    """
    n_source, n_target = connectivity_graph.shape
    n_source_types = len(cell_type_signs)

    assert source_cell_indices.shape[0] == n_source, (
        "Mismatch between source indices and graph rows."
    )
    assert target_cell_indices.shape[0] == n_target, (
        "Mismatch between target indices and graph columns."
    )

    # Get the number of unique target types
    n_target_types = max(target_cell_indices) + 1

    assert low_matrix.shape == (n_source_types, n_target_types), (
        f"low_matrix must be {n_source_types}x{n_target_types} for {n_source_types} source types and {n_target_types} target types."
    )
    assert high_matrix.shape == (n_source_types, n_target_types), (
        f"high_matrix must be {n_source_types}x{n_target_types} for {n_source_types} source types and {n_target_types} target types."
    )

    # Convert cell_type_signs to numpy array if it's not already one
    cell_type_signs_arr = np.array(cell_type_signs)
    source_signs = cell_type_signs_arr[source_cell_indices]
    low = low_matrix[source_cell_indices][:, target_cell_indices]
    high = high_matrix[source_cell_indices][:, target_cell_indices]

    weight_magnitudes = np.random.uniform(low, high)
    weights = weight_magnitudes * source_signs[:, None]

    return weights * connectivity_graph
