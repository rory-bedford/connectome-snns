"""Analysis functions for membrane voltage statistics."""

import torch


def compute_membrane_potential_by_cell_type(
    voltages: torch.Tensor,
    cell_type_indices: torch.Tensor,
) -> dict[int, dict[str, float]]:
    """
    Compute mean and standard deviation of membrane potential statistics by cell type.

    For each neuron, computes the mean and standard deviation of its membrane potential
    across time. Then, for each cell type, computes the mean and standard deviation
    of these per-neuron statistics.

    Args:
        voltages (torch.Tensor): Membrane voltages of shape (batch_size, n_steps, n_neurons).
        cell_type_indices (torch.Tensor): Cell type indices of shape (n_neurons,).
            Each value indicates the cell type index for that neuron.

    Returns:
        dict[int, dict[str, float]]: Dictionary mapping cell type index to a dict with:
            - "mean_of_means": Mean of per-neuron mean voltages for this cell type
            - "std_of_means": Standard deviation of per-neuron mean voltages
            - "mean_of_stds": Mean of per-neuron voltage standard deviations
            - "std_of_stds": Standard deviation of per-neuron voltage standard deviations
    """
    batch_size, n_steps, n_neurons = voltages.shape

    # Compute mean voltage per neuron across time and batches
    # Shape: (n_neurons,)
    mean_voltage_per_neuron = voltages.mean(dim=(0, 1))

    # Compute std voltage per neuron across time and batches
    # Shape: (n_neurons,)
    std_voltage_per_neuron = voltages.std(dim=(0, 1), unbiased=True)

    # Get unique cell types
    unique_cell_types = torch.unique(cell_type_indices)

    # Compute statistics by cell type
    stats_by_type = {}
    for cell_type in unique_cell_types.tolist():
        # Get indices of neurons belonging to this cell type
        mask = cell_type_indices == cell_type

        # Extract statistics for neurons of this cell type
        cell_type_means = mean_voltage_per_neuron[mask]
        cell_type_stds = std_voltage_per_neuron[mask]

        # Compute mean and std of means
        mean_of_means = cell_type_means.mean().item()
        std_of_means = (
            cell_type_means.std(unbiased=True).item()
            if len(cell_type_means) > 1
            else 0.0
        )

        # Compute mean and std of stds
        mean_of_stds = cell_type_stds.mean().item()
        std_of_stds = (
            cell_type_stds.std(unbiased=True).item() if len(cell_type_stds) > 1 else 0.0
        )

        stats_by_type[cell_type] = {
            "mean_of_means": mean_of_means,
            "std_of_means": std_of_means,
            "mean_of_stds": mean_of_stds,
            "std_of_stds": std_of_stds,
        }

    return stats_by_type
