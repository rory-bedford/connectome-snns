"""
Analyze network activity driven by odourant patterns vs homogeneous Poisson noise.

This script loads a teacher network and generates spike trains using homogeneous
Poisson inputs with two different odour patterns. It then generates:
1. Input firing rate histogram for odourant 1
2. Cross-correlation scatter plots comparing:
   - Network responses to different odours (odourant 1 vs odourant 2)
   - Network responses to same odour repeated (odourant 1 vs odourant 1 with different noise)
"""

import argparse
import numpy as np
import torch
import toml
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
from network_inputs.unsupervised import HomogeneousPoissonSpikeDataLoader
from network_inputs.odourants import (
    generate_odour_firing_rates,
    generate_baseline_firing_rates,
)
from network_simulators.conductance_based.simulator import ConductanceLIFNetwork
from parameter_loaders import TeacherActivityParams
from visualization.firing_statistics import plot_cross_correlation_scatter
from visualization.odours import plot_input_firing_rate_histogram


def main(experiment_dir):
    """Generate comparison plots for odourant-driven network activity.

    Args:
        experiment_dir (Path): Directory containing input/ with network_structure.npz
            and parameters.toml
    """

    # ======================================
    # Device Selection and Parameter Loading
    # ======================================

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load parameters from the experiment directory
    params_file = experiment_dir / "parameters.toml"
    with open(params_file, "r") as f:
        data = toml.load(f)
    params = TeacherActivityParams(**data)

    simulation = params.simulation
    recurrent = params.recurrent
    feedforward = params.feedforward

    # Set random seed for reproducibility
    if simulation.seed is not None:
        np.random.seed(simulation.seed)
        torch.manual_seed(simulation.seed)
        print(f"Using seed: {simulation.seed}")

    # ================================
    # Load Network Structure from Disk
    # ================================

    input_dir = experiment_dir / "inputs"
    network_structure = np.load(input_dir / "network_structure.npz")

    # Extract network components
    weights = network_structure["recurrent_weights"]
    feedforward_weights = network_structure["feedforward_weights"]
    cell_type_indices = network_structure["cell_type_indices"]
    input_source_indices = network_structure["feedforward_cell_type_indices"]
    assembly_ids = network_structure["assembly_ids"]

    print(f"✓ Loaded network with {len(cell_type_indices)} neurons")

    # =========================
    # Create Firing Rate Patterns
    # =========================

    # Calculate number of odour patterns from assemblies
    num_odours = len(np.unique(assembly_ids[assembly_ids >= 0]))

    # Generate odour-modulated firing rate patterns (one per assembly)
    input_firing_rates_odour = generate_odour_firing_rates(
        feedforward_weights=feedforward_weights,
        input_source_indices=input_source_indices,
        cell_type_indices=cell_type_indices,
        assembly_ids=assembly_ids,
        target_cell_type_idx=0,
        cell_type_names=feedforward.cell_types.names,
        odour_configs=params.get_odour_configs_dict(),
    )

    print(f"✓ Generated {num_odours} odour patterns")
    print(f"  Firing rate patterns shape: {input_firing_rates_odour.shape}")

    # ======================
    # Initialize LIF Network
    # ======================

    model = ConductanceLIFNetwork(
        dt=simulation.dt,
        weights=weights,
        cell_type_indices=cell_type_indices,
        cell_params=recurrent.get_cell_params(),
        synapse_params=recurrent.get_synapse_params(),
        surrgrad_scale=1.0,
        batch_size=1,
        weights_FF=feedforward_weights,
        cell_type_indices_FF=input_source_indices,
        cell_params_FF=feedforward.get_cell_params(),
        synapse_params_FF=feedforward.get_synapse_params(),
        optimisable=None,
        track_variables=True,  # Enable tracking for visualization
        use_tqdm=False,
    ).to(device)

    print("✓ Network initialized")

    # ========================================
    # Generate Baseline Firing Rates
    # ========================================

    print("\nGenerating baseline firing rates...")

    baseline_firing_rates = generate_baseline_firing_rates(
        n_input_neurons=feedforward_weights.shape[0],
        input_source_indices=input_source_indices,
        cell_type_names=feedforward.cell_types.names,
        odour_configs=params.get_odour_configs_dict(),
    )

    print(f"✓ Generated baseline firing rates: {baseline_firing_rates.shape}")

    # ========================================
    # Generate Responses: All patterns in parallel batch
    # ========================================

    print(
        "\nGenerating network responses for Odourant 1, Odourant 1 (repeat), and Baseline..."
    )

    # Create firing rate patterns: Odourant 1, Odourant 1 repeat (different seed), Baseline
    firing_rates_pattern_1 = input_firing_rates_odour[0:1]  # Odourant 1

    # Generate Odourant 1 with different seed for repeat
    torch.manual_seed(simulation.seed + 1 if simulation.seed is not None else 42)
    np.random.seed((simulation.seed + 1) if simulation.seed is not None else 42)
    firing_rates_pattern_2 = generate_odour_firing_rates(
        feedforward_weights=feedforward_weights,
        input_source_indices=input_source_indices,
        cell_type_indices=cell_type_indices,
        assembly_ids=assembly_ids,
        target_cell_type_idx=0,
        cell_type_names=feedforward.cell_types.names,
        odour_configs=params.get_odour_configs_dict(),
    )[0:1]  # Get odourant 1 again with different random seed

    firing_rates_pattern_3 = baseline_firing_rates

    # Stack all 3 patterns: (3, n_input_neurons)
    all_firing_rates = np.vstack(
        [firing_rates_pattern_1, firing_rates_pattern_2, firing_rates_pattern_3]
    )

    # Reset seed back to original for dataloader
    torch.manual_seed(simulation.seed)
    np.random.seed(simulation.seed)

    chunk_size = int(simulation.chunk_size)
    dt = simulation.dt

    dataloader_all = HomogeneousPoissonSpikeDataLoader(
        firing_rates=all_firing_rates,
        chunk_size=chunk_size,
        dt=dt,
        batch_size=1,
        device=device,
    )

    # Run network for all 3 patterns in parallel
    num_chunks = simulation.plot_size if hasattr(simulation, "plot_size") else 10
    all_spikes_chunks = []

    with torch.inference_mode():
        for chunk_idx, (input_spikes_chunk, pattern_indices) in enumerate(
            tqdm(
                dataloader_all,
                total=num_chunks,
                desc="Processing patterns",
                leave=False,
            )
        ):
            input_spikes_chunk = input_spikes_chunk.to(device)

            # 4D case: (batch=1, n_patterns=3, time, inputs)
            batch_size, n_patterns, time_steps, n_inputs = input_spikes_chunk.shape

            # Process each pattern separately
            chunk_outputs = []
            for pattern_idx in range(n_patterns):
                # Reset state for each independent pattern (at start of each pattern)
                if chunk_idx == 0:
                    model.reset_state()

                input_pattern = input_spikes_chunk[
                    :, pattern_idx, :, :
                ]  # (batch, time, inputs)

                # Run network
                outputs = model.forward(input_spikes=input_pattern)

                # Extract spikes from dict
                output_spikes_pattern = outputs["spikes"]

                # Store output
                chunk_outputs.append(output_spikes_pattern.cpu())

            # Stack patterns: (batch, n_patterns, time, neurons)
            chunk_stacked = torch.stack(chunk_outputs, dim=1)
            all_spikes_chunks.append(chunk_stacked)

            if chunk_idx + 1 >= num_chunks:
                break

    # Concatenate chunks: (batch=1, n_patterns=3, total_time, n_neurons)
    spikes_all_patterns = torch.cat(all_spikes_chunks, dim=2)

    # Extract individual patterns
    spikes_odour_1 = spikes_all_patterns[:, 0, :, :]  # Odourant 1
    spikes_odour_1_repeat = spikes_all_patterns[:, 1, :, :]  # Odourant 1 (repeat)
    spikes_baseline = spikes_all_patterns[:, 2, :, :]  # Baseline

    print("✓ Network simulations completed")

    # ====================================
    # Plot Input Firing Rate Histogram
    # ====================================

    print("\nGenerating input firing rate histogram...")

    # Plot histogram of odourant 1 firing rates
    fig_input_histogram = plot_input_firing_rate_histogram(
        firing_rates=input_firing_rates_odour[0:1],  # Odourant 1 only
        bins=30,
    )

    # ====================================
    # Convert spikes to numpy and create comparison plots
    # ====================================

    print("\nGenerating comparison plots...")

    # Convert spike tensors to numpy
    spikes_odour_1_np = spikes_odour_1.cpu().numpy()
    spikes_baseline_np = spikes_baseline.cpu().numpy()
    spikes_odour_1_repeat_np = spikes_odour_1_repeat.cpu().numpy()

    output_dir = experiment_dir / "figures"

    # Plot 1: Odourant 1 vs Baseline
    print("  Creating odourant 1 vs baseline scatter plot...")
    fig_odour_comparison = plot_cross_correlation_scatter(
        spike_trains_trial1=spikes_odour_1_np,
        spike_trains_trial2=spikes_baseline_np,
        window_size=10.0,  # 10 second windows
        dt=dt,
        title="Network Activity: Odourant 1 vs Baseline",
        x_label="Odourant 1 Firing Rate (Hz)",
        y_label="Baseline Firing Rate (Hz)",
    )

    # Plot 2: Odourant 1 vs Odourant 1 (repeat with different noise)
    print("  Creating odourant 1 vs 1 scatter plot (noise comparison)...")
    fig_noise_comparison = plot_cross_correlation_scatter(
        spike_trains_trial1=spikes_odour_1_np,
        spike_trains_trial2=spikes_odour_1_repeat_np,
        window_size=10.0,  # 10 second windows
        dt=dt,
        title="Network Activity: Same Odourant, Different Noise",
        x_label="Odourant 1 Firing Rate (Hz)",
        y_label="Odourant 1 Firing Rate (noise variant) (Hz)",
    )

    # ====================================
    # Create Dashboard with both plots
    # ====================================

    print("  Creating combined dashboard...")

    # Create a dashboard figure with both correlograms side-by-side
    dashboard_fig = plt.figure(figsize=(16, 6))

    # Helper function to copy axes content
    def copy_axes_to_subplot(source_ax, subplot_pos):
        """Copy content from source axis to new subplot."""
        new_ax = dashboard_fig.add_subplot(subplot_pos)

        # Copy scatter data
        for collection in source_ax.collections:
            offsets = collection.get_offsets()
            colors = collection.get_facecolors()
            if len(colors) > 0:
                new_ax.scatter(
                    offsets[:, 0], offsets[:, 1], color=colors[0], alpha=0.3, s=10
                )
            else:
                new_ax.scatter(offsets[:, 0], offsets[:, 1], alpha=0.3, s=10)

        # Copy lines (like diagonal reference lines)
        for line in source_ax.get_lines():
            xdata = line.get_xdata()
            ydata = line.get_ydata()
            new_ax.plot(
                xdata,
                ydata,
                linestyle=line.get_linestyle(),
                color=line.get_color(),
                alpha=line.get_alpha(),
            )

        # Copy labels and limits
        new_ax.set_xlim(source_ax.get_xlim())
        new_ax.set_ylim(source_ax.get_ylim())
        new_ax.set_xlabel(source_ax.get_xlabel())
        new_ax.set_ylabel(source_ax.get_ylabel())
        new_ax.set_title(source_ax.get_title())
        new_ax.grid(True, alpha=0.3)

    # Add first plot to dashboard
    if fig_odour_comparison is not None:
        for ax in fig_odour_comparison.axes:
            copy_axes_to_subplot(ax, 121)
            break

    # Add second plot to dashboard
    if fig_noise_comparison is not None:
        for ax in fig_noise_comparison.axes:
            copy_axes_to_subplot(ax, 122)
            break

    # Save dashboard
    dashboard_path = output_dir / "odourant_vs_baseline_correlogram.png"
    dashboard_fig.savefig(dashboard_path, dpi=150, bbox_inches="tight")
    plt.close(dashboard_fig)
    print(f"✓ Saved: {dashboard_path}")

    # Close individual plots
    if fig_odour_comparison is not None:
        plt.close(fig_odour_comparison)
    if fig_noise_comparison is not None:
        plt.close(fig_noise_comparison)
    if fig_input_histogram is not None:
        fig_input_histogram.savefig(
            output_dir / "input_firing_rates_histogram.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig_input_histogram)
        print(f"✓ Saved: {output_dir / 'input_firing_rates_histogram.png'}")

    print(f"\n✓ All plots saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze network activity driven by odourants"
    )
    parser.add_argument(
        "experiment_dir",
        type=Path,
        help="Path to experiment directory (must contain inputs/ and parameters.toml)",
    )

    args = parser.parse_args()

    main(experiment_dir=args.experiment_dir)
