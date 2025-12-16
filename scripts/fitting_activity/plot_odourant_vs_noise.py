"""
Compare network activity driven by odourant patterns vs homogeneous Poisson noise.

This script loads a teacher network and generates spike trains using homogeneous
Poisson inputs with two different odour patterns. It then compares:
1. Network responses to different odours (odourant 1 vs odourant 2)
2. Network responses to same odour repeated (odourant 1 vs odourant 1 with different noise)

The comparison uses cross-correlation scatter plots of firing rates to visualize
how odour-specific the network response is.
"""

import argparse
import numpy as np
import torch
import toml
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
from network_inputs.unsupervised import HomogeneousPoissonSpikeDataLoader
from network_inputs.odourants import generate_odour_firing_rates
from network_simulators.conductance_based.simulator import ConductanceLIFNetwork
from parameter_loaders import TeacherActivityParams
from visualization.firing_statistics import plot_cross_correlation_scatter


def main(experiment_dir):
    """Generate comparison plots between odourant-driven and noise-driven activity.

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
        weights_FF=feedforward_weights,
        cell_type_indices_FF=input_source_indices,
        cell_params_FF=feedforward.get_cell_params(),
        synapse_params_FF=feedforward.get_synapse_params(),
        optimisable=None,
        use_tqdm=False,
    ).to(device)

    print("✓ Network initialized")

    # ========================================
    # Generate Responses: Odourant 1 vs 2
    # ========================================

    print("\nGenerating responses to different odours (Odourant 1 vs 2)...")

    # Create dataloaders for odourant 1 and odourant 2
    # Use only first batch element (batch_size=1)
    chunk_size = int(simulation.chunk_size)
    dt = simulation.dt

    dataloader_odour_1 = HomogeneousPoissonSpikeDataLoader(
        firing_rates=input_firing_rates_odour[0:1],  # Odourant 1
        chunk_size=chunk_size,
        dt=dt,
        batch_size=1,
        device=device,
    )

    if num_odours > 1:
        dataloader_odour_2 = HomogeneousPoissonSpikeDataLoader(
            firing_rates=input_firing_rates_odour[1:2],  # Odourant 2
            batch_size=1,
            chunk_size=chunk_size,
            dt=dt,
            device=device,
        )
    else:
        print("Warning: Only 1 odour pattern found, cannot compare odourant 1 vs 2")
        dataloader_odour_2 = None

    # Generate responses for a fixed number of chunks
    num_chunks = simulation.plot_size if hasattr(simulation, "plot_size") else 10

    def run_network_chunks(dataloader, num_chunks_to_run, description):
        """Run network for specified number of chunks and return spike data."""
        all_spikes = []

        with torch.inference_mode():
            for chunk_idx, (input_spikes_chunk, pattern_indices) in enumerate(
                tqdm(
                    dataloader,
                    total=num_chunks_to_run,
                    desc=description,
                    leave=False,
                )
            ):
                # Extract only first batch element: (1, chunk_size, n_input_neurons)
                input_spikes_chunk = input_spikes_chunk[0:1]

                # Initialize state variables on first chunk
                if chunk_idx == 0:
                    initial_v = None
                    initial_g = None
                    initial_g_FF = None

                # Run network
                (
                    output_spikes_chunk,
                    output_voltages,
                    output_currents,
                    output_currents_FF,
                    output_currents_leak,
                    output_conductances,
                    output_conductances_FF,
                ) = model(
                    input_spikes=input_spikes_chunk,
                    initial_v=initial_v,
                    initial_g=initial_g,
                    initial_g_FF=initial_g_FF,
                )

                # Store final states for next chunk
                initial_v = output_voltages[:, -1, :].clone()
                initial_g = output_conductances[:, -1, :, :, :].clone()
                initial_g_FF = output_conductances_FF[:, -1, :, :, :].clone()

                # Move to CPU and accumulate
                if device == "cuda":
                    all_spikes.append(output_spikes_chunk.cpu())
                else:
                    all_spikes.append(output_spikes_chunk)

                if chunk_idx + 1 >= num_chunks_to_run:
                    break

        # Concatenate chunks: (batch_size=1, total_time, n_neurons)
        spikes = torch.cat(all_spikes, dim=1)
        return spikes

    # Generate responses
    spikes_odour_1 = run_network_chunks(dataloader_odour_1, num_chunks, "Odourant 1")

    if dataloader_odour_2 is not None:
        spikes_odour_2 = run_network_chunks(
            dataloader_odour_2, num_chunks, "Odourant 2"
        )
    else:
        spikes_odour_2 = None

    # Generate second instance of odourant 1 for comparison (different random seed)
    print("\nGenerating second instance of Odourant 1 (with different noise)...")
    # Reset seed for different random instance
    torch.manual_seed(simulation.seed + 1 if simulation.seed is not None else 42)
    np.random.seed((simulation.seed + 1) if simulation.seed is not None else 42)

    dataloader_odour_1_repeat = HomogeneousPoissonSpikeDataLoader(
        firing_rates=input_firing_rates_odour[0:1],  # Odourant 1 again
        chunk_size=chunk_size,
        dt=dt,
        batch_size=1,
        device=device,
    )

    spikes_odour_1_repeat = run_network_chunks(
        dataloader_odour_1_repeat, num_chunks, "Odourant 1 (repeat)"
    )

    print("✓ Network simulations completed")

    # ====================================
    # Convert spikes to numpy and create comparison plots
    # ====================================

    print("\nGenerating comparison plots...")

    # Convert spike tensors to numpy
    spikes_odour_1_np = spikes_odour_1.cpu().numpy()
    if spikes_odour_2 is not None:
        spikes_odour_2_np = spikes_odour_2.cpu().numpy()
    spikes_odour_1_repeat_np = spikes_odour_1_repeat.cpu().numpy()

    output_dir = experiment_dir / "figures"

    # Plot 1: Odourant 1 vs Odourant 2 (if available)
    if spikes_odour_2_np is not None:
        print("  Creating odourant 1 vs 2 scatter plot...")
        fig_odour_comparison = plot_cross_correlation_scatter(
            spike_trains_trial1=spikes_odour_1_np,
            spike_trains_trial2=spikes_odour_2_np,
            window_size=10.0,  # 10 second windows
            dt=dt,
            title="Network Activity: Odourant 1 vs Odourant 2",
            x_label="Odourant 1 Firing Rate (Hz)",
            y_label="Odourant 2 Firing Rate (Hz)",
        )
    else:
        fig_odour_comparison = None

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
    dashboard_path = output_dir / "odourant_vs_noise_correlogram.png"
    dashboard_fig.savefig(dashboard_path, dpi=150, bbox_inches="tight")
    plt.close(dashboard_fig)
    print(f"✓ Saved: {dashboard_path}")

    # Close individual plots
    if fig_odour_comparison is not None:
        plt.close(fig_odour_comparison)
    if fig_noise_comparison is not None:
        plt.close(fig_noise_comparison)

    print(f"\n✓ All plots saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare network activity driven by odourants vs homogeneous Poisson noise"
    )
    parser.add_argument(
        "experiment_dir",
        type=Path,
        help="Path to experiment directory (must contain inputs/ and parameters.toml)",
    )

    args = parser.parse_args()

    main(experiment_dir=args.experiment_dir)
