"""
Compare trained student network activity against teacher network activity.

This script loads a trained student network from a checkpoint and generates
spike trains using the same input patterns as the teacher network. It then
compares the student's output spike trains with the teacher's target spike trains
using cross-correlation analysis and visualization to evaluate training success.

The script generates comparison plots including cross-correlation histograms
and scatter plots to quantify how well the student matches the teacher.
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
from parameter_loaders import StudentTrainingParams, TeacherActivityParams
from visualization.firing_statistics import (
    plot_cross_correlation_scatter,
)
from visualization.neuronal_dynamics import plot_spike_trains
from visualization.dashboards import (
    create_connectivity_dashboard,
    create_activity_dashboard,
)


def main(experiment_dir, output_dir, teacher_params_file):
    """Generate comparison plots between student and teacher network activity.

    Args:
        experiment_dir (Path): Directory containing training run outputs
            Expected structure:
                - input/ : Contains teacher network_structure.npz and spike_data.zarr
                - checkpoints/ : Contains checkpoint_best.pt
                - parameters.toml : Training parameters
        output_dir (Path): Directory where comparison plots will be saved
        teacher_params_file (Path): Path to generate-teacher-activity.toml file
            containing odour configurations
    """

    # ======================================
    # Device Selection and Parameter Loading
    # ======================================

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load parameters from the training run
    params_file = experiment_dir / "parameters.toml"
    with open(params_file, "r") as f:
        data = toml.load(f)
    params = StudentTrainingParams(**data)

    # Load teacher parameters for odour configurations
    with open(teacher_params_file, "r") as f:
        teacher_data = toml.load(f)
    teacher_params = TeacherActivityParams(**teacher_data)

    simulation = params.simulation
    recurrent = params.recurrent
    feedforward = params.feedforward

    # Set random seed for reproducibility
    if simulation.seed is not None:
        np.random.seed(simulation.seed)
        torch.manual_seed(simulation.seed)
        print(f"Using seed: {simulation.seed}")

    # ================================
    # Load Network Structures
    # ================================

    # Load ORIGINAL (unperturbed) network for target model
    input_dir = experiment_dir / "inputs"
    original_network_structure = np.load(input_dir / "network_structure.npz")

    original_weights = original_network_structure["recurrent_weights"]
    original_feedforward_weights = original_network_structure["feedforward_weights"]
    cell_type_indices = original_network_structure["cell_type_indices"]
    feedforward_cell_type_indices = original_network_structure[
        "feedforward_cell_type_indices"
    ]
    assembly_ids = original_network_structure["assembly_ids"]

    # Load TRAINED network for student model (from final_state)
    final_state_dir = experiment_dir / "final_state"
    trained_network_structure = np.load(final_state_dir / "network_structure.npz")

    trained_weights = trained_network_structure["recurrent_weights"]
    trained_feedforward_weights = trained_network_structure["feedforward_weights"]

    # Load assembly structure
    n_assemblies = len(np.unique(assembly_ids[assembly_ids >= 0]))
    print(
        f"✓ Loaded original (target) network with {len(cell_type_indices)} neurons ({n_assemblies} assemblies)"
    )
    print(
        "✓ Loaded trained (final) network weights with learned scaling factors applied"
    )

    # ========================================
    # Generate Odour 1 Firing Rates (repeated)
    # ========================================

    print("\nGenerating odour 1 firing rates...")

    # Generate odour-modulated firing rate patterns (one per assembly)
    input_firing_rates_odour = generate_odour_firing_rates(
        feedforward_weights=original_feedforward_weights,
        input_source_indices=feedforward_cell_type_indices,
        cell_type_indices=cell_type_indices,
        assembly_ids=assembly_ids,
        target_cell_type_idx=0,
        cell_type_names=feedforward.cell_types.names,
        odour_configs=teacher_params.get_odour_configs_dict(),
    )

    # Extract odour 1 firing rates
    firing_rates_odour_1 = input_firing_rates_odour[0]  # Shape: (n_input_neurons,)

    print(f"✓ Generated odour 1 firing rates: {firing_rates_odour_1.shape}")

    # Use teacher simulation parameters for dt and chunk_size
    teacher_simulation = teacher_params.simulation
    dt = teacher_simulation.dt
    chunk_size = int(teacher_simulation.chunk_size)

    # Calculate number of chunks to run based on training.plot_size
    training = params.training
    chunks_to_run = training.plot_size if hasattr(training, "plot_size") else 10

    # Calculate simulation duration
    simulation_duration = chunks_to_run * chunk_size

    # Create HomogeneousPoissonSpikeDataLoader for generating input spikes
    dataloader = HomogeneousPoissonSpikeDataLoader(
        firing_rates=firing_rates_odour_1,
        chunk_size=chunk_size,
        dt=dt,
        batch_size=1,
        device=device,
    )

    print("\n✓ Created HomogeneousPoissonSpikeDataLoader")
    print(
        f"  Running {chunks_to_run} chunks ({simulation_duration:.1f} ms) for comparison"
    )
    print(f"  Chunk size: {chunk_size:.1f} ms, Timestep: {dt:.2f} ms")
    print(f"  Input neurons: {firing_rates_odour_1.shape[0]}")

    # ====================================
    # Initialize Original Network and Run
    # ====================================

    print("\nInitializing original target network...")

    target_model = ConductanceLIFNetwork(
        dt=dt,
        weights=original_weights,
        cell_type_indices=cell_type_indices,
        cell_params=recurrent.get_cell_params(),
        synapse_params=recurrent.get_synapse_params(),
        weights_FF=original_feedforward_weights,
        cell_type_indices_FF=feedforward_cell_type_indices,
        cell_params_FF=feedforward.get_cell_params(),
        synapse_params_FF=feedforward.get_synapse_params(),
        surrgrad_scale=1.0,  # Not used for inference
        optimisable=None,  # No optimization for target network
        use_tqdm=False,
    ).to(device)

    print(
        "  Running target network simulation (odour 1, two independent runs for Poisson variability)..."
    )

    # We'll run the target network twice with independent Poisson noise
    # Run 1 for comparison with trained, Run 2 for Poisson variability analysis
    target_spikes_runs = []
    all_input_spikes_run1 = []  # Save input from run 1 for trained network
    all_target_voltages = []
    all_target_currents = []
    all_target_currents_FF = []
    all_target_currents_leak = []
    all_target_conductances = []
    all_target_conductances_FF = []

    for run_idx in range(2):
        print(f"    Run {run_idx + 1}/2...")

        # Reset states for each independent run
        initial_v = None
        initial_g = None
        initial_g_FF = None

        run_spikes = []

        with torch.inference_mode():
            for chunk_idx, (input_spikes, _) in enumerate(
                tqdm(
                    dataloader,
                    desc=f"Target run {run_idx + 1}",
                    total=chunks_to_run,
                    leave=False,
                )
            ):
                if chunk_idx >= chunks_to_run:
                    break

                # input_spikes shape: (1, time, n_inputs)
                input_spikes = input_spikes.to(device)

                # Save input from first run for trained network
                if run_idx == 0:
                    all_input_spikes_run1.append(input_spikes.cpu())

                # Run simulation
                (
                    target_spikes_chunk,
                    target_voltages,
                    target_currents,
                    target_currents_FF,
                    target_currents_leak,
                    target_conductances,
                    target_conductances_FF,
                ) = target_model(
                    input_spikes=input_spikes,
                    initial_v=initial_v,
                    initial_g=initial_g,
                    initial_g_FF=initial_g_FF,
                )

                # Store states for next chunk
                initial_v = target_voltages[:, -1, :].clone()
                initial_g = target_conductances[:, -1, :, :, :].clone()
                initial_g_FF = target_conductances_FF[:, -1, :, :, :].clone()

                # Accumulate spikes
                run_spikes.append(target_spikes_chunk.cpu())

                # Save intermediate results from first run only (for dashboard)
                if run_idx == 0:
                    all_target_voltages.append(target_voltages.cpu())
                    all_target_currents.append(target_currents.cpu())
                    all_target_currents_FF.append(target_currents_FF.cpu())
                    all_target_currents_leak.append(target_currents_leak.cpu())
                    all_target_conductances.append(target_conductances.cpu())
                    all_target_conductances_FF.append(target_conductances_FF.cpu())

        # Concatenate chunks for this run: (1, total_time, n_neurons)
        target_spikes_runs.append(torch.cat(run_spikes, dim=1))

    print(f"✓ Generated target spikes (2 runs): {target_spikes_runs[0].shape} each")

    # ====================================
    # Initialize Trained Network and Run
    # ====================================

    print("\nInitializing trained network with final weights...")

    # Initialize trained model with TRAINED weights from final_state
    # These weights already have scaling factors applied (perturbed_weights * learned_scaling_factors)
    trained_model = ConductanceLIFNetwork(
        dt=dt,
        weights=trained_weights,  # Use final trained weights from final_state
        cell_type_indices=cell_type_indices,
        cell_params=recurrent.get_cell_params(),
        synapse_params=recurrent.get_synapse_params(),
        weights_FF=trained_feedforward_weights,  # Use final trained weights from final_state
        cell_type_indices_FF=feedforward_cell_type_indices,
        cell_params_FF=feedforward.get_cell_params(),
        synapse_params_FF=feedforward.get_synapse_params(),
        surrgrad_scale=1.0,  # Not used for inference
        optimisable=None,  # Inference mode - scaling factors already applied to weights
        use_tqdm=False,
    ).to(device)

    print(
        "  Running trained network simulation (using identical input spikes as target run 1)..."
    )

    # Concatenate saved input spikes along time dimension: (1, total_time, n_inputs)
    input_spikes_for_trained = torch.cat(all_input_spikes_run1, dim=1)
    print(f"  Reusing input spikes: {input_spikes_for_trained.shape}")

    # Initialize lists to accumulate results across chunks
    all_trained_spikes = []
    # Store intermediate results for dashboard
    all_trained_voltages = []
    all_trained_currents = []
    all_trained_currents_FF = []
    all_trained_currents_leak = []
    all_trained_conductances = []
    all_trained_conductances_FF = []
    initial_v = None
    initial_g = None
    initial_g_FF = None

    # Run inference in chunks, using the saved input spikes
    with torch.inference_mode():
        for chunk_idx in tqdm(
            range(chunks_to_run), desc="Trained network chunks", leave=False
        ):
            # Extract chunk from saved input spikes
            start_idx = chunk_idx * int(chunk_size / dt)
            end_idx = (chunk_idx + 1) * int(chunk_size / dt)
            input_pattern = input_spikes_for_trained[:, start_idx:end_idx, :].to(device)

            # Run one chunk of simulation
            (
                trained_spikes_chunk,
                trained_voltages,
                trained_currents,
                trained_currents_FF,
                trained_currents_leak,
                trained_conductances,
                trained_conductances_FF,
            ) = trained_model(
                input_spikes=input_pattern,
                initial_v=initial_v,
                initial_g=initial_g,
                initial_g_FF=initial_g_FF,
            )

            # Store final states for next chunk
            initial_v = trained_voltages[:, -1, :].clone()
            initial_g = trained_conductances[:, -1, :, :, :].clone()
            initial_g_FF = trained_conductances_FF[:, -1, :, :, :].clone()

            # Move to CPU and accumulate results
            all_trained_spikes.append(trained_spikes_chunk.cpu())
            all_trained_voltages.append(trained_voltages.cpu())
            all_trained_currents.append(trained_currents.cpu())
            all_trained_currents_FF.append(trained_currents_FF.cpu())
            all_trained_currents_leak.append(trained_currents_leak.cpu())
            all_trained_conductances.append(trained_conductances.cpu())
            all_trained_conductances_FF.append(trained_conductances_FF.cpu())

    # Concatenate all chunks along time dimension: (batch=1, total_time, n_neurons)
    trained_spikes = torch.cat(all_trained_spikes, dim=1)

    print(f"✓ Generated trained spikes: {trained_spikes.shape}")

    print("\nGenerating comparison plots...")

    # Extract individual spike trains
    # Target spikes: Two runs of (1, total_time, n_neurons)
    target_run1 = target_spikes_runs[0].cpu().numpy()  # First run
    target_run2 = (
        target_spikes_runs[1].cpu().numpy()
    )  # Second run (independent Poisson)
    # Trained spikes: (1, total_time, n_neurons)
    trained_spikes_np = trained_spikes.cpu().numpy()

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # dt is already loaded from teacher_params.simulation earlier

    # ====================================
    # Plot 1: Poisson Variability (Target Run 1 vs Run 2)
    # ====================================

    print("  Creating Poisson variability scatter plot (target run 1 vs run 2)...")
    fig_poisson = plot_cross_correlation_scatter(
        spike_trains_trial1=target_run1,
        spike_trains_trial2=target_run2,
        window_size=10.0,  # 10 second windows
        dt=dt,
        title="Poisson Variability: Target Run 1 vs Run 2",
        x_label="Target Run 1 Firing Rate (Hz)",
        y_label="Target Run 2 Firing Rate (Hz)",
    )

    # ====================================
    # Plot 2: Learning Variability (Target vs Trained)
    # ====================================

    print("  Creating learning variability scatter plot (target vs trained)...")
    fig_learning = plot_cross_correlation_scatter(
        spike_trains_trial1=target_run1,
        spike_trains_trial2=trained_spikes_np,
        window_size=10.0,  # 10 second windows
        dt=dt,
        title="Learning Variability: Target vs Trained",
        x_label="Target Network Firing Rate (Hz)",
        y_label="Trained Network Firing Rate (Hz)",
    )

    # ====================================
    # Create Side-by-Side Dashboard
    # ====================================

    print("  Creating combined side-by-side dashboard...")

    # Create a dashboard figure with both correlograms side-by-side (square plots)
    dashboard_fig = plt.figure(figsize=(14, 7))

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
        new_ax.set_aspect("equal", adjustable="box")  # Make plot square
        new_ax.grid(True, alpha=0.3)

    # Add Poisson variability plot to left side of dashboard
    if fig_poisson is not None:
        for ax in fig_poisson.axes:
            copy_axes_to_subplot(ax, 121)
            break

    # Add learning variability plot to right side of dashboard
    if fig_learning is not None:
        for ax in fig_learning.axes:
            copy_axes_to_subplot(ax, 122)
            break

    # Save dashboard
    dashboard_path = output_dir / "firing_rate_correlogram.png"
    dashboard_fig.savefig(dashboard_path, dpi=150, bbox_inches="tight")
    plt.close(dashboard_fig)
    print(f"  ✓ Saved: {dashboard_path}")

    # Close individual plots
    if fig_poisson is not None:
        plt.close(fig_poisson)
    if fig_learning is not None:
        plt.close(fig_learning)

    # ====================================
    # Plot 3: Spike Raster (Interleaved Target and Trained)
    # ====================================

    print("  Creating interleaved spike raster plot (first 10 neurons)...")

    # Extract first 10 neurons from target (run 1) and trained networks
    n_neurons_to_plot = 10
    target_spikes_10 = target_run1[:, :, :n_neurons_to_plot]  # (1, time, 10)
    trained_spikes_10 = trained_spikes_np[:, :, :n_neurons_to_plot]  # (1, time, 10)

    # Interleave: neuron 0 target, neuron 0 trained, neuron 1 target, neuron 1 trained, etc.
    # Create array of shape (1, time, 20) where even indices are target, odd are trained
    interleaved_spikes = np.zeros(
        (1, target_spikes_10.shape[1], n_neurons_to_plot * 2), dtype=np.int32
    )
    for i in range(n_neurons_to_plot):
        interleaved_spikes[:, :, 2 * i] = target_spikes_10[:, :, i]  # Target
        interleaved_spikes[:, :, 2 * i + 1] = trained_spikes_10[:, :, i]  # Trained

    # Create cell type indices: 0 for target, 1 for trained
    interleaved_cell_types = np.array([i % 2 for i in range(n_neurons_to_plot * 2)])
    cell_type_names_raster = ["Target", "Trained"]

    fig_raster = plot_spike_trains(
        spikes=interleaved_spikes,
        dt=dt,
        cell_type_indices=interleaved_cell_types,
        cell_type_names=cell_type_names_raster,
        n_neurons_plot=n_neurons_to_plot * 2,
        fraction=1.0,
        random_seed=None,  # Don't shuffle - keep interleaved order
        title="Spike Raster: Target vs Trained (First 10 Neurons, Interleaved)",
        ylabel="Neuron Pair",
        figsize=(14, 8),
    )

    if fig_raster is not None:
        # Add y-axis labels showing neuron pairs
        ax = fig_raster.axes[0]
        ytick_positions = []
        ytick_labels = []
        for i in range(n_neurons_to_plot):
            # Add tick at the middle of each pair (between target and trained)
            ytick_positions.append(2 * i + 0.5)
            ytick_labels.append(f"{i}")
        ax.set_yticks(ytick_positions)
        ax.set_yticklabels(ytick_labels)
        ax.set_ylabel("Neuron ID", fontsize=10)

        fig_raster.savefig(
            output_dir / "spike_raster_comparison.png", dpi=150, bbox_inches="tight"
        )
        plt.close(fig_raster)
        print(f"  ✓ Saved: {output_dir / 'spike_raster_comparison.png'}")
    # ====================================
    # Plot 4 & 5: Connectivity and Activity Dashboards
    # ====================================

    print("\n  Creating connectivity dashboard...")

    # Derive connectivity masks
    connectome_mask = (original_weights != 0).astype(np.bool_)
    feedforward_mask = (original_feedforward_weights != 0).astype(np.bool_)

    # Get trained weights from the model
    trained_weights = trained_model.weights.detach().cpu().numpy()
    trained_feedforward_weights = trained_model.weights_FF.detach().cpu().numpy()

    # Create connectivity dashboard for target network
    connectivity_fig_target = create_connectivity_dashboard(
        weights=original_weights,
        feedforward_weights=original_feedforward_weights,
        cell_type_indices=cell_type_indices,
        input_cell_type_indices=feedforward_cell_type_indices,
        cell_type_names=recurrent.cell_types.names,
        input_cell_type_names=feedforward.cell_types.names,
        connectome_mask=connectome_mask,
        feedforward_mask=feedforward_mask,
    )
    connectivity_fig_target.savefig(
        output_dir / "connectivity_dashboard_target.png", dpi=150, bbox_inches="tight"
    )
    plt.close(connectivity_fig_target)
    print(f"  ✓ Saved: {output_dir / 'connectivity_dashboard_target.png'}")

    # Create connectivity dashboard for trained network
    connectivity_fig_trained = create_connectivity_dashboard(
        weights=trained_weights,
        feedforward_weights=trained_feedforward_weights,
        cell_type_indices=cell_type_indices,
        input_cell_type_indices=feedforward_cell_type_indices,
        cell_type_names=recurrent.cell_types.names,
        input_cell_type_names=feedforward.cell_types.names,
        connectome_mask=connectome_mask,
        feedforward_mask=feedforward_mask,
    )
    connectivity_fig_trained.savefig(
        output_dir / "connectivity_dashboard_trained.png", dpi=150, bbox_inches="tight"
    )
    plt.close(connectivity_fig_trained)
    print(f"  ✓ Saved: {output_dir / 'connectivity_dashboard_trained.png'}")

    print("\n  Creating activity dashboards...")

    # Concatenate intermediate results for dashboards (from first run only)
    target_voltages_full = torch.cat(all_target_voltages, dim=1)
    target_currents_full = torch.cat(all_target_currents, dim=1)
    target_currents_FF_full = torch.cat(all_target_currents_FF, dim=1)
    target_currents_leak_full = torch.cat(all_target_currents_leak, dim=1)
    target_conductances_full = torch.cat(all_target_conductances, dim=1)
    target_conductances_FF_full = torch.cat(all_target_conductances_FF, dim=1)

    trained_voltages_full = torch.cat(all_trained_voltages, dim=1)
    trained_currents_full = torch.cat(all_trained_currents, dim=1)
    trained_currents_FF_full = torch.cat(all_trained_currents_FF, dim=1)
    trained_currents_leak_full = torch.cat(all_trained_currents_leak, dim=1)
    trained_conductances_full = torch.cat(all_trained_conductances, dim=1)
    trained_conductances_FF_full = torch.cat(all_trained_conductances_FF, dim=1)

    # Concatenate input spikes
    input_spikes_full = torch.cat(all_input_spikes_run1, dim=1)

    # Create activity dashboard for target network (using run 1)
    activity_fig_target = create_activity_dashboard(
        output_spikes=target_run1,
        input_spikes=input_spikes_full.numpy(),
        cell_type_indices=cell_type_indices,
        cell_type_names=recurrent.cell_types.names,
        dt=dt,
        voltages=target_voltages_full.numpy(),
        neuron_types=cell_type_indices,
        neuron_params=recurrent.get_neuron_params_for_plotting(),
        recurrent_currents=target_currents_full.numpy(),
        feedforward_currents=target_currents_FF_full.numpy(),
        leak_currents=target_currents_leak_full.numpy(),
        recurrent_conductances=target_conductances_full.numpy(),
        feedforward_conductances=target_conductances_FF_full.numpy(),
        input_cell_type_names=feedforward.cell_types.names,
        recurrent_synapse_names=recurrent.get_synapse_names(),
        feedforward_synapse_names=feedforward.get_synapse_names(),
    )
    activity_fig_target.savefig(
        output_dir / "activity_dashboard_target.png", dpi=150, bbox_inches="tight"
    )
    plt.close(activity_fig_target)
    print(f"  ✓ Saved: {output_dir / 'activity_dashboard_target.png'}")

    # Create activity dashboard for trained network
    activity_fig_trained = create_activity_dashboard(
        output_spikes=trained_spikes_np,
        input_spikes=input_spikes_full.numpy(),
        cell_type_indices=cell_type_indices,
        cell_type_names=recurrent.cell_types.names,
        dt=dt,
        voltages=trained_voltages_full.numpy(),
        neuron_types=cell_type_indices,
        neuron_params=recurrent.get_neuron_params_for_plotting(),
        recurrent_currents=trained_currents_full.numpy(),
        feedforward_currents=trained_currents_FF_full.numpy(),
        leak_currents=trained_currents_leak_full.numpy(),
        recurrent_conductances=trained_conductances_full.numpy(),
        feedforward_conductances=trained_conductances_FF_full.numpy(),
        input_cell_type_names=feedforward.cell_types.names,
        recurrent_synapse_names=recurrent.get_synapse_names(),
        feedforward_synapse_names=feedforward.get_synapse_names(),
    )
    activity_fig_trained.savefig(
        output_dir / "activity_dashboard_trained.png", dpi=150, bbox_inches="tight"
    )
    plt.close(activity_fig_trained)
    print(f"  ✓ Saved: {output_dir / 'activity_dashboard_trained.png'}")
    print(f"\n✓ All plots saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare trained network activity with target network activity"
    )
    parser.add_argument(
        "experiment_dir",
        type=Path,
        help="Path to experiment directory containing input/, checkpoints/, and parameters.toml",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Path to directory where comparison plots will be saved",
    )
    parser.add_argument(
        "teacher_params_file",
        type=Path,
        help="Path to generate-teacher-activity.toml file with odour configurations",
    )

    args = parser.parse_args()

    main(
        experiment_dir=args.experiment_dir,
        output_dir=args.output_dir,
        teacher_params_file=args.teacher_params_file,
    )
