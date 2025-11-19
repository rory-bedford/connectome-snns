"""
Generating spike trains from a synthetic connectome

This script generates spiketrains from a predefined synthetic connectome
to be used as teacher activity for fitting recurrent networks.
"""

import numpy as np
from inputs.dataloaders import PoissonSpikeDataset
from network_simulators.conductance_based.simulator import ConductanceLIFNetwork
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from parameter_loaders import TeacherActivityParams
import toml
from analysis import compute_network_statistics
from visualization.dashboards import (
    create_connectivity_dashboard,
    create_activity_dashboard,
)


def main(input_dir, output_dir, params_file):
    """Main execution function for Dp network simulation.

    Args:
        input_dir (Path, optional): Directory containing input data files (may be None)
        output_dir (Path): Directory where output files will be saved
        params_file (Path): Path to the file containing network parameters
    """

    # ======================================
    # Device Selection and Parameter Loading
    # ======================================

    # Select device (CPU/GPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load all network parameters from TOML file
    with open(params_file, "r") as f:
        data = toml.load(f)
    params = TeacherActivityParams(**data)

    # Extract commonly used parameter groups
    simulation = params.simulation
    recurrent = params.recurrent
    feedforward = params.feedforward

    # Set random seed if provided
    if simulation.seed is not None:
        np.random.seed(simulation.seed)
        torch.manual_seed(simulation.seed)
        print(f"Using seed: {simulation.seed}")
    else:
        print("No seed specified - using random initialization")

    # ================================
    # Load Network Structure from Disk
    # ================================

    print("Loading network structure from disk...")
    network_structure = np.load(input_dir / "network_structure.npz")

    # Extract network components
    weights = network_structure["recurrent_weights"]
    feedforward_weights = network_structure["feedforward_weights"]
    cell_type_indices = network_structure["cell_type_indices"]
    input_source_indices = network_structure["feedforward_cell_type_indices"]

    print(f"✓ Loaded network with {len(cell_type_indices)} neurons")
    print(f"✓ Loaded recurrent weights: {weights.shape}")
    print(f"✓ Loaded feedforward weights: {feedforward_weights.shape}")

    # =========================
    # Create Feedforward Inputs
    # =========================

    # Calculate simulation parameters
    print(
        f"Simulation: {simulation.duration:.1f} ms in {simulation.num_chunks} chunks of {simulation.chunk_size:.1f} ms each (batch_size={simulation.batch_size})"
    )
    print(
        f"Timestep: {simulation.dt:.2f} ms ({int(simulation.chunk_size / simulation.dt)} steps per chunk)"
    )

    # Create firing rates array for input neurons
    input_firing_rates = np.zeros(feedforward_weights.shape[0])
    for ct_idx, ct_name in enumerate(feedforward.cell_types.names):
        mask = input_source_indices == ct_idx
        input_firing_rates[mask] = feedforward.activity[ct_name].firing_rate

    # Create Poisson spike generator dataset
    spike_dataset = PoissonSpikeDataset(
        firing_rates=input_firing_rates,
        chunk_size=simulation.chunk_size,
        dt=simulation.dt,
        device=device,
    )

    # Create DataLoader with batch_size from parameters
    spike_dataloader = DataLoader(
        spike_dataset,
        batch_size=simulation.batch_size,
        shuffle=False,
        num_workers=0,  # Keep 0 for GPU generation
    )

    # ======================
    # Initialize LIF Network
    # ======================

    print(
        f"Initializing conductance-based LIF network with {len(recurrent.get_cell_params())} cell types and {len(recurrent.get_synapse_params())} synapse types..."
    )

    # Initialize conductance-based LIF network model
    model = ConductanceLIFNetwork(
        dt=simulation.dt,
        weights=weights,
        cell_type_indices=cell_type_indices,
        cell_params=recurrent.get_cell_params(),
        synapse_params=recurrent.get_synapse_params(),
        surrgrad_scale=1.0,  # Not used for inference, but required parameter
        weights_FF=feedforward_weights,
        cell_type_indices_FF=input_source_indices,
        cell_params_FF=feedforward.get_cell_params(),
        synapse_params_FF=feedforward.get_synapse_params(),
        optimisable=None,  # No optimization for inference
        use_tqdm=True,  # Enable model's internal progress bar for each chunk
    )

    # Move model to device for GPU acceleration
    model.to(device)
    print(f"Model moved to device: {device}")

    # Jit compile the model for faster execution
    model.compile_step()
    print("✓ Model JIT compiled for faster execution")

    print("\n" + "=" * len("STARTING CHUNKED NETWORK SIMULATION"))
    print("STARTING CHUNKED NETWORK SIMULATION")
    print("=" * len("STARTING CHUNKED NETWORK SIMULATION"))

    # ==============================
    # Run Chunked Network Simulation
    # ==============================

    print(f"Running simulation in {simulation.num_chunks} chunks...")

    # Initialize lists to accumulate results across chunks
    all_output_spikes = []
    all_output_voltages = []
    all_output_currents = []
    all_output_currents_FF = []
    all_output_conductances = []
    all_output_conductances_FF = []
    all_input_spikes = []

    # Initialize state variables (will be passed between chunks)
    initial_v = None
    initial_g = None
    initial_g_FF = None

    # Run inference in chunks using DataLoader
    with torch.inference_mode():
        dataloader_iter = iter(spike_dataloader)
        for chunk_idx in range(simulation.num_chunks):
            print(f"Processing chunk {chunk_idx + 1}/{simulation.num_chunks}...")

            # Get next chunk from dataloader
            input_spikes_chunk = next(dataloader_iter)

            # Run one chunk of simulation
            (
                output_spikes,
                output_voltages,
                output_currents,
                output_currents_FF,
                output_conductances,
                output_conductances_FF,
            ) = model.forward(
                input_spikes=input_spikes_chunk,
                initial_v=initial_v,
                initial_g=initial_g,
                initial_g_FF=initial_g_FF,
            )

            # Store final states for next chunk (average across batch dimension)
            initial_v = output_voltages[:, -1, :].clone()  # Last timestep voltages
            initial_g = output_conductances[
                :, -1, :, :
            ].clone()  # Last timestep conductances
            initial_g_FF = output_conductances_FF[
                :, -1, :, :
            ].clone()  # Last timestep FF conductances

            # Move to CPU and accumulate results
            # Store all batches for spikes, but only first batch for voltages/currents/conductances
            if device == "cuda":
                all_output_spikes.append(output_spikes.bool().cpu())
                all_input_spikes.append(input_spikes_chunk.cpu())
                # Only store first batch for detailed data
                all_output_voltages.append(output_voltages[0:1, ...].cpu())
                all_output_currents.append(output_currents[0:1, ...].cpu())
                all_output_currents_FF.append(output_currents_FF[0:1, ...].cpu())
                all_output_conductances.append(output_conductances[0:1, ...].cpu())
                all_output_conductances_FF.append(
                    output_conductances_FF[0:1, ...].cpu()
                )
            else:
                all_output_spikes.append(output_spikes.bool())
                all_input_spikes.append(input_spikes_chunk)
                # Only store first batch for detailed data
                all_output_voltages.append(output_voltages[0:1, ...])
                all_output_currents.append(output_currents[0:1, ...])
                all_output_currents_FF.append(output_currents_FF[0:1, ...])
                all_output_conductances.append(output_conductances[0:1, ...])
                all_output_conductances_FF.append(output_conductances_FF[0:1, ...])

    # Concatenate all batches along time dimension (dim=1) and convert to numpy
    output_spikes = torch.cat(all_output_spikes, dim=1).numpy()
    output_voltages = torch.cat(all_output_voltages, dim=1).numpy()
    output_currents = torch.cat(all_output_currents, dim=1).numpy()
    output_currents_FF = torch.cat(all_output_currents_FF, dim=1).numpy()
    output_conductances = torch.cat(all_output_conductances, dim=1).numpy()
    output_conductances_FF = torch.cat(all_output_conductances_FF, dim=1).numpy()
    input_spikes = torch.cat(all_input_spikes, dim=1).numpy()

    print("✓ Network simulation completed!")

    # =====================================
    # Save Output Data for Further Analysis
    # =====================================
    print("Saving simulation data...")

    # Create results directory if it doesn't exist
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save spike trains (the main output for analysis)
    np.savez_compressed(
        results_dir / "spike_data.npz",
        output_spikes=output_spikes.astype(np.bool_),
        input_spikes=input_spikes.astype(np.bool_),
        cell_type_indices=cell_type_indices,
        input_cell_type_indices=input_source_indices,
    )

    print(f"✓ Saved data to {results_dir}")

    # Free GPU memory before generating dashboards
    if device == "cuda":
        del model
        torch.cuda.empty_cache()
        print("✓ Freed GPU memory")

    # =====================================
    # Generate All Plots and Visualizations
    # =====================================

    print("Generating dashboards...")

    # Generate connectivity dashboard
    connectivity_fig = create_connectivity_dashboard(
        weights=weights,
        feedforward_weights=feedforward_weights,
        cell_type_indices=cell_type_indices,
        input_cell_type_indices=input_source_indices,
        cell_type_names=params.recurrent.cell_types.names,
        input_cell_type_names=params.feedforward.cell_types.names,
        recurrent_g_bar_by_type=params.recurrent.get_g_bar_by_type(),
        feedforward_g_bar_by_type=params.feedforward.get_g_bar_by_type(),
    )

    # Compute total excitatory and inhibitory currents from recurrent and feedforward
    # Generate activity dashboard
    activity_fig = create_activity_dashboard(
        output_spikes=output_spikes[0:1, ...],
        input_spikes=input_spikes[0:1, ...],
        cell_type_indices=cell_type_indices,
        cell_type_names=params.recurrent.cell_types.names,
        dt=params.simulation.dt,
        voltages=output_voltages,
        neuron_types=cell_type_indices,
        neuron_params=params.recurrent.get_neuron_params_for_plotting(),
        recurrent_currents=output_currents,
        feedforward_currents=output_currents_FF,
        recurrent_conductances=output_conductances,
        feedforward_conductances=output_conductances_FF,
        input_cell_type_names=params.feedforward.cell_types.names,
        recurrent_synapse_names=params.recurrent.get_synapse_names(),
        feedforward_synapse_names=params.feedforward.get_synapse_names(),
        window_size=50.0,
        n_neurons_plot=20,
        fraction=1.0,
        random_seed=42,
    )

    # Save the two dashboards
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    connectivity_fig.savefig(
        figures_dir / "connectivity_dashboard.png", dpi=300, bbox_inches="tight"
    )
    plt.close(connectivity_fig)

    activity_fig.savefig(
        figures_dir / "activity_dashboard.png", dpi=300, bbox_inches="tight"
    )
    plt.close(activity_fig)

    print(f"✓ Saved dashboard plots to {figures_dir}")

    # ============================================
    # Compute and Save Combined Statistics to CSV
    # ============================================

    print("Computing network statistics...")
    stats_df = compute_network_statistics(
        output_spikes=output_spikes,
        output_voltages=output_voltages,
        cell_type_indices=cell_type_indices,
        cell_type_names=params.recurrent.cell_types.names,
        duration=params.simulation.duration,
        dt=params.simulation.dt,
    )

    stats_csv_path = output_dir / "network_statistics.csv"
    stats_df.to_csv(stats_csv_path, index=False)
    print(f"✓ Saved network statistics to {stats_csv_path}")

    # ========
    # Clean Up
    # ========

    print("\n" + "=" * len("SIMULATION COMPLETE!"))
    print("SIMULATION COMPLETE!")
    print("=" * len("SIMULATION COMPLETE!"))
    print(f"✓ Results saved to {output_dir / 'results'}")
    print(f"✓ Figures saved to {output_dir / 'figures'}")
    print(f"✓ Statistics saved to {output_dir / 'network_statistics.csv'}")
    print("=" * len("SIMULATION COMPLETE!"))
