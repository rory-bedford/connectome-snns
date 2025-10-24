"""
Generating spike trains from a synthetic connectome

This script generates spiketrains from a recurrent E-I network with conductance-based
LIF neurons, and assembly connectivity inspired by zebrafish Dp.

Overview:
1. First we generate a biologically plausible recurrent weight matrix with a
   Dp-inspired assembly structure.
2. Next we generate excitatory mitral cell inputs from the OB with Poisson
   statistics and sparse projections.
3. Then we initialise our network with parameters adapted from
   Meissner-Bernard et al. (2025) https://doi.org/10.1016/j.celrep.2025.115330
4. Finally we run our network and examine the output dynamics.
"""

# ruff: noqa

import numpy as np
import toml
from synthetic_connectome import topology_generators, weight_assigners, cell_types
from network_simulators.conductance_lif_network import ConductanceLIFNetwork
import torch
import sys
from pathlib import Path

# Import local scripts
sys.path.append(str(Path(__file__).resolve().parent))
import conductance_based_Dp_plots


def main(output_dir, params_file):
    """Main execution function for Dp network simulation.

    Args:
        output_dir (Path): Directory where output files will be saved
        params_file (Path): Path to the file containing network parameters
    """
    # =============================================
    # SETUP: Device selection and parameter loading
    # =============================================

    # Select device (CPU/GPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load all network parameters from TOML file
    with open(params_file, "r") as f:
        params = toml.load(f)

    # Extract and convert all parameters at the top
    dt = float(params["simulation"]["dt"])
    duration = float(params["simulation"]["duration"])
    seed = params["simulation"].get("seed", None)

    # ========== FEEDFORWARD LAYER PARAMETERS ==========
    # Feedforward topology
    input_num_neurons = int(params["feedforward"]["topology"]["num_neurons"])
    input_conn_inputs = np.array(
        params["feedforward"]["topology"]["conn_inputs"], dtype=float
    )

    # Feedforward cell types
    input_cell_type_names = params["feedforward"]["cell_types"]["names"]
    input_cell_type_proportions = np.array(
        params["feedforward"]["cell_types"]["proportion"], dtype=float
    )

    # Feedforward weights
    input_w_mu = np.array(params["feedforward"]["weights"]["w_mu"], dtype=float)
    input_w_sigma = np.array(params["feedforward"]["weights"]["w_sigma"], dtype=float)

    # Feedforward activity (firing rates)
    input_firing_rates = {}
    for ct in input_cell_type_names:
        input_firing_rates[ct] = float(
            params["feedforward"]["activity"][ct]["firing_rate"]
        )

    # Feedforward synapses (cell type specific)
    synapse_params_FF = {}
    for ct in input_cell_type_names:
        synapse_params_FF[ct] = {
            "names": params["feedforward"]["synapses"][ct]["names"],
            "tau_rise": np.array(
                params["feedforward"]["synapses"][ct]["tau_rise"], dtype=float
            ),
            "tau_decay": np.array(
                params["feedforward"]["synapses"][ct]["tau_decay"], dtype=float
            ),
            "reversal_potential": np.array(
                params["feedforward"]["synapses"][ct]["reversal_potential"],
                dtype=float,
            ),
            "g_bar": np.array(
                params["feedforward"]["synapses"][ct]["g_bar"], dtype=float
            ),
        }

    # ========== RECURRENT LAYER PARAMETERS ==========
    # Recurrent topology
    num_neurons = int(params["recurrent"]["topology"]["num_neurons"])
    num_assemblies = int(params["recurrent"]["topology"]["num_assemblies"])
    conn_within = np.array(params["recurrent"]["topology"]["conn_within"])
    conn_between = np.array(params["recurrent"]["topology"]["conn_between"])

    # Recurrent cell types
    cell_type_names = params["recurrent"]["cell_types"]["names"]
    cell_type_proportions = np.array(
        params["recurrent"]["cell_types"]["proportion"], dtype=float
    )

    # Recurrent weights
    w_mu = np.array(params["recurrent"]["weights"]["w_mu"], dtype=float)
    w_sigma = np.array(params["recurrent"]["weights"]["w_sigma"], dtype=float)

    # Recurrent physiology (cell type specific)
    physiology_params = {}
    for ct in cell_type_names:
        physiology_params[ct] = {
            "tau_mem": float(params["recurrent"]["physiology"][ct]["tau_mem"]),
            "theta": float(params["recurrent"]["physiology"][ct]["theta"]),
            "U_reset": float(params["recurrent"]["physiology"][ct]["U_reset"]),
            "E_L": float(params["recurrent"]["physiology"][ct]["E_L"]),
            "g_L": float(params["recurrent"]["physiology"][ct]["g_L"]),
            "tau_ref": float(params["recurrent"]["physiology"][ct]["tau_ref"]),
        }

    # Recurrent synapses (cell type specific)
    synapse_params = {}
    for ct in cell_type_names:
        synapse_params[ct] = {
            "names": params["recurrent"]["synapses"][ct]["names"],
            "tau_rise": np.array(
                params["recurrent"]["synapses"][ct]["tau_rise"], dtype=float
            ),
            "tau_decay": np.array(
                params["recurrent"]["synapses"][ct]["tau_decay"], dtype=float
            ),
            "reversal_potential": np.array(
                params["recurrent"]["synapses"][ct]["reversal_potential"],
                dtype=float,
            ),
            "g_bar": np.array(
                params["recurrent"]["synapses"][ct]["g_bar"], dtype=float
            ),
        }

    # ========== OPTIMIZATION PARAMETERS ==========
    scaling_factors = np.array(params["optimisation"]["scaling_factors"], dtype=float)
    scaling_factors_FF = np.array(
        params["optimisation"]["scaling_factors_FF"], dtype=float
    )

    # ========== HYPERPARAMETERS ==========
    surrgrad_scale = float(params["hyperparameters"]["surrgrad_scale"])

    # Set global random seed for reproducibility (only if specified in config)
    if seed is not None:
        np.random.seed(seed)
        print(f"Using seed: {seed}")
    else:
        print("No seed specified - using random initialization")

    # ==========================================================
    # STEP 1: Generate Assembly-Based Topology and Visualization
    # ==========================================================

    # First assign cell types to source and target neurons (same for recurrent)
    cell_type_indices = cell_types.assign_cell_types(
        num_neurons=num_neurons,
        cell_type_proportions=cell_type_proportions,
    )

    # Generate assembly-based connectivity graph
    connectivity_graph = topology_generators.assembly_generator(
        source_cell_types=cell_type_indices,
        target_cell_types=cell_type_indices,  # Same for recurrent connections
        num_assemblies=num_assemblies,
        conn_within=conn_within,
        conn_between=conn_between,
    )

    # Store cell type indices for later use (no signs needed for conductance-based model)
    neuron_types = cell_type_indices

    # ========================================================
    # STEP 2: Assign Synaptic Weights and Analyze Connectivity
    # ========================================================

    # Assign log-normal weights to connectivity graph (no signs for conductance-based)
    weights = weight_assigners.assign_weights_lognormal(
        connectivity_graph=connectivity_graph,
        source_cell_indices=cell_type_indices,
        target_cell_indices=cell_type_indices,  # Same for recurrent connections
        cell_type_signs=None,  # No signs for conductance-based model
        w_mu_matrix=w_mu,
        w_sigma_matrix=w_sigma,
    )

    # ===================================================
    # STEP 3: Create Feedforward Inputs from Mitral Cells
    # ===================================================

    # Generate Poisson spike trains for feedforward inputs
    n_steps = int(duration / dt)
    shape = (1, n_steps, input_num_neurons)

    # Assign cell types to input layer
    input_source_indices = cell_types.assign_cell_types(
        num_neurons=input_num_neurons,
        cell_type_proportions=input_cell_type_proportions,
    )

    # Generate spikes for each input neuron based on its cell type firing rate
    input_spikes = np.zeros(shape, dtype=bool)
    for i, ct_idx in enumerate(input_source_indices):
        ct_name = input_cell_type_names[ct_idx]
        firing_rate = input_firing_rates[ct_name]
        p_spike = firing_rate * dt * 1e-3  # rate in Hz, dt in ms
        input_spikes[:, :, i] = np.random.rand(1, n_steps) < p_spike

    # Generate feedforward connectivity graph
    feedforward_connectivity_graph = topology_generators.sparse_graph_generator(
        source_cell_types=input_source_indices,
        target_cell_types=cell_type_indices,  # Connectome layer cell assignments
        conn_matrix=input_conn_inputs,  # N_input_types x N_recurrent_types matrix
        allow_self_loops=True,  # Allow for feedforward connections
    )

    # Assign log-normal weights to feedforward connectivity (no signs for conductance-based)
    feedforward_weights = weight_assigners.assign_weights_lognormal(
        connectivity_graph=feedforward_connectivity_graph,
        source_cell_indices=input_source_indices,
        target_cell_indices=cell_type_indices,
        cell_type_signs=None,  # No signs for conductance-based model
        w_mu_matrix=input_w_mu,
        w_sigma_matrix=input_w_sigma,
    )

    # =================================================
    # STEP 4: Initialize and Run LIF Network Simulation
    # =================================================
    # STEP 4: Initialize and Run LIF Network Simulation
    # =================================================

    # TODO: Update model initialization once model accepts new synapse parameters
    # Initialize LIF network model with corrected arguments
    # model = ConductanceLIFNetwork(
    #     weights=weights,
    #     cell_types=cell_type_names,
    #     cell_type_indices=cell_type_indices,
    #     physiology_params=physiology_params,
    #     synapse_params=synapse_params,
    #     scaling_factors=scaling_factors,
    #     surrgrad_scale=surrgrad_scale,
    #     weights_FF=feedforward_weights,
    #     cell_types_FF=input_cell_type_names,
    #     cell_type_indices_FF=input_source_indices,
    #     synapse_params_FF=synapse_params_FF,
    #     scaling_factors_FF=scaling_factors_FF,
    # )

    # # Move model to device for GPU acceleration
    # model.to(device)

    # # Run inference
    # with torch.inference_mode():
    #     output_spikes, output_voltages, output_g, output_g_FF = model.forward(
    #         n_steps=n_steps,
    #         dt=dt,
    #         inputs=input_spikes,
    #     )

    # # Move tensors to CPU for further processing and saving
    # output_spikes = output_spikes.cpu()
    # output_voltages = output_voltages.cpu()
    # output_g = output_g.cpu()
    # output_g_FF = output_g_FF.cpu()

    # ============================================
    # STEP 5: Save Output Data for Further Analysis
    # ============================================

    # # Save output arrays
    # np.save(output_dir / "output_spikes.npy", output_spikes.numpy())
    # np.save(output_dir / "output_voltages.npy", output_voltages.numpy())
    # np.save(output_dir / "output_g.npy", output_g.numpy())
    # np.save(output_dir / "output_g_FF.npy", output_g_FF.numpy())

    # Save input data and network structure
    np.save(output_dir / "input_spikes.npy", input_spikes)
    np.save(output_dir / "neuron_types.npy", neuron_types)
    np.save(output_dir / "cell_type_indices.npy", cell_type_indices)
    np.save(output_dir / "connectivity_graph.npy", connectivity_graph)
    np.save(output_dir / "weights.npy", weights)
    np.save(output_dir / "feedforward_weights.npy", feedforward_weights)

    print("\n✓ Successfully loaded all parameters from TOML file")
    print(
        f"✓ Generated {num_neurons} neuron recurrent network with {num_assemblies} assemblies"
    )
    print(f"✓ Generated {input_num_neurons} feedforward inputs")
    print(f"✓ Generated {n_steps} timesteps of input spikes")
    print(f"✓ Saved network structure to {output_dir}")
    print("\nNext step: Update model initialization to accept new synapse parameters")

    # =============================================
    # STEP 6: Generate All Plots and Visualizations
    # =============================================

    # # Call the plotting script to generate all visualizations
    # conductance_based_Dp_plots.main(output_dir)

    print("✓ Skipping plotting until model simulation is complete")
