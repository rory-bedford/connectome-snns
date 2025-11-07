"""
Simulating Dp with homeostatic plasticity to achieve target activity regime.

This script sets up and runs a conductance-based leaky integrate-and-fire
network model of the zebrafish dorsal pallium (Dp) with homeostatic plasticity
mechanisms to regulate neuron firing rates and spike train statistics.

Overview:
1. First we generate a biologically plausible recurrent weight matrix with a
   Dp-inspired assembly structure.
2. Next we generate excitatory mitral cell inputs from the OB with Poisson
   statistics and sparse projections.
3. Then we initialise our network with parameters adapted from
   Meissner-Bernard et al. (2025) https://doi.org/10.1016/j.celrep.2025.115330
4. We then run our network simulation in a training loop, applying updates to
   the connectome-constrained weights every so often to optimise the activity
   towards target firing rates and spike train CVs.
"""

import numpy as np
from synthetic_connectome import (
    topology_generators,
    weight_assigners,
    cell_types,
    spike_generators,
)
from network_simulators.conductance_based.simulator import ConductanceLIFNetwork
import torch
from torch.amp import autocast, GradScaler
import sys
from pathlib import Path
from tqdm import tqdm
from optimisation.loss_functions import CVLoss, FiringRateLoss
from optimisation.utils import save_checkpoint, load_checkpoint, AsyncLogger
from network_simulators.conductance_based.parameter_loader import (
    HomeostaticPlasticityParams,
)
import toml
import wandb
from matplotlib import pyplot as plt


# Import local scripts
sys.path.append(str(Path(__file__).resolve().parent))
import homeostatic_plots


def main(
    output_dir,
    params_file,
    wandb_config=None,
    resume_from=None,
    resumed_output_dir=None,
):
    """Main execution function for Dp network homeostatic training.

    Args:
        output_dir (Path): Directory where output files will be saved
        params_file (Path): Path to the file containing network parameters
        wandb_config (dict, optional): W&B configuration from experiment.toml
        resume_from (Path, optional): Path to checkpoint to resume from
        resumed_output_dir (Path, optional): Separate directory for plots when resuming training
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
    params = HomeostaticPlasticityParams(**data)

    # Extract commonly used parameter groups
    simulation = params.simulation
    training = params.training
    targets = params.targets
    hyperparameters = params.hyperparameters
    recurrent = params.recurrent
    feedforward = params.feedforward

    # Determine output directory for plots (use resumed_output_dir if resuming)
    plots_output_dir = resumed_output_dir if resumed_output_dir else output_dir

    # Set random seed if provided
    if simulation.seed is not None:
        np.random.seed(simulation.seed)
        torch.manual_seed(simulation.seed)
        print(f"Using seed: {simulation.seed}")
    else:
        print("No seed specified - using random initialization")

    # ===================================
    # Create Recurrent Connectivity Graph
    # ===================================

    # Assign cell types to recurrent layer
    cell_type_indices = cell_types.assign_cell_types(
        num_neurons=recurrent.topology.num_neurons,
        cell_type_proportions=recurrent.cell_types.proportion,
    )

    # Generate assembly-based connectivity graph
    connectivity_graph = topology_generators.assembly_generator(
        source_cell_types=cell_type_indices,
        target_cell_types=cell_type_indices,  # Same for recurrent connections
        num_assemblies=recurrent.topology.num_assemblies,
        conn_within=recurrent.topology.conn_within,
        conn_between=recurrent.topology.conn_between,
        method="configuration",  # Ensures exact in-degree/out-degree distributions
    )

    # =======================
    # Assign Synaptic Weights
    # =======================

    # Assign log-normal weights to connectivity graph (no signs for conductance-based)
    weights = weight_assigners.assign_weights_lognormal(
        connectivity_graph=connectivity_graph,
        source_cell_indices=cell_type_indices,
        target_cell_indices=cell_type_indices,  # Same for recurrent connections
        cell_type_signs=None,  # No signs for conductance-based model
        w_mu_matrix=recurrent.weights.w_mu,
        w_sigma_matrix=recurrent.weights.w_sigma,
        parameter_space="linear",
    )

    # ==========================================
    # Create Feedforward Connections and Weights
    # ==========================================

    # Assign cell types to input layer
    input_source_indices = cell_types.assign_cell_types(
        num_neurons=feedforward.topology.num_neurons,
        cell_type_proportions=feedforward.cell_types.proportion,
    )

    # Generate feedforward connectivity graph
    feedforward_connectivity_graph = topology_generators.sparse_graph_generator(
        source_cell_types=input_source_indices,
        target_cell_types=cell_type_indices,  # Connectome layer cell assignments
        conn_matrix=feedforward.topology.conn_inputs,  # N_input_types x N_recurrent_types matrix
        allow_self_loops=True,  # Allow for feedforward connections
        method="configuration",  # Ensures exact in-degree/out-degree distributions
    )

    # Assign log-normal weights to feedforward connectivity (no signs for conductance-based)
    feedforward_weights = weight_assigners.assign_weights_lognormal(
        connectivity_graph=feedforward_connectivity_graph,
        source_cell_indices=input_source_indices,
        target_cell_indices=cell_type_indices,
        cell_type_signs=None,  # No signs for conductance-based model
        w_mu_matrix=feedforward.weights.w_mu,
        w_sigma_matrix=feedforward.weights.w_sigma,
        parameter_space="linear",
    )

    # ======================
    # Initialize LIF Network
    # ======================

    # Initialize conductance-based LIF network model
    model = ConductanceLIFNetwork(
        weights=weights,
        cell_type_indices=cell_type_indices,
        cell_params=recurrent.get_cell_params(),
        synapse_params=recurrent.get_synapse_params(),
        surrgrad_scale=hyperparameters.surrgrad_scale,
        weights_FF=feedforward_weights,
        cell_type_indices_FF=input_source_indices,
        cell_params_FF=feedforward.get_cell_params(),
        synapse_params_FF=feedforward.get_synapse_params(),
        optimisable="weights",
        use_tqdm=False,  # Disable tqdm progress bar for training loop
    )

    # Move model to device for GPU acceleration
    model.to(device)

    # ==============================
    # Setup Optimiser and DataLoader
    # ==============================

    optimiser = torch.optim.Adam(model.parameters(), lr=hyperparameters.learning_rate)

    # Mixed precision training scaler (only used if enabled)
    scaler = GradScaler("cuda", enabled=training.mixed_precision and device == "cuda")

    # Define loss functions
    target_cv_tensor = (
        torch.ones(recurrent.topology.num_neurons, device=device) * targets.cv
    )
    target_rate_tensor = (
        torch.ones(recurrent.topology.num_neurons, device=device) * targets.firing_rate
    )
    cv_loss_fn = CVLoss(
        target_cv=target_cv_tensor, penalty_value=hyperparameters.cv_high_loss
    )
    firing_rate_loss_fn = FiringRateLoss(
        target_rate=target_rate_tensor, dt=simulation.dt
    )

    # Initial arrays for chunk inputs
    initial_v = None
    initial_g = None
    initial_g_FF = None

    # Initialize gradient accumulation
    optimiser.zero_grad(set_to_none=True)  # Faster than zeroing to 0

    # Accumulators for loss computation (GPU tensors with gradients)
    loss_spikes = []

    # Accumulators for plotting (CPU numpy arrays, only stored when needed)
    # These are only populated in the last plot_size epochs before a checkpoint
    plot_spikes = []
    plot_voltages = []
    plot_currents = []
    plot_currents_FF = []
    plot_conductances = []
    plot_conductances_FF = []
    plot_input_spikes = []

    start_epoch = 0
    best_loss = float("inf")

    # Load checkpoint if resuming
    if resume_from is not None:
        start_epoch, initial_v, initial_g, initial_g_FF, best_loss = load_checkpoint(
            checkpoint_path=resume_from,
            model=model,
            optimiser=optimiser,
            scaler=scaler,
            device=device,
        )

    # =============
    # Setup Loggers
    # =============

    if wandb_config and wandb_config.get("enabled", False):
        # Build wandb config from network parameters
        wandb_config_dict = {
            **params.model_dump(),  # Convert all params to dict
            "output_dir": str(output_dir),
            "device": device,
        }

        # Build init kwargs with only non-None optional parameters
        wandb_init_kwargs = {
            "project": wandb_config["project"],
            "name": output_dir.name,
            "config": wandb_config_dict,
            "dir": str(output_dir),
            **{
                k: v
                for k, v in wandb_config.items()
                if k in ("entity", "tags", "notes") and v is not None
            },
        }

        wandb.init(**wandb_init_kwargs)
        wandb.watch(model, log="all", log_freq=training.log_interval)

    # Initialize async logger for non-blocking metric logging
    metrics_logger = AsyncLogger(log_dir=output_dir / "metrics", flush_interval=120.0)

    # =================
    # Run Training Loop
    # =================

    print(f"\nStarting training from chunk {start_epoch}...")
    print(f"Target firing rate: {targets.firing_rate} Hz")
    print(f"Target CV: {targets.cv}")
    print(
        f"Total chunks: {simulation.num_chunks} ({simulation.num_chunks * simulation.chunk_duration / 1000:.2f} s)"
    )
    print(
        f"Losses per update: {training.losses_per_update} losses ({training.losses_per_update * simulation.chunk_duration / 1000:.2f} s)"
    )
    print(
        f"Log interval: {training.log_interval} chunks ({training.log_interval * simulation.chunk_duration / 1000:.2f} s)"
    )
    print(
        f"Checkpoint interval: {training.checkpoint_interval} chunks ({training.checkpoint_interval * simulation.chunk_duration / 1000:.2f} s)"
    )

    # Initialize progress bar for training loop
    pbar = tqdm(
        range(start_epoch, simulation.num_chunks),
        desc="Training",
        unit="chunk",
        initial=start_epoch,
        total=simulation.num_chunks,
    )

    for epoch in pbar:
        # Generate new feedforward spikes for this chunk
        input_spikes = spike_generators.generate_poisson_spikes(
            n_steps=simulation.chunk_size,
            dt=simulation.dt,
            num_neurons=feedforward.topology.num_neurons,
            cell_type_indices=input_source_indices,
            cell_type_names=feedforward.cell_types.names,
            firing_rates=feedforward.activity.firing_rates,
            batch_size=1,
            device=device,
        )

        # Run network simulation for this chunk (with mixed precision)
        with autocast(
            device_type="cuda", enabled=training.mixed_precision and device == "cuda"
        ):
            chunk_s, chunk_v, chunk_I, chunk_I_FF, chunk_g, chunk_g_FF = model.forward(
                n_steps=simulation.chunk_size,
                dt=simulation.dt,
                inputs=input_spikes,
                initial_v=initial_v,
                initial_g=initial_g,
                initial_g_FF=initial_g_FF,
            )

        # Carry state forward to next chunk (detach to stop gradient flow)
        initial_v = chunk_v[:, -1, :].detach()
        initial_g = chunk_g[:, -1, :, :, :].detach()
        initial_g_FF = chunk_g_FF[:, -1, :, :, :].detach()

        # Accumulate spikes for loss computation (keep on GPU with gradients)
        loss_spikes.append(chunk_s)

        # Check if we're in the window before a checkpoint (last plot_size epochs before a checkpoint_interval)
        epochs_until_checkpoint = training.checkpoint_interval - (
            (epoch + 1) % training.checkpoint_interval
        )
        if epochs_until_checkpoint == training.checkpoint_interval:
            epochs_until_checkpoint = 0  # We're at a checkpoint point
        should_store_for_plot = epochs_until_checkpoint < training.plot_size

        # Accumulate data for plotting only when in the window before a checkpoint
        if should_store_for_plot:
            # Use pinned memory for faster transfer to CPU
            plot_spikes.append(chunk_s.detach().cpu().pin_memory().numpy())
            plot_voltages.append(chunk_v.detach().cpu().pin_memory().numpy())
            plot_currents.append(chunk_I.detach().cpu().pin_memory().numpy())
            plot_currents_FF.append(chunk_I_FF.detach().cpu().pin_memory().numpy())
            plot_conductances.append(chunk_g.detach().cpu().pin_memory().numpy())
            plot_conductances_FF.append(chunk_g_FF.detach().cpu().pin_memory().numpy())
            plot_input_spikes.append(input_spikes.detach().cpu().pin_memory().numpy())

        # Delete chunk tensors to free GPU memory immediately
        del chunk_v, chunk_I, chunk_I_FF, chunk_g, chunk_g_FF

        # Compute loss every chunk
        # Indicate gradient computation is starting
        pbar.set_description("Training [Computing gradients...]")

        # Concatenate accumulated chunks for loss computation
        full_spikes = torch.cat(
            loss_spikes, dim=1
        )  # (batch, accumulated_chunks * n_steps, n_neurons)

        # Compute loss over full trajectory with mixed precision
        with autocast(
            device_type="cuda",
            enabled=training.mixed_precision and device == "cuda",
        ):
            cv_loss = cv_loss_fn(full_spikes)
            fr_loss = firing_rate_loss_fn(full_spikes)
            total_loss = (
                hyperparameters.loss_ratio * fr_loss
                + (1 - hyperparameters.loss_ratio) * cv_loss
            )

            # Scale loss by number of loss computations per update (so gradient magnitude is consistent)
            total_loss = total_loss / training.losses_per_update

        # Backward pass with gradient scaling (accumulates gradients)
        scaler.scale(total_loss).backward()

        # Reset description after gradient computation
        pbar.set_description("Training")

        # Convert to numpy for logging (detach first)
        cv_loss_np = cv_loss.detach().cpu().item()
        fr_loss_np = fr_loss.detach().cpu().item()
        total_loss_np = total_loss.detach().cpu().item()

        # Delete loss tensors and clear loss_spikes accumulator to free GPU memory
        del cv_loss, fr_loss, total_loss, full_spikes
        loss_spikes.clear()

        # Update progress bar with loss information
        pbar.set_postfix(
            {
                "CV": f"{cv_loss_np:.4f}",
                "FR": f"{fr_loss_np:.4f}",
                "Total": f"{total_loss_np:.4f}",
            }
        )

        # Check if it's time to update weights
        if (epoch + 1) % training.losses_per_update == 0:
            # Indicate weight update is starting
            pbar.set_description("Training [Updating weights...]")

            # Unscale gradients before clipping
            scaler.unscale_(optimiser)

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Update weights with scaled gradients
            scaler.step(optimiser)
            scaler.update()
            optimiser.zero_grad(set_to_none=True)

            # Clear CUDA cache after weight update to release orphaned memory
            if device == "cuda":
                torch.cuda.empty_cache()

            # Brief indication that weights were updated
            pbar.set_description("Training [✓ Weights updated]")

            # Reset back to normal after a moment
            pbar.set_description("Training")

        # Log metrics to CSV and wandb periodically
        if (epoch + 1) % training.log_interval == 0:
            # Compute statistics on most recent chunk for logging
            stats = homeostatic_plots.compute_network_statistics(
                spikes=chunk_s.detach().cpu().numpy(),
                cell_type_indices=cell_type_indices,
                cell_type_names=recurrent.cell_types.names,
                dt=simulation.dt,
            )

            # Log all metrics
            metrics_logger.log(
                epoch=epoch + 1,
                cv_loss=cv_loss_np,
                fr_loss=fr_loss_np,
                total_loss=total_loss_np,
                **stats,
            )

            # Log to wandb (scalars only, no plots)
            if wandb_config and wandb_config.get("enabled", False):
                wandb.log(
                    {
                        "loss/cv": cv_loss_np,
                        "loss/firing_rate": fr_loss_np,
                        "loss/total": total_loss_np,
                        **stats,
                    },
                    step=epoch + 1,
                )

        # Save checkpoint and generate plots periodically
        if (epoch + 1) % training.checkpoint_interval == 0:
            # Clear progress bar and print checkpoint header
            pbar.clear()
            print(f"\n{'=' * 60}")
            print(f"Checkpoint at chunk {epoch + 1}/{simulation.num_chunks}")
            print("=" * 60)

            # Concatenate all accumulated data for visualization and checkpointing
            vis_spikes = np.concatenate(plot_spikes, axis=1)
            vis_voltages = np.concatenate(plot_voltages, axis=1)
            vis_currents = np.concatenate(plot_currents, axis=1)
            vis_currents_FF = np.concatenate(plot_currents_FF, axis=1)
            vis_conductances = np.concatenate(plot_conductances, axis=1)
            vis_conductances_FF = np.concatenate(plot_conductances_FF, axis=1)
            vis_input_spikes = np.concatenate(plot_input_spikes, axis=1)

            # Save checkpoint
            is_best = save_checkpoint(
                output_dir=output_dir,
                epoch=epoch + 1,
                model=model,
                optimiser=optimiser,
                scaler=scaler,
                initial_v=initial_v.cpu().pin_memory().numpy(),
                initial_g=initial_g.cpu().pin_memory().numpy(),
                initial_g_FF=initial_g_FF.cpu().pin_memory().numpy(),
                input_spikes=vis_input_spikes,
                cv_loss=cv_loss_np,
                fr_loss=fr_loss_np,
                total_loss=total_loss_np,
                best_loss=best_loss,
            )
            if is_best:
                best_loss = total_loss_np

            # Compute statistics on accumulated trajectory for display
            stats = homeostatic_plots.compute_network_statistics(
                spikes=vis_spikes,
                cell_type_indices=cell_type_indices,
                cell_type_names=recurrent.cell_types.names,
                dt=simulation.dt,
            )

            # Log to console
            print(f"  CV Loss: {cv_loss_np:.6f}")
            print(f"  FR Loss: {fr_loss_np:.6f}")
            print(f"  Total Loss: {total_loss_np:.6f}")
            # Print statistics by cell type
            for cell_type_name in recurrent.cell_types.names:
                mean_fr = stats[f"firing_rate/{cell_type_name}/mean"]
                mean_cv = stats[f"cv/{cell_type_name}/mean"]
                frac_active = stats[f"fraction_active/{cell_type_name}"]
                print(
                    f"  {cell_type_name}: FR={mean_fr:.3f} Hz, CV={mean_cv:.3f}, active={frac_active:.3f}"
                )

            # Generate and log plots
            print("  Generating plots...")
            figures_dir = plots_output_dir / "figures" / f"chunk_{epoch + 1:06d}"
            figures_dir.mkdir(parents=True, exist_ok=True)

            figures = homeostatic_plots.generate_training_plots(
                spikes=vis_spikes,
                voltages=vis_voltages,
                conductances=vis_conductances,
                conductances_FF=vis_conductances_FF,
                currents=vis_currents,
                currents_FF=vis_currents_FF,
                input_spikes=vis_input_spikes,
                cell_type_indices=cell_type_indices,
                input_cell_type_indices=input_source_indices,
                cell_type_names=recurrent.cell_types.names,
                input_cell_type_names=feedforward.cell_types.names,
                weights=weights,
                feedforward_weights=feedforward_weights,
                connectivity_graph=connectivity_graph,
                num_assemblies=recurrent.topology.num_assemblies,
                params=params.model_dump(),
                dt=simulation.dt,
            )

            # Log plots to wandb
            if wandb_config and wandb_config.get("enabled", False):
                # Create wandb Images directly from matplotlib figures
                wandb_plots = {
                    f"plots/{plot_name}": wandb.Image(fig)
                    for plot_name, fig in figures.items()
                }

                # Log plots at the checkpoint step
                wandb.log(wandb_plots, step=epoch + 1)

            # Save figures to disk after wandb has processed them
            for plot_name, fig in figures.items():
                fig_path = figures_dir / f"{plot_name}.png"
                fig.savefig(fig_path, dpi=300, bbox_inches="tight")
                plt.close(fig)

            print(f"  ✓ Saved plots to {figures_dir}")
            print("=" * 60)

            # Clear plot lists to free memory
            plot_spikes.clear()
            plot_voltages.clear()
            plot_currents.clear()
            plot_currents_FF.clear()
            plot_conductances.clear()
            plot_conductances_FF.clear()
            plot_input_spikes.clear()

            # Force garbage collection and clear CUDA cache to ensure memory is freed
            import gc

            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()

            # Refresh the progress bar after checkpoint output
            pbar.refresh()
            pbar.set_description("Training")

    # Close progress bar
    pbar.close()

    # =============================
    # Clean Up
    # =============================

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)

    # Finish wandb run
    if wandb_config and wandb_config.get("enabled", False):
        wandb.finish()

    # Close async logger and flush all remaining data
    print("  Flushing metrics to disk...")
    metrics_logger.close()

    print(f"✓ Checkpoints saved to {output_dir / 'checkpoints'}")
    print(f"✓ Metrics saved to {output_dir / 'metrics'}")
    print(f"✓ Best loss achieved: {best_loss:.6f}")
