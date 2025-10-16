"""Neuronal dynamics visualization functions."""

import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy.typing import NDArray


def plot_membrane_voltages(
    voltages: torch.Tensor,
    spikes: torch.Tensor,
    neuron_types: NDArray[np.int32],
    model: torch.nn.Module,
    delta_t: float,
    duration: float,
    n_neurons_plot: int = 10,
    fraction: float = 1.0,
    y_min: float = -100.0,
    y_max: float = 0.0,
    y_tick_step: float = 50.0,
    figsize: tuple[float, float] = (12, 12),
    save_path: str | None = None,
) -> None:
    """
    Visualize membrane voltage traces with spike markers.

    Args:
        voltages (torch.Tensor): Voltage tensor with shape (batch, time, neurons).
        spikes (torch.Tensor): Spike tensor with shape (batch, time, neurons).
        neuron_types (NDArray[np.int32]): Array indicating neuron type (1 for excitatory, -1 for inhibitory).
        model (torch.nn.Module): The neural network model with threshold and rest parameters.
        delta_t (float): Time step in milliseconds.
        duration (float): Total duration in milliseconds.
        n_neurons_plot (int): Number of neurons to plot. Defaults to 10.
        fraction (float): Fraction of duration to plot (0-1). Defaults to 1.0.
        y_min (float): Minimum y-axis value in mV. Defaults to -100.0.
        y_max (float): Maximum y-axis value in mV. Defaults to 0.0.
        y_tick_step (float): Step size for y-axis ticks. Defaults to 50.0.
        figsize (tuple[float, float]): Figure size. Defaults to (12, 12).
        save_path (str | None): Path to save the figure. If None, figure is not saved. Defaults to None.
    """
    n_steps = voltages.shape[1]
    n_steps_plot = int(n_steps * fraction)

    y_ticks = np.arange(y_min, y_max + 1, y_tick_step)

    fig, axes = plt.subplots(n_neurons_plot, 1, figsize=figsize, sharex=True)
    time_axis = np.arange(n_steps_plot) * delta_t * 1e-3  # Convert to seconds

    for neuron_id in range(n_neurons_plot):
        voltage_trace = voltages[0, :n_steps_plot, neuron_id].detach().cpu().numpy()
        spike_times_neuron = np.where(
            spikes[0, :n_steps_plot, neuron_id].cpu().numpy()
        )[0]

        # Plot voltage trace
        axes[neuron_id].plot(time_axis, voltage_trace, linewidth=0.5, color="black")

        # Get neuron-specific parameters
        is_excitatory = neuron_types[neuron_id] == 1
        threshold = model.theta_E.item() if is_excitatory else model.theta_I.item()
        rest = model.U_rest_E.item() if is_excitatory else model.U_rest_I.item()

        # Add threshold and rest lines
        axes[neuron_id].axhline(
            y=threshold,
            color="#FF0000",
            linestyle="--",
            linewidth=0.8,
            alpha=0.6,
            label="Threshold",
        )
        axes[neuron_id].axhline(
            y=rest,
            color="#0000FF",
            linestyle="--",
            linewidth=0.8,
            alpha=0.6,
            label="Rest",
        )

        # Mark spike times with vertical lines from threshold to zero
        if len(spike_times_neuron) > 0:
            spike_times_s = spike_times_neuron * delta_t * 1e-3
            for spike_t in spike_times_s:
                axes[neuron_id].plot(
                    [spike_t, spike_t],
                    [threshold, 0],
                    color="black",
                    linewidth=1.5,
                    alpha=0.7,
                    zorder=5,
                )

        axes[neuron_id].set_ylabel(f"U_{neuron_id}", fontsize=8)
        axes[neuron_id].set_xlim(0, duration * 1e-3 * fraction)
        axes[neuron_id].set_ylim(y_min, y_max)
        axes[neuron_id].set_yticks(y_ticks)
        axes[neuron_id].grid(True, alpha=0.3)

        # Add legend to first subplot only
        if neuron_id == 0:
            axes[neuron_id].legend(loc="upper right", fontsize=7)

    axes[-1].set_xlabel("Time (s)")
    axes[0].set_ylabel("U_0 (mV)", fontsize=8)  # Add unit to first ylabel only
    fig.suptitle(
        f"Membrane Potential Traces (First {n_neurons_plot} Neurons)", fontsize=12
    )
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_synaptic_currents(
    I_exc: torch.Tensor,
    I_inh: torch.Tensor,
    delta_t: float,
    duration: float,
    n_neurons_plot: int = 10,
    fraction: float = 1.0,
    show_total: bool = False,
    figsize: tuple[float, float] = (12, 12),
    save_path: str | None = None,
) -> None:
    """
    Visualize excitatory and inhibitory synaptic currents.

    Args:
        I_exc (torch.Tensor): Excitatory current tensor with shape (batch, time, neurons).
        I_inh (torch.Tensor): Inhibitory current tensor with shape (batch, time, neurons).
        delta_t (float): Time step in milliseconds.
        duration (float): Total duration in milliseconds.
        n_neurons_plot (int): Number of neurons to plot. Defaults to 10.
        fraction (float): Fraction of duration to plot (0-1). Defaults to 1.0.
        show_total (bool): Whether to show total current trace in grey. Defaults to False.
        figsize (tuple[float, float]): Figure size. Defaults to (12, 12).
        save_path (str | None): Path to save the figure. If None, figure is not saved. Defaults to None.
    """
    n_steps = I_exc.shape[1]
    n_steps_plot = int(n_steps * fraction)

    # Calculate max absolute value across all neurons for consistent scaling
    max_current = 0
    for neuron_id in range(n_neurons_plot):
        I_exc_trace = I_exc[0, :n_steps_plot, neuron_id].detach().cpu().numpy()
        I_inh_trace = I_inh[0, :n_steps_plot, neuron_id].detach().cpu().numpy()
        max_current = max(
            max_current, np.abs(I_exc_trace).max(), np.abs(I_inh_trace).max()
        )

    # Round up to nearest 0.1
    y_lim = np.ceil(max_current / 0.1) * 0.1

    fig, axes = plt.subplots(n_neurons_plot, 1, figsize=figsize, sharex=True)
    time_axis = np.arange(n_steps_plot) * delta_t * 1e-3  # Convert to seconds

    for neuron_id in range(n_neurons_plot):
        # Extract excitatory and inhibitory currents for this neuron
        I_exc_trace = I_exc[0, :n_steps_plot, neuron_id].detach().cpu().numpy()
        I_inh_trace = I_inh[0, :n_steps_plot, neuron_id].detach().cpu().numpy()
        I_total_trace = I_exc_trace + I_inh_trace

        # Compute mean total current over full simulation (not just plotted portion)
        I_total_full = (
            (I_exc[0, :, neuron_id] + I_inh[0, :, neuron_id]).detach().cpu().numpy()
        )
        mean_total = I_total_full.mean()

        # Plot total current in grey (optional)
        if show_total:
            axes[neuron_id].plot(
                time_axis,
                I_total_trace,
                linewidth=0.8,
                color="gray",
                alpha=0.7,
                label="Total",
            )

        # Plot excitatory and inhibitory currents
        axes[neuron_id].plot(
            time_axis,
            I_exc_trace,
            linewidth=0.8,
            color="#0000FF",
            alpha=0.7,
            label="Excitatory",
        )
        axes[neuron_id].plot(
            time_axis,
            I_inh_trace,
            linewidth=0.8,
            color="#FF0000",
            alpha=0.7,
            label="Inhibitory",
        )

        # Add zero line
        axes[neuron_id].axhline(
            y=0, color="black", linestyle="-", linewidth=1.0, alpha=0.3
        )

        # Add horizontal line at mean total current
        axes[neuron_id].axhline(
            y=mean_total, color="black", linestyle="--", linewidth=0.8, alpha=0.5
        )

        # Add mean total current as text annotation on the right
        axes[neuron_id].text(
            1.04,
            0.5,
            f"mean current = {mean_total:.2f} pA",
            transform=axes[neuron_id].transAxes,
            fontsize=7,
            va="center",
            ha="left",
            color="black",
        )

        # Add units only to first ylabel
        ylabel = "I_0 (pA)" if neuron_id == 0 else f"I_{neuron_id}"
        axes[neuron_id].set_ylabel(ylabel, fontsize=8)
        axes[neuron_id].set_xlim(0, duration * 1e-3 * fraction)
        axes[neuron_id].set_ylim(-y_lim, y_lim)
        axes[neuron_id].grid(True, alpha=0.3)

        # Add legend to first subplot only
        if neuron_id == 0:
            axes[neuron_id].legend(loc="upper right", fontsize=7)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(
        f"Synaptic Current Inputs (First {n_neurons_plot} Neurons)", fontsize=12
    )
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
