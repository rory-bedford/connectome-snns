"""Neuronal dynamics visualization functions."""

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
import torch
from typing import Union


def plot_membrane_voltages(
    voltages: Union[NDArray[np.float32], torch.Tensor],
    spikes: Union[NDArray[np.int32], torch.Tensor],
    neuron_types: Union[NDArray[np.int32], torch.Tensor],
    delta_t: float,
    duration: float,
    neuron_params: dict,
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
        voltages (Union[NDArray[np.float32], torch.Tensor]): Voltage array with shape (batch, time, neurons).
        spikes (Union[NDArray[np.int32], torch.Tensor]): Spike array with shape (batch, time, neurons).
        neuron_types (Union[NDArray[np.int32], torch.Tensor]): Array indicating neuron type indices (0, 1, 2, ...).
        delta_t (float): Time step in milliseconds.
        duration (float): Total duration in milliseconds.
        neuron_params (dict): Dictionary mapping cell type indices to parameters
            {'threshold': float, 'rest': float, 'name': str, 'sign': int}.
        n_neurons_plot (int): Number of neurons to plot. Defaults to 10.
        fraction (float): Fraction of duration to plot (0-1). Defaults to 1.0.
        y_min (float): Minimum y-axis value in mV. Defaults to -100.0.
        y_max (float): Maximum y-axis value in mV. Defaults to 0.0.
        y_tick_step (float): Step size for y-axis ticks. Defaults to 50.0.
        figsize (tuple[float, float]): Figure size. Defaults to (12, 12).
        save_path (str | None): Path to save the figure. If None, figure is not saved. Defaults to None.
    """
    # Convert PyTorch tensors to NumPy arrays if needed
    if isinstance(voltages, torch.Tensor):
        voltages = voltages.detach().cpu().numpy()
    if isinstance(spikes, torch.Tensor):
        spikes = spikes.detach().cpu().numpy()
    if isinstance(neuron_types, torch.Tensor):
        neuron_types = neuron_types.detach().cpu().numpy()

    n_steps = voltages.shape[1]
    n_steps_plot = int(n_steps * fraction)

    y_ticks = np.arange(y_min, y_max + 1, y_tick_step)

    fig, axes = plt.subplots(n_neurons_plot, 1, figsize=figsize, sharex=True)
    time_axis = np.arange(n_steps_plot) * delta_t * 1e-3  # Convert to seconds

    for neuron_id in range(n_neurons_plot):
        voltage_trace = voltages[0, :n_steps_plot, neuron_id]
        spike_times_neuron = np.where(spikes[0, :n_steps_plot, neuron_id])[0]

        # Plot voltage trace
        axes[neuron_id].plot(time_axis, voltage_trace, linewidth=0.5, color="black")

        # Get neuron-specific parameters
        cell_type_idx = neuron_types[neuron_id]
        params = neuron_params[cell_type_idx]
        threshold = params["threshold"]
        rest = params["rest"]
        cell_name = params["name"]

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
                    linewidth=0.5,
                    alpha=0.7,
                    zorder=5,
                )

        # Create ylabel with cell type info
        ylabel = f"Neuron {neuron_id} ({cell_name})\nVoltage (mV)"
        axes[neuron_id].set_ylabel(ylabel, fontsize=9)
        axes[neuron_id].set_xlim(0, duration * 1e-3 * fraction)
        axes[neuron_id].set_ylim(y_min, y_max)
        axes[neuron_id].set_yticks(y_ticks)
        axes[neuron_id].grid(True, alpha=0.3)

        # Add legend to first subplot only
        if neuron_id == 0:
            axes[neuron_id].legend(loc="upper right", fontsize=7)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(
        f"Membrane Potential Traces (First {n_neurons_plot} Neurons)", fontsize=12
    )
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=600, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_synaptic_currents(
    I_exc: Union[NDArray[np.float32], torch.Tensor],
    I_inh: Union[NDArray[np.float32], torch.Tensor],
    delta_t: float,
    duration: float,
    n_neurons_plot: int = 10,
    fraction: float = 1.0,
    show_total: bool = False,
    neuron_types: Union[NDArray[np.int32], torch.Tensor, None] = None,
    neuron_params: dict | None = None,
    figsize: tuple[float, float] = (12, 12),
    save_path: str | None = None,
) -> None:
    """
    Visualize excitatory and inhibitory synaptic currents.

    Args:
        I_exc (Union[NDArray[np.float32], torch.Tensor]): Excitatory current array with shape (batch, time, neurons).
        I_inh (Union[NDArray[np.float32], torch.Tensor]): Inhibitory current array with shape (batch, time, neurons).
        delta_t (float): Time step in milliseconds.
        duration (float): Total duration in milliseconds.
        n_neurons_plot (int): Number of neurons to plot. Defaults to 10.
        fraction (float): Fraction of duration to plot (0-1). Defaults to 1.0.
        show_total (bool): Whether to show total current trace in grey. Defaults to False.
        neuron_types (Union[NDArray[np.int32], torch.Tensor, None]): Array indicating neuron type indices. Defaults to None.
        neuron_params (dict | None): Dictionary mapping cell type indices to parameters. Defaults to None.
        figsize (tuple[float, float]): Figure size. Defaults to (12, 12).
        save_path (str | None): Path to save the figure. If None, figure is not saved. Defaults to None.
    """
    # Convert PyTorch tensors to NumPy arrays if needed
    if isinstance(I_exc, torch.Tensor):
        I_exc = I_exc.detach().cpu().numpy()
    if isinstance(I_inh, torch.Tensor):
        I_inh = I_inh.detach().cpu().numpy()
    if isinstance(neuron_types, torch.Tensor):
        neuron_types = neuron_types.detach().cpu().numpy()

    n_steps = I_exc.shape[1]
    n_steps_plot = int(n_steps * fraction)

    # Fixed y-axis limits at ±1000 pA
    y_lim = 1000.0

    fig, axes = plt.subplots(n_neurons_plot, 1, figsize=figsize, sharex=True)
    time_axis = np.arange(n_steps_plot) * delta_t * 1e-3  # Convert to seconds

    for neuron_id in range(n_neurons_plot):
        # Extract excitatory and inhibitory currents for this neuron
        I_exc_trace = I_exc[0, :n_steps_plot, neuron_id]
        I_inh_trace = I_inh[0, :n_steps_plot, neuron_id]
        I_total_trace = I_exc_trace + I_inh_trace

        # Compute mean total current over full simulation (not just plotted portion)
        I_total_full = I_exc[0, :, neuron_id] + I_inh[0, :, neuron_id]
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

        # Add ylabel with cell type info
        cell_type_idx = neuron_types[neuron_id] if neuron_types is not None else None
        if cell_type_idx is not None and neuron_params is not None:
            cell_name = neuron_params[cell_type_idx]["name"]
            ylabel = f"Neuron {neuron_id} ({cell_name})\nCurrent (pA)"
        else:
            ylabel = f"Neuron {neuron_id}\nCurrent (pA)"

        axes[neuron_id].set_ylabel(ylabel, fontsize=9)
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
        plt.savefig(save_path, dpi=600, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
