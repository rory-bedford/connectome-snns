"""Visualization functions for odour-related input patterns and activity."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from scipy import stats


def plot_input_firing_rate_histogram(
    firing_rates: NDArray[np.float64],
    ax: plt.Axes | None = None,
    bins: int = 30,
) -> Figure | None:
    """
    Plot histogram of firing rates for input neurons across odour patterns.

    Displays all firing rate values as a histogram with fraction (density) on the y-axis.

    Args:
        firing_rates (NDArray[np.float64]): Firing rates in Hz with shape
            (batch_size, n_patterns, n_input_neurons) or (n_patterns, n_input_neurons).
        ax (plt.Axes | None): Matplotlib axes to plot on. If None, creates new figure.
        bins (int): Number of histogram bins. Defaults to 30.

    Returns:
        Figure | None: Matplotlib figure object if ax is None, otherwise None.
    """
    # Handle both 2D and 3D input
    if firing_rates.ndim == 3:
        # Average over batch dimension: (batch_size, n_patterns, n_input_neurons) -> (n_patterns, n_input_neurons)
        firing_rates = firing_rates.mean(axis=0)

    # Extract dimensions
    n_patterns, n_input_neurons = firing_rates.shape

    # Flatten to get all firing rate values for histogram
    # Shape: (n_patterns * n_input_neurons,)
    firing_rates_flat = firing_rates.flatten()

    # Create figure if no axis provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        return_fig = True
    else:
        fig = ax.get_figure()
        return_fig = False

    # Plot histogram with density=True for fraction on y-axis
    ax.hist(
        firing_rates_flat,
        bins=bins,
        alpha=0.7,
        color="#1f77b4",
        edgecolor="black",
        linewidth=1.2,
        density=True,
    )

    # Add mean line
    mean_rate = firing_rates_flat.mean()
    ax.axvline(
        mean_rate,
        color="red",
        linestyle="--",
        linewidth=2,
        alpha=0.7,
        label=f"Mean = {mean_rate:.2f} Hz",
    )

    # Formatting
    ax.set_xlabel("Firing Rate (Hz)", fontsize=12)
    ax.set_ylabel("Fraction (Density)", fontsize=12)
    ax.set_title(
        f"Input Neuron Firing Rate Distribution\n({n_input_neurons} neurons × {n_patterns} patterns)",
        fontsize=13,
    )
    ax.tick_params(labelsize=10)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    if return_fig:
        plt.tight_layout()

    return fig if return_fig else None


def plot_firing_rate_variance_comparison(
    output_firing_rates: NDArray[np.float64],
    output_firing_rates_control: NDArray[np.float64],
    ax: plt.Axes | None = None,
    n_spaghetti: int = 50,
) -> Figure | None:
    """
    Compare standard deviation in neuronal firing rates between odour-modulated and control conditions.

    For each neuron, computes the standard deviation in firing rates across all trials (odours × repeats)
    and compares it between the experimental (odour-modulated) and control (baseline) conditions.
    Displays as violin plots with paired spaghetti lines connecting the same neurons.

    Args:
        output_firing_rates (NDArray[np.float64]): Firing rates for main experiment with shape
            (batch_size, n_patterns, n_neurons). Multiple patterns (odours).
        output_firing_rates_control (NDArray[np.float64]): Firing rates for control experiment
            with shape (batch_size, 1, n_neurons). Single baseline pattern.
        ax (plt.Axes | None): Matplotlib axes to plot on. If None, creates new figure.
        n_spaghetti (int): Number of randomly sampled neurons to show as spaghetti lines.
            Defaults to 50.

    Returns:
        Figure | None: Matplotlib figure object if ax is None, otherwise None.

    Notes:
        - Control: std over batch dimension measures trial-to-trial noise with fixed input
        - Odour: std over patterns per batch (signal variance), averaged across batches
        - This compares noise variance vs signal variance (how much responses differ across inputs)
        - Higher std in odour condition indicates neurons respond differently to different inputs
        - Spaghetti lines show paired data (same neuron in both conditions)
    """
    # Control: std over batch dimension (noise variance with fixed baseline input)
    # (batch_size, 1, n_neurons) -> (batch_size, n_neurons) -> std over axis 0 -> (n_neurons,)
    std_control = np.std(output_firing_rates_control[:, 0, :], axis=0)

    # Odour: std over pattern dimension for each batch, then average across batches
    # (batch_size, n_patterns, n_neurons) -> std over axis 1 -> (batch_size, n_neurons) -> mean over axis 0 -> (n_neurons,)
    std_per_batch = np.std(output_firing_rates, axis=1)  # (batch_size, n_neurons)
    std_main = np.mean(std_per_batch, axis=0)  # (n_neurons,)

    # Compute mean firing rates (average over batch and pattern dimensions)
    mean_main = np.mean(output_firing_rates, axis=(0, 1))  # (n_neurons,)
    mean_control = np.mean(output_firing_rates_control, axis=(0, 1))  # (n_neurons,)

    n_neurons = std_main.shape[0]

    # Statistical tests: Paired t-test
    # Compare per-neuron statistics between conditions
    stat_std, p_std = stats.ttest_rel(std_control, std_main, alternative="two-sided")
    stat_mean, p_mean = stats.ttest_rel(
        mean_control, mean_main, alternative="two-sided"
    )

    # Helper function to format p-values with significance stars
    def format_pvalue(p):
        # Format p-value in scientific notation if very small
        if p < 1e-10:
            p_str = "p < 1e-10"
            stars = "***"
        elif p < 0.001:
            p_str = f"p = {p:.2e}"
            stars = "***"
        elif p < 0.01:
            p_str = f"p = {p:.4f}"
            stars = "**"
        elif p < 0.05:
            p_str = f"p = {p:.4f}"
            stars = "*"
        else:
            p_str = f"p = {p:.4f}"
            stars = "n.s."
        return f"{p_str} (paired t-test)", stars

    # Create figure if no axis provided
    if ax is None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        return_fig = True
    else:
        # If single axis provided, we need two axes - ignore provided ax
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        return_fig = False

    # Prepare data for violin plots (control on left)
    data_std = [std_control, std_main]
    data_mean = [mean_control, mean_main]
    positions = [1, 2]
    labels = ["Control (Baseline)", "Odour-Modulated"]

    # Sample neurons once for both plots
    if n_spaghetti > 0 and n_neurons > 0:
        n_sample = min(n_spaghetti, n_neurons)
        sampled_indices = np.random.choice(n_neurons, size=n_sample, replace=False)
    else:
        sampled_indices = []

    # Color scheme (orange for control, blue for main)
    colors = ["#ff7f0e", "#1f77b4"]

    # ========== LEFT PLOT: Standard Deviation ==========
    ax_std = axes[0]

    # Create violin plots for standard deviation
    parts_std = ax_std.violinplot(
        data_std,
        positions=positions,
        widths=0.6,
        showmeans=False,
        showextrema=False,
    )

    # Color the violins
    for pc, color in zip(parts_std["bodies"], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.6)
        pc.set_edgecolor("black")
        pc.set_linewidth(1.5)

    # Style the violin plot elements
    for partname in ("cbars", "cmins", "cmaxes", "cmeans"):
        if partname in parts_std:
            vp = parts_std[partname]
            vp.set_edgecolor("black")
            vp.set_linewidth(1.5)

    # Add spaghetti lines for sampled neurons
    for idx in sampled_indices:
        ax_std.plot(
            positions,
            [std_control[idx], std_main[idx]],
            color="gray",
            alpha=0.5,
            linewidth=0.5,
            zorder=1,
        )

    # Formatting
    ax_std.set_xticks(positions)
    ax_std.set_xticklabels(labels, fontsize=11)
    ax_std.set_ylabel("Standard Deviation in Firing Rate (Hz)", fontsize=12)
    ax_std.set_title(
        f"Response Variability (Std Dev)\n({n_neurons} neurons, {len(sampled_indices)} paired samples)",
        fontsize=13,
    )
    ax_std.tick_params(labelsize=10)
    ax_std.grid(True, alpha=0.3, axis="y")

    # Add significance bracket and stars
    p_text_std, stars_std = format_pvalue(p_std)
    y_max = max(std_control.max(), std_main.max())
    y_bracket = y_max * 1.1
    # Lower the bracket for n.s. to avoid overlap
    if stars_std == "n.s.":
        y_bracket = y_max * 1.05
    ax_std.plot(
        [1, 1, 2, 2],
        [y_bracket, y_bracket * 1.02, y_bracket * 1.02, y_bracket],
        "k-",
        linewidth=1.5,
    )
    ax_std.text(1.5, y_bracket * 1.03, stars_std, ha="center", va="bottom", fontsize=14)

    # Add statistics text
    mean_std_control = std_control.mean()
    mean_std_main = std_main.mean()
    ax_std.text(
        0.98,
        0.02,
        f"Noise (fixed input): {mean_std_control:.2f} Hz\n"
        f"Signal (varied input): {mean_std_main:.2f} Hz\n"
        f"{p_text_std}",
        transform=ax_std.transAxes,
        verticalalignment="bottom",
        horizontalalignment="right",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # ========== RIGHT PLOT: Mean ==========
    ax_mean = axes[1]

    # Create violin plots for mean
    parts_mean = ax_mean.violinplot(
        data_mean,
        positions=positions,
        widths=0.6,
        showmeans=False,
        showextrema=False,
    )

    # Color the violins
    for pc, color in zip(parts_mean["bodies"], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.6)
        pc.set_edgecolor("black")
        pc.set_linewidth(1.5)

    # Style the violin plot elements
    for partname in ("cbars", "cmins", "cmaxes", "cmeans"):
        if partname in parts_mean:
            vp = parts_mean[partname]
            vp.set_edgecolor("black")
            vp.set_linewidth(1.5)

    # Add spaghetti lines for sampled neurons
    for idx in sampled_indices:
        ax_mean.plot(
            positions,
            [mean_control[idx], mean_main[idx]],
            color="gray",
            alpha=0.5,
            linewidth=0.5,
            zorder=1,
        )

    # Formatting
    ax_mean.set_xticks(positions)
    ax_mean.set_xticklabels(labels, fontsize=11)
    ax_mean.set_ylabel("Mean Firing Rate (Hz)", fontsize=12)
    ax_mean.set_title(
        f"Mean Firing Rate\n({n_neurons} neurons, {len(sampled_indices)} paired samples)",
        fontsize=13,
    )
    ax_mean.tick_params(labelsize=10)
    ax_mean.grid(True, alpha=0.3, axis="y")

    # Add significance bracket and stars
    p_text_mean, stars_mean = format_pvalue(p_mean)
    y_max_mean = max(mean_control.max(), mean_main.max())
    y_bracket_mean = y_max_mean * 1.1
    # Lower the bracket for n.s. to avoid overlap
    if stars_mean == "n.s.":
        y_bracket_mean = y_max_mean * 1.05
    ax_mean.plot(
        [1, 1, 2, 2],
        [y_bracket_mean, y_bracket_mean * 1.02, y_bracket_mean * 1.02, y_bracket_mean],
        "k-",
        linewidth=1.5,
    )
    ax_mean.text(
        1.5, y_bracket_mean * 1.03, stars_mean, ha="center", va="bottom", fontsize=14
    )

    # Add statistics text
    mean_mean_control = mean_control.mean()
    mean_mean_main = mean_main.mean()
    ax_mean.text(
        0.98,
        0.02,
        f"Mean (control): {mean_mean_control:.2f} Hz\n"
        f"Mean (odour-modulated): {mean_mean_main:.2f} Hz\n"
        f"{p_text_mean}",
        transform=ax_mean.transAxes,
        verticalalignment="bottom",
        horizontalalignment="right",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    if return_fig:
        plt.tight_layout()

    return fig if return_fig else None
