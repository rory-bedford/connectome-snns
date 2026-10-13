"""Private helpers shared by the plotting modules.

Nothing here is part of the public API. The helpers cover the boilerplate every plot
function repeats: resolving the ``ax`` argument into a figure and axes, finishing a
figure, the per-cell-type colour palette, splitting synapse types by sign, and a few
axis-styling idioms.
"""

from collections.abc import Sequence

import matplotlib.pyplot as plt

from connectome_snns.configs.conductance_based import (
    EXCITATORY_SYNAPSE_TYPES,
    INHIBITORY_SYNAPSE_TYPES,
)

from .colors import (
    EXCITATORY_COLOR,
    EXTRA_CELL_TYPE_CMAP,
    FEEDFORWARD_COLOR,
    INHIBITORY_COLOR,
)

#: Small extension past the last time point so the final x tick stays visible.
TIME_AXIS_PAD_S = 0.01


# ── Figure / axes resolution ─────────────────────────────────────────────────


def figure_and_axes(
    ax: plt.Axes | None, figsize: tuple[float, float]
) -> tuple[plt.Figure, plt.Axes, bool]:
    """Return ``(fig, ax, owns_figure)`` for a single-axes plot.

    Creates a new figure when ``ax`` is None (``owns_figure`` is then True),
    otherwise draws on ``ax`` and its figure.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        return fig, ax, True
    return ax.get_figure(), ax, False


def figure_and_axes_grid(
    ax: plt.Axes | Sequence[plt.Axes] | None,
    n_panels: int,
    *,
    vertical: bool,
    figsize: tuple[float, float],
    what: str = "",
) -> tuple[plt.Figure, Sequence[plt.Axes], bool]:
    """Return ``(fig, axes, owns_figure)`` for a plot with one panel per item.

    Panels are stacked vertically with a shared x axis (``vertical=True``) or laid out
    in a row with a shared y axis. ``ax`` may be:

    - None: a new figure is created and returned (``owns_figure`` is True);
    - a list of exactly ``n_panels`` axes to draw on;
    - a single Axes: kept for backward compatibility -- a new figure is created but
      ``owns_figure`` is False, so the caller does not return it.

    Args:
        ax: See above.
        n_panels: Number of panels needed.
        vertical: Stack panels vertically (sharex) instead of in a row (sharey).
        figsize: Size of a newly created figure.
        what: Optional suffix for the error message, e.g. ``" (E, I, FF)"``.
    """
    if isinstance(ax, list):
        if len(ax) != n_panels:
            raise ValueError(f"Expected {n_panels} axes{what}, got {len(ax)}")
        return ax[0].get_figure(), ax, False

    if vertical:
        fig, axes = plt.subplots(n_panels, 1, figsize=figsize, sharex=True)
    else:
        fig, axes = plt.subplots(1, n_panels, figsize=figsize, sharey=True)
    if n_panels == 1:
        axes = [axes]
    return fig, axes, ax is None


def finish_figure(fig: plt.Figure, owns_figure: bool) -> plt.Figure | None:
    """Tight-layout and return ``fig`` if the plot created it, else return None.

    Uses ``plt.tight_layout()`` (the current figure), as the plot functions always
    have; the figure the plot just created is the current one.
    """
    if owns_figure:
        plt.tight_layout()
        return fig
    return None


# ── Colours ──────────────────────────────────────────────────────────────────


def cell_type_palette(n_cell_types: int) -> list:
    """One colour per recurrent cell type, indexed by cell-type index.

    Excitatory red and inhibitory blue first, then ``tab10`` colours for any further
    types.
    """
    base_colors = [EXCITATORY_COLOR, INHIBITORY_COLOR]
    if n_cell_types <= len(base_colors):
        return base_colors[:n_cell_types]
    cmap = plt.colormaps[EXTRA_CELL_TYPE_CMAP]
    return base_colors + [cmap(i) for i in range(n_cell_types - len(base_colors))]


def raster_color_map(n_recurrent_types: int) -> dict[int, object]:
    """Cell-type index -> colour for rasters: ``-1`` (feedforward) is gray."""
    colors_map: dict[int, object] = {-1: FEEDFORWARD_COLOR}
    for i, color in enumerate(cell_type_palette(n_recurrent_types)):
        colors_map[i] = color
    return colors_map


# ── Synapses ─────────────────────────────────────────────────────────────────


def split_synapses_by_sign(
    cell_type_names: Sequence[str], synapse_names: dict[str, list[str]]
) -> tuple[list[tuple[int, str]], list[tuple[int, str]]]:
    """Split the flattened synapse axis into excitatory and inhibitory synapses.

    Synapse arrays are ordered by presynaptic cell type, then by that type's synapse
    names. Returns ``(excitatory, inhibitory)`` lists of ``(synapse_index, name)``;
    synapses of neither sign are skipped but still advance the index.
    """
    excitatory, inhibitory = [], []
    synapse_idx = 0
    for cell_name in cell_type_names:
        for syn_name in synapse_names[cell_name]:
            if syn_name in EXCITATORY_SYNAPSE_TYPES:
                excitatory.append((synapse_idx, syn_name))
            elif syn_name in INHIBITORY_SYNAPSE_TYPES:
                inhibitory.append((synapse_idx, syn_name))
            synapse_idx += 1
    return excitatory, inhibitory


# ── Axis styling ─────────────────────────────────────────────────────────────


def set_time_xlim(axes: Sequence[plt.Axes], start_s: float, end_s: float) -> None:
    """Tight x limits over ``[start_s, end_s]`` with a small pad for the last tick."""
    for ax in axes:
        ax.set_xlim(start_s, end_s + TIME_AXIS_PAD_S)
        ax.margins(x=0)


def floating_legend(ax: plt.Axes, *args) -> None:
    """Legend floating just above the top-right corner of ``ax``."""
    ax.legend(
        *args,
        loc="lower right",
        bbox_to_anchor=(1.0, 1.0),
        fancybox=True,
        facecolor="white",
        edgecolor="gray",
        framealpha=0.9,
    )


def color_tick_labels(tick_labels, colors) -> None:
    """Colour each tick label and make it bold."""
    for tick_label, color in zip(tick_labels, colors):
        tick_label.set_color(color)
        tick_label.set_fontweight("bold")


def centered_message(ax: plt.Axes, text: str) -> None:
    """Write ``text`` in the middle of an (empty) axes."""
    ax.text(0.5, 0.5, text, ha="center", va="center", transform=ax.transAxes)


def title_above_axes(
    fig: plt.Figure, axes: Sequence[plt.Axes], text: str, pad: float = 0.01, **kwargs
) -> None:
    """Figure text centred above a row of axes (from the first to the last)."""
    first, last = axes[0].get_position(), axes[-1].get_position()
    fig.text(
        (first.x0 + last.x1) / 2,
        first.y1 + pad,
        text,
        ha="center",
        va="bottom",
        **kwargs,
    )
