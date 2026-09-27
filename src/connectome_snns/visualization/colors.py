"""Canonical colours used across all visualization modules."""

# ── Cell-type colours ────────────────────────────────────────────────────────

EXCITATORY_COLOR = "#CC3333"  # muted red
INHIBITORY_COLOR = "#3366AA"  # muted blue
FEEDFORWARD_COLOR = "#808080"  # gray
CELL_TYPE_COLORS = [EXCITATORY_COLOR, INHIBITORY_COLOR]  # indexed by cell_type_indices

# ── Synapse-type colours ─────────────────────────────────────────────────────

AMPA_COLOR = "#CC3333"  # muted red (matches excitatory)
NMDA_COLOR = "#339955"  # muted green
GABA_A_COLOR = "#3366AA"  # muted blue (matches inhibitory)
GABA_B_COLOR = "#884499"  # muted purple
SYNAPSE_COLORS = {
    "AMPA": AMPA_COLOR,
    "NMDA": NMDA_COLOR,
    "GABA_A": GABA_A_COLOR,
    "GABA_B": GABA_B_COLOR,
}

# ── Teacher / student comparison colours ─────────────────────────────────────

TEACHER_COLOR = "#333333"  # dark gray
STUDENT_COLOR = "#5588cc"  # muted blue
NEUTRAL_BAR_COLOR = "#888888"  # medium gray (for teacher bars in bar charts)
RASTER_BAND_COLOR = "#eeeeee"  # very light gray (alternating raster bands)

# ── Training phase colours (fully-observed experiment) ───────────────────────

CMA_PHASE_COLOR = "#e8e8f0"  # pale lavender (background shading)
CMA_LOSS_COLOR = "#5555cc"  # purple-blue (CMA-ES loss curve)
GRADIENT_LOSS_COLOR = "#cc5555"  # dusty rose (gradient loss curve)

# ── Strategy colours (inferring-inputs experiment) ───────────────────────────

STRATEGY_COLORS = {
    "ou-rates": "#5588cc",  # muted blue
    "uniform-inputs": "#cc7744",  # warm rust
}

# ── Highlight colours ────────────────────────────────────────────────────────

HIGHLIGHT_COLOR = "#e74c3c"  # vivid red (active odourant, emphasis)
SCATTER_NEUTRAL_COLOR = "#555555"  # dark gray (neutral scatter points)
BASELINE_BAR_COLOR = "steelblue"  # baseline/default bars

# ── Qualitative palette for grid searches ────────────────────────────────────
# Colorblind-friendly, visually distinctive (Paul Tol's qualitative scheme).

QUALITATIVE_COLORS = [
    "#4477AA",  # blue
    "#EE6677",  # rose
    "#228833",  # green
    "#CCBB44",  # yellow
    "#66CCEE",  # cyan
    "#AA3377",  # magenta
    "#BBBBBB",  # grey
    "#EE8866",  # orange
]

# ── Paper figure palette ─────────────────────────────────────────────────────
# Colours used in the teacher-student paper figures.

FIGURE_BLUE = "#457b9d"  # steel blue
FIGURE_ORANGE = "#e07b2e"  # orange
FIGURE_TEAL = "#2a9d8f"  # teal
FIGURE_CORAL = "#e76f51"  # coral
FIGURE_COLORS = [FIGURE_BLUE, FIGURE_ORANGE, FIGURE_TEAL, FIGURE_CORAL]

# ── Bernstein talk figures: one fixed colour per condition, never reassigned ──

FULL_CONNECTOME_COLOR = FIGURE_BLUE
LEARNT_RECURRENCE_COLOR = FIGURE_ORANGE
SHUFFLE_WEIGHTS_COLOR = FIGURE_TEAL
CONFIGURATION_MODEL_COLOR = "#8e6bb3"  # muted violet
FIXED_TOPOLOGY_COLOR = "#a6761d"  # ochre brown
OBSERVED_COLOR = "#1d3557"  # navy
UNOBSERVED_COLOR = "#e9a23b"  # amber
FLOOR_COLOR = "#9e9e9e"  # grey
NEURON_REMOVAL_COLOR = "#b5446e"  # raspberry
SYNAPSE_DROPOUT_COLOR = "#3c8d5a"  # green
WEIGHT_NOISE_COLOR = FIGURE_CORAL

# ── Talk slide scheme (teacher-student/COLORSCHEME.txt) ──────────────────────
# The deck's own palette. Used for the roles that recur across its slides, so a colour
# means one thing throughout: cell type in the scatters, and teacher vs student wherever
# the two are drawn together.
SLIDE_INK = "#263640"  # dark slate: the deck's background, used here for ground truth
SLIDE_BLUE = "#5096c0"  # accent blue
SLIDE_RED = "#e87878"  # accent red
SLIDE_YELLOW = "#e9c46a"  # accent yellow
#: The teacher's dimensionality marked on a sweep -- its own colour, so it is never
#: confused with SHUFFLE_WEIGHTS_COLOR (both were teal until 2026-09-21).
DIMENSIONALITY_COLOR = "#7d8597"  # slate grey
