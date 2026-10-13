"""Small seeded networks shared by the fixture generator and the tests.

Every network is rebuilt from its seed, so a fixture stores only what the simulator
*produced*, plus a fingerprint of the parameters it was recorded with. If a rebuilt
network does not match its fingerprint (numpy changed its random streams, or someone
edited a builder) the tests fail loudly instead of comparing against the wrong network.

The physiology is the Bernstein teacher's (``fig00-teacher-activity/parameters.toml``);
the networks are ~30x smaller, with connection strengths scaled up so both populations
are genuinely active: the recurrent net runs at E ~2 Hz, I ~21 Hz, against the teacher's
E 4.3 Hz, I 19 Hz (E cannot reach 4 Hz here without I saturating its GABA clip).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from torch import nn

from connectome_snns.network_simulators.conductance.recurrent import (
    ConductanceLIFNetwork,
)
from connectome_snns.network_simulators.current.recurrent import (
    CurrentLIFNetwork,
)
from connectome_snns.network_simulators.conductance.feedforward import (
    FeedforwardConductanceLIFNetwork,
)
from connectome_snns.network_simulators.current.feedforward import (
    FeedforwardCurrentLIFNetwork,
)
from connectome_snns.network_simulators.projections import (
    FullRankProjection,
    LowRankProjection,
    ScalingFactorProjection,
    make_frozen_projections,
)
from connectome_snns.network_simulators.two_layer import TwoLayerSNN

DT = 1.0  # ms

# =====================================================================
# Physiology (Bernstein teacher)
# =====================================================================

E, I = 0, 1
CELLS = [
    dict(name="excitatory", cell_id=E, tau_mem=85.0, theta=-38.0, U_reset=-60.0,
         E_L=-60.0, g_L=1.35),
    dict(name="inhibitory", cell_id=I, tau_mem=50.0, theta=-45.0, U_reset=-60.0,
         E_L=-60.0, g_L=0.9),
]  # fmt: skip
SYNAPSES = [
    dict(name="AMPA", synapse_id=0, cell_id=E, tau_rise=0.1, tau_decay=5.0,
         E_syn=0.0, g_bar=4.0, g_clip=40.0),
    dict(name="NMDA", synapse_id=1, cell_id=E, tau_rise=10.0, tau_decay=60.0,
         E_syn=0.0, g_bar=0.5, g_clip=4.0),
    dict(name="GABA_A", synapse_id=2, cell_id=I, tau_rise=0.5, tau_decay=6.0,
         E_syn=-70.0, g_bar=8.0, g_clip=80.0),
]  # fmt: skip
MITRAL_CELLS = [dict(name="mitral", cell_id=0)]
MITRAL_SYNAPSES = [
    dict(name="AMPA", synapse_id=0, cell_id=0, tau_rise=0.1, tau_decay=5.0,
         E_syn=0.0, g_bar=6.0, g_clip=40.0),
    dict(name="NMDA", synapse_id=1, cell_id=0, tau_rise=10.0, tau_decay=70.0,
         E_syn=0.0, g_bar=0.1, g_clip=4.0),
]  # fmt: skip


def combined_physiology():
    """Input namespace of a teacher-forced layer: mitral, then E and I (shifted ids)."""
    cells = [dict(c) for c in MITRAL_CELLS]
    for c in CELLS:
        cells.append({**c, "cell_id": c["cell_id"] + 1})
    synapses = [dict(s) for s in MITRAL_SYNAPSES]
    for s in SYNAPSES:
        synapses.append(
            {**s, "cell_id": s["cell_id"] + 1, "synapse_id": s["synapse_id"] + 2}
        )
    return cells, synapses


# =====================================================================
# Random structure
# =====================================================================

# Mean synaptic weight and connection probability by source class, tuned for the rates
# in the module docstring.
W_MEAN = {"mitral": 0.18, "excitatory": 0.3, "inhibitory": 1.8}
P_CONN = {"mitral": 0.3, "excitatory": 0.15, "inhibitory": 0.3}
SF_SPREAD = 0.15  # log-normal spread of the perturbed scaling factors
RATE_HZ = {"mitral": 10.0, "excitatory": 8.0, "inhibitory": 25.0}


def cell_types(rng, n, frac_e=0.8):
    types = np.full(n, I, dtype=np.int64)
    types[: round(frac_e * n)] = E
    return rng.permutation(types)


def sparse_lognormal(rng, n_src, n_tgt, p, mean, cv=1.0):
    """Bernoulli(p) topology with log-normal weights of the given mean and CV."""
    sigma2 = np.log1p(cv**2)
    w = rng.lognormal(np.log(mean) - sigma2 / 2, np.sqrt(sigma2), (n_src, n_tgt))
    return (w * (rng.random((n_src, n_tgt)) < p)).astype(np.float32)


def weights_by_source(rng, source_names, n_tgt, p_scale=1.0):
    """Stack one weight block per source row, each drawn with its class statistics."""
    return np.concatenate(
        [
            sparse_lognormal(rng, 1, n_tgt, P_CONN[s] * p_scale, W_MEAN[s])
            for s in source_names
        ]
    )


def poisson(rng, batch, n_steps, rates_hz):
    """Independent Poisson spike trains, ``(batch, n_steps, len(rates_hz))`` bool."""
    p = np.asarray(rates_hz) * DT / 1000.0
    return rng.random((batch, n_steps, len(p))) < p


def move_scaling_factors(rng, projections):
    """Draw new scaling factors for already-built projections, as training would.

    Models cache their connectome when constructed, so scaling factors start at 1
    (as in every experiment) and only move afterwards. Returns the values drawn.
    """
    values = rng.lognormal(0.0, SF_SPREAD, len(projections))
    with torch.no_grad():
        for projection, value in zip(projections, values):
            projection.log_sf.fill_(float(np.log(value)))
    return values


def per_type_blocks(weights, src_types, tgt_types, src_names, tgt_names):
    """Slice a matrix into ``{(src_name, tgt_name): block}`` by cell type."""
    out = {}
    for a, src in enumerate(src_names):
        for b, tgt in enumerate(tgt_names):
            rows, cols = np.flatnonzero(src_types == a), np.flatnonzero(tgt_types == b)
            if rows.size and cols.size:
                out[(src, tgt)] = weights[np.ix_(rows, cols)]
    return out


# =====================================================================
# Scenarios
# =====================================================================


@dataclass(frozen=True)
class Built:
    """A built network: the model, its input and the arrays that define it."""

    model: nn.Module
    input_spikes: np.ndarray  # (batch, n_steps, n_inputs) bool
    structure: dict[str, np.ndarray]


@dataclass(frozen=True)
class Scenario:
    """A seeded network and its input.

    ``build(batch_size)`` returns a ``Built``; ``state_keys`` are the keys
    of ``model.get_checkpoint_state()`` (empty when the model cannot be restarted from a
    recorded state, which rules out the lockstep tests).
    """

    name: str
    build: Callable[[int], "Built"]
    n_steps: int = 200
    state_keys: tuple[str, ...] = ()
    returns_probabilities: bool = False


N, N_FF = 150, 80
REC_NAMES = [c["name"] for c in CELLS]


def _teacher(seed, n=N, n_ff=N_FF):
    """Cell types and weights of a teacher with ``n`` neurons and ``n_ff`` inputs.

    Connection probabilities are scaled by ``N / n`` and ``N_FF / n_ff``, so every
    neuron has the same expected in-degree, and so the same input statistics and rates,
    at any size.
    """
    rng = np.random.default_rng(seed)
    types = cell_types(rng, n)
    names = [REC_NAMES[t] for t in types]
    w_rec = weights_by_source(rng, names, n, p_scale=N / n)
    w_ff = weights_by_source(rng, ["mitral"] * n_ff, n, p_scale=N_FF / n_ff)
    return rng, types, w_rec, w_ff


def _input_rng(seed, input_seed):
    """Inputs get their own stream, so one network can be driven by several inputs."""
    return np.random.default_rng([seed, input_seed])


def recurrent(
    batch_size,
    seed,
    scaling_factors=False,
    n=N,
    n_ff=N_FF,
    n_steps=200,
    input_seed=0,
):
    """``ConductanceLIFNetwork`` with frozen weights or perturbed scaling factors."""
    rng, types, w_rec, w_ff = _teacher(seed, n, n_ff)
    ff_types = np.zeros(n_ff, dtype=np.int64)
    if scaling_factors:
        rec = {
            k: ScalingFactorProjection(b)
            for k, b in per_type_blocks(
                w_rec, types, types, REC_NAMES, REC_NAMES
            ).items()
        }
        ff = {
            k: ScalingFactorProjection(b)
            for k, b in per_type_blocks(
                w_ff, ff_types, types, ["mitral"], REC_NAMES
            ).items()
        }
    else:
        rec, ff = make_frozen_projections(
            w_rec, w_ff, types, ff_types, REC_NAMES, ["mitral"]
        )
    model = ConductanceLIFNetwork(
        dt=DT,
        rec_projections=rec,
        ff_projections=ff,
        cell_type_indices=types,
        cell_type_indices_FF=ff_types,
        cell_params=CELLS,
        cell_params_FF=MITRAL_CELLS,
        synapse_params=SYNAPSES,
        synapse_params_FF=MITRAL_SYNAPSES,
        surrgrad_scale=5.0,
        batch_size=batch_size,
    )
    sfs = (
        move_scaling_factors(rng, [*rec.values(), *ff.values()])
        if scaling_factors
        else np.array([])
    )
    inputs = poisson(
        _input_rng(seed, input_seed), batch_size, n_steps, [RATE_HZ["mitral"]] * n_ff
    )
    structure = dict(types=types, w_rec=w_rec, w_ff=w_ff, sf=sfs)
    return Built(model, inputs, structure)


def feedforward(
    batch_size,
    seed,
    learnt_weights=False,
    probabilities=False,
    n=N,
    n_ff=N_FF,
    n_steps=200,
    input_seed=0,
):
    """Teacher-forced ``FeedforwardConductanceLIFNetwork`` (the fully observed student).

    Scaling factors on every pathway, or (``learnt_weights``) full-rank mitral, low-rank
    excitatory and scaling-factor inhibitory pathways — one of each caching mode.
    """
    rng, types, w_rec, w_ff = _teacher(seed, n, n_ff)
    cells, synapses = combined_physiology()
    names = [c["name"] for c in cells]
    # Rows: [mitral inputs, teacher neurons]; teacher types shift past "mitral".
    row_types = np.concatenate([np.zeros(n_ff, dtype=np.int64), types + 1])
    weights = np.concatenate([w_ff, w_rec])
    blocks = per_type_blocks(weights, row_types, types, names, REC_NAMES)
    projections, drawn = {}, {}
    for (src, tgt), block in blocks.items():
        if learnt_weights and src == "mitral":
            init = block * rng.lognormal(0.0, 0.2, block.shape)
            drawn[f"{src}->{tgt}"] = init
            projections[(src, tgt)] = FullRankProjection(init, mask=block != 0)
        elif learnt_weights and src == "excitatory":
            n_src, n_tgt = block.shape
            U = np.stack([np.ones(n_src), rng.normal(0, 0.3, n_src)], axis=1)
            V = np.stack(
                [np.full(n_tgt, np.log(W_MEAN[src])), rng.normal(0, 0.3, n_tgt)]
            )
            drawn[f"{src}->{tgt}"] = U @ V
            projections[(src, tgt)] = LowRankProjection(
                n_src, n_tgt, rank=2, U_init=U, V_init=V, mask=block != 0
            )
        else:
            projections[(src, tgt)] = ScalingFactorProjection(block)
    model = FeedforwardConductanceLIFNetwork(
        dt=DT,
        projections=projections,
        cell_type_indices=types,
        cell_type_indices_FF=row_types,
        cell_params=CELLS,
        cell_params_FF=cells,
        synapse_params_FF=synapses,
        surrgrad_scale=5.0,
        batch_size=batch_size,
        return_probabilities=probabilities,
    )
    scaled = [p for p in projections.values() if isinstance(p, ScalingFactorProjection)]
    drawn["sf"] = move_scaling_factors(rng, scaled)
    rates = [RATE_HZ["mitral"]] * n_ff + [RATE_HZ[REC_NAMES[t]] for t in types]
    inputs = poisson(_input_rng(seed, input_seed), batch_size, n_steps, rates)
    return Built(model, inputs, dict(types=types, weights=weights, **drawn))


def two_layer(batch_size, seed, n_hidden=40):
    """``TwoLayerSNN``: recurrent hidden layer + teacher-forced visible layer.

    One scaling factor per (source type, target type) pair, shared by both layers —
    the arrangement the Bernstein student uses.
    """
    rng, types, w_rec, w_ff = _teacher(seed)
    cells, synapses = combined_physiology()
    names = [c["name"] for c in cells]
    hidden = np.sort(rng.choice(N, n_hidden, replace=False))
    visible = np.setdiff1d(np.arange(N), hidden)
    log_sf = {(a, b): nn.Parameter(torch.zeros(())) for a in names for b in REC_NAMES}

    def shared(blocks):
        out = {}
        for key, block in blocks.items():
            projection = ScalingFactorProjection(block)
            projection.log_sf = log_sf[key]
            out[key] = projection
        return out

    ff_types = np.zeros(N_FF, dtype=np.int64)
    # Layer 1 rows: [mitral, visible]; recurrence among hidden.
    l1_rows = np.concatenate([w_ff[:, hidden], w_rec[np.ix_(visible, hidden)]])
    l1_types = np.concatenate([ff_types, types[visible] + 1])
    layer1 = ConductanceLIFNetwork(
        dt=DT,
        rec_projections=shared(
            per_type_blocks(
                w_rec[np.ix_(hidden, hidden)],
                types[hidden],
                types[hidden],
                REC_NAMES,
                REC_NAMES,
            )
        ),
        ff_projections=shared(
            per_type_blocks(l1_rows, l1_types, types[hidden], names, REC_NAMES)
        ),
        cell_type_indices=types[hidden],
        cell_type_indices_FF=l1_types,
        cell_params=CELLS,
        cell_params_FF=cells,
        synapse_params=SYNAPSES,
        synapse_params_FF=synapses,
        surrgrad_scale=5.0,
        batch_size=batch_size,
    )
    # Layer 2 rows: [mitral, hidden (layer-1 spikes), visible].
    l2_rows = np.concatenate(
        [
            w_ff[:, visible],
            w_rec[np.ix_(hidden, visible)],
            w_rec[np.ix_(visible, visible)],
        ]
    )
    l2_types = np.concatenate([ff_types, types[hidden] + 1, types[visible] + 1])
    layer2 = FeedforwardConductanceLIFNetwork(
        dt=DT,
        projections=shared(
            per_type_blocks(l2_rows, l2_types, types[visible], names, REC_NAMES)
        ),
        cell_type_indices=types[visible],
        cell_type_indices_FF=l2_types,
        cell_params=CELLS,
        cell_params_FF=cells,
        synapse_params_FF=synapses,
        surrgrad_scale=5.0,
        batch_size=batch_size,
    )
    model = TwoLayerSNN(layer1, layer2, n_ff=N_FF, return_hidden_spikes=True)
    holders = [_LogSF(p) for p in log_sf.values()]
    move_scaling_factors(rng, holders)
    rates = [RATE_HZ["mitral"]] * N_FF + [RATE_HZ[REC_NAMES[t]] for t in types[visible]]
    inputs = poisson(rng, batch_size, 200, rates)
    log_sf_values = np.array([p.item() for p in log_sf.values()])
    structure = dict(
        types=types, w_rec=w_rec, w_ff=w_ff, hidden=hidden, log_sf=log_sf_values
    )
    return Built(model, inputs, structure)


class _LogSF:
    """Lets ``move_scaling_factors`` set a shared ``log_sf`` parameter directly."""

    def __init__(self, parameter):
        self.log_sf = parameter


# Current-based networks use natural units (reset 0, threshold 1) and the older
# weight-matrix API. They are covered by free-run and gradient fixtures only.
CURRENT_CELLS = [
    dict(name="excitatory", cell_id=E, tau_mem=20.0),
    dict(name="inhibitory", cell_id=I, tau_mem=10.0),
]
CURRENT_SYNAPSES = [
    dict(name="exc", synapse_id=0, cell_id=E, tau_syn=5.0, type="excitatory"),
    dict(name="inh", synapse_id=1, cell_id=I, tau_syn=10.0, type="inhibitory"),
]
CURRENT_SYNAPSES_FF = [
    dict(name="ff", synapse_id=0, cell_id=0, tau_syn=5.0, type="excitatory"),
]


def current_recurrent(batch_size, seed):
    rng = np.random.default_rng(seed)
    types = cell_types(rng, N)
    p = {E: (0.15, 0.08), I: (0.3, 0.25)}
    w_rec = np.concatenate([sparse_lognormal(rng, 1, N, *p[t]) for t in types])
    w_ff = sparse_lognormal(rng, N_FF, N, 0.3, 0.05)
    sf, sf_ff = (
        rng.lognormal(0.0, SF_SPREAD, (2, 2)),
        rng.lognormal(0.0, SF_SPREAD, (1, 2)),
    )
    model = CurrentLIFNetwork(
        dt=DT,
        weights=w_rec,
        weights_FF=w_ff,
        cell_type_indices=types,
        cell_type_indices_FF=np.zeros(N_FF, dtype=np.int64),
        cell_params=CURRENT_CELLS,
        synapse_params=CURRENT_SYNAPSES,
        synapse_params_FF=CURRENT_SYNAPSES_FF,
        surrgrad_scale=5.0,
        batch_size=batch_size,
        scaling_factors=sf,
        scaling_factors_FF=sf_ff,
        optimisable="scaling_factors",
        use_tqdm=False,
    )
    inputs = poisson(rng, batch_size, 200, [RATE_HZ["mitral"]] * N_FF)
    return Built(
        model, inputs, dict(types=types, w_rec=w_rec, w_ff=w_ff, sf=sf, sf_ff=sf_ff)
    )


def current_feedforward(batch_size, seed):
    rng = np.random.default_rng(seed)
    types = cell_types(rng, N)
    w_ff = sparse_lognormal(rng, N_FF, N, 0.3, 0.05)
    sf_ff = rng.lognormal(0.0, SF_SPREAD, (1, 2))
    model = FeedforwardCurrentLIFNetwork(
        dt=DT,
        weights_FF=w_ff,
        cell_type_indices=types,
        cell_type_indices_FF=np.zeros(N_FF, dtype=np.int64),
        cell_params=CURRENT_CELLS,
        synapse_params_FF=CURRENT_SYNAPSES_FF,
        surrgrad_scale=5.0,
        batch_size=batch_size,
        scaling_factors_FF=sf_ff,
        optimisable="weights",
        feedforward_mask=w_ff != 0,
        use_tqdm=False,
    )
    inputs = poisson(rng, batch_size, 200, [RATE_HZ["mitral"]] * N_FF)
    return Built(model, inputs, dict(types=types, w_ff=w_ff, sf_ff=sf_ff))


SEED = 2026
SCENARIOS = {
    s.name: s
    for s in [
        Scenario(
            "recurrent_frozen",
            lambda b: recurrent(b, SEED),
            state_keys=("v", "g"),
        ),
        Scenario(
            "recurrent_scaling_factors",
            lambda b: recurrent(b, SEED, scaling_factors=True),
            state_keys=("v", "g"),
        ),
        Scenario(
            "feedforward_scaling_factors",
            lambda b: feedforward(b, SEED),
            state_keys=("v", "g_FF"),
        ),
        Scenario(
            "feedforward_learnt_weights",
            lambda b: feedforward(b, SEED, learnt_weights=True),
            state_keys=("v", "g_FF"),
        ),
        Scenario(
            "feedforward_probabilities",
            lambda b: feedforward(b, SEED, probabilities=True),
            state_keys=("v", "g_FF"),
            returns_probabilities=True,
        ),
        Scenario("two_layer", lambda b: two_layer(b, SEED)),
        Scenario("current_recurrent", lambda b: current_recurrent(b, SEED)),
        Scenario("current_feedforward", lambda b: current_feedforward(b, SEED)),
    ]
}
BATCH_SIZE = 2


def fingerprint(built: Built) -> str:
    """SHA-256 of the arrays that define a built network and of its input."""
    digest = hashlib.sha256()
    for key in sorted(built.structure):
        digest.update(key.encode())
        digest.update(np.ascontiguousarray(built.structure[key]).tobytes())
    digest.update(np.ascontiguousarray(built.input_spikes).tobytes())
    return digest.hexdigest()


def loss_readout(seed, shape):
    """Fixed random readout ``R`` for the gradient tests' loss ``sum(spikes * R)``."""
    rng = np.random.default_rng(seed)
    return torch.from_numpy(rng.normal(size=shape).astype(np.float32))
