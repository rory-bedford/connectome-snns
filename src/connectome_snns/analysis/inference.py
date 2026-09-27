"""Inference wrappers for analysis notebooks.

Each wrapper builds the appropriate model from a TOML config and checkpoint,
runs inference via :class:`~snn_runners.SNNInference`, and returns
teacher/student spike arrays. All wrappers share the same return format
and support caching.

Available wrappers:

- :func:`run_feedforward_inference` — ``FeedforwardConductanceLIFNetwork``
  with ``[FF, rec]`` input. Used by fully-observed, noisy-weights,
  inferring-inputs, and hidden-units experiments.
- :func:`run_recurrent_inference` — ``ConductanceLIFNetwork`` with
  FF-only input (recurrence is internal). Used by fully-recurrent
  hidden-activity strategy.
- :func:`run_visible_driven_inference` — ``ConductanceLIFNetwork`` with
  restructured weights and ``[FF, teacher_visible]`` input. Used by
  visible-driven hidden-activity strategy.
- :func:`run_two_layer_inference` — ``TwoLayerSNN`` with Layer 1
  (hidden, recurrent) + Layer 2 (visible, feedforward). Used by the
  full-inference experiment.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import toml
import torch
import zarr
from numpy.typing import NDArray
from torch.utils.data import DataLoader

from connectome_snns.configs import StudentSimulationConfig, StudentHyperparameters
from connectome_snns.configs.conductance_based import (
    RecurrentLayerConfig,
    FeedforwardLayerConfig,
)
from connectome_snns.dataloaders.supervised import (
    ExactFFDataset,
    CyclicSampler,
    SpikeData,
    FeedforwardCollate,
    VisibleSubsetCollate,
)
from connectome_snns.network_simulators.conductance_based.simulator import (
    ConductanceLIFNetwork,
)
from connectome_snns.network_simulators.feedforward_conductance_based.simulator import (
    FeedforwardConductanceLIFNetwork,
)
from connectome_snns.network_simulators.projections import (
    FrozenProjection,
    make_frozen_chunked_ff_projections,
    make_frozen_projections,
)
from connectome_snns.network_simulators.two_layer import TwoLayerSNN
from connectome_snns.snn_runners import SNNInference


# ======================================================================
# Shared helpers
# ======================================================================


def _load_config(params_file):
    """Load TOML config and return parsed objects."""
    with open(params_file) as f:
        cfg = toml.load(f)

    sim = StudentSimulationConfig(**cfg["simulation"])
    hp = StudentHyperparameters(**cfg["hyperparameters"])
    rec_cfg = RecurrentLayerConfig(**cfg["recurrent"])
    ff_cfg = FeedforwardLayerConfig(**cfg["feedforward"])

    return cfg, sim, hp, rec_cfg, ff_cfg


def _build_combined_ff_params(
    ff_cell_params, rec_cell_params, ff_syn_params, rec_syn_params
):
    """Build combined FF cell/synapse params with offset IDs."""
    n_ff_ct = len(ff_cell_params)
    n_ff_st = len(ff_syn_params)

    combined_cell = ff_cell_params.copy()
    for cp in rec_cell_params:
        o = cp.copy()
        o["cell_id"] = cp["cell_id"] + n_ff_ct
        combined_cell.append(o)

    combined_syn = list(ff_syn_params)
    for sp in rec_syn_params:
        o = sp.copy()
        o["cell_id"] = sp["cell_id"] + n_ff_ct
        o["synapse_id"] = sp["synapse_id"] + n_ff_st
        combined_syn.append(o)

    return combined_cell, combined_syn


def _load_network_structure(
    run_dir,
    network_structure_path=None,
    structure_subpath="inputs/network_structure.npz",
):
    """Load network structure arrays."""
    if network_structure_path is not None:
        return np.load(network_structure_path)
    return np.load(run_dir / structure_subpath)


def _check_cache(cache_path):
    """Return cached result if it exists, else None."""
    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists():
            return dict(np.load(cache_path))
    return None


def _save_cache(cache_path, result, dt, desc):
    """Save result dict to .npz cache."""
    if cache_path is not None:
        cache_path = Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, **result)
        # Find a spike array to report duration
        for k in ("teacher_spikes", "teacher_visible_spikes"):
            if k in result:
                duration_s = result[k].shape[0] * dt / 1000.0
                print(f"  [{desc}] Saved ({duration_s:.1f}s)")
                break


def _run_and_extract(model, dataloader, device, total_chunks, burnin_ts):
    """Run SNNInference and extract post-burnin student spikes."""
    runner = SNNInference(
        model=model,
        dataloader=dataloader,
        device=device,
        output_mode="memory",
        save_tracked_variables=False,
        max_chunks=total_chunks,
        progress_bar=True,
    )
    results = runner.run()
    return results["output_spikes"][0, burnin_ts:, :]


# ======================================================================
# Feedforward inference (FeedforwardConductanceLIFNetwork)
# ======================================================================


def run_feedforward_inference(
    params_file: str | Path,
    run_dir: str | Path,
    scaling_factors_FF: NDArray,
    *,
    zarr_subpath: str = "inputs/spike_data.zarr",
    ff_input: NDArray | None = None,
    rec_input: NDArray | None = None,
    output_indices: NDArray | None = None,
    n_burnin_chunks: int = 25,
    n_analysis_chunks: int = 25,
    cache_path: str | Path | None = None,
    device: str = "cpu",
    desc: str = "inference",
    network_structure_path: str | Path | None = None,
) -> dict:
    """Run inference with a FeedforwardConductanceLIFNetwork.

    Model input is ``[FF spikes, recurrent spikes]`` concatenated along the
    neuron dimension. Used by fully-observed, noisy-weights, inferring-inputs,
    hidden-units, and EM hidden-activity experiments.

    Args:
        params_file: Path to ``parameters.toml``.
        run_dir: Experiment output directory.
        scaling_factors_FF: Scaling factor matrix ``(n_source_types, n_target_types)``.
        zarr_subpath: Relative path to spike zarr within *run_dir*.
        ff_input: Optional override for FF input ``(time, n_ff)``.
        rec_input: Optional override for recurrent input ``(time, n_rec)``.
        output_indices: Optional neuron indices for slicing teacher spikes.
        n_burnin_chunks: Chunks to discard for warm-up.
        n_analysis_chunks: Chunks to keep for analysis.
        cache_path: Optional ``.npz`` cache path.
        device: PyTorch device string.
        desc: Progress bar description.
        network_structure_path: Optional override for network structure file.

    Returns:
        Dict with ``teacher_spikes``, ``student_spikes``, ``cell_type_indices``, ``dt``.
    """
    run_dir = Path(run_dir)
    cached = _check_cache(cache_path)
    if cached is not None:
        return cached

    cfg, sim, hp, rec_cfg, ff_cfg = _load_config(params_file)
    chunk_size = sim.chunk_size
    rec_cell_params = rec_cfg.get_cell_params()
    ff_cell_params = ff_cfg.get_cell_params()
    rec_syn_params = rec_cfg.get_synapse_params()
    ff_syn_params = ff_cfg.get_synapse_params()
    combined_cell, combined_syn = _build_combined_ff_params(
        ff_cell_params, rec_cell_params, ff_syn_params, rec_syn_params
    )

    ns = _load_network_structure(run_dir, network_structure_path)
    cell_type_indices = ns["cell_type_indices"]

    zarr_root = zarr.open_group(run_dir / zarr_subpath, mode="r")
    dt = float(zarr_root.attrs["dt"])

    # When the student was trained with ``missing_unit_fraction > 0`` the
    # teacher zarr has more recurrent neurons than the student tracks; the
    # recurrent input fed back into the student and the teacher reference
    # must both be sliced down to the visible subset.
    visible_indices = None
    nc_path = Path(run_dir) / "targets" / "neuron_classification.npz"
    if nc_path.exists():
        vi = np.load(nc_path).get("structural_visible_indices")
        if vi is not None and vi.shape[0] != int(zarr_root["output_spikes"].shape[2]):
            visible_indices = vi

    total_chunks = n_burnin_chunks + n_analysis_chunks
    has_custom_input = ff_input is not None or rec_input is not None

    # Determine batch size: from dataset for standard path, 1 for custom input
    if not has_custom_input:
        dataset = ExactFFDataset(
            spike_data_path=run_dir / zarr_subpath,
            chunk_size=chunk_size,
            device=device,
        )
        model_batch_size = dataset.batch_size
    else:
        model_batch_size = 1

    # Split combined feedforward_weights into FF-source (mitral) and
    # recurrent-source halves. The first n_ff rows are mitral (cell type
    # ids 0..n_ff_ct-1); the rest are recurrent neurons (cell types
    # offset by n_ff_ct), with rows in the same order as cell_type_indices.
    n_ff_ct = len(ff_cell_params)
    ff_ct_ids = ns["feedforward_cell_type_indices"]
    ff_rows = ff_ct_ids < n_ff_ct
    ff_conn = ns["feedforward_connectivity"]
    ff_block = ns["feedforward_weights"][ff_rows] * ff_conn[ff_rows].astype(np.float32)
    rec_block = ns["feedforward_weights"][~ff_rows] * ff_conn[~ff_rows].astype(
        np.float32
    )

    cell_type_names = [c["name"] for c in rec_cell_params]
    ff_cell_type_names = [c["name"] for c in ff_cell_params]
    projections = make_frozen_chunked_ff_projections(
        rec_weights=rec_block,
        ff_weights=ff_block,
        cell_type_indices=cell_type_indices,
        ff_cell_type_indices=ff_ct_ids[ff_rows].astype(np.int64),
        cell_type_names=cell_type_names,
        ff_cell_type_names=ff_cell_type_names,
        scaling_factors=scaling_factors_FF,
    )

    model = FeedforwardConductanceLIFNetwork(
        dt=dt,
        projections=projections,
        cell_type_indices=cell_type_indices,
        cell_type_indices_FF=ff_ct_ids,
        cell_params=rec_cell_params,
        cell_params_FF=combined_cell,
        synapse_params_FF=combined_syn,
        surrgrad_scale=hp.surrgrad_scale if hp.surrgrad_scale is not None else 1.0,
        batch_size=model_batch_size,
        track_variables=False,
    )
    model.to(device)
    model.eval()

    if has_custom_input:
        total_ts = total_chunks * chunk_size
        teacher_out_full = np.array(zarr_root["output_spikes"][0, :total_ts, :])
        teacher_out = (
            teacher_out_full[:, output_indices]
            if output_indices is not None
            else teacher_out_full
        )
        ff_all = (
            ff_input[:total_ts]
            if ff_input is not None
            else np.array(zarr_root["input_spikes"][0, :total_ts, :])
        )
        rec_all = rec_input[:total_ts] if rec_input is not None else teacher_out

        def _iter():
            for ci in range(total_chunks):
                t0, t1 = ci * chunk_size, (ci + 1) * chunk_size
                ff_t = torch.from_numpy(ff_all[t0:t1].astype(np.float32)).unsqueeze(0)
                rec_t = torch.from_numpy(rec_all[t0:t1].astype(np.float32)).unsqueeze(0)
                yield SpikeData(
                    input_spikes=torch.cat([ff_t, rec_t], dim=2),
                    target_spikes=torch.from_numpy(
                        teacher_out[t0:t1].astype(np.float32)
                    ).unsqueeze(0),
                )

        dataloader = _iter()
    else:
        # dataset already created above for batch_size detection
        if visible_indices is not None:
            collate_fn = VisibleSubsetCollate(
                torch.from_numpy(visible_indices.astype(np.int64))
            )
        else:
            collate_fn = FeedforwardCollate(dt=dt)
        dataloader = DataLoader(
            dataset,
            batch_size=None,
            sampler=CyclicSampler(dataset),
            num_workers=0,
            collate_fn=collate_fn,
        )

    burnin_ts = n_burnin_chunks * chunk_size
    student_spikes = _run_and_extract(
        model, dataloader, device, total_chunks, burnin_ts
    )

    if has_custom_input:
        teacher_spikes = teacher_out[burnin_ts:, :]
    else:
        total_ts = total_chunks * chunk_size
        teacher_out = np.array(zarr_root["output_spikes"][0, :total_ts, :])
        # Pick the indices to use for the teacher reference. The caller's
        # ``output_indices`` wins; otherwise fall back to ``visible_indices``
        # so the teacher matches the student's neuron set when structural
        # masking was applied.
        teacher_indices = output_indices
        if teacher_indices is None and visible_indices is not None:
            teacher_indices = visible_indices
        teacher_spikes = (
            teacher_out[burnin_ts:, teacher_indices]
            if teacher_indices is not None
            else teacher_out[burnin_ts:, :]
        )

    result = dict(
        teacher_spikes=teacher_spikes,
        student_spikes=student_spikes,
        cell_type_indices=cell_type_indices,
        dt=dt,
    )
    _save_cache(cache_path, result, dt, desc)
    return result


# ======================================================================
# Recurrent inference (ConductanceLIFNetwork, FF-only input)
# ======================================================================


def run_recurrent_inference(
    params_file: str | Path,
    run_dir: str | Path,
    scaling_factors: NDArray,
    scaling_factors_FF: NDArray,
    *,
    zarr_subpath: str = "inputs/spike_data.zarr",
    chunk_size: int | None = None,
    n_burnin_chunks: int = 25,
    n_analysis_chunks: int = 25,
    cache_path: str | Path | None = None,
    device: str = "cpu",
    desc: str = "recurrent inference",
    network_structure_path: str | Path | None = None,
) -> dict:
    """Run inference with a ConductanceLIFNetwork (FF-only input).

    The model simulates all neurons with full internal recurrence.
    Input is feedforward spikes only — recurrent dynamics are generated
    by the model itself. Used by the fully-recurrent hidden-activity strategy.

    Args:
        params_file: Path to ``parameters.toml``.
        run_dir: Experiment output directory.
        scaling_factors: Recurrent scaling factors ``(n_rec_ct, n_rec_ct)``.
        scaling_factors_FF: Feedforward scaling factors ``(n_ff_ct, n_rec_ct)``.
        zarr_subpath: Relative path to spike zarr within *run_dir*.
        n_burnin_chunks: Chunks to discard for warm-up.
        n_analysis_chunks: Chunks to keep for analysis.
        cache_path: Optional ``.npz`` cache path.
        device: PyTorch device string.
        desc: Progress bar description.
        network_structure_path: Optional override for network structure file.

    Returns:
        Dict with ``teacher_spikes``, ``student_spikes``, ``cell_type_indices``, ``dt``.
    """
    run_dir = Path(run_dir)
    cached = _check_cache(cache_path)
    if cached is not None:
        return cached

    cfg, sim, hp, rec_cfg, ff_cfg = _load_config(params_file)
    if chunk_size is None:
        chunk_size = sim.chunk_size

    ns = _load_network_structure(run_dir, network_structure_path)
    cell_type_indices = ns["cell_type_indices"]

    zarr_root = zarr.open_group(run_dir / zarr_subpath, mode="r")
    dt = float(zarr_root.attrs["dt"])

    # FF-only dataloader (no collate — just pass input_spikes through)
    total_chunks = n_burnin_chunks + n_analysis_chunks
    dataset = ExactFFDataset(
        spike_data_path=run_dir / zarr_subpath,
        chunk_size=chunk_size,
        device=device,
    )

    rec_cell_params = rec_cfg.get_cell_params()
    ff_cell_params = ff_cfg.get_cell_params()
    rec_projections, ff_projections = make_frozen_projections(
        rec_weights=ns["recurrent_weights"]
        * ns["recurrent_connectivity"].astype(np.float32),
        ff_weights=ns["feedforward_weights"]
        * ns["feedforward_connectivity"].astype(np.float32),
        cell_type_indices=cell_type_indices,
        ff_cell_type_indices=ns["feedforward_cell_type_indices"],
        cell_type_names=[c["name"] for c in rec_cell_params],
        ff_cell_type_names=[c["name"] for c in ff_cell_params],
        scaling_factors_rec=scaling_factors,
        scaling_factors_ff=scaling_factors_FF,
    )
    model = ConductanceLIFNetwork(
        dt=dt,
        rec_projections=rec_projections,
        ff_projections=ff_projections,
        cell_type_indices=cell_type_indices,
        cell_type_indices_FF=ns["feedforward_cell_type_indices"],
        cell_params=rec_cell_params,
        cell_params_FF=ff_cell_params,
        synapse_params=rec_cfg.get_synapse_params(),
        synapse_params_FF=ff_cfg.get_synapse_params(),
        surrgrad_scale=hp.surrgrad_scale if hp.surrgrad_scale is not None else 1.0,
        batch_size=dataset.batch_size,
        track_variables=False,
    )
    model.to(device)
    model.eval()
    dataloader = DataLoader(
        dataset,
        batch_size=None,
        sampler=CyclicSampler(dataset),
        num_workers=0,
    )

    burnin_ts = n_burnin_chunks * chunk_size
    student_spikes = _run_and_extract(
        model, dataloader, device, total_chunks, burnin_ts
    )

    total_ts = total_chunks * chunk_size
    teacher_spikes = np.array(zarr_root["output_spikes"][0, burnin_ts:total_ts, :])

    result = dict(
        teacher_spikes=teacher_spikes,
        student_spikes=student_spikes,
        cell_type_indices=cell_type_indices,
        dt=dt,
    )
    _save_cache(cache_path, result, dt, desc)
    return result


# ======================================================================
# Visible-driven inference (ConductanceLIFNetwork, restructured weights)
# ======================================================================


def run_visible_driven_inference(
    params_file: str | Path,
    run_dir: str | Path,
    scaling_factors: NDArray,
    scaling_factors_FF: NDArray,
    visible_indices: NDArray,
    *,
    zarr_subpath: str = "inputs/spike_data.zarr",
    n_burnin_chunks: int = 25,
    n_analysis_chunks: int = 25,
    cache_path: str | Path | None = None,
    device: str = "cpu",
    desc: str = "visible-driven inference",
    network_structure_path: str | Path | None = None,
) -> dict:
    """Run inference with a visible-driven ConductanceLIFNetwork.

    The model has restructured weights: visible→all connections are moved from
    the recurrent pathway to feedforward (visible rows zeroed in recurrent).
    Input is ``[FF spikes, teacher visible spikes]``. Used by the visible-driven
    hidden-activity strategy.

    Args:
        params_file: Path to ``parameters.toml``.
        run_dir: Experiment output directory.
        scaling_factors: Recurrent scaling factors ``(n_rec_ct, n_rec_ct)``.
        scaling_factors_FF: Feedforward scaling factors ``(n_ff_ct + n_rec_ct, n_rec_ct)``.
        visible_indices: Array of visible neuron indices.
        zarr_subpath: Relative path to spike zarr within *run_dir*.
        n_burnin_chunks: Chunks to discard for warm-up.
        n_analysis_chunks: Chunks to keep for analysis.
        cache_path: Optional ``.npz`` cache path.
        device: PyTorch device string.
        desc: Progress bar description.
        network_structure_path: Optional override for network structure file.

    Returns:
        Dict with ``teacher_spikes``, ``student_spikes``, ``cell_type_indices``, ``dt``.
    """
    run_dir = Path(run_dir)
    cached = _check_cache(cache_path)
    if cached is not None:
        return cached

    cfg, sim, hp, rec_cfg, ff_cfg = _load_config(params_file)
    chunk_size = sim.chunk_size
    rec_cell_params = rec_cfg.get_cell_params()
    ff_cell_params = ff_cfg.get_cell_params()
    rec_syn_params = rec_cfg.get_synapse_params()
    ff_syn_params = ff_cfg.get_synapse_params()
    combined_cell, combined_syn = _build_combined_ff_params(
        ff_cell_params, rec_cell_params, ff_syn_params, rec_syn_params
    )

    ns = _load_network_structure(run_dir, network_structure_path)
    cell_type_indices = ns["cell_type_indices"]
    n_ff_ct = len(ff_cell_params)

    # Restructure weights: visible→all from recurrent to FF
    rec_weights = ns["recurrent_weights"]
    ff_weights = ns["feedforward_weights"]
    rec_mask = ns["recurrent_connectivity"]
    ff_mask = ns["feedforward_connectivity"]

    model_ff_weights = np.concatenate(
        [ff_weights, rec_weights[visible_indices, :]], axis=0
    )
    model_rec_weights = rec_weights.copy()
    model_rec_weights[visible_indices, :] = 0.0

    model_ff_mask = np.concatenate([ff_mask, rec_mask[visible_indices, :]], axis=0)
    model_rec_mask = rec_mask.copy()
    model_rec_mask[visible_indices, :] = False

    model_ff_ct = np.concatenate(
        [
            ns["feedforward_cell_type_indices"],
            cell_type_indices[visible_indices] + n_ff_ct,
        ]
    )

    zarr_root = zarr.open_group(run_dir / zarr_subpath, mode="r")
    dt = float(zarr_root.attrs["dt"])

    # Dataloader: uses ExactFFDataset with a collate that builds [FF, teacher_visible]
    total_chunks = n_burnin_chunks + n_analysis_chunks
    dataset = ExactFFDataset(
        spike_data_path=run_dir / zarr_subpath,
        chunk_size=chunk_size,
        device=device,
    )

    vis_tensor = torch.from_numpy(visible_indices).long()

    class _VisibleDrivenCollate:
        def __call__(self, batch):
            vis_rec = batch.target_spikes[:, :, vis_tensor]
            return SpikeData(
                input_spikes=torch.cat([batch.input_spikes, vis_rec], dim=2),
                target_spikes=batch.target_spikes,
            )

    rec_projections, ff_projections = make_frozen_projections(
        rec_weights=model_rec_weights * model_rec_mask.astype(np.float32),
        ff_weights=model_ff_weights * model_ff_mask.astype(np.float32),
        cell_type_indices=cell_type_indices,
        ff_cell_type_indices=model_ff_ct,
        cell_type_names=[c["name"] for c in rec_cell_params],
        ff_cell_type_names=[c["name"] for c in combined_cell],
        scaling_factors_rec=scaling_factors,
        scaling_factors_ff=scaling_factors_FF,
    )
    model = ConductanceLIFNetwork(
        dt=dt,
        rec_projections=rec_projections,
        ff_projections=ff_projections,
        cell_type_indices=cell_type_indices,
        cell_type_indices_FF=model_ff_ct,
        cell_params=rec_cell_params,
        cell_params_FF=combined_cell,
        synapse_params=rec_syn_params,
        synapse_params_FF=combined_syn,
        surrgrad_scale=hp.surrgrad_scale if hp.surrgrad_scale is not None else 1.0,
        batch_size=dataset.batch_size,
        track_variables=False,
    )
    model.to(device)
    model.eval()

    dataloader = DataLoader(
        dataset,
        batch_size=None,
        sampler=CyclicSampler(dataset),
        num_workers=0,
        collate_fn=_VisibleDrivenCollate(),
    )

    burnin_ts = n_burnin_chunks * chunk_size
    student_spikes = _run_and_extract(
        model, dataloader, device, total_chunks, burnin_ts
    )

    total_ts = total_chunks * chunk_size
    teacher_spikes = np.array(zarr_root["output_spikes"][0, burnin_ts:total_ts, :])

    result = dict(
        teacher_spikes=teacher_spikes,
        student_spikes=student_spikes,
        cell_type_indices=cell_type_indices,
        dt=dt,
    )
    _save_cache(cache_path, result, dt, desc)
    return result


# ======================================================================
# Two-layer inference (TwoLayerSNN: hidden recurrent + visible FF)
# ======================================================================


def _build_two_layer_model(
    params_file: str | Path,
    run_dir: str | Path,
    scaling_factors: NDArray,
    scaling_factors_FF: NDArray,
    *,
    ff_weights: NDArray | None = None,
    zarr_subpath: str = "inputs/spike_data.zarr",
    n_burnin_chunks: int = 25,
    n_analysis_chunks: int = 25,
    device: str = "cpu",
) -> dict:
    """Build a TwoLayerSNN model and dataloader for inference.

    Returns a dict with all the objects needed to run inference:
    model, dataloader, chunk_size, cell type indices, etc.
    """
    run_dir = Path(run_dir)

    cfg, sim, hp, rec_cfg, ff_cfg = _load_config(params_file)
    chunk_size = sim.chunk_size
    rec_cell_params = rec_cfg.get_cell_params()
    ff_cell_params = ff_cfg.get_cell_params()
    rec_syn_params = rec_cfg.get_synapse_params()
    ff_syn_params = ff_cfg.get_synapse_params()
    n_ff_cell_types = len(ff_cell_params)
    combined_cell, combined_syn = _build_combined_ff_params(
        ff_cell_params, rec_cell_params, ff_syn_params, rec_syn_params
    )

    # Load saved state
    nc = np.load(run_dir / "targets" / "neuron_classification.npz")
    visible_indices = nc["visible_indices"]
    unobserved_indices = nc["unobserved_indices"]
    missing_unit_fraction = float(nc["missing_unit_fraction"])

    l1_state = np.load(run_dir / "initial_state" / "layer1.npz")
    l2_state = np.load(run_dir / "initial_state" / "layer2.npz")

    # Derive cell type indices from network structure + masking
    ns = _load_network_structure(run_dir)
    full_cell_type_indices = ns["cell_type_indices"]
    ff_cell_type_indices = ns["feedforward_cell_type_indices"]
    n_ff_inputs = ns["feedforward_weights"].shape[0]

    structural_visible = None
    if missing_unit_fraction > 0:
        structural_hidden = nc.get("structural_hidden_indices", np.array([], dtype=int))
        structural_visible = np.setdiff1d(
            np.arange(nc["n_neurons_original"]), structural_hidden
        )
        cell_type_indices = full_cell_type_indices[structural_visible]
    else:
        cell_type_indices = full_cell_type_indices

    layer1_ct = cell_type_indices[unobserved_indices]
    layer2_ct = cell_type_indices[visible_indices]

    layer1_ff_ct = np.concatenate(
        [
            ff_cell_type_indices,
            cell_type_indices[visible_indices] + n_ff_cell_types,
        ]
    )
    layer2_ff_ct = np.concatenate(
        [
            ff_cell_type_indices,
            cell_type_indices[unobserved_indices] + n_ff_cell_types,
            cell_type_indices[visible_indices] + n_ff_cell_types,
        ]
    )

    # Start from initial weights
    l1_ff_w = l1_state["ff_weights"].copy()
    l1_rec_w = l1_state["rec_weights"].copy()
    l2_ff_w = l2_state["ff_weights"].copy()

    # Override learnable FF rows if trained weights provided.
    if ff_weights is not None:
        linear_ff = np.exp(ff_weights)
        l1_ff_w[:n_ff_inputs, :] = linear_ff[:, unobserved_indices]
        l2_ff_w[:n_ff_inputs, :] = linear_ff[:, visible_indices]

        # Zero out FF-unique SF rows so learned weights aren't double-scaled
        scaling_factors_FF = scaling_factors_FF.copy()
        scaling_factors_FF[:n_ff_cell_types, :] = 1.0

    # Apply structural masks before passing to projection builders; SFs are
    # baked in by the builders themselves.
    l1_rec_w = l1_rec_w * l1_state["rec_mask"].astype(np.float32)
    l1_ff_w = l1_ff_w * l1_state["ff_mask"].astype(np.float32)
    l2_ff_w = l2_ff_w * l2_state["ff_mask"].astype(np.float32)

    rec_names = [c["name"] for c in rec_cell_params]
    combined_names = [c["name"] for c in combined_cell]

    l1_rec_projs, l1_ff_projs = make_frozen_projections(
        rec_weights=l1_rec_w,
        ff_weights=l1_ff_w,
        cell_type_indices=layer1_ct,
        ff_cell_type_indices=layer1_ff_ct,
        cell_type_names=rec_names,
        ff_cell_type_names=combined_names,
        scaling_factors_rec=scaling_factors,
        scaling_factors_ff=scaling_factors_FF,
    )

    # Layer 2 is chunked-FF; its source rows mix FF and (hidden, visible)
    # recurrent neurons in the same combined cell-type namespace. The shared
    # builder assumes source == target neuron set, so build per-pair here.
    layer2_projs: dict[tuple[str, str], FrozenProjection] = {}
    for src_id, src_name in enumerate(combined_names):
        src_rows = np.flatnonzero(layer2_ff_ct == src_id)
        for tgt_id, tgt_name in enumerate(rec_names):
            tgt_rows = np.flatnonzero(layer2_ct == tgt_id)
            block = l2_ff_w[np.ix_(src_rows, tgt_rows)].astype(np.float32)
            block = block * float(scaling_factors_FF[src_id, tgt_id])
            layer2_projs[(src_name, tgt_name)] = FrozenProjection(block)

    zarr_root = zarr.open_group(run_dir / zarr_subpath, mode="r")
    dt = float(zarr_root.attrs["dt"])

    total_chunks = n_burnin_chunks + n_analysis_chunks
    dataset = ExactFFDataset(
        spike_data_path=run_dir / zarr_subpath,
        chunk_size=chunk_size,
        device=device,
    )
    batch_size = dataset.batch_size
    surrgrad = hp.surrgrad_scale if hp.surrgrad_scale is not None else 1.0

    layer1 = ConductanceLIFNetwork(
        dt=dt,
        rec_projections=l1_rec_projs,
        ff_projections=l1_ff_projs,
        cell_type_indices=layer1_ct,
        cell_type_indices_FF=layer1_ff_ct,
        cell_params=rec_cell_params,
        cell_params_FF=combined_cell,
        synapse_params=rec_syn_params,
        synapse_params_FF=combined_syn,
        surrgrad_scale=surrgrad,
        batch_size=batch_size,
        track_variables=False,
    )

    layer2 = FeedforwardConductanceLIFNetwork(
        dt=dt,
        projections=layer2_projs,
        cell_type_indices=layer2_ct,
        cell_type_indices_FF=layer2_ff_ct,
        cell_params=rec_cell_params,
        cell_params_FF=combined_cell,
        synapse_params_FF=combined_syn,
        surrgrad_scale=surrgrad,
        batch_size=batch_size,
        track_variables=False,
    )

    model = TwoLayerSNN(layer1, layer2, n_ff=n_ff_inputs)
    model.track_variables = True
    model.to(device)
    model.eval()

    # Visible-driven collate: extract visible teacher spikes for input
    if missing_unit_fraction > 0:
        visible_original = structural_visible[visible_indices]
    else:
        visible_original = visible_indices
    vis_tensor = torch.from_numpy(visible_original).long()

    class _Collate:
        def __call__(self, batch):
            vis_rec = batch.target_spikes[:, :, vis_tensor]
            return SpikeData(
                input_spikes=torch.cat([batch.input_spikes, vis_rec], dim=2),
                target_spikes=batch.target_spikes,
            )

    dataloader = DataLoader(
        dataset,
        batch_size=None,
        sampler=CyclicSampler(dataset),
        num_workers=0,
        collate_fn=_Collate(),
    )

    return dict(
        model=model,
        dataloader=dataloader,
        chunk_size=chunk_size,
        total_chunks=total_chunks,
        visible_indices=visible_indices,
        unobserved_indices=unobserved_indices,
        visible_original=visible_original,
        missing_unit_fraction=missing_unit_fraction,
        structural_visible=structural_visible,
        layer1_ct=layer1_ct,
        layer2_ct=layer2_ct,
        zarr_root=zarr_root,
        dt=dt,
        rec_cfg=rec_cfg,
        ff_cfg=ff_cfg,
    )


def run_two_layer_inference(
    params_file: str | Path,
    run_dir: str | Path,
    scaling_factors: NDArray,
    scaling_factors_FF: NDArray,
    *,
    ff_weights: NDArray | None = None,
    zarr_subpath: str = "inputs/spike_data.zarr",
    n_burnin_chunks: int = 25,
    n_analysis_chunks: int = 25,
    cache_path: str | Path | None = None,
    device: str = "cpu",
    desc: str = "two-layer inference",
    record_voltage_stats: bool = False,
) -> dict:
    """Run inference with a TwoLayerSNN (hidden recurrent + visible feedforward).

    Loads the two-layer weight matrices from ``initial_state/`` and neuron
    classification from ``targets/``. Applies scaling factors to the base
    weights. Optionally overrides the learnable FF portion with trained weights.

    Args:
        params_file: Path to ``parameters.toml``.
        run_dir: Experiment output directory.
        scaling_factors: Recurrent scaling factors ``(n_rec_ct, n_rec_ct)``.
        scaling_factors_FF: Feedforward scaling factors
            ``(n_ff_ct + n_rec_ct, n_rec_ct)``.
        ff_weights: Optional trained FF weights ``(n_ff_inputs, n_neurons)``
            in log-space. Replaces the learnable FF rows in both layers.
        zarr_subpath: Relative path to spike zarr within *run_dir*.
        n_burnin_chunks: Chunks to discard for warm-up.
        n_analysis_chunks: Chunks to keep for analysis.
        cache_path: Optional ``.npz`` cache path.
        device: PyTorch device string.
        desc: Progress bar description.
        record_voltage_stats: If True, compute per-neuron membrane voltage
            mean and std online during inference and include them in the
            result (keys ``visible_v_mean``, ``visible_v_std``,
            ``hidden_v_mean``, ``hidden_v_std``).

    Returns:
        Dict with ``teacher_visible_spikes``, ``student_visible_spikes``,
        ``teacher_hidden_spikes``, ``student_hidden_spikes``,
        ``visible_cell_types``, ``hidden_cell_types``, ``dt``.
        When *record_voltage_stats* is True, also includes per-neuron
        voltage statistics.
    """
    run_dir = Path(run_dir)
    cached = _check_cache(cache_path)
    if cached is not None:
        return cached

    ctx = _build_two_layer_model(
        params_file=params_file,
        run_dir=run_dir,
        scaling_factors=scaling_factors,
        scaling_factors_FF=scaling_factors_FF,
        ff_weights=ff_weights,
        zarr_subpath=zarr_subpath,
        n_burnin_chunks=n_burnin_chunks,
        n_analysis_chunks=n_analysis_chunks,
        device=device,
    )
    model = ctx["model"]
    dataloader = ctx["dataloader"]
    chunk_size = ctx["chunk_size"]
    visible_indices = ctx["visible_indices"]
    unobserved_indices = ctx["unobserved_indices"]
    visible_original = ctx["visible_original"]
    missing_unit_fraction = ctx["missing_unit_fraction"]
    structural_visible = ctx.get("structural_visible")
    layer1_ct = ctx["layer1_ct"]
    layer2_ct = ctx["layer2_ct"]
    zarr_root = ctx["zarr_root"]
    dt = ctx["dt"]
    total_chunks = ctx["total_chunks"]

    # Run inference with a lean loop — only accumulate visible + hidden spikes,
    # not the full dataloader fields (which include all-neuron target_spikes).
    from tqdm import tqdm

    vis_chunks = []
    hid_chunks = []
    data_iter = iter(dataloader)

    # Online voltage statistics (sum + sum-of-squares) — only for analysis chunks
    if record_voltage_stats:
        n_vis_neurons = len(visible_indices)
        n_hid_neurons = len(unobserved_indices)
        vis_v_sum = np.zeros(n_vis_neurons, dtype=np.float64)
        vis_v_sq_sum = np.zeros(n_vis_neurons, dtype=np.float64)
        hid_v_sum = np.zeros(n_hid_neurons, dtype=np.float64)
        hid_v_sq_sum = np.zeros(n_hid_neurons, dtype=np.float64)
        v_timesteps = 0

    with torch.inference_mode():
        for ci in tqdm(range(total_chunks), desc=desc):
            batch = next(data_iter)
            inp = batch.input_spikes.to(device, non_blocking=True)
            out = model.forward(input_spikes=inp)

            vis_chunks.append(out["spikes"][0].bool().cpu().numpy())
            hid_chunks.append(out["hidden_spikes"][0].bool().cpu().numpy())

            # Accumulate voltage statistics for analysis chunks only
            if record_voltage_stats and ci >= n_burnin_chunks:
                vis_v = out["voltages"][0].float().cpu().numpy()
                hid_v = out["hidden_voltages"][0].float().cpu().numpy()
                vis_v_sum += vis_v.sum(axis=0)
                vis_v_sq_sum += (vis_v**2).sum(axis=0)
                hid_v_sum += hid_v.sum(axis=0)
                hid_v_sq_sum += (hid_v**2).sum(axis=0)
                v_timesteps += vis_v.shape[0]

    burnin_ts = n_burnin_chunks * chunk_size
    student_visible = np.concatenate(vis_chunks, axis=0)[burnin_ts:]
    student_hidden = np.concatenate(hid_chunks, axis=0)[burnin_ts:]

    # Teacher spikes
    total_ts = total_chunks * chunk_size
    teacher_all = np.array(zarr_root["output_spikes"][0, :total_ts, :])

    if missing_unit_fraction > 0:
        teacher_visible = teacher_all[burnin_ts:, visible_original]
        unobserved_original = structural_visible[unobserved_indices]
        teacher_hidden = teacher_all[burnin_ts:, unobserved_original]
    else:
        teacher_visible = teacher_all[burnin_ts:, visible_indices]
        teacher_hidden = teacher_all[burnin_ts:, unobserved_indices]

    result = dict(
        teacher_visible_spikes=teacher_visible,
        student_visible_spikes=student_visible,
        teacher_hidden_spikes=teacher_hidden,
        student_hidden_spikes=student_hidden,
        visible_cell_types=layer2_ct,
        hidden_cell_types=layer1_ct,
        dt=dt,
    )

    if record_voltage_stats:
        vis_mean = vis_v_sum / v_timesteps
        hid_mean = hid_v_sum / v_timesteps
        result["visible_v_mean"] = vis_mean.astype(np.float32)
        result["visible_v_std"] = np.sqrt(
            vis_v_sq_sum / v_timesteps - vis_mean**2
        ).astype(np.float32)
        result["hidden_v_mean"] = hid_mean.astype(np.float32)
        result["hidden_v_std"] = np.sqrt(
            hid_v_sq_sum / v_timesteps - hid_mean**2
        ).astype(np.float32)

    _save_cache(cache_path, result, dt, desc)
    return result


def run_two_layer_tracked_inference(
    params_file: str | Path,
    run_dir: str | Path,
    scaling_factors: NDArray,
    scaling_factors_FF: NDArray,
    *,
    ff_weights: NDArray | None = None,
    zarr_subpath: str = "inputs/spike_data.zarr",
    n_burnin_chunks: int = 25,
    n_analysis_chunks: int = 25,
    device: str = "cpu",
    desc: str = "tracked inference",
) -> dict:
    """Run two-layer inference with full variable tracking via SNNInference.

    Returns all tracked variables (voltages, currents, conductances) from
    both layers, suitable for passing to ``create_activity_dashboard``.

    Args:
        params_file: Path to ``parameters.toml``.
        run_dir: Experiment output directory.
        scaling_factors: Recurrent scaling factors ``(n_rec_ct, n_rec_ct)``.
        scaling_factors_FF: Feedforward scaling factors
            ``(n_ff_ct + n_rec_ct, n_rec_ct)``.
        ff_weights: Optional trained FF weights in log-space.
        zarr_subpath: Relative path to spike zarr within *run_dir*.
        n_burnin_chunks: Chunks to discard for warm-up.
        n_analysis_chunks: Chunks to keep for analysis.
        device: PyTorch device string.
        desc: Progress bar description.

    Returns:
        Dict with SNNInference memory-mode results (all tracked variables
        concatenated along time axis, with batch dim), plus ``rec_cfg``,
        ``ff_cfg``, ``layer1_ct``, ``layer2_ct``, ``dt``, ``n_burnin_chunks``.
    """
    ctx = _build_two_layer_model(
        params_file=params_file,
        run_dir=run_dir,
        scaling_factors=scaling_factors,
        scaling_factors_FF=scaling_factors_FF,
        ff_weights=ff_weights,
        zarr_subpath=zarr_subpath,
        n_burnin_chunks=n_burnin_chunks,
        n_analysis_chunks=n_analysis_chunks,
        device=device,
    )

    model = ctx["model"]
    dataloader = ctx["dataloader"]

    from tqdm import tqdm

    data_iter = iter(dataloader)
    storage: dict[str, list] = {}

    with torch.inference_mode():
        # Burnin: tracking off, discard outputs
        model.track_variables = False
        for _ in tqdm(range(n_burnin_chunks), desc=f"{desc} (burnin)"):
            batch = next(data_iter)
            inp = batch.input_spikes.to(device, non_blocking=True)
            model.forward(input_spikes=inp)

        # Analysis: tracking on, accumulate all variables
        model.track_variables = True
        for _ in tqdm(range(n_analysis_chunks), desc=f"{desc} (analysis)"):
            batch = next(data_iter)
            inp = batch.input_spikes.to(device, non_blocking=True)
            out = model.forward(input_spikes=inp)

            # Accumulate dataloader fields (batch 0 only)
            for field_name in batch._fields:
                data = getattr(batch, field_name)
                if isinstance(data, torch.Tensor):
                    storage.setdefault(field_name, []).append(data[0:1].cpu().numpy())

            # Accumulate model outputs (batch 0 only)
            if isinstance(out, dict):
                for key, value in out.items():
                    if isinstance(value, torch.Tensor):
                        if key == "spikes":
                            storage.setdefault("output_spikes", []).append(
                                value[0:1].bool().cpu().numpy()
                            )
                        else:
                            storage.setdefault(key, []).append(value[0:1].cpu().numpy())

    # Concatenate along time axis
    result = {k: np.concatenate(v, axis=1) for k, v in storage.items()}

    # Attach metadata
    result["model"] = model
    result["layer1_ct"] = ctx["layer1_ct"]
    result["layer2_ct"] = ctx["layer2_ct"]
    result["dt"] = ctx["dt"]
    result["rec_cfg"] = ctx["rec_cfg"]
    result["ff_cfg"] = ctx["ff_cfg"]
    return result
