# TODO: what is still messy or untouched

The state after the `perf/conductance-speedups` cleanup. Items are roughly in order of
how much they matter. "Unused" means not referenced in this repo or in teacher-student
— other projects may still use it (cleanups tidy, never delete).

## Plotting quirks (kept as they were)

- A single Axes where a list is expected makes a new figure but returns None;
  `plot_firing_rate_variance_comparison` ignores `ax`; the default title
  "Sample  Spike Trains" has a double space; several arguments are accepted but unused
  (documented as such).

## Library code not yet tidied

- **`visualization/__init__`** imports every plotting module up front (the modules
  themselves are tidied: 5,800 → 5,000 lines, no function over 140 lines).
- **`configs/`.** The classes no code uses are in `configs/legacy.py` (still importable
  from their old modules); `configs/training.py` keeps only the ones the archived
  bernstein run scripts import. `configs/current_based/` is never imported.
- **Unused losses**, untested: `VanRossumRateLoss`, `AsymmetricSilencePenalty`,
  `MomentMatchLoss`. `FiringRateLoss` and `SilentNeuronLoss` repeat
  the rate computation.
- **Between-assembly blocks drop a diagonal.** The configuration-model assembly
  generator builds each between-assembly block with `allow_self_loops=False`, so edge
  k -> k *within the block* (neuron `i*size + k` to `j*size + k`, different cells) is
  never placed. Kept so existing teachers regenerate identically; fixing it changes them.

## Training loop

- Checkpoints are skipped during burn-in chunks, and checkpoints that fall between
  updates drop the pending gradients on resume (the trainer warns).
- Teacher-student resumes with `load_checkpoint` + `set_checkpoint_state`, which now
  realigns the data but cannot restore the van Rossum filter state; `trainer.resume`
  restores both.
- The legacy gradient-masking branch in `_update_weights` serves only weight-trained
  current-based models.

## Performance left on the table

- The recurrent step kernel is ~2x from memory bandwidth (22 MB of state per step).
  Ideas: per-synapse 1-D tiles instead of padded 2-D, fusing the spike compaction,
  CUDA graphs for the two launches per step. Matters for teacher simulation; ~10% of
  half-observed training.
- The backward pass saves the summed conductances of every step (`B x T x N x S`);
  recomputing them in segments would cut memory and ~10–15% of training time, at the
  cost of harder-to-read kernels.
- Current-based networks run the PyTorch path only.
- `SNNInference` to zarr is bound by compressing and writing float rate arrays.

## Housekeeping

- Teacher-student's `ObservedOnlySNN` does not forward `track_batch_idx` to its layer
  (harmless now that tracking is off without a plot generator).
