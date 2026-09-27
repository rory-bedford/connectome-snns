"""
Gradient-free parameter search using CMA-ES.

Searches over a parameter vector using CMA-ES, evaluating candidates
by running the model forward and computing losses. The caller provides
the initial parameter values and an update function that writes candidate
values into the model.

Loss functions use the same required_inputs/requires_target interface
as SNNTrainer.
"""

from typing import Callable

import numpy as np
import torch
import cma
from tqdm import tqdm


class EvolutionarySearch:
    """
    Gradient-free parameter search using CMA-ES.

    Args:
        model: The initialized SNN model (should use optimisable=None for speed).
        initial_params: 1D numpy array of initial parameter values for the search.
        update_fn: Callable that takes a 1D numpy array and writes candidate
            values into the model (e.g. reshaping and calling
            model.update_scaling_factors).
        dataloader: Iterator providing input spike data (must return named tuples).
        loss_functions: Dict with individual loss functions.
        loss_weights: Dict with weights for each loss function.
        device: Device string ('cpu' or 'cuda').
        n_chunks: Number of chunks to preload from the dataloader.
        initial_sigma: Initial step size for CMA-ES.
        max_evaluations: Maximum number of function evaluations.
        popsize: Population size per generation (default: CMA-ES auto).
        batch_size: Number of batch elements to use per evaluation. If None,
            uses the full batch from the dataloader.
        seed: Random seed for CMA-ES.
        callback: Optional function called after each generation with a dict
            containing: generation, n_evals, best_loss, mean_loss, sigma.
        burn_in_chunks: Number of initial chunks to skip for loss computation.
            The model still runs forward (to build up internal state) but those
            chunks don't contribute to the fitness value.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        initial_params: np.ndarray,
        update_fn: Callable[[np.ndarray], None],
        dataloader,
        loss_functions: dict[str, Callable],
        loss_weights: dict[str, float],
        device: str,
        n_chunks: int,
        initial_sigma: float = 0.1,
        max_evaluations: int = 200,
        popsize: int | None = None,
        batch_size: int | None = None,
        seed: int | None = None,
        callback: Callable | None = None,
        burn_in_chunks: int = 0,
    ):
        self.model = model
        self.update_fn = update_fn
        self.loss_functions = loss_functions
        self.loss_weights = loss_weights
        self.device = device
        self.initial_sigma = initial_sigma
        self.max_evaluations = max_evaluations
        self.popsize = popsize
        self.seed = seed
        self.callback = callback
        self.burn_in_chunks = burn_in_chunks

        self._initial_params = initial_params

        # Compile model for speed on GPU
        if device == "cuda":
            try:
                self.model = torch.compile(model)
                print("  ✓ Model compiled with torch.compile")
            except Exception as e:
                print(f"  ⚠ torch.compile failed ({e}), running in eager mode")

        # Preload data chunks. Move to the target device (GPU) to avoid
        # costly CPU↔GPU round-trips during the evaluation loop. Batch slicing
        # is applied first to limit memory usage.
        self._chunks = []
        spike_iter = iter(dataloader)
        for _ in range(n_chunks):
            batch_data = next(spike_iter)
            input_spikes = batch_data.input_spikes.to(device)
            target_spikes = getattr(batch_data, "target_spikes", None)
            if target_spikes is not None:
                target_spikes = target_spikes.to(device)
            if batch_size is not None:
                input_spikes = input_spikes[:batch_size]
                if target_spikes is not None:
                    target_spikes = target_spikes[:batch_size]
            self._chunks.append((input_spikes, target_spikes))

        self._n_chunks = len(self._chunks)
        self._eval_batch_size = self._chunks[0][0].shape[0]

    def _evaluate(self, flat_params: np.ndarray) -> dict[str, float]:
        """Evaluate a candidate parameter vector. Returns dict of losses."""
        self.update_fn(flat_params)

        # Reset model state
        if hasattr(self.model, "reset_state"):
            self.model.reset_state(batch_size=self._eval_batch_size)

        # Reset loss function states
        for loss_fn in self.loss_functions.values():
            if hasattr(loss_fn, "reset_state"):
                loss_fn.reset_state()

        # Disable variable tracking for speed
        self.model.track_variables = False
        self.model.track_batch_idx = None

        accumulated = None

        with torch.inference_mode():
            for i, (input_spikes, target_spikes) in enumerate(self._chunks):
                # Forward pass (always runs to build model state)
                spikes = self.model.forward(input_spikes=input_spikes)

                # During burn-in: warm up loss state but skip accumulation
                if i < self.burn_in_chunks:
                    for loss_fn in self.loss_functions.values():
                        inputs = {}
                        if hasattr(loss_fn, "required_inputs"):
                            if "output_spikes" in loss_fn.required_inputs:
                                inputs["output_spikes"] = spikes
                        else:
                            inputs["output_spikes"] = spikes
                        if (
                            hasattr(loss_fn, "requires_target")
                            and loss_fn.requires_target
                        ):
                            inputs["target_spikes"] = target_spikes
                        if inputs:
                            loss_fn(**inputs)
                    continue

                # Build chunk_outputs for loss dispatch
                chunk_outputs = {
                    "spikes": spikes,
                    "target_spikes": target_spikes,
                }

                # Compute losses (same dispatch as SNNTrainer._compute_losses)
                chunk_losses = self._compute_losses(chunk_outputs)
                if accumulated is None:
                    accumulated = {k: 0.0 for k in chunk_losses}
                for k, v in chunk_losses.items():
                    accumulated[k] += v

        n = self._n_chunks - self.burn_in_chunks
        return {k: v / n for k, v in accumulated.items()}

    def _compute_losses(self, chunk_outputs: dict) -> dict[str, float]:
        """Compute losses for a single chunk. Returns dict with per-loss and total."""
        spikes = chunk_outputs["spikes"]

        # Check if any loss function needs weights
        needs_weights = any(
            hasattr(loss_fn, "required_inputs")
            and (
                "recurrent_weights" in loss_fn.required_inputs
                or "feedforward_weights" in loss_fn.required_inputs
            )
            for loss_fn in self.loss_functions.values()
        )

        if needs_weights:
            weights_for_loss = self.model.weights
            weights_FF_for_loss = self.model.weights_FF
        else:
            weights_for_loss = None
            weights_FF_for_loss = None

        losses = {}
        total_loss = 0.0

        for loss_name, loss_fn in self.loss_functions.items():
            # Build inputs based on loss function's required_inputs metadata
            inputs = {}
            if hasattr(loss_fn, "required_inputs"):
                for req_input in loss_fn.required_inputs:
                    if req_input == "output_spikes":
                        inputs["output_spikes"] = spikes
                    elif req_input == "voltages":
                        inputs["voltages"] = chunk_outputs.get("voltages")
                    elif req_input == "dt":
                        inputs["dt"] = self.model.dt
                    elif req_input == "recurrent_weights":
                        inputs["recurrent_weights"] = weights_for_loss
                    elif req_input == "feedforward_weights":
                        inputs["feedforward_weights"] = weights_FF_for_loss
                    elif req_input == "cell_type_indices":
                        inputs["cell_type_indices"] = self.model.cell_type_indices
                    elif req_input == "connectome_mask":
                        inputs["connectome_mask"] = getattr(
                            self.model, "connectome_mask", None
                        )
                    elif req_input == "feedforward_mask":
                        inputs["feedforward_mask"] = getattr(
                            self.model, "feedforward_mask", None
                        )
                    elif req_input == "scaling_factors":
                        inputs["scaling_factors"] = self.model.scaling_factors
                    elif req_input == "scaling_factors_FF":
                        inputs["scaling_factors_FF"] = self.model.scaling_factors_FF
            else:
                inputs["output_spikes"] = spikes

            # Add target spikes if the loss requires them
            if hasattr(loss_fn, "requires_target") and loss_fn.requires_target:
                if chunk_outputs["target_spikes"] is not None:
                    inputs["target_spikes"] = chunk_outputs["target_spikes"]
                else:
                    raise ValueError(
                        f"Loss function '{loss_name}' requires target spikes, "
                        "but none were provided by the dataloader."
                    )

            loss_value = loss_fn(**inputs)
            weight = self.loss_weights.get(loss_name, 0.0)
            weighted = weight * loss_value.item()
            losses[loss_name] = weighted
            total_loss += weighted

        losses["total"] = total_loss
        return losses

    def search(self) -> float:
        """
        Run CMA-ES search.

        Uses the initial parameter values as the starting point, runs the
        search, writes the best values back via update_fn, and returns the
        best loss.
        """
        x0 = self._initial_params
        n_evals = 0
        generation = 0

        opts = {
            "maxfevals": self.max_evaluations,
            "verbose": -1,  # suppress CMA-ES output, we print our own
        }
        if self.popsize is not None:
            opts["popsize"] = self.popsize
        if self.seed is not None:
            opts["seed"] = self.seed

        es = cma.CMAEvolutionStrategy(x0.tolist(), self.initial_sigma, opts)

        pbar = tqdm(
            total=self.max_evaluations,
            desc="CMA-ES",
            unit="eval",
        )

        while not es.stop():
            candidates = es.ask()
            fitnesses = []
            all_losses = []
            for candidate in candidates:
                losses = self._evaluate(np.array(candidate))
                fitnesses.append(losses["total"])
                all_losses.append(losses)
                n_evals += 1
                pbar.update(1)

            es.tell(candidates, fitnesses)
            generation += 1

            best_loss = es.result.fbest
            mean_losses = {
                k: float(np.mean([l[k] for l in all_losses])) for k in all_losses[0]
            }
            pbar.set_postfix(best=f"{best_loss:.6f}", sigma=f"{es.sigma:.4f}")

            if self.callback is not None:
                # Write best-so-far params to model so callback can inspect/evaluate
                self.update_fn(np.array(es.result.xbest))

                # Run forward pass with best params to get output spikes (streamed)
                if hasattr(self.model, "reset_state"):
                    self.model.reset_state(batch_size=self._eval_batch_size)
                self.model.track_variables = False
                spike_chunks = []
                with torch.inference_mode():
                    for input_spikes, _ in self._chunks:
                        spikes = self.model.forward(input_spikes=input_spikes)
                        spike_chunks.append(spikes.cpu())
                output_spikes = torch.cat(spike_chunks, dim=1)

                self.callback(
                    {
                        "generation": generation,
                        "n_evals": n_evals,
                        "best_loss": best_loss,
                        "mean_losses": mean_losses,
                        "sigma": es.sigma,
                        "output_spikes": output_spikes,
                    }
                )

        pbar.close()

        # Write best parameters back to the model
        self.update_fn(np.array(es.result.xbest))

        best_loss = es.result.fbest
        print(f"  CMA-ES finished: {n_evals} evaluations, best loss: {best_loss:.6f}")
        return best_loss
