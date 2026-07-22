# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
import sys
import platform
if platform.machine().lower() in ("arm64", "aarch64"):
    print(
        "Warning: stim is not supported on manylinux ARM64/aarch64. Skipping this example..."
    )
    sys.exit(0)

if sys.version_info < (3, 11):
    print(
        "Warning: The tensor network noise learner requires Python 3.11 or higher. Exiting..."
    )
    sys.exit(0)

# [Begin Documentation]
"""
Online noise learning with NMOptimizer on a small repetition code.

This example follows the workflow from Nico's original noise-learning
prototype, using the productized NMOptimizer API:

1. Build a distance-3 repetition-code memory circuit with 3 syndrome rounds.
2. Convert the Stim detector error model (DEM) into a parity-check matrix,
   a logical-observable matrix, and reference per-error probabilities.
3. Start from uniform error probabilities and learn one probability per DEM
   error from fresh batches of detector events and observed logical flips.
4. Compare uniform, learned, and true DEM priors on one held-out dataset.

The true per-error probabilities are not supplied to the optimizer. Their mean
is used only to choose a fair uniform starting point, and the individual values
are retained for the final reference comparison. Data errors are deliberately
ten times more likely than measurement errors, giving the optimizer visible
non-uniform structure to learn.

Requirements:
    pip install cudaq-qec[tensor-network-decoder] stim beliefmatching
"""

import numpy as np
import torch
import stim
from beliefmatching.belief_matching import detector_error_model_to_check_matrices

from cudaq_qec import NMOptimizer, make_compiled_step

CODE_DISTANCE = 3
NUM_ROUNDS = 3
DATA_ERROR_PROBABILITY = 0.05
MEASUREMENT_ERROR_PROBABILITY = 0.005

TRAIN_SHOTS = 5000
EVAL_SHOTS = 20000
ITERS = 100
LR = 1e-2
DTYPE = "float64"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
EXECUTE = "codegen"


def parse_detector_error_model(dem):
    """Convert a Stim DEM into the matrices and priors NMOptimizer uses."""
    matrices = detector_error_model_to_check_matrices(dem)
    H = np.zeros(matrices.check_matrix.shape)
    matrices.check_matrix.astype(np.float64).toarray(out=H)
    L = np.zeros(matrices.observables_matrix.shape)
    matrices.observables_matrix.astype(np.float64).toarray(out=L)
    priors = np.array([float(p) for p in matrices.priors], dtype=np.float64)
    return H, L, priors


def make_sampler(circuit, seed):
    """Create a reproducible sampler when supported by the Stim version."""
    try:
        return circuit.compile_detector_sampler(seed=seed)
    except TypeError:
        return circuit.compile_detector_sampler()


def sample_batch(sampler, shots):
    """Sample detector events and their corresponding logical-flip labels."""
    det_events, obs_flips = sampler.sample(shots, separate_observables=True)
    return det_events.astype(float), obs_flips.ravel().astype(bool)


def probs_to_logits(probs):
    """Map probabilities to unconstrained values for stable optimization."""
    probs = np.clip(probs, 1e-7, 1.0 - 1e-7)
    return np.log(probs / (1.0 - probs))


def evaluate_ler(H, L, priors, det_events, obs_flips):
    """Evaluate one prior model on a fixed held-out dataset."""
    opt = NMOptimizer(H,
                      L,
                      priors.tolist(),
                      det_events,
                      obs_flips,
                      dtype=DTYPE,
                      device=DEVICE,
                      execute=EXECUTE)
    return opt.logical_error_rate()


def main():
    # d=3 is the smallest repetition code that corrects one data error.
    # Three rounds add repeated syndrome information while keeping the
    # tensor network small enough for a quick-start example.
    circuit = stim.Circuit.generated(
        "repetition_code:memory",
        rounds=NUM_ROUNDS,
        distance=CODE_DISTANCE,
        before_round_data_depolarization=DATA_ERROR_PROBABILITY,
        before_measure_flip_probability=MEASUREMENT_ERROR_PROBABILITY,
    )
    dem = circuit.detector_error_model(decompose_errors=True)
    H, L, true_priors = parse_detector_error_model(dem)
    n_checks, n_errors = H.shape

    print(f"Circuit: repetition code, distance={CODE_DISTANCE}, "
          f"rounds={NUM_ROUNDS}")
    print(f"DEM: {n_checks} checks, {n_errors} errors")
    print(f"True priors:  mean={true_priors.mean():.4e}  "
          f"min={true_priors.min():.4e}  max={true_priors.max():.4e}")
    print(f"Device: {DEVICE}, execute={EXECUTE}")

    # Refreshing the batch at every step models online calibration from a
    # stream of experimental syndromes instead of fitting one finite batch.
    train_sampler = make_sampler(circuit, seed=1234)
    det_events, obs_flips = sample_batch(train_sampler, TRAIN_SHOTS)

    # The optimizer sees only this uniform initial model, not the individual
    # true DEM probabilities.
    uniform = np.full(n_errors, true_priors.mean(), dtype=np.float64)
    opt = NMOptimizer(H,
                      L,
                      uniform.tolist(),
                      det_events,
                      obs_flips,
                      dtype=DTYPE,
                      device=DEVICE,
                      execute=EXECUTE)

    # Optimize logits so sigmoid(logits) always remains a valid probability.
    logits = torch.tensor(probs_to_logits(uniform),
                          dtype=getattr(torch, DTYPE),
                          device=opt.torch_device,
                          requires_grad=True)
    adam = torch.optim.Adam([logits], lr=LR)
    step_fn = make_compiled_step(opt, logits, adam)

    losses = []
    mae_history = []
    true_probs = torch.tensor(true_priors,
                              dtype=getattr(torch, DTYPE),
                              device=opt.torch_device)
    for it in range(ITERS):
        if it > 0:
            det_events, obs_flips = sample_batch(train_sampler, TRAIN_SHOTS)
            opt.update_dataset(det_events, obs_flips)
        loss = step_fn()
        learned_probs = torch.sigmoid(logits)
        losses.append(float(loss.detach().cpu()))
        mae_history.append(
            float(
                torch.mean(torch.abs(learned_probs -
                                     true_probs)).detach().cpu()))

    learned = torch.sigmoid(logits).detach().cpu().numpy()

    print(f"Loss:      {losses[0]:.4e} -> {losses[-1]:.4e} "
          f"({ITERS} online Adam steps, {TRAIN_SHOTS} shots/step)")
    print(f"Prior MAE: {mae_history[0]:.4e} -> {mae_history[-1]:.4e}")
    print(f"Learned priors: mean={learned.mean():.4e}  "
          f"min={learned.min():.4e}  max={learned.max():.4e}")

    # Reuse one unseen dataset for all models so only the priors differ.
    eval_sampler = make_sampler(circuit, seed=4321)
    eval_events, eval_flips = sample_batch(eval_sampler, EVAL_SHOTS)
    ler_uniform = evaluate_ler(H, L, uniform, eval_events, eval_flips)
    ler_learned = evaluate_ler(H, L, learned, eval_events, eval_flips)
    ler_true = evaluate_ler(H, L, true_priors, eval_events, eval_flips)

    print(f"Held-out LER (uniform priors): {ler_uniform:.4f}")
    print(f"Held-out LER (learned priors): {ler_learned:.4f}")
    print(f"Held-out LER (true DEM priors): {ler_true:.4f}")
    print(f"Absolute LER improvement: {ler_uniform - ler_learned:+.4f}")


if __name__ == "__main__":
    main()
