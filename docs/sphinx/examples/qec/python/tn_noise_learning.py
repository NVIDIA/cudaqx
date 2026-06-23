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
Online noise learning with NMOptimizer on a Stim surface-code DEM.

This example demonstrates how to use NMOptimizer to fit per-error noise
probabilities from syndrome data sampled from a Stim detector error model.
The optimizer is initialized away from the DEM priors, then each training
iteration resamples a fresh syndrome batch and tracks cross-entropy, prior
recovery, and logical error-rate dynamics.

For a low-noise code, LER is a noisy metric; representative experiments often
need 10k-30k shots per iteration.  The default below is smaller so the example
stays quick to run.

Requirements:
    pip install cudaq-qec[tensor-network-decoder] stim beliefmatching
"""

import numpy as np
import torch
import stim
from beliefmatching.belief_matching import detector_error_model_to_check_matrices

from cudaq_qec import NMOptimizer, make_compiled_step

BATCH_SHOTS = 1000
ITERS = 20
LR = 1e-2
DTYPE = "float64"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def parse_detector_error_model(dem):
    matrices = detector_error_model_to_check_matrices(dem)
    H = np.zeros(matrices.check_matrix.shape)
    matrices.check_matrix.astype(np.float64).toarray(out=H)
    L = np.zeros(matrices.observables_matrix.shape)
    matrices.observables_matrix.astype(np.float64).toarray(out=L)
    priors = np.array([float(p) for p in matrices.priors], dtype=np.float64)
    return H, L, priors


def sample_batch(circuit, shots):
    det_events, obs_flips = circuit.compile_detector_sampler().sample(
        shots, separate_observables=True)
    return det_events.astype(float), obs_flips.ravel().astype(bool)


def probs_to_logits(probs):
    probs = np.clip(probs, 1e-7, 1.0 - 1e-7)
    return np.log(probs / (1.0 - probs))


def logical_error_rate(opt, probs):
    original = opt.noise_params[0]
    try:
        opt._noise_probs = probs
        return opt.logical_error_rate()
    finally:
        opt._noise_probs = original


def main():
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        rounds=3,
        distance=3,
        before_round_data_depolarization=0.005,
    )
    dem = circuit.detector_error_model(decompose_errors=True)
    H, L, true_priors = parse_detector_error_model(dem)
    n_checks, n_errors = H.shape

    print(f"DEM: {n_checks} checks, {n_errors} errors")
    print(f"True priors:  mean={true_priors.mean():.4e}  "
          f"min={true_priors.min():.4e}  max={true_priors.max():.4e}")
    print(f"Device: {DEVICE}")

    det_events, obs_flips = sample_batch(circuit, BATCH_SHOTS)
    uniform = np.full(n_errors, true_priors.mean(), dtype=np.float64)
    opt = NMOptimizer(H,
                      L,
                      uniform.tolist(),
                      det_events,
                      obs_flips,
                      dtype=DTYPE,
                      device=DEVICE,
                      execute="codegen")

    logits = torch.tensor(probs_to_logits(uniform),
                          dtype=getattr(torch, DTYPE),
                          device=opt.torch_device,
                          requires_grad=True)
    uniform_probs = torch.tensor(uniform,
                                 dtype=getattr(torch, DTYPE),
                                 device=opt.torch_device)
    true_probs = torch.tensor(true_priors,
                              dtype=getattr(torch, DTYPE),
                              device=opt.torch_device)
    adam = torch.optim.Adam([logits], lr=LR)
    step_fn = make_compiled_step(opt, logits, adam)

    print(
        "iter | loss       | learned LER | uniform LER | true-prior LER | prior MAE"
    )
    for it in range(1, ITERS + 1):
        if it > 1:
            det_events, obs_flips = sample_batch(circuit, BATCH_SHOTS)
            opt.update_dataset(det_events, obs_flips)

        loss = step_fn()
        learned_probs = torch.sigmoid(logits)
        prior_mae = float(
            torch.mean(torch.abs(learned_probs - true_probs)).detach().cpu())
        ler_learned = logical_error_rate(opt, learned_probs)
        ler_uniform = logical_error_rate(opt, uniform_probs)
        ler_true = logical_error_rate(opt, true_probs)

        print(f"{it:4d} | {float(loss.detach().cpu()):.4e} | "
              f"{ler_learned:.4f}      | {ler_uniform:.4f}     | "
              f"{ler_true:.4f}         | {prior_mae:.4e}")

    learned = torch.sigmoid(logits).detach().cpu().numpy()
    print(f"Learned priors: mean={learned.mean():.4e}  "
          f"min={learned.min():.4e}  max={learned.max():.4e}")


if __name__ == "__main__":
    main()
