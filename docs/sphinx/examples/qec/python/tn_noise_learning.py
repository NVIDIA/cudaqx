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
Noise learning with NMOptimizer on a Stim repetition-code circuit.

This script demonstrates how to use NMOptimizer to fit per-error noise
probabilities to syndrome data sampled from a Stim repetition-code memory
experiment.  Starting from uniform initial priors, Adam optimization on
logits drives the cross-entropy loss down toward the true DEM error rates,
and a held-out evaluation compares the learned model's logical error rate
against the static uniform-prior baseline.

Requirements:
    pip install cudaq-qec[tensor-network-decoder] stim beliefmatching
"""

import numpy as np
import torch
import stim
from beliefmatching.belief_matching import detector_error_model_to_check_matrices

import cudaq_qec as qec
from cudaq_qec import NMOptimizer, make_compiled_step


def parse_detector_error_model(dem):
    matrices = detector_error_model_to_check_matrices(dem)
    H = np.zeros(matrices.check_matrix.shape)
    matrices.check_matrix.astype(np.float64).toarray(out=H)
    L = np.zeros(matrices.observables_matrix.shape)
    matrices.observables_matrix.astype(np.float64).toarray(out=L)
    priors = [float(p) for p in matrices.priors]
    return H, L, priors


def main():
    # Asymmetric noise (data 10x measurement) so the uniform initial
    # prior is meaningfully wrong and the optimizer has signal to
    # learn; with symmetric noise, uniform is already near-optimal.
    circuit = stim.Circuit.generated(
        "repetition_code:memory",
        rounds=5,
        distance=3,
        before_round_data_depolarization=0.05,
        before_measure_flip_probability=0.005,
    )
    dem = circuit.detector_error_model(decompose_errors=True)
    H, L, true_priors = parse_detector_error_model(dem)
    true_probs = np.array(true_priors)
    n_checks, n_errors = H.shape
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"DEM: {n_checks} checks, {n_errors} errors")
    print(f"True priors:  mean={true_probs.mean():.4e}  "
          f"min={true_probs.min():.4e}  max={true_probs.max():.4e}  "
          f"(spread {true_probs.max() / true_probs.min():.1f}x)")

    num_shots = 1000
    sampler = circuit.compile_detector_sampler()
    det_events, obs_flips = sampler.sample(num_shots, separate_observables=True)
    det_events = det_events.astype(float)
    obs_flips = obs_flips.ravel().astype(bool)

    uniform = float(true_probs.mean())
    # precontract_noise=True is the recommended reduced-topology path
    # for larger detector-error models.  Set precontract_noise=False
    # only when explicitly checking the full tensor-network contraction.
    opt = NMOptimizer(H,
                      L, [uniform] * n_errors,
                      det_events,
                      obs_flips,
                      dtype="float64",
                      device=device,
                      execute="opt_einsum",
                      precontract_noise=True)

    # Optimize in logit space — numerically stabler than raw probs.
    def _to_logits(p):
        p = np.clip(p, 1e-7, 1 - 1e-7)
        return -np.log(1.0 / p - 1.0)

    logits = torch.tensor(
        _to_logits(np.full(n_errors, uniform)),
        dtype=torch.float64,
        device=opt.torch_device,
        requires_grad=True,
    )
    adam = torch.optim.Adam([logits], lr=1e-2)
    step_fn = make_compiled_step(opt, logits, adam)

    iters = 300
    losses = [float(step_fn().detach().cpu()) for _ in range(iters)]
    learned = torch.sigmoid(logits).detach().cpu().numpy()

    print(f"Loss:           {losses[0]:.2f} -> {losses[-1]:.2f} "
          f"({iters} Adam steps)")
    print(f"True priors:    mean={true_probs.mean():.4e}  "
          f"min={true_probs.min():.4e}  max={true_probs.max():.4e}")
    print(f"Learned priors: mean={learned.mean():.4e}  "
          f"min={learned.min():.4e}  max={learned.max():.4e}")

    if losses[-1] >= losses[0]:
        raise RuntimeError(f"Training did not reduce loss at all: "
                           f"{losses[0]:.2f} -> {losses[-1]:.2f}")

    # Held-out LER comparison is the real gate: a noise model is only
    # useful if it decodes better than uniform priors.  20k shots keeps
    # the per-run std of the (static - learned) difference around 0.001,
    # so the +0.002 gate sits many sigmas below the expected gain even
    # without a fixed RNG seed.
    num_test = 20000
    test_events, test_flips = sampler.sample(num_test,
                                             separate_observables=True)
    test_events = test_events.astype(float)
    test_flips_bool = test_flips.ravel().astype(bool)

    def _ler(noise: list[float]) -> float:
        decoder = qec.get_decoder(
            "tensor_network_decoder",
            H,
            logical_obs=L,
            noise_model=noise,
            contract_noise_model=True,
        )
        res = decoder.decode_batch(test_events)
        pred = np.array([r.result[0] > 0.5 for r in res], dtype=bool)
        return float(np.mean(pred != test_flips_bool))

    ler_static = _ler([uniform] * n_errors)
    ler_learned = _ler(learned.tolist())

    print(f"LER (static uniform priors):  {ler_static:.4f}  ({num_test} shots)")
    print(
        f"LER (learned priors):         {ler_learned:.4f}  ({num_test} shots)")
    print(f"Absolute improvement:          {ler_static - ler_learned:+.4f}")

    min_improvement = 0.002
    if ler_static - ler_learned < min_improvement:
        raise RuntimeError(
            f"Learned LER ({ler_learned:.4f}) did not beat the static "
            f"baseline ({ler_static:.4f}) by at least {min_improvement:.4f}.")


if __name__ == "__main__":
    main()
