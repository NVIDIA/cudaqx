# ============================================================================ #
# Copyright (c) 2025 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
import sys
# Check Python version
# Remove this check once the pipeline advances to Python 3.11
if sys.version_info < (3, 11):
    print("Skipping tensor network decoder example")
    sys.exit(0)

# [Begin Documentation]
"""
Example usage of TensorNetworkDecoder from cudaq_qec.

This script demonstrates how to instantiate and use the tensor network decoder
to decode a circuit level noise problem derived from a Stim surface code experiment.

This example requires the `cudaq-qec` package and the optional tensor-network dependencies.
To install the required dependencies, run:

pip install cudaq-qec[tn_decoder]

Additionaly, you will need `stim` and `beliefmatching` packages:
pip install stim beliefmatching

"""
import cudaq_qec as qec
import numpy as np

import stim

from beliefmatching.belief_matching import detector_error_model_to_check_matrices


def parse_detector_error_model(detector_error_model):
    matrices = detector_error_model_to_check_matrices(detector_error_model)

    out_H = np.zeros(matrices.check_matrix.shape)
    matrices.check_matrix.astype(np.float64).toarray(out=out_H)
    out_L = np.zeros(matrices.observables_matrix.shape)
    matrices.observables_matrix.astype(np.float64).toarray(out=out_L)

    return out_H, out_L, [float(p) for p in matrices.priors]


circuit = stim.Circuit.generated("surface_code:rotated_memory_z",
                                 rounds=3,
                                 distance=3,
                                 after_clifford_depolarization=0.001,
                                 after_reset_flip_probability=0.01,
                                 before_measure_flip_probability=0.01,
                                 before_round_data_depolarization=0.01)

detector_error_model = circuit.detector_error_model(decompose_errors=True)

H, logicals, noise_model = parse_detector_error_model(detector_error_model)

decoder = qec.get_decoder(
    "tensor_network_decoder",
    H,
    logical_obs=logicals,
    noise_model=noise_model,
    contract_noise_model=True,
)

num_shots = 5
sampler = circuit.compile_detector_sampler()
detection_events, observable_flips = sampler.sample(num_shots,
                                                    separate_observables=True)

res = decoder.decode_batch(detection_events)

print("Tensor network prediction: ", [r.result[0] > 0.5 for r in res])
print("Actual observable flips: ", [bool(o[0]) for o in observable_flips])
