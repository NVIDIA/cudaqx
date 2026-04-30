# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# This is a draft test script that should be improved and run by the CI (where possible).

import stim
import cudaq_qec as qec
from beliefmatching.belief_matching import detector_error_model_to_check_matrices
import numpy as np
import time

d = 13
e = 0.003
# Generate a Stim circuit for the surface code
# circuit = stim.Circuit.generated(
#     "surface_code:rotated_memory_z",
#     distance=d,
#     rounds=1*d,
#     after_clifford_depolarization=e,
#     before_round_data_depolarization=e,
#     before_measure_flip_probability=e,
#     after_reset_flip_probability=e,
# )

circuit = stim.Circuit.from_file("/workspaces/pre-decoder/circuit_Z.stim")

# Get the Detector Error Model (DEM) from the Stim circuit
dem = circuit.detector_error_model(decompose_errors=True,
                                   approximate_disjoint_errors=True)
# print(dem)
# exit()

matrices = detector_error_model_to_check_matrices(dem)
# H = matrices.check_matrix
# O = matrices.observables_matrix
H = matrices.edge_check_matrix
O = matrices.edge_observables_matrix
# priors = matrices.priors
edge_probs = matrices.hyperedge_to_edge_matrix @ matrices.priors
eps = 1e-14
edge_probs[edge_probs > 1 - eps] = 1 - eps
edge_probs[edge_probs < eps] = eps
priors = edge_probs
print(f"Shape of priors: {priors.shape}")
print(f"Shape of H: {H.shape}")
print(f"Shape of O: {O.shape}")
print(
    f"Shape of matrices.hyperedge_to_edge_matrix: {matrices.hyperedge_to_edge_matrix.shape}"
)
H_dense = H.todense(order="C").astype(np.uint8)
O_dense = O.todense(order="C").astype(np.uint8)

# If there is a global decoder, the H_dense will be passed to the global
# decoder. Additionally, when there is a global decoder, it is assumed that the
# last portion of the syndrome corresponds to the boundary detectors.
dec = qec.get_decoder(
    "trt_decoder",
    H_dense,
    O=O_dense,
    #engine_load_path="/workspaces/pre-decoder/predecoder_memory_d13_T13_Z.engine",
    onnx_load_path="/workspaces/pre-decoder/predecoder_memory_d13_T13_Z.onnx",
    use_cuda_graph=False,
    batch_size=7,
    global_decoder="pymatching",
    global_decoder_params={
        "merge_strategy": "independent",
        "O": O_dense,
    })

sampler = circuit.compile_detector_sampler(seed=42)
dets, obs = sampler.sample(2048, separate_observables=True)
# Print the shape of dets and obs
print(f"Shape of dets: {dets.shape}")
print(f"Shape of obs: {obs.shape}")

results = dec.decode_batch(dets)
# for i in range(min(20, len(results))):
#     print(f"Result {i}: {results[i].result[0]}, len: {len(results[i].result)}, obs: {obs[i]}")

num_mismatches = 0
for i in range(min(len(results), len(obs))):
    if results[i].result[0] != obs[i]:
        num_mismatches += 1
print(
    f"Number of mismatches: {num_mismatches} out of {min(len(results), len(obs))}"
)
