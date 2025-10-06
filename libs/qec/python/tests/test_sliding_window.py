# ============================================================================ #
# Copyright (c) 2025 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import numpy as np
import cudaq
import cudaq_qec as qec
import pytest


@pytest.fixture(scope="function", autouse=True)
def setTarget():
    old_target = cudaq.get_target()
    cudaq.set_target('stim')
    yield
    cudaq.set_target(old_target)


@pytest.mark.parametrize("decoder_name", ["single_error_lut"])
@pytest.mark.parametrize("batched", [True, False])
@pytest.mark.parametrize("num_rounds", [5, 10])
@pytest.mark.parametrize("num_windows", [1, 2, 3])
def test_sliding_window_1(decoder_name, batched, num_rounds, num_windows):
    cudaq.set_random_seed(13)
    code = qec.get_code('surface_code', distance=5)
    p = 0.001
    noise = cudaq.NoiseModel()
    noise.add_all_qubit_channel("x", cudaq.Depolarization2(p), 1)
    statePrep = qec.operation.prep0
    nShots = 1000

    dem = qec.z_dem_from_memory_circuit(code, statePrep, num_rounds, noise)
    num_syndromes_per_round = dem.detector_error_matrix.shape[0] // num_rounds

    # Inject only one error per shot. This will keep the number of mismatches
    # low, and any debug should be straightforward.
    syndromes = np.zeros((nShots, dem.detector_error_matrix.shape[0]),
                         dtype=np.uint8)
    np.random.seed(13)
    for shot in range(nShots):
        # Pick a single random error to inject
        col = np.random.randint(0, dem.detector_error_matrix.shape[1])
        syndromes[shot, :] = dem.detector_error_matrix[:, col].T

    # First compare the results of the full decoder to the sliding window
    # decoder using an inner decoder of the full window size. The results should
    # be the same.
    full_decoder = qec.get_decoder(decoder_name, dem.detector_error_matrix)
    num_syndromes_per_round = dem.detector_error_matrix.shape[0] // num_rounds
    sw_as_full_decoder = qec.get_decoder(
        "sliding_window",
        dem.detector_error_matrix,
        window_size=num_rounds - num_windows + 1,
        step_size=1,
        num_syndromes_per_round=num_syndromes_per_round,
        straddle_start_round=False,
        straddle_end_round=True,
        error_rate_vec=np.array(dem.error_rates),
        inner_decoder_name=decoder_name,
        inner_decoder_params={'dummy_parm': 1})
    if batched:
        full_results = full_decoder.decode_batch(syndromes)
        sw_results = sw_as_full_decoder.decode_batch(syndromes)
        num_mismatches = 0
        for r1, r2 in zip(full_results, sw_results):
            if r1.result != r2.result:
                num_mismatches += 1
        assert num_mismatches == 0

    else:
        full_results = []
        sw_results = []
        num_mismatches = 0
        for syndrome in syndromes:
            r1 = full_decoder.decode(syndrome)
            r2 = sw_as_full_decoder.decode(syndrome)
            if r1.result != r2.result:
                num_mismatches += 1
        assert num_mismatches == 0
