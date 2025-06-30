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
    Lz = code.get_observables_z()
    p = 0.001
    noise = cudaq.NoiseModel()
    noise.add_all_qubit_channel("x", cudaq.Depolarization2(p), 1)
    statePrep = qec.operation.prep0
    nShots = 200

    # Sample the memory circuit with errors
    syndromes, data = qec.sample_memory_circuit(code, statePrep, nShots,
                                                num_rounds, noise)

    logical_measurements = (Lz @ data.transpose()) % 2
    # only one logical qubit, so do not need the second axis
    logical_measurements = logical_measurements.flatten()

    # Reshape and drop the X stabilizers, keeping just the Z stabilizers (since this is prep0)
    syndromes = syndromes.reshape((nShots, num_rounds, -1))
    syndromes = syndromes[:, :, :syndromes.shape[2] // 2]
    # Now flatten to two dimensions again
    syndromes = syndromes.reshape((nShots, -1))

    dem = qec.z_dem_from_memory_circuit(code, statePrep, num_rounds, noise)

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
        straddle_end_round=False,
        error_rate_vec=np.array(dem.error_rates),
        inner_decoder_name=decoder_name,
        inner_decoder_params={})
    if batched:
        full_results = full_decoder.decode_batch(syndromes)
        sw_results = sw_as_full_decoder.decode_batch(syndromes)
        num_mismatches = 0
        for r1, r2 in zip(full_results, sw_results):
            if r1.result != r2.result:
                num_mismatches += 1
        # The 0.05 is a fudge factor to account for the fact that the results
        # are not exactly the same when using sliding windows.
        assert num_mismatches <= (num_windows - 1) * 0.05 * nShots

    else:
        full_results = []
        sw_results = []
        num_mismatches = 0
        for syndrome in syndromes:
            r1 = full_decoder.decode(syndrome)
            r2 = sw_as_full_decoder.decode(syndrome)
            if r1.result != r2.result:
                num_mismatches += 1
        # The 0.05 is a fudge factor to account for the fact that the results
        # are not exactly the same when using sliding windows.
        assert num_mismatches <= (num_windows - 1) * 0.05 * nShots
