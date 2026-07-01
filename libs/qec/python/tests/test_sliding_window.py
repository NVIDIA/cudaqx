# ============================================================================ #
# Copyright (c) 2025 - 2026 NVIDIA Corporation & Affiliates.                   #
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


@pytest.mark.parametrize("decoder_name", ["single_error_lut", "pymatching"])
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

    dem = qec.z_dem_from_memory_circuit(code,
                                        statePrep,
                                        num_rounds,
                                        noise,
                                        decompose_errors=True)
    num_syndromes_per_round = code.get_num_z_stabilizers()
    effective_num_rounds = (dem.detector_error_matrix.shape[0] //
                            num_syndromes_per_round)

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
    # be the same. The full decoder must use the same merge_strategy as the
    # sliding window's inner decoder (below): a faithful DEM can contain
    # parallel matching edges, and the two decoders only agree if they combine
    # those edges identically.
    full_decoder = qec.get_decoder(decoder_name,
                                   dem.detector_error_matrix,
                                   merge_strategy='smallest_weight')

    sw_as_full_decoder = qec.get_decoder(
        "sliding_window",
        dem.detector_error_matrix,
        window_size=effective_num_rounds - num_windows + 1,
        step_size=1,
        num_syndromes_per_round=num_syndromes_per_round,
        straddle_start_round=False,
        straddle_end_round=True,
        error_rate_vec=np.array(dem.error_rates),
        inner_decoder_name=decoder_name,
        inner_decoder_params={
            'dummy_param': 1,
            'merge_strategy': 'smallest_weight'
        })

    if batched:
        full_results = full_decoder.decode_batch(syndromes)
        sw_results = sw_as_full_decoder.decode_batch(syndromes)
        num_mismatches = np.count_nonzero(
            np.any(full_results.result != sw_results.result, axis=1))
        assert num_mismatches == 0

    else:
        num_mismatches = 0
        for syndrome in syndromes:
            r1 = full_decoder.decode(syndrome)
            r2 = sw_as_full_decoder.decode(syndrome)
            if not np.array_equal(r1.result, r2.result):
                num_mismatches += 1
        assert num_mismatches == 0


def test_pymatching_parallel_edges_use_observable_faults():
    # Same detector syndrome with different observable flips are distinct
    # logical fault mechanisms. After #610 these stay as separate DEM columns,
    # which are parallel edges in the matching graph. This guards the two
    # PyMatching code paths: H-only construction must reject parallel edges,
    # while the observable-aware path must merge them and still decode.
    H = np.array([[1, 1]], dtype=np.uint8)
    O = np.array([[1, 0], [0, 1]], dtype=np.uint8)
    error_rates = np.array([0.1, 0.2], dtype=np.float64)

    # ASSERT: the H-only path keeps the default 'disallow' strategy, so building
    # a graph with two columns on the same edge raises rather than silently
    # collapsing the mechanisms.
    with pytest.raises(ValueError, match="Parallel edges not permitted"):
        qec.get_decoder("pymatching", H)

    # ASSERT: providing O with merge_strategy='independent' combines the
    # parallel edges and yields a converged observable-space decode.
    decoder = qec.get_decoder("pymatching",
                              H,
                              O=O,
                              error_rate_vec=error_rates,
                              merge_strategy="independent")
    result = decoder.decode_batch(np.array([[1]], dtype=np.uint8))

    assert isinstance(result, qec.BatchDecoderResult)
    assert result.result.shape[0] == 1
    assert result.result.shape[1] > 0
    assert result.converged.tolist() == [True]
