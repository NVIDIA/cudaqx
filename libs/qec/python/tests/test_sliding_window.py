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

    # Compare the full decoder against the sliding window decoder (configured as
    # a single full-size window). For column-space decoders (single_error_lut)
    # the corrections must be identical; for matching decoders (pymatching),
    # same-H/different-O degeneracy makes the exact error column non-unique, so
    # we instead require each correction to reproduce the input syndrome.
    if decoder_name == "pymatching":
        # canonicalize_for_rounds keeps same-syndrome/different-observable
        # columns distinct; these are parallel edges in the matching graph that
        # PyMatching's default 'disallow' strategy rejects, so merge them.
        full_decoder = qec.get_decoder(decoder_name,
                                       dem.detector_error_matrix,
                                       merge_strategy="independent")
    else:
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
        inner_decoder_params={
            'dummy_param': 1,
            'merge_strategy': 'smallest_weight'
        })

    # H maps an error (column space) to the syndrome it produces (mod 2).
    H = np.asarray(dem.detector_error_matrix, dtype=np.int64)
    # pymatching is a graph/matching decoder: for degenerate same-H/different-O
    # groups it cannot return a unique column, so validate by syndrome
    # reproduction rather than exact column equality.
    check_syndrome_only = decoder_name == "pymatching"

    if batched:
        full_results = full_decoder.decode_batch(syndromes)
        sw_results = sw_as_full_decoder.decode_batch(syndromes)
        if check_syndrome_only:
            full_e = np.asarray(full_results.result, dtype=np.int64)
            sw_e = np.asarray(sw_results.result, dtype=np.int64)
            target = syndromes.astype(np.int64)
            # ASSERT: every correction (full and windowed) explains the observed
            # syndrome, i.e. (H @ e) % 2 == syndrome for all shots.
            assert np.array_equal((full_e @ H.T) % 2, target)
            assert np.array_equal((sw_e @ H.T) % 2, target)
        else:
            # ASSERT: column-space decoders produce identical corrections.
            num_mismatches = np.count_nonzero(
                np.any(full_results.result != sw_results.result, axis=1))
            assert num_mismatches == 0

    else:
        num_mismatches = 0
        for syndrome in syndromes:
            r1 = full_decoder.decode(syndrome)
            r2 = sw_as_full_decoder.decode(syndrome)
            if check_syndrome_only:
                # A correction is valid if it reproduces the observed syndrome
                # via H (mod 2); count a mismatch if either decoder fails to.
                target = np.asarray(syndrome, dtype=np.int64)
                e1 = np.asarray(r1.result, dtype=np.int64)
                e2 = np.asarray(r2.result, dtype=np.int64)
                if not (np.array_equal((H @ e1) % 2, target) and np.array_equal(
                    (H @ e2) % 2, target)):
                    num_mismatches += 1
            elif not np.array_equal(r1.result, r2.result):
                num_mismatches += 1
        assert num_mismatches == 0


def test_pymatching_parallel_edges_use_observable_faults():
    # Same detector syndrome with different observable flips represents
    # distinct logical fault mechanisms. H-only PyMatching cannot distinguish
    # them, so the observable-aware path must preserve O and merge parallel
    # graph edges inside PyMatching rather than dropping DEM columns.
    H = np.array([[1, 1]], dtype=np.uint8)
    O = np.array([[1, 0], [0, 1]], dtype=np.uint8)
    error_rates = np.array([0.1, 0.2], dtype=np.float64)

    with pytest.raises(ValueError, match="Parallel edges not permitted"):
        qec.get_decoder("pymatching", H)

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
