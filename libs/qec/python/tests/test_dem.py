# ============================================================================ #
# Copyright (c) 2024 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
import pytest
import numpy as np
import cudaq
import cudaq_qec as qec

import subprocess


def is_nvidia_gpu_available():
    """Check if NVIDIA GPU is available using nvidia-smi command."""
    try:
        result = subprocess.run(["nvidia-smi"],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                text=True)
        if result.returncode == 0 and "GPU" in result.stdout:
            return True
    except FileNotFoundError:
        # The nvidia-smi command is not found, indicating no NVIDIA GPU drivers
        return False
    return False


# Helper function to convert a binary matrix to a convenient string
def mat_to_str(mat):
    s = ''
    if mat.ndim == 1:
        for col in mat:
            if col == 0:
                s += '.'
            elif col < 0:
                s += '-'
            elif col >= 10:
                s += '*'
            else:
                s += str(int(col))
        s += '\n'
        return s
    else:
        for row in mat:
            for col in row:
                if col == 0:
                    s += '.'
                elif col < 0:
                    s += '-'
                elif col >= 10:
                    s += '*'
                else:
                    s += str(int(col))
            s += '\n'
    return s


# Use the fixture to set the target
@pytest.fixture(scope="module", autouse=True)
def set_target():
    cudaq.set_target("stim")
    yield
    cudaq.reset_target()


@pytest.mark.parametrize(
    "dem_fn",
    [
        qec.dem_from_memory_circuit,
        qec.x_dem_from_memory_circuit,
        qec.z_dem_from_memory_circuit,
    ],
)
def test_dem_from_memory_circuit_requires_noise_model(dem_fn):
    # DEM generation needs noise mechanisms, so omitted/None noise must fail
    # before the binding dereferences an empty optional noise model.
    code = qec.get_code('steane')
    statePrep = qec.operation.prep0

    with pytest.raises(RuntimeError, match="requires a noise model"):
        dem_fn(code, statePrep, 1)

    with pytest.raises(RuntimeError, match="requires a noise model"):
        dem_fn(code, statePrep, 1, None)


def test_dem_from_memory_circuit():
    code = qec.get_code('steane')
    p = 0.01
    noise = cudaq.NoiseModel()
    noise.add_all_qubit_channel("x", cudaq.Depolarization2(p), 1)
    statePrep = qec.operation.prep0
    nRounds = 2

    dem = qec.z_dem_from_memory_circuit(code, statePrep, nRounds, noise)
    # Mechanisms that share a detector syndrome but flip a different observable
    # are kept as distinct columns (they are not merged), so the same detector
    # column pattern can appear more than once with a distinct observable row.
    expected_detector_error_matrix = """
11111......11111111111111111......................................
..1.11111...11..1111.111....111111111111111111....................
...11..1111..11..1.1.11111.....111...11111....1111111111..........
...........1111...111.11.1111....11....1.111..1..1..11..1111......
...............111111..1...1.111111.....11.111.111...111.1.11111..
.....................1111111.......11111111111....111111..11..1111
"""
    # Uncomment the following line to get a string representation of the DEM
    # that you can compare to expected_detector_error_matrix.
    # print(mat_to_str(dem.detector_error_matrix), end='')
    assert '\n' + mat_to_str(
        dem.detector_error_matrix) == expected_detector_error_matrix

    # Uncomment the following line to get a string representation of the error
    # rates that you can compare to expected_error_rates.
    # print(np.round(dem.error_rates, 4))

    # The following error rates were captured from the above print statement and
    # are considered "truth" data now.
    expected_error_rates = [
        0.0119, 0.0053, 0.0145, 0.0086, 0.0100, 0.0229, 0.0047, 0.0013, 0.0165,
        0.0100, 0.0093, 0.0073, 0.0040, 0.0027, 0.0047, 0.0013, 0.0027, 0.0013,
        0.0027, 0.0020, 0.0007, 0.0020, 0.0013, 0.0020, 0.0007, 0.0020, 0.0020,
        0.0007, 0.0040, 0.0020, 0.0020, 0.0007, 0.0047, 0.0007, 0.0040, 0.0027,
        0.0007, 0.0007, 0.0007, 0.0007, 0.0020, 0.0013, 0.0007, 0.0007, 0.0027,
        0.0007, 0.0007, 0.0040, 0.0013, 0.0013, 0.0027, 0.0053, 0.0027, 0.0047,
        0.0027, 0.0007, 0.0267, 0.0100, 0.0067, 0.0093, 0.0248, 0.0053, 0.0100,
        0.0027, 0.0305, 0.0093
    ]
    assert np.allclose(dem.error_rates, expected_error_rates, atol=1e-4)

    expected_observables_flips_matrix = (
        '.1....1.1.1...................1.1...1.1.1....1..1..1..1......1.1.1\n')
    # Uncomment the following line to get a string representation of the
    # observables flips matrix that you can compare to
    # expected_observables_flips_matrix.
    assert mat_to_str(
        dem.observables_flips_matrix) == expected_observables_flips_matrix


def test_x_dem_from_memory_circuit():
    code = qec.get_code('steane')
    p = 0.01
    noise = cudaq.NoiseModel()
    # X stabilizers detect Z errors, but we need X errors for X basis prep
    noise.add_all_qubit_channel("x", cudaq.Depolarization2(p), 1)
    statePrep = qec.operation.prepp  # X basis preparation
    nRounds = 2

    dem = qec.x_dem_from_memory_circuit(code, statePrep, nRounds, noise)

    # X DEM should have different structure from Z DEM
    # Verify basic properties
    assert dem.detector_error_matrix.shape[0] > 0
    assert dem.detector_error_matrix.shape[1] > 0
    assert len(dem.error_rates) == dem.detector_error_matrix.shape[1]
    assert dem.observables_flips_matrix.shape[0] > 0
    assert dem.observables_flips_matrix.shape[
        1] == dem.detector_error_matrix.shape[1]

    # Error rates should be positive
    assert all(rate >= 0 for rate in dem.error_rates)
    # At least some non-zero rates
    assert any(rate > 0 for rate in dem.error_rates)


def test_decoding_from_dem_from_memory_circuit():
    cudaq.set_random_seed(13)
    code = qec.get_code('steane')
    Lz = code.get_observables_z()
    p = 0.001
    noise = cudaq.NoiseModel()
    noise.add_all_qubit_channel("x", cudaq.Depolarization2(p), 1)
    statePrep = qec.operation.prep0
    nRounds = 6
    nShots = 200

    dem = qec.dem_from_memory_circuit(code, statePrep, nRounds, noise)
    syndromes, sampled_errors = qec.dem_sampling(
        dem.detector_error_matrix, nShots, dem.error_rates)
    syndromes = np.asarray(syndromes, dtype=np.uint8)
    sampled_errors = np.asarray(sampled_errors, dtype=np.uint8)

    logical_measurements = (
        (dem.observables_flips_matrix @ sampled_errors.T) % 2
    ).flatten().astype(np.uint8)

    decoder = qec.get_decoder('single_error_lut', dem.detector_error_matrix)
    dr = decoder.decode_batch(syndromes)
    error_predictions = np.asarray(dr.result, dtype=np.uint8)
    dr_converged = np.asarray(dr.converged, dtype=np.uint8)

    data_predictions = (dem.observables_flips_matrix @ error_predictions.T) % 2
    print(f'data_predictions.shape: {data_predictions.shape}')
    print(f'dr_converged        : {mat_to_str(dr_converged)}', end='')
    print(f'data_predictions    : {mat_to_str(data_predictions)}', end='')
    print(f'logical_measurements: {mat_to_str(logical_measurements)}', end='')

    nLogicalErrorsWithoutDecoding = np.sum(logical_measurements)
    nLogicalErrorsWithDecoding = np.sum(data_predictions ^ logical_measurements)
    print(f'nLogicalErrorsWithoutDecoding : {nLogicalErrorsWithoutDecoding}')
    print(f'nLogicalErrorsWithDecoding    : {nLogicalErrorsWithDecoding}')
    assert nLogicalErrorsWithDecoding < nLogicalErrorsWithoutDecoding


# TODO: Enable tensor network decoder once that goes into main.
@pytest.mark.parametrize(
    "decoder_name,error_rate",
    [
        ("single_error_lut", 0.003),
        ("nv-qldpc-decoder", 0.003),
        # ("tensor_network_decoder", 0.003),
    ])
def test_decoding_from_surface_code_dem_from_memory_circuit(
        decoder_name, error_rate):
    # If this machine only has a CPU, then skip the nv-qldpc-decoder test.
    if decoder_name == "nv-qldpc-decoder" and not is_nvidia_gpu_available():
        pytest.skip("nv-qldpc-decoder requires a GPU")

    cudaq.set_random_seed(13)
    code = qec.get_code('surface_code', distance=5)
    p = error_rate
    noise = cudaq.NoiseModel()
    noise.add_all_qubit_channel("x", cudaq.Depolarization2(p), 1)
    statePrep = qec.operation.prep0
    nRounds = 5
    nShots = 2000

    dem = qec.dem_from_memory_circuit(code, statePrep, nRounds, noise)
    syndromes, sampled_errors = qec.dem_sampling(
        dem.detector_error_matrix, nShots, dem.error_rates)
    syndromes = np.asarray(syndromes, dtype=np.uint8)
    sampled_errors = np.asarray(sampled_errors, dtype=np.uint8)

    logical_measurements = (
        (dem.observables_flips_matrix @ sampled_errors.T) % 2
    ).flatten().astype(np.uint8)

    num_syn_triggered_per_shot = np.sum(syndromes, axis=1)

    try:
        if decoder_name == "tensor_network_decoder":
            # Print the shape of the matrices
            decoder = qec.get_decoder('tensor_network_decoder',
                                      dem.detector_error_matrix,
                                      logical_obs=dem.observables_flips_matrix,
                                      noise_model=dem.error_rates,
                                      contractor_name="cutensornet",
                                      dtype="float32",
                                      device="cuda:0")
        elif decoder_name == "nv-qldpc-decoder":
            osd_method = 3  # Combination Sweep
            decoder = qec.get_decoder('nv-qldpc-decoder',
                                      dem.detector_error_matrix,
                                      max_iterations=50,
                                      error_rate_vec=np.array(dem.error_rates),
                                      use_sparsity=True,
                                      use_osd=osd_method > 0,
                                      osd_order=1000,
                                      osd_method=osd_method)
        elif decoder_name == "single_error_lut":
            decoder = qec.get_decoder('single_error_lut',
                                      dem.detector_error_matrix)
        else:
            raise ValueError(f'Invalid decoder name: {decoder_name}')
    except Exception as e:
        print(
            f'Error getting decoder: {e}, probably not available in this build')
        pytest.skip(f'{decoder_name} not available in this build')
    print(f'decoder_name: {decoder_name}')

    dr = decoder.decode_batch(syndromes)
    error_predictions = np.asarray(dr.result, dtype=np.uint8)
    dr_converged = np.asarray(dr.converged, dtype=np.uint8)
    if decoder_name == "tensor_network_decoder":
        # Tensor network decoder returns the observable flips, not the error predictions.
        data_predictions = np.round(dr.result).astype(np.uint8).T
    else:
        data_predictions = (
            dem.observables_flips_matrix @ error_predictions.T) % 2

    print(f'data_predictions.shape: {data_predictions.shape}')
    if nShots < 200:
        print(f'num_syn_per_shot    : {mat_to_str(num_syn_triggered_per_shot)}',
              end='')
        print(f'dr_converged        : {mat_to_str(dr_converged)}', end='')
        print(f'data_predictions    : {mat_to_str(data_predictions)}', end='')
        print(f'logical_measurements: {mat_to_str(logical_measurements)}',
              end='')

    nLogicalErrorsWithoutDecoding = np.sum(logical_measurements)
    nLogicalErrorsWithDecoding = np.sum(data_predictions ^ logical_measurements)
    print(f'nLogicalErrorsWithoutDecoding : {nLogicalErrorsWithoutDecoding}')
    print(f'nLogicalErrorsWithDecoding    : {nLogicalErrorsWithDecoding}')
    assert nLogicalErrorsWithDecoding < nLogicalErrorsWithoutDecoding


def test_pymatching_decode_to_observable_surface_code_dem():
    """Test PyMatching with O (observables) matrix: decoder returns observable
    flips directly.cpp)."""
    cudaq.set_random_seed(13)
    code = qec.get_code('surface_code', distance=5)
    p = 0.003
    noise = cudaq.NoiseModel()
    noise.add_all_qubit_channel("x", cudaq.Depolarization2(p), 1)
    statePrep = qec.operation.prep0
    nRounds = 5
    nShots = 2000

    dem = qec.z_dem_from_memory_circuit(code, statePrep, nRounds, noise,
                                            decompose_errors=True)

    # Verify decomposition: every column of the detector error matrix must have
    # weight ≤ 2 (graph-like) so that PyMatching can accept it.
    H = np.asarray(dem.detector_error_matrix)
    max_col_weight = int(H.sum(axis=0).max())
    assert max_col_weight <= 2, (
        f"decompose_errors=True should yield max column weight 2, got {max_col_weight}"
    )

    syndromes, sampled_errors = qec.dem_sampling(
        dem.detector_error_matrix, nShots, dem.error_rates)
    syndromes = np.asarray(syndromes, dtype=np.uint8)
    sampled_errors = np.asarray(sampled_errors, dtype=np.uint8)

    logical_measurements = (
        (dem.observables_flips_matrix @ sampled_errors.T) % 2
    ).flatten().astype(np.uint8)

    # Decode syndromes using the graph-like decomposed DEM with PyMatching.
    decoder = qec.get_decoder(
        'pymatching',
        dem.detector_error_matrix,
        O=dem.observables_flips_matrix,
        error_rate_vec=np.array(dem.error_rates),
    )

    dr = decoder.decode_batch(syndromes)
    # With decode_to_observables=True, each row is observable flips
    # (length num_observables), not error predictions.
    obs_per_shot = np.asarray(dr.result, dtype=np.float64)
    data_predictions = np.round(obs_per_shot).astype(np.uint8).T

    nLogicalErrorsWithoutDecoding = np.sum(logical_measurements)
    nLogicalErrorsWithDecoding = np.sum(data_predictions ^ logical_measurements)
    assert nLogicalErrorsWithDecoding < nLogicalErrorsWithoutDecoding


def test_pcm_extend_to_n_rounds():
    # This test independently compares the functionality of dem_from_memory_circuit
    # (of two different numbers of rounds) to pcm_extend_to_n_rounds.
    statePrep = qec.operation.prep0
    nRounds = 5
    noise = cudaq.NoiseModel()
    noise.add_all_qubit_channel("x", cudaq.Depolarization2(0.003), 1)
    code = qec.get_code('surface_code', distance=5)
    dem05 = qec.z_dem_from_memory_circuit(code, statePrep, nRounds, noise)
    nSyndromesPerRound = dem05.detector_error_matrix.shape[0] // nRounds
    dem15 = qec.z_dem_from_memory_circuit(code, statePrep, 3 * nRounds, noise)
    H05 = dem05.detector_error_matrix
    H15 = dem15.detector_error_matrix
    H15_new, column_list = qec.pcm_extend_to_n_rounds(H05, nSyndromesPerRound,
                                                      3 * nRounds)
    # Check if H15 == H15_new, one column at a time.
    for c in range(H15.shape[1]):
        if not np.allclose(H15[:, c], H15_new[:, c]):
            print(f'Column {c} is not equal')
            # Use join to print the columns as a string without spaces
            print(f'H15    [:, c] : {"".join(map(str, H15[:, c]))}')
            print(f'H15_new[:, c] : {"".join(map(str, H15_new[:, c]))}')
            assert False
    assert len(column_list) == H15_new.shape[1]

    # Check the extended error rates.
    H15_new_error_rates = np.array(dem05.error_rates)[column_list]
    assert np.allclose(H15_new_error_rates, dem15.error_rates, atol=1e-6)


if __name__ == "__main__":
    pytest.main()
