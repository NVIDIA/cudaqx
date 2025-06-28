# ============================================================================ #
# Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
import pytest
import numpy as np
import cudaq
import cudaq_qec as qec


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


def test_dem_from_memory_circuit():
    code = qec.get_code('steane')
    # observables = code.get_pauli_observables_matrix()
    p = 0.01
    noise = cudaq.NoiseModel()
    noise.add_all_qubit_channel("x", cudaq.Depolarization2(p), 1)
    statePrep = qec.operation.prep0
    nRounds = 2

    dem = qec.z_dem_from_memory_circuit(code, statePrep, nRounds, noise)
    expected_detector_error_matrix = """
1111...1..............
.1.111..111...........
..11.11..1.1111.......
.......111.1.1.1111...
..........1.11..1.111.
..............1..11.11
"""
    # print(mat_to_str(dem.detector_error_matrix), end='')
    assert '\n' + mat_to_str(
        dem.detector_error_matrix) == expected_detector_error_matrix

    # Round to 4 decimal places
    # print(np.round(dem.error_rates, 4))
    expected_error_rates = [
        0.0183, 0.0235, 0.0158, 0.0209, 0.0310, 0.0235, 0.0183, 0.0106, 0.0053,
        0.0053, 0.0106, 0.0053, 0.0053, 0.0053, 0.0106, 0.0235, 0.0158, 0.0158,
        0.0183, 0.0335, 0.0209, 0.0434
    ]
    assert np.allclose(dem.error_rates, expected_error_rates, atol=1e-4)

    expected_observables_flips_matrix = '1....11.....1......111\n'
    # print(mat_to_str(dem.observables_flips_matrix), end='')
    assert mat_to_str(
        dem.observables_flips_matrix) == expected_observables_flips_matrix


def test_decoding_from_dem_from_memory_circuit():
    cudaq.set_random_seed(13)
    code = qec.get_code('steane')
    observables = code.get_pauli_observables_matrix()
    Lz = code.get_observables_z()
    p = 0.001
    noise = cudaq.NoiseModel()
    noise.add_all_qubit_channel("x", cudaq.Depolarization2(p), 1)
    statePrep = qec.operation.prep0
    nRounds = 6
    nShots = 200

    dem = qec.dem_from_memory_circuit(code, statePrep, nRounds, noise)
    # Sample the memory circuit with errors
    syndromes, data = qec.sample_memory_circuit(code, statePrep, nShots,
                                                nRounds, noise)

    logical_measurements = (Lz @ data.transpose()) % 2
    # only one logical qubit, so do not need the second axis
    logical_measurements = logical_measurements.flatten()

    # Use nShots for the first dimension and whatever is left for the second
    syndromes = syndromes.reshape((nShots, -1))
    decoder = qec.get_decoder('single_error_lut', dem.detector_error_matrix)
    dr = decoder.decode_batch(syndromes)
    error_predictions = np.array([e.result for e in dr], dtype=np.uint8)
    dr_converged = np.array([e.converged for e in dr], dtype=np.uint8)

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
    cudaq.set_random_seed(13)
    code = qec.get_code('surface_code', distance=5)
    observables = code.get_pauli_observables_matrix()
    Lz = code.get_observables_z()
    p = error_rate
    noise = cudaq.NoiseModel()
    noise.add_all_qubit_channel("x", cudaq.Depolarization2(p), 1)
    statePrep = qec.operation.prep0
    nRounds = 5
    nShots = 2000

    # Sample the memory circuit with errors
    syndromes, data = qec.sample_memory_circuit(code, statePrep, nShots,
                                                nRounds, noise)

    logical_measurements = (Lz @ data.transpose()) % 2
    # only one logical qubit, so do not need the second axis
    logical_measurements = logical_measurements.flatten()

    # Reshape and drop the X stabilizers, keeping just the Z stabilizers (since this is prep0)
    syndromes = syndromes.reshape((nShots, nRounds, -1))
    syndromes = syndromes[:, :, :syndromes.shape[2] // 2]
    # Now flatten to two dimensions again
    syndromes = syndromes.reshape((nShots, -1))
    # Sum syndromes across the second dimension
    num_syn_triggered_per_shot = np.sum(syndromes, axis=1)

    dem = qec.z_dem_from_memory_circuit(code, statePrep, nRounds, noise)
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
    error_predictions = np.array([e.result for e in dr], dtype=np.uint8)
    dr_converged = np.array([e.converged for e in dr], dtype=np.uint8)
    if decoder_name == "tensor_network_decoder":
        # Tensor network decoder returns the observable flips, not the error predictions.
        data_predictions = np.array([np.round(e.result) for e in dr],
                                    dtype=np.uint8).T
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
