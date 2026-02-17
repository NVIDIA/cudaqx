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


def test_get_code():
    steane = qec.get_code("steane")
    assert isinstance(steane, qec.Code)


def test_get_available_codes():
    codes = qec.get_available_codes()
    assert isinstance(codes, list)
    assert "steane" in codes


def test_code_parity_matrices():
    steane = qec.get_code("steane")

    parity = steane.get_parity()
    assert isinstance(parity, np.ndarray)
    assert parity.shape == (6, 14)

    parity_x = steane.get_parity_x()
    assert isinstance(parity, np.ndarray)
    assert parity_x.shape == (3, 7)

    parity_z = steane.get_parity_z()
    assert isinstance(parity, np.ndarray)
    assert parity_z.shape == (3, 7)


def test_code_stabilizers():
    steane = qec.get_code("steane")
    stabilizers = steane.get_stabilizers()
    assert isinstance(stabilizers, list)
    assert len(stabilizers) == 6
    assert all(isinstance(stab, cudaq.Operator) for stab in stabilizers)
    stabStrings = [term.get_pauli_word() for term in stabilizers]
    expected = [
        "ZZZZIII", "XXXXIII", "IXXIXXI", "IIXXIXX", "IZZIZZI", "IIZZIZZ"
    ]
    assert set(expected) == set(stabStrings)


def test_sample_memory_circuit():
    steane = qec.get_code("steane")

    syndromes, dataResults = qec.sample_memory_circuit(steane,
                                                       numShots=10,
                                                       numRounds=4)
    assert isinstance(syndromes, np.ndarray)
    assert syndromes.shape == (40, 6)
    print(syndromes)

    syndromes_with_op, dataResults = qec.sample_memory_circuit(
        steane, qec.operation.prep1, 10, 4)
    assert isinstance(syndromes_with_op, np.ndarray)
    print(syndromes_with_op)
    assert syndromes_with_op.shape == (40, 6)


def test_custom_steane_code():
    ops = ["ZZZZIII", "XXXXIII", "IXXIXXI", "IIXXIXX", "IZZIZZI", "IIZZIZZ"]
    custom_steane = qec.get_code("steane", stabilizers=ops)
    assert isinstance(custom_steane, qec.Code)

    parity = custom_steane.get_parity()
    assert parity.shape == (6, 14)

    expected_parity = np.array([
        1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1
    ])
    print(parity)
    np.testing.assert_array_equal(parity, expected_parity.reshape(6, 14))


def test_noisy_simulation():
    cudaq.set_target('stim')

    noise = cudaq.NoiseModel()
    noise.add_all_qubit_channel('x',
                                qec.TwoQubitDepolarization(.1),
                                num_controls=1)
    steane = qec.get_code("steane")
    syndromes, dataResults = qec.sample_memory_circuit(steane,
                                                       numShots=10,
                                                       numRounds=4,
                                                       noise=noise)
    assert isinstance(syndromes, np.ndarray)
    assert syndromes.shape == (40, 6)
    print(syndromes)
    assert np.any(syndromes)
    cudaq.reset_target()


@pytest.mark.skip(reason="PYTHON-REFACTOR")
def test_python_code():
    steane = qec.get_code("py-steane-example")
    syndromes, dataResults = qec.sample_memory_circuit(steane,
                                                       numShots=10,
                                                       numRounds=4)
    assert isinstance(syndromes, np.ndarray)
    assert syndromes.shape == (40, 6)
    print(syndromes)
    assert not np.any(syndromes)


def test_invalid_code():
    with pytest.raises(RuntimeError):
        qec.get_code("invalid_code_name")


def test_invalid_operation():
    steane = qec.get_code("steane")
    with pytest.raises(TypeError):
        qec.sample_memory_circuit(steane, "invalid_op", 10, 4)


def test_generate_random_bit_flips():
    # Test case 1: error_prob = 0
    nBits = 10
    error_prob = 0

    data = qec.generate_random_bit_flips(nBits, error_prob)
    print(f"data shape: {data.shape}")

    assert len(data.shape) == 1
    assert data.shape[0] == 10
    assert np.all(data == 0)


def test_steane_code_capacity():
    # Test case 1: error_prob = 0
    steane = qec.get_code("steane")
    Hz = steane.get_parity_z()
    n_shots = 10
    error_prob = 0

    syndromes, data = qec.sample_code_capacity(Hz, n_shots, error_prob)

    assert len(Hz.shape) == 2
    assert Hz.shape[0] == 3
    assert Hz.shape[1] == 7
    assert syndromes.shape[0] == n_shots
    assert syndromes.shape[1] == Hz.shape[0]
    assert data.shape[0] == n_shots
    assert data.shape[1] == Hz.shape[1]

    # Error prob = 0 should be all zeros
    assert np.all(data == 0)
    assert np.all(syndromes == 0)

    # Test case 2: error_prob = 0.15
    error_prob = 0.15
    seed = 1337

    syndromes, data = qec.sample_code_capacity(Hz,
                                               n_shots,
                                               error_prob,
                                               seed=seed)

    assert len(Hz.shape) == 2
    assert Hz.shape[0] == 3
    assert Hz.shape[1] == 7
    assert syndromes.shape[0] == n_shots
    assert syndromes.shape[1] == Hz.shape[0]
    assert data.shape[0] == n_shots
    assert data.shape[1] == Hz.shape[1]

    # Known seeded data for error_prob = 0.15
    seeded_data = np.array([[0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0]])

    checked_syndromes = np.array([[1, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 0],
                                  [0, 0, 0], [0, 0, 0], [0, 1, 0], [0, 0, 0],
                                  [0, 1, 1], [0, 0, 0]])

    assert np.array_equal(data, seeded_data)
    assert np.array_equal(syndromes, checked_syndromes)

    # Test case 3: error_prob = 0.25
    error_prob = 0.25
    seed = 1337

    syndromes, data = qec.sample_code_capacity(Hz,
                                               n_shots,
                                               error_prob,
                                               seed=seed)

    assert len(Hz.shape) == 2
    assert Hz.shape[0] == 3
    assert Hz.shape[1] == 7
    assert syndromes.shape[0] == n_shots
    assert syndromes.shape[1] == Hz.shape[0]
    assert data.shape[0] == n_shots
    assert data.shape[1] == Hz.shape[1]

    # Known seeded data for error_prob = 0.25
    seeded_data = np.array([[0, 1, 0, 1, 1, 0, 0], [0, 0, 0, 1, 0, 0, 1],
                            [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0]])

    checked_syndromes = np.array([[0, 0, 1], [1, 0, 0], [1, 1, 1], [0, 1, 0],
                                  [0, 0, 0], [0, 0, 0], [0, 1, 0], [0, 0, 0],
                                  [0, 1, 1], [0, 0, 0]])

    assert np.array_equal(data, seeded_data)
    assert np.array_equal(syndromes, checked_syndromes)


def test_het_map_from_kwargs_bool():
    steane = qec.get_code("steane", bool_true=True, bool_false=False)
    assert isinstance(steane, qec.Code)


def test_version():
    assert "CUDA-Q QEC" in qec.__version__


if __name__ == "__main__":
    pytest.main()
