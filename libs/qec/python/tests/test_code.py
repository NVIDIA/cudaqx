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
from cudaq_qec import patch


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


def test_repetition_empty_x_matrices_preserve_rank():
    repetition = qec.get_code("repetition", distance=3)

    # Repetition is Z-only; empty X-side results must still be matrices with
    # one column per data qubit so Python callers can inspect their shape.
    parity_x = repetition.get_parity_x()
    assert isinstance(parity_x, np.ndarray)
    assert parity_x.dtype == np.uint8
    assert parity_x.shape == (0, 3)

    observables_x = repetition.get_observables_x()
    assert isinstance(observables_x, np.ndarray)
    assert observables_x.dtype == np.uint8
    assert observables_x.shape == (0, 3)

    parity_z = repetition.get_parity_z()
    assert parity_z.shape == (2, 3)

    observables_z = repetition.get_observables_z()
    assert observables_z.shape == (1, 3)


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
    nShots, nRounds = 10, 4
    # Shape: (nShots, k) where k = 2*numAncz + (nRounds-1)*numCols
    # For Steane: numAncz=3, numCols=6, nRounds=4 → k = 2*3 + 3*6 = 24
    nDetectors = 24

    syndromes, dataResults = qec.sample_memory_circuit(steane,
                                                       numShots=nShots,
                                                       numRounds=nRounds)
    assert isinstance(syndromes, np.ndarray)
    assert syndromes.shape == (nShots, nDetectors)
    print(syndromes)

    syndromes_with_op, dataResults = qec.sample_memory_circuit(
        steane, qec.operation.prep1, nShots, nRounds)
    assert isinstance(syndromes_with_op, np.ndarray)
    print(syndromes_with_op)
    assert syndromes_with_op.shape == (nShots, nDetectors)


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
    nShots, nRounds = 10, 4
    nDetectors = 24
    syndromes, dataResults = qec.sample_memory_circuit(steane,
                                                       numShots=nShots,
                                                       numRounds=nRounds,
                                                       noise=noise)
    assert isinstance(syndromes, np.ndarray)
    assert syndromes.shape == (nShots, nDetectors)
    print(syndromes)
    assert np.any(syndromes)
    cudaq.reset_target()


def test_python_code():
    steane = qec.get_code("py-steane-example")
    syndromes, dataResults = qec.sample_memory_circuit(steane,
                                                       numShots=10,
                                                       numRounds=4)
    assert isinstance(syndromes, np.ndarray)
    assert syndromes.shape == (10, 24)
    print(syndromes)
    assert not np.any(syndromes)


# stabilizer_round kernels with invalid (non list[cudaq.measure_handle])
# return annotations. The bodies are minimal; only the annotation matters.
@cudaq.kernel
def _stab_list_bool(q: patch, xs: list[int], zs: list[int]) -> list[bool]:
    return mz([*q.ancx, *q.ancz])


@cudaq.kernel
def _stab_bool(q: patch, xs: list[int], zs: list[int]) -> bool:
    return mz(q.ancx[0])


@cudaq.kernel
def _stab_int(q: patch, xs: list[int], zs: list[int]) -> int:
    return cudaq.to_integer(cudaq.to_bools(mz([*q.ancx, *q.ancz])))


@cudaq.kernel
def _stab_void(q: patch, xs: list[int], zs: list[int]):
    mz([*q.ancx, *q.ancz])


@cudaq.kernel
def _stab_none(q: patch, xs: list[int], zs: list[int]) -> None:
    mz([*q.ancx, *q.ancz])


@pytest.mark.parametrize("kernel,name", [
    (_stab_list_bool, 'py-bad-stab-list-bool'),
    (_stab_bool, 'py-bad-stab-bool'),
    (_stab_int, 'py-bad-stab-int'),
    (_stab_void, 'py-bad-stab-void'),
    (_stab_none, 'py-bad-stab-none'),
])
def test_stabilizer_round_invalid_annotation(kernel, name):
    # A stabilizer_round must return list[cudaq.measure_handle]; any other
    # annotation drops the measurement handles, so registration must reject it.
    @qec.code(name)
    class BadCode:

        def __init__(self, **kwargs):
            qec.Code.__init__(self, **kwargs)
            self.stabilizers = [
                cudaq.SpinOperator.from_word(w) for w in [
                    "XXXXIII", "IXXIXXI", "IIXXIXX", "ZZZZIII", "IZZIZZI",
                    "IIZZIZZ"
                ]
            ]
            self.pauli_observables = [
                cudaq.SpinOperator.from_word(p) for p in ["IIIIXXX", "IIIIZZZ"]
            ]
            self.operation_encodings = {qec.operation.stabilizer_round: kernel}

        def get_num_data_qubits(self):
            return 7

        def get_num_ancilla_x_qubits(self):
            return 3

        def get_num_ancilla_z_qubits(self):
            return 3

        def get_num_ancilla_qubits(self):
            return 6

        def get_num_x_stabilizers(self):
            return 3

        def get_num_z_stabilizers(self):
            return 3

    with pytest.raises(RuntimeError, match="list\\[cudaq.measure_handle\\]"):
        qec.get_code(name)


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


# Kernels for the inlined-feedback toy code: two data qubits with stabilizers
# XX and ZZ, both extracted through a single superdense (Bell-pair) ancilla
# pair (ancx[0] carries the XX record, ancz[0] the ZZ record). The round ends
# with an *uncorrected* record-conditioned byproduct: after the Bell decode,
# ancx[0] sits in the computational basis holding its future measurement
# outcome r_X, so the trailing CX(ancx[0], data[1]) is the unitary equivalent
# of the classically controlled byproduct X^{r_X} on data[1] that a hardware
# frame update would otherwise track. Same gadget and feedback matrices as the
# C++ toy in unittests/feedback_toy_device.cpp / test_qec.cpp (see the full
# derivation there): feedback = [[0, 1], [0, 0]], observable feedback =
# [[0, 1]], with records per round in [Z][X] order.
@cudaq.kernel
def _feedback_toy_prep0(q: patch):
    reset(q.data[0])
    reset(q.data[1])


@cudaq.kernel
def _feedback_toy_round(q: patch, x_stabilizers: list[int],
                        z_stabilizers: list[int]) -> list[cudaq.measure_handle]:
    # Bell-prepare the superdense ancilla pair.
    h(q.ancx[0])
    x.ctrl(q.ancx[0], q.ancz[0])
    # Couple XX through the X ancilla and ZZ through the Z ancilla.
    x.ctrl(q.ancx[0], q.data[0])
    x.ctrl(q.ancx[0], q.data[1])
    x.ctrl(q.data[0], q.ancz[0])
    x.ctrl(q.data[1], q.ancz[0])
    # Decode the Bell pair.
    x.ctrl(q.ancx[0], q.ancz[0])
    h(q.ancx[0])
    # Uncorrected record-conditioned byproduct: X^{r_X} on data[1].
    x.ctrl(q.ancx[0], q.data[1])
    # Records in [Z][X] order.
    results = mz([*q.ancz, *q.ancx])
    reset(q.ancz[0])
    reset(q.ancx[0])
    return results


def test_python_inlined_feedback_toy():

    @qec.code('py-feedback-toy')
    class FeedbackToy:

        def __init__(self):
            qec.Code.__init__(self)
            self.stabilizers = [
                cudaq.SpinOperator.from_word(w) for w in ["XX", "ZZ"]
            ]
            self.pauli_observables = [cudaq.SpinOperator.from_word("ZZ")]
            self.operation_encodings = {
                qec.operation.prep0: _feedback_toy_prep0,
                qec.operation.stabilizer_round: _feedback_toy_round
            }

        def get_num_data_qubits(self):
            return 2

        def get_num_ancilla_qubits(self):
            return 2

        def get_num_ancilla_x_qubits(self):
            return 1

        def get_num_ancilla_z_qubits(self):
            return 1

        def get_num_x_stabilizers(self):
            return 1

        def get_num_z_stabilizers(self):
            return 1

        def get_inlined_feedback(self):
            return np.array([[0, 1], [0, 0]], dtype=np.uint8)

        def get_observable_inlined_feedback(self):
            # Exercise the bridge's dtype coercion: int64 instead of uint8.
            return np.array([[0, 1]])

    # Sample on stim, mirroring the C++ unit tests (test_qec links the stim
    # target). The toy's X record is genuinely random in round 1, and the
    # default (qpp) target does not preserve cross-round mid-circuit
    # measurement correlations when sampling Python kernels - independent of
    # the inlined-feedback machinery, but this toy (unlike the steane
    # example, whose records are all deterministically 0) exposes it.
    cudaq.set_target('stim')

    numShots, numRounds = 20, 4
    syndromes, data = qec.sample_memory_circuit(qec.get_code('py-feedback-toy'),
                                                numShots=numShots,
                                                numRounds=numRounds)

    # 1 first-round boundary + 2 * (numRounds - 1) cross-round + 1 final
    # boundary detectors, all deterministic (zero) in the noiseless circuit.
    assert syndromes.shape == (numShots, 2 * numRounds)
    assert not np.any(syndromes)

    # The individual data qubits are randomized by the XX measurement, but
    # the ZZ parity must close deterministically every shot.
    assert data.shape == (numShots, 2)
    assert not np.any(data[:, 0] ^ data[:, 1])
    cudaq.reset_target()


def test_python_inlined_feedback_toy_negative_control():
    # The identical toy without the feedback declarations: the uncorrected
    # byproduct makes the record-0 detectors non-deterministic and stim must
    # reject the circuit. This proves the Python-declared feedback is doing
    # the work in the positive test above.
    @qec.code('py-feedback-toy-nofb')
    class FeedbackToyNoFeedback:

        def __init__(self):
            qec.Code.__init__(self)
            self.stabilizers = [
                cudaq.SpinOperator.from_word(w) for w in ["XX", "ZZ"]
            ]
            self.pauli_observables = [cudaq.SpinOperator.from_word("ZZ")]
            self.operation_encodings = {
                qec.operation.prep0: _feedback_toy_prep0,
                qec.operation.stabilizer_round: _feedback_toy_round
            }

        def get_num_data_qubits(self):
            return 2

        def get_num_ancilla_qubits(self):
            return 2

        def get_num_ancilla_x_qubits(self):
            return 1

        def get_num_ancilla_z_qubits(self):
            return 1

        def get_num_x_stabilizers(self):
            return 1

        def get_num_z_stabilizers(self):
            return 1

    # stim's determinism analysis (a std::invalid_argument surfaced as
    # ValueError) rejects the circuit inside dem_from_kernel, before any
    # sampling happens.
    with pytest.raises(ValueError, match="non-deterministic detectors"):
        qec.sample_memory_circuit(qec.get_code('py-feedback-toy-nofb'),
                                  numShots=20,
                                  numRounds=4)


if __name__ == "__main__":
    pytest.main()
