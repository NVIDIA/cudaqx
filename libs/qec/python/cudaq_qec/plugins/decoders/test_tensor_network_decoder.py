# ============================================================================ #
# Copyright (c) 2024 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import numpy as np
import pytest
import stim
from quimb.tensor import TensorNetwork

from cudaq_qec.plugins.decoders.tensor_network_decoder import (
    tensor_network_from_parity_check, tensor_network_from_single_syndrome,
    prepare_syndrome_data_batch, tensor_network_from_syndrome_batch,
    tensor_network_from_logical_observable, tensor_to_cpu)


def test_tensor_network_from_parity_check_basic():
    mat = np.array([[1, 1, 0], [0, 1, 1]])
    row_inds = ['r0', 'r1']
    col_inds = ['c0', 'c1', 'c2']
    tags = ['tag0', 'tag1', 'tag2', 'tag3']
    tn = tensor_network_from_parity_check(mat, row_inds, col_inds, tags=tags)
    assert isinstance(tn, TensorNetwork)
    assert len(tn.tensors) == 4
    expected_inds = [('r0', 'c0'), ('r0', 'c1'), ('r1', 'c1'), ('r1', 'c2')]
    inds = [t.inds for t in tn.tensors]
    assert set(inds) == set(expected_inds)
    for i, t in enumerate(tn.tensors):
        assert t.tags.pop() in tags
        assert t.inds in expected_inds
        np.testing.assert_array_equal(t.data, np.array([[1.0, 1.0], [1.0,
                                                                     -1.0]]))


def test_tensor_network_from_parity_check_no_tags():
    mat = np.array([[1, 0], [0, 1]])
    row_inds = ['r0', 'r1']
    col_inds = ['c0', 'c1']
    tn = tensor_network_from_parity_check(mat, row_inds, col_inds)
    assert isinstance(tn, TensorNetwork)
    assert len(tn.tensors) == 2
    for t in tn.tensors:
        assert len(t.tags) == 0


def test_tensor_network_from_parity_check_empty():
    mat = np.zeros((2, 2), dtype=int)
    row_inds = ['r0', 'r1']
    col_inds = ['c0', 'c1']
    tn = tensor_network_from_parity_check(mat, row_inds, col_inds)
    assert isinstance(tn, TensorNetwork)
    assert len(tn.tensors) == 0


def test_tensor_network_from_single_syndrome_all_false():
    syndrome = [False, False, False]
    check_inds = ['c0', 'c1', 'c2']
    tn = tensor_network_from_single_syndrome(syndrome, check_inds)
    assert len(tn.tensors) == 3
    for i, t in enumerate(tn.tensors):
        np.testing.assert_array_equal(t.data, np.array([1.0, 1.0]))
        assert t.inds == (check_inds[i],)
        assert f"SYN_{i}" in t.tags
        assert "SYNDROME" in t.tags


def test_tensor_network_from_single_syndrome_mixed():
    syndrome = [True, False, True]
    check_inds = ['a', 'b', 'c']
    tn = tensor_network_from_single_syndrome(syndrome, check_inds)
    assert len(tn.tensors) == 3
    for i, t in enumerate(tn.tensors):
        expected = np.array([1.0, -1.0]) if syndrome[i] else np.array(
            [1.0, 1.0])
        np.testing.assert_array_equal(t.data, expected)
        assert t.inds == (check_inds[i],)
        assert f"SYN_{i}" in t.tags
        assert "SYNDROME" in t.tags


def test_prepare_syndrome_data_batch_shape_and_values_randomized():
    np.random.seed(123)
    data = np.random.choice([False, True],
                            size=(4, 5))  # syndrome length 4, 5 syndromes
    arr = prepare_syndrome_data_batch(data)
    assert arr.shape == (5, 4, 2)
    # Check that each entry is [1, 1] if False, [1, -1] if True
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            expected = np.array([1, -1]) if data[i, j] else np.array([1, 1])
            np.testing.assert_array_equal(arr[j, i], expected)


def test_tensor_network_from_syndrome_batch_tags_and_inds_randomized():
    np.random.seed(42)
    batch_size = 5
    n_synd = 4
    detection_events = np.random.choice([False, True],
                                        size=(batch_size, n_synd))
    syndrome_inds = [f's{i}' for i in range(n_synd)]
    tags = [f'tag{i}' for i in range(n_synd)]
    tn = tensor_network_from_syndrome_batch(detection_events,
                                            syndrome_inds,
                                            batch_index="batch",
                                            tags=tags)
    assert len(tn.tensors) == n_synd
    for i, t in enumerate(tn.tensors):
        assert t.inds == ("batch", syndrome_inds[i])
        assert tags[i] in t.tags
        assert "SYNDROME" in t.tags
        # Check tensor data for each batch
        for b in range(batch_size):
            expected = np.array(
                [1.0, -1.0]) if detection_events[b, i] else np.array([1.0, 1.0])
            np.testing.assert_array_equal(t.data[b], expected)


def test_tensor_network_from_logical_observable():
    obs = np.array([[True, False, True]])
    obs_inds = ['o0']
    tn = tensor_network_from_logical_observable(obs,
                                                obs_inds, ["l0"],
                                                logicals_tags=["OBS_0"])
    assert len(tn.tensors) == obs.shape[0]
    for i, t in enumerate(tn.tensors):
        expected = np.array([[1.0, 1.0], [1.0, -1.0]])
        np.testing.assert_array_equal(t.data, expected)
        assert t.inds == (obs_inds[i], "l0")
        assert f"OBS_{i}" in t.tags


def test_tensor_to_cpu_numpy():
    arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    out = tensor_to_cpu(arr, backend="numpy", dtype="float64")
    assert isinstance(out, np.ndarray)
    np.testing.assert_array_equal(out, arr)
    assert out.dtype == np.float64


def test_tensor_to_cpu_torch_cpu_to_numpy():
    import torch
    t = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    out = tensor_to_cpu(t, backend="numpy", dtype="float64")
    assert isinstance(out, np.ndarray)
    np.testing.assert_array_equal(out, t.numpy())
    assert out.dtype == np.float64


def test_tensor_to_cpu_torch_gpu_to_numpy():
    import torch
    if not torch.cuda.is_available():
        pytest.skip('No GPUs available, skip test')
    t = torch.tensor([[1.0, 2.0], [3.0, 4.0]],
                     dtype=torch.float32,
                     device="cuda")
    out = tensor_to_cpu(t, backend="numpy", dtype="float64")
    assert isinstance(out, np.ndarray)
    np.testing.assert_array_equal(out, t.cpu().numpy())
    assert out.dtype == np.float64


def test_tensor_to_cpu_torch_cpu_to_torch():
    import torch
    t = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    out = tensor_to_cpu(t, backend="torch", dtype="float64")
    assert isinstance(out, torch.Tensor)
    np.testing.assert_array_equal(out.numpy(), t.numpy())
    assert out.dtype == torch.float64


def test_tensor_to_cpu_torch_gpu_to_torch():
    import torch
    if not torch.cuda.is_available():
        pytest.skip('No GPUs available, skip test')
    t = torch.tensor([[1.0, 2.0], [3.0, 4.0]],
                     dtype=torch.float32,
                     device="cuda")
    out = tensor_to_cpu(t, backend="torch", dtype="float64")
    assert isinstance(out, torch.Tensor)
    np.testing.assert_array_equal(out.cpu().numpy(), t.cpu().numpy())
    assert out.dtype == torch.float64
    assert not out.is_cuda


def test_parse_detector_error_model_with_real_stim():
    from .tensor_network_decoder import parse_detector_error_model

    # Generate a real stim DetectorErrorModel
    circuit = stim.Circuit.generated("surface_code:rotated_memory_z",
                                     rounds=3,
                                     distance=3,
                                     after_clifford_depolarization=0.001,
                                     after_reset_flip_probability=0.01,
                                     before_measure_flip_probability=0.01,
                                     before_round_data_depolarization=0.01)
    detector_error_model = circuit.detector_error_model(decompose_errors=True)

    # Call the function under test
    out_H, out_L, priors = parse_detector_error_model(detector_error_model)

    # Check types and shapes
    assert isinstance(out_H, np.ndarray)
    assert isinstance(out_L, np.ndarray)
    assert isinstance(priors, list)
    assert all(isinstance(p, float) for p in priors)
    assert len(priors) == out_H.shape[1] or len(priors) == out_L.shape[1]
