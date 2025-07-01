# ============================================================================ #
# Copyright (c) 2025 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import numpy as np
import pytest
from quimb.tensor import TensorNetwork
import cudaq_qec as qec


def make_simple_code():
    # [[1, 1, 0], [0, 1, 1]] parity check, 1 logical, depolarizing noise
    H = np.array([[1, 1, 0], [0, 1, 1]])
    logical = np.array([[1, 0, 1]])
    noise = [0.1, 0.2, 0.3]
    return H, logical, noise


def test_decoder_init_and_attributes():
    H, logical, noise = make_simple_code()
    decoder = qec.get_decoder("tensor_network_decoder",
                              H,
                              logical_obs=logical,
                              noise_model=noise)
    assert isinstance(decoder.code_tn, TensorNetwork)
    assert isinstance(decoder.logical_tn, TensorNetwork)
    assert isinstance(decoder.syndrome_tn, TensorNetwork)
    assert isinstance(decoder.full_tn, TensorNetwork)
    assert hasattr(decoder, "noise_model")

    import torch
    if torch.cuda.is_available():
        assert decoder._contractor_name == "cutensornet"
        assert decoder._backend == "torch"
        assert decoder._device == "cuda:0"
    else:
        assert decoder._contractor_name == "numpy"
        assert decoder._backend == "numpy"
        assert decoder._device == "cpu"
    assert decoder._dtype == "float32"


def test_decoder_replace_logical_observable():
    H, logical, noise = make_simple_code()
    import cudaq_qec as qec

    decoder = qec.get_decoder("tensor_network_decoder",
                              H,
                              logical_obs=logical,
                              noise_model=noise)

    # New logical observable and indices
    new_logical = np.array([[0, 1, 1]])
    new_logical_inds = ["l_1"]
    new_logical_tags = ["LOG_1"]

    decoder.replace_logical_observable(logical_obs=new_logical,
                                       logical_inds=new_logical_inds,
                                       logical_tags=new_logical_tags)

    # Check that the logical observable and indices are updated
    assert np.array_equal(decoder.logical_obs, new_logical)
    assert decoder.logical_inds == new_logical_inds
    assert decoder.logical_tags == new_logical_tags


def test_decoder_replace_logical_observable_shape_error():
    H, logical, noise = make_simple_code()
    import cudaq_qec as qec

    decoder = qec.get_decoder("tensor_network_decoder",
                              H,
                              logical_obs=logical,
                              noise_model=noise)

    # new_logical with wrong shape: first dimension != 1
    new_logical = np.array([[0, 1, 1], [1, 0, 0]])  # shape (2, 3)
    new_logical_inds = ["l_1"]
    new_logical_obs_inds = ["e_0", "e_1", "e_2"]
    new_logical_tags = ["LOG_1"]

    with pytest.raises(Exception):
        decoder.replace_logical_observable(
            logical_obs=new_logical,
            logical_inds=new_logical_inds,
            logical_obs_inds=new_logical_obs_inds,
            logical_tags=new_logical_tags)


def test_decoder_flip_syndromes():
    H, logical, noise = make_simple_code()
    decoder = qec.get_decoder("tensor_network_decoder",
                              H,
                              logical_obs=logical,
                              noise_model=noise,
                              contractor_name="numpy",
                              dtype="float64",
                              device="cpu")

    new_syndromes = [1.0] * H.shape[0]
    decoder.flip_syndromes(new_syndromes)
    for i, t in enumerate(decoder.syndrome_tn.tensors):
        np.testing.assert_array_equal(t.data, np.array([1.0, -1.0]))

    new_syndromes = [0.0] * H.shape[0]
    decoder.flip_syndromes(new_syndromes)
    for i, t in enumerate(decoder.syndrome_tn.tensors):
        np.testing.assert_array_equal(t.data, np.array([1.0, 1.0]))


def test_decoder_decode_single():
    H, logical, noise = make_simple_code()
    decoder = qec.get_decoder("tensor_network_decoder",
                              H,
                              logical_obs=logical,
                              noise_model=noise)
    syndrome = [1.0, 0.0]
    res = decoder.decode(syndrome)
    assert hasattr(res, "converged")
    assert hasattr(res, "result")
    assert isinstance(res.result, list)
    assert 0.0 <= res.result[0] <= 1.0


def test_decoder_decode_batch():
    H, logical, noise = make_simple_code()
    decoder = qec.get_decoder("tensor_network_decoder",
                              H,
                              logical_obs=logical,
                              noise_model=noise,
                              contractor_name="numpy",
                              dtype="float64",
                              device="cpu")
    batch = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    res = decoder.decode_batch(batch)
    print([r.result for r in res])
    assert isinstance(res, list)
    assert all(hasattr(r, "converged") and hasattr(r, "result") for r in res)
    assert all(
        isinstance(r.result, list) and 0.0 <= r.result[0] <= 1.0 for r in res)


def test_decoder_set_contractor_invalid():
    H, logical, noise = make_simple_code()
    decoder = qec.get_decoder("tensor_network_decoder",
                              H,
                              logical_obs=logical,
                              noise_model=noise)
    with pytest.raises(ValueError):
        decoder.set_contractor("not_a_contractor")


def test_TensorNetworkDecoder_optimize_path_all_variants():
    import cotengra
    from cuquantum import tensornet as cutn
    from opt_einsum.contract import PathInfo
    from cuquantum.tensornet.configuration import OptimizerInfo
    import torch

    # Simple code setup
    H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
    logical = np.array([[1, 0, 1]], dtype=np.uint8)
    noise = [0.1, 0.2, 0.3]
    decoder = qec.get_decoder("tensor_network_decoder",
                              H,
                              logical_obs=logical,
                              noise_model=noise)

    # optimize="auto" (opt_einsum)
    info = decoder.optimize_path(output_inds=("l_0",), optimize="auto")
    assert isinstance(decoder.path_single, (list, tuple))
    assert decoder.slicing_single is not None
    assert isinstance(info, PathInfo)

    if not torch.cuda.is_available():
        pytest.skip("No GPUs available, skip cuQuantum test.")

    # optimize=cuQuantum OptimizerOptions
    opt = cutn.OptimizerOptions()
    info2 = decoder.optimize_path(output_inds=("l_0",), optimize=opt)
    assert isinstance(decoder.path_single, (list, tuple))
    assert decoder.slicing_single is not None
    assert isinstance(info2, OptimizerInfo)

    # optimize=cotengra.HyperOptimizer()
    hyper = cotengra.HyperOptimizer()
    info3 = decoder.optimize_path(output_inds=("l_0",), optimize=hyper)
    assert isinstance(decoder.path_single, (list, tuple))
    assert decoder.slicing_single is not None
    assert isinstance(info3, PathInfo)


@pytest.mark.parametrize("contractor,dtype,device,expect_gpu", [
    ("numpy", "float64", "cpu", False),
    ("torch", "float32", "cpu", False),
    ("torch", "float64", "cpu", False),
    ("torch_compiled_opt_einsum", "float64", "cpu", False),
    ("torch_compiled_opt_einsum", "float32", "cpu", False),
    ("cutensornet", "float32", "cuda:0", True),
    ("cutensornet", "float64", "cuda:0", True),
])
def test_set_contractor_variants(contractor, dtype, device, expect_gpu):
    H, logical, noise = make_simple_code()
    decoder = qec.get_decoder("tensor_network_decoder",
                              H,
                              logical_obs=logical,
                              noise_model=noise)
    import torch

    if expect_gpu and not torch.cuda.is_available():
        pytest.skip("No GPUs available, skip GPU contractor test.")

    if contractor == "cutensornet" and not torch.cuda.is_available():
        with pytest.raises(AssertionError):
            decoder.set_contractor(contractor, dtype=dtype, device=device)
        return

    if expect_gpu and "cuda" not in device:
        with pytest.raises(ValueError):
            decoder.set_contractor(contractor, dtype=dtype, device=device)
        return

    if not expect_gpu and contractor == "cutensornet":
        with pytest.raises(ValueError):
            decoder.set_contractor(contractor, dtype=dtype, device="cpu")
        return

    decoder.set_contractor(contractor, dtype=dtype, device=device)
    assert decoder._contractor_name == contractor
    assert decoder._dtype == dtype
    assert decoder._device == (device if expect_gpu else "cpu")
    if expect_gpu:
        assert decoder._backend == "torch"
    elif "torch" in contractor:
        assert decoder._backend == "torch"
    else:
        assert decoder._backend == "numpy"


@pytest.mark.parametrize(
    "init_contractor,change_contractor,init_device,change_device,expect_gpu", [
        ("numpy", "torch", "cpu", "cpu", False),
        ("torch", "numpy", "cpu", "cpu", False),
        ("torch", "torch_compiled_opt_einsum", "cpu", "cpu", False),
        ("torch", "cutensornet", "cpu", "cuda:0", True),
        ("cutensornet", "numpy", "cuda:0", "cpu", False),
        ("cutensornet", "torch", "cuda:0", "cpu", False),
        ("cutensornet", "cutensornet", "cuda:0", "cpu", True),
    ])
def test_decoder_change_contractor(init_contractor, change_contractor,
                                   init_device, change_device, expect_gpu):
    H, logical, noise = make_simple_code()
    import torch

    if ("cuda" in init_device or expect_gpu or
            "cuda" in change_device) and not torch.cuda.is_available():
        pytest.skip("No GPUs available, skip GPU contractor test.")

    # Initialize decoder with initial contractor and device
    decoder = qec.get_decoder("tensor_network_decoder",
                              H,
                              logical_obs=logical,
                              noise_model=noise,
                              contractor_name=init_contractor,
                              device=init_device)

    assert decoder._contractor_name == init_contractor
    assert decoder._device == init_device

    # Change contractor and device
    if expect_gpu and "cuda" not in change_device:
        with pytest.raises(ValueError):
            decoder.set_contractor(change_contractor, device=change_device)
        return

    if change_contractor == "cutensornet" and not torch.cuda.is_available():
        with pytest.raises(AssertionError):
            decoder.set_contractor(change_contractor, device=change_device)
        return

    decoder.set_contractor(change_contractor, device=change_device)
    assert decoder._contractor_name == change_contractor
    assert decoder._device == (change_device if expect_gpu else "cpu")
    if expect_gpu:
        assert decoder._backend == "torch"
    elif "torch" in change_contractor:
        assert decoder._backend == "torch"
    else:
        assert decoder._backend == "numpy"


def test_decoder_batch_vs_single_and_expected_results_with_contractors():
    np.random.seed(42)
    n_checks = 5
    n_errors = 8
    n_logical = 1
    n_batch = 10

    # Generate random binary parity check matrix and logical
    H = np.random.randint(0, 2, size=(n_checks, n_errors)).astype(np.float64)
    logical = np.random.randint(0, 2,
                                size=(n_logical, n_errors)).astype(np.float64)
    noise = np.random.uniform(0.01, 0.2, size=n_errors).tolist()

    import cudaq_qec as qec
    import torch

    # Provided expected results
    expected = [
        0.9604944927882665, 0.9796816612788876, 0.020709125507417103,
        0.35314051570803995, 0.3616138088105539, 0.01979825044290266,
        0.01979825044290266, 0.6381641010485968, 0.01979825044290266,
        0.3616795232730325
    ]

    contractors = [
        ("numpy", "float64", "cpu"),
        ("torch", "float64", "cpu"),
        ("cutensornet", "float32", "cuda:0"),
    ]

    decoder = qec.get_decoder("tensor_network_decoder",
                              H,
                              logical_obs=logical,
                              noise_model=noise)

    # Generate a batch of random syndromes
    batch = np.random.choice([False, True], size=(n_batch, n_checks))
    batch = batch.astype(np.float64, copy=False)  # Ensure float64 dtype

    for contractor, dtype, device in contractors:
        if "cuda" in device and not torch.cuda.is_available():
            # Skip cutensornet tests if no GPU is available
            continue
        try:
            decoder.set_contractor(contractor, dtype=dtype, device=device)
        except Exception as e:
            print(f"Skipping contractor {contractor} ({dtype}, {device}): {e}")
            continue

        # Decode each syndrome individually
        single_results = []
        for syndrome in batch:
            res = decoder.decode(syndrome.tolist())
            # Use float32 for float32 contractors, float64 otherwise
            if dtype == "float32":
                single_results.append(np.float32(res.result[0]))
            else:
                single_results.append(np.float64(res.result[0]))

        # Decode the batch
        res_batch = decoder.decode_batch(batch)
        if dtype == "float32":
            batch_results = [np.float32(r.result[0]) for r in res_batch]
            expected_cast = np.array(expected, dtype=np.float32)
            rtol = 1e-5
            atol = 1e-5
        else:
            batch_results = [np.float64(r.result[0]) for r in res_batch]
            expected_cast = np.array(expected, dtype=np.float64)
            rtol = 1e-5
            atol = 1e-5

        # Compare single and batch results
        np.testing.assert_allclose(single_results,
                                   batch_results,
                                   rtol=rtol,
                                   atol=atol)

        # Compare single and batch results
        np.testing.assert_allclose(single_results,
                                   expected_cast,
                                   rtol=rtol,
                                   atol=atol)

        # Compare to expected results
        np.testing.assert_allclose(batch_results,
                                   expected_cast,
                                   rtol=rtol,
                                   atol=atol)
