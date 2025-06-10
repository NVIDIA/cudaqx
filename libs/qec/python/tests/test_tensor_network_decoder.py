# ============================================================================ #
# Copyright (c) 2024 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import numpy as np
import pytest
from quimb.tensor import TensorNetwork
import stim
import cudaq_qec as qec


def make_simple_code():
    # [[1, 1, 0], [0, 1, 1]] parity check, 1 logical, depolarizing noise
    H = np.array([[1, 1, 0], [0, 1, 1]])
    logicals = np.array([[1, 0, 1]])
    noise = [0.1, 0.2, 0.3]
    return H, logicals, noise


def test_decoder_init_and_attributes():
    H, logicals, noise = make_simple_code()
    decoder = qec.get_decoder("tensor_network_decoder",
                              H,
                              logicals=logicals,
                              noise_model=noise)
    assert isinstance(decoder.code_tn, TensorNetwork)
    assert isinstance(decoder.logicals_tn, TensorNetwork)
    assert isinstance(decoder.syndrome_tn, TensorNetwork)
    assert isinstance(decoder.full_tn, TensorNetwork)
    assert hasattr(decoder, "noise_model")
    assert decoder._contractor_name == "numpy"
    assert decoder._backend == "numpy"
    assert decoder._device == "cpu"
    assert decoder._dtype == "float64"


def test_decoder_flip_syndromes():
    H, logicals, noise = make_simple_code()
    decoder = qec.get_decoder("tensor_network_decoder",
                              H,
                              logicals=logicals,
                              noise_model=noise)
    # Flip all to True
    new_syndromes = [True] * H.shape[0]
    decoder.flip_syndromes(new_syndromes)
    for i, t in enumerate(decoder.syndrome_tn.tensors):
        np.testing.assert_array_equal(t.data, np.array([1.0, -1.0]))
    # Flip all to False
    new_syndromes = [False] * H.shape[0]
    decoder.flip_syndromes(new_syndromes)
    for i, t in enumerate(decoder.syndrome_tn.tensors):
        np.testing.assert_array_equal(t.data, np.array([1.0, 1.0]))


def test_decoder_decode_single():
    H, logicals, noise = make_simple_code()
    decoder = qec.get_decoder("tensor_network_decoder",
                              H,
                              logicals=logicals,
                              noise_model=noise)
    syndrome = [False, True]
    res = decoder.decode(syndrome)
    assert hasattr(res, "converged")
    assert hasattr(res, "result")
    assert isinstance(res.result, list)
    assert 0.0 <= res.result[0] <= 1.0


def test_decoder_decode_batch():
    H, logicals, noise = make_simple_code()
    decoder = qec.get_decoder("tensor_network_decoder",
                              H,
                              logicals=logicals,
                              noise_model=noise)
    batch = np.array([[False, True], [True, False], [False, False]])
    res = decoder.decode_batch(batch)
    assert isinstance(res, list)
    assert all(hasattr(r, "converged") and hasattr(r, "result") for r in res)
    assert all(
        isinstance(r.result, list) and 0.0 <= r.result[0] <= 1.0 for r in res)


def test_decoder_set_contractor_invalid():
    H, logicals, noise = make_simple_code()
    decoder = qec.get_decoder("tensor_network_decoder",
                              H,
                              logicals=logicals,
                              noise_model=noise)
    with pytest.raises(ValueError):
        decoder.set_contractor("not_a_contractor")


def test_TensorNetworkDecoder_optimize_path_all_variants():
    import cotengra
    from cuquantum import tensornet as cutn
    from opt_einsum.contract import PathInfo
    from cuquantum.tensornet.configuration import OptimizerInfo

    # Simple code setup
    H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
    logicals = np.array([[1, 0, 1]], dtype=np.uint8)
    noise = [0.1, 0.2, 0.3]
    decoder = qec.get_decoder("tensor_network_decoder",
                              H,
                              logicals=logicals,
                              noise_model=noise)

    # optimize="auto" (opt_einsum)
    info = decoder.optimize_path(output_inds=("l_0",), optimize="auto")
    assert isinstance(decoder.path_single, (list, tuple))
    assert decoder.slicing_single is not None
    assert isinstance(info, PathInfo)

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
    H, logicals, noise = make_simple_code()
    decoder = qec.get_decoder("tensor_network_decoder",
                              H,
                              logicals=logicals,
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
    H, logicals, noise = make_simple_code()
    import torch

    if ("cuda" in init_device or expect_gpu or
            "cuda" in change_device) and not torch.cuda.is_available():
        pytest.skip("No GPUs available, skip GPU contractor test.")

    # Initialize decoder with initial contractor and device
    decoder = qec.get_decoder("tensor_network_decoder",
                              H,
                              logicals=logicals,
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


def test_parse_detector_error_model_real_stim_and_decoder_init():

    # import stim
    # Generate a real stim DetectorErrorModel
    circuit = stim.Circuit.generated("surface_code:rotated_memory_z",
                                     rounds=3,
                                     distance=3,
                                     after_clifford_depolarization=0.001,
                                     after_reset_flip_probability=0.01,
                                     before_measure_flip_probability=0.01,
                                     before_round_data_depolarization=0.01)
    detector_error_model = circuit.detector_error_model(decompose_errors=True)

    # import cudaq_qec as qec
    from cudaq_qec.plugins.decoders.tensor_network_decoder import (
        parse_detector_error_model,
        TensorNetworkDecoder,
    )
    # Call the function under test
    out_H, out_L, priors = parse_detector_error_model(detector_error_model)

    # Check types and shapes
    assert isinstance(out_H, np.ndarray)
    assert isinstance(out_L, np.ndarray)
    assert isinstance(priors, list)
    assert out_H.shape[1] == len(priors)
    # Try to initialize the TensorNetworkDecoder with the output
    decoder = qec.get_decoder("tensor_network_decoder",
                              out_H,
                              logicals=out_L,
                              noise_model=priors)
    assert isinstance(decoder, TensorNetworkDecoder)
    assert decoder.parity_check_matrix.shape == out_H.shape
    assert decoder.logicals.shape == out_L.shape
    assert hasattr(decoder, "noise_model")


def test_decoder_batch_vs_single_and_expected_results_with_contractors():
    np.random.seed(42)
    n_checks = 5
    n_errors = 8
    n_logicals = 1
    n_batch = 10

    # Generate random binary parity check matrix and logicals
    H = np.random.randint(0, 2, size=(n_checks, n_errors)).astype(np.float64)
    logicals = np.random.randint(0, 2,
                                 size=(n_logicals, n_errors)).astype(np.float64)
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
                              logicals=logicals,
                              noise_model=noise)

    # Generate a batch of random syndromes
    batch = np.random.choice([False, True], size=(n_batch, n_checks))

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
        # Compare to expected results
        np.testing.assert_allclose(batch_results,
                                   expected_cast,
                                   rtol=rtol,
                                   atol=atol)
