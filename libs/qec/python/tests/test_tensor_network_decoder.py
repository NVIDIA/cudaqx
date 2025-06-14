import numpy as np
import pytest
from quimb.tensor import TensorNetwork
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
    import torch
    if not torch.cuda.is_available():
        pytest.skip("No GPUs available, skip cuQuantum test.")

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
