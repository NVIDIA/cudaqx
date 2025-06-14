import numpy as np
import pytest

from quimb.tensor import TensorNetwork

from cudaq_qec.plugins.decoders.tensor_network_utils.contractors import BACKENDS, CONTRACTORS, optimize_path
from cudaq_qec.plugins.decoders.tensor_network_utils.noise_models import factorized_noise_model, error_pairs_noise_model


def test_cutn_contractor_matches_numpy_gpu_check():
    import numpy as np
    import torch
    from .contractors import cutn_contractor
    from cuquantum import tensornet as cutn

    # Check for GPU availability
    if not torch.cuda.is_available():
        pytest.skip("No GPUs available, skip cuQuantum test.")

    a = np.arange(4).reshape(2, 2).astype(np.float32)
    b = np.arange(4, 8).reshape(2, 2).astype(np.float32)
    subscripts = "ij,jk->ik"

    # cuQuantum result
    result_cutn = cutn_contractor(subscripts, [a, b])
    # Numpy result
    result_np = np.einsum(subscripts, a, b)

    np.testing.assert_allclose(result_cutn, result_np, rtol=1e-5, atol=1e-7)


def test_backends_and_contractors_dicts():
    assert isinstance(BACKENDS, list)
    assert "numpy" in BACKENDS
    assert "torch" in BACKENDS
    assert isinstance(CONTRACTORS, dict)
    for key in ["numpy", "torch", "torch_compiled_opt_einsum", "cutensornet"]:
        assert key in CONTRACTORS
    # Check that at least one CONTRACTORS key contains a BACKENDS key as a substring (except "cutensornet")
    assert any(
        any(b in k
            for k in CONTRACTORS
            if k != "cutensornet")
        for b in BACKENDS)


def test_optimize_path_numpy_variants():
    from quimb.tensor import TensorNetwork, Tensor
    from cuquantum import tensornet as cutn
    from opt_einsum.contract import PathInfo

    tn = TensorNetwork([
        Tensor(np.ones((2, 2)), inds=("a", "b")),
        Tensor(np.ones((2, 2)), inds=("b", "c")),
        Tensor(np.ones((2, 2)), inds=("c", "a")),
    ])

    # Case 1: optimize="auto"
    path, info = optimize_path("auto", output_inds=("a",), tn=tn)
    assert isinstance(path, (list, tuple))
    assert isinstance(info, PathInfo)

    # Case 2: optimize=None (should use cuQuantum path finder)
    path2, info2 = optimize_path(None, output_inds=("a",), tn=tn)
    assert path2 is not None
    from cuquantum.tensornet.configuration import OptimizerInfo
    assert isinstance(info2, OptimizerInfo)

    # Case 3: optimize=OptimizerOptions
    opt = cutn.OptimizerOptions()
    path3, info3 = optimize_path(opt, output_inds=("a",), tn=tn)
    assert path3 is not None
    assert isinstance(info3, OptimizerInfo)


def test_factorized_noise_model_basic():
    error_indices = ['e0', 'e1', 'e2']
    error_probabilities = [0.1, 0.5, 0.9]
    tags = ['tag0', 'tag1', 'tag2']
    tn = factorized_noise_model(error_indices,
                                error_probabilities,
                                tensors_tags=tags)
    assert isinstance(tn, TensorNetwork)
    assert len(tn.tensors) == 3
    for i, t in enumerate(tn.tensors):
        np.testing.assert_array_equal(
            t.data,
            np.array([1.0 - error_probabilities[i], error_probabilities[i]]))
        assert t.inds == (error_indices[i],)
        assert tags[i] in t.tags


def test_factorized_noise_model_default_tags():
    error_indices = ['e0', 'e1']
    error_probabilities = [0.2, 0.8]
    tn = factorized_noise_model(error_indices, error_probabilities)
    for t in tn.tensors:
        assert "NOISE" in t.tags


def test_error_pairs_noise_model_basic():
    error_index_pairs = [('e0', 'e1'), ('e2', 'e3')]
    error_probabilities = [
        np.array([[0.9, 0.1], [0.2, 0.8]]),
        np.array([[0.7, 0.3], [0.4, 0.6]])
    ]
    tags = ['tagA', 'tagB']
    tn = error_pairs_noise_model(error_index_pairs,
                                 error_probabilities,
                                 tensors_tags=tags)
    assert isinstance(tn, TensorNetwork)
    assert len(tn.tensors) == 2
    for i, t in enumerate(tn.tensors):
        np.testing.assert_array_equal(t.data, error_probabilities[i])
        assert t.inds == error_index_pairs[i]
        assert tags[i] in t.tags


def test_error_pairs_noise_model_default_tags():
    error_index_pairs = [('x', 'y')]
    error_probabilities = [np.array([[0.6, 0.4], [0.3, 0.7]])]
    tn = error_pairs_noise_model(error_index_pairs, error_probabilities)
    for t in tn.tensors:
        assert "NOISE" in t.tags
