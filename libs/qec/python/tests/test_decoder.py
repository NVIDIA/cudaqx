# ============================================================================ #
# Copyright (c) 2024 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
import pytest
import numpy as np
import cudaq_qec as qec


def create_test_matrix():
    np.random.seed(42)
    return np.random.randint(0, 2, (10, 20)).astype(np.uint8)


def create_test_syndrome():
    np.random.seed(42)
    return np.random.random(10).tolist()


H = create_test_matrix()


def test_decoder_initialization():
    decoder = qec.get_decoder('example_byod', H)
    assert decoder is not None
    assert hasattr(decoder, 'decode')


def test_decoder_initialization_with_error():
    # We do not support column-major order (Fortran order)
    H_bad = np.zeros((10, 20), dtype=np.uint8, order='F')
    with pytest.raises(RuntimeError) as e:
        decoder = qec.get_decoder('single_error_lut_example', H_bad)


def test_decoder_api():
    # Test decode_batch
    decoder = qec.get_decoder('example_byod', H)
    result = decoder.decode_batch(
        [create_test_syndrome(), create_test_syndrome()])
    assert isinstance(result, qec.BatchDecoderResult)
    assert len(result) == 2
    assert hasattr(result, 'converged')
    assert hasattr(result, 'result')
    assert hasattr(result, 'opt_results')
    assert isinstance(result.converged, np.ndarray)
    assert result.converged.shape == (2,)
    assert isinstance(result.result, np.ndarray)
    assert result.result.shape == (2, 10)
    assert len(result.opt_results) == 2
    assert all(opt is None for opt in result.opt_results)
    first = result[0]
    assert isinstance(first, qec.DecoderResult)
    assert first.converged == result.converged[0]
    np.testing.assert_array_equal(first.result, result.result[0])
    assert first.opt_results is None
    last = result[-1]
    np.testing.assert_array_equal(last.result, result.result[-1])
    sliced = result[:1]
    assert isinstance(sliced, qec.BatchDecoderResult)
    assert sliced.result.shape == (1, 10)
    assert sliced.converged.shape == (1,)
    assert len(sliced.opt_results) == 1
    iterated = list(result)
    assert len(iterated) == 2
    assert all(isinstance(r, qec.DecoderResult) for r in iterated)
    np.testing.assert_array_equal(iterated[1].result, result.result[1])

    # Empty batch: shape (0, 0); per-shot width is undefined without input.
    empty_result = decoder.decode_batch([])
    assert isinstance(empty_result, qec.BatchDecoderResult)
    assert empty_result.result.shape == (0, 0)
    assert empty_result.converged.shape == (0,)
    assert len(empty_result) == 0
    assert len(empty_result.opt_results) == 0

    # Test decode_async
    decoder = qec.get_decoder('example_byod', H)
    result_async = decoder.decode_async(create_test_syndrome())
    assert hasattr(result_async, 'get')
    assert hasattr(result_async, 'ready')

    result = result_async.get()
    assert hasattr(result, 'converged')
    assert hasattr(result, 'result')
    assert isinstance(result.converged, bool)
    assert isinstance(result.result, np.ndarray)
    assert result.result.shape == (10,)


def test_decoder_result_structure():
    decoder = qec.get_decoder('example_byod', H)
    result = decoder.decode(create_test_syndrome())

    # Test basic structure
    assert hasattr(result, 'converged')
    assert hasattr(result, 'result')
    assert hasattr(result, 'opt_results')
    assert isinstance(result.converged, bool)
    assert isinstance(result.result, np.ndarray)
    assert result.result.shape == (10,)

    # Test opt_results functionality
    assert result.opt_results is None  # Default should be None

    # Test that opt_results is preserved in async decode
    async_result = decoder.decode_async(create_test_syndrome())
    result = async_result.get()
    assert hasattr(result, 'opt_results')
    assert result.opt_results is None


def test_batch_decoder_result_constructor():
    result = np.zeros((2, 3), dtype=np.float64)
    converged = np.array([True, False], dtype=np.bool_)
    batch_result = qec.BatchDecoderResult(result, converged)

    assert isinstance(batch_result.result, np.ndarray)
    assert batch_result.result.shape == (2, 3)
    assert batch_result.converged.tolist() == [True, False]
    assert batch_result.opt_results == [None, None]
    assert batch_result[np.int64(0)].converged is True
    assert batch_result[::2].result.shape == (1, 3)
    assert np.shares_memory(batch_result[::2].result, batch_result.result)

    empty_result = qec.BatchDecoderResult(np.empty((0, 3), dtype=np.float64),
                                          np.array([], dtype=np.bool_))
    assert len(empty_result) == 0
    assert empty_result.result.shape == (0, 3)
    assert empty_result.converged.shape == (0,)
    assert list(empty_result) == []
    with pytest.raises(IndexError):
        empty_result[0]

    # Cross-array invariants we still enforce.
    with pytest.raises(RuntimeError, match="row count must match"):
        qec.BatchDecoderResult(result,
                               np.array([True, False, True], dtype=np.bool_))

    with pytest.raises(RuntimeError, match="opt_results length must match"):
        qec.BatchDecoderResult(result, converged, [None])

    # Rank is enforced by nanobind, surfaced as TypeError.
    with pytest.raises(TypeError):
        qec.BatchDecoderResult(np.zeros(3, dtype=np.float64), converged)

    # dtype and contiguity are coerced silently, not rejected.
    int_result = qec.BatchDecoderResult(np.zeros((2, 3), dtype=np.int32),
                                        converged)
    assert int_result.result.dtype == batch_result.result.dtype
    assert int_result.result.flags.c_contiguous

    f_order = np.asfortranarray(np.zeros((2, 3), dtype=np.float64))
    assert not f_order.flags.c_contiguous
    f_result = qec.BatchDecoderResult(f_order, converged)
    assert f_result.result.flags.c_contiguous


def test_python_decoder_batch_preserves_opt_results():

    @qec.decoder("python_opt_results_byod")
    class PythonOptResultsDecoder:

        def __init__(self, H, **kwargs):
            qec.Decoder.__init__(self, H)
            self.H = H

        def decode(self, syndrome):
            res = qec.DecoderResult()
            res.converged = True
            res.result = np.arange(self.H.shape[1], dtype=np.float64)
            res.opt_results = {
                "syndrome_weight": int(np.count_nonzero(syndrome)),
                "tag": "python"
            }
            return res

    decoder = qec.get_decoder("python_opt_results_byod", H)
    batch_result = decoder.decode_batch(
        [np.zeros(H.shape[0]), np.ones(H.shape[0])])

    assert isinstance(batch_result, qec.BatchDecoderResult)
    assert batch_result.result.shape == (2, H.shape[1])
    assert batch_result.converged.tolist() == [True, True]
    assert batch_result.opt_results[0]["syndrome_weight"] == 0
    assert batch_result.opt_results[1]["syndrome_weight"] == H.shape[0]
    assert batch_result[1].opt_results["tag"] == "python"


def test_python_decoder_batch_override_must_return_batch_decoder_result():

    @qec.decoder("python_bad_batch_byod")
    class PythonBadBatchDecoder:

        def __init__(self, H, **kwargs):
            qec.Decoder.__init__(self, H)
            self.H = H

        def decode(self, syndrome):
            res = qec.DecoderResult()
            res.converged = True
            res.result = np.zeros(self.H.shape[1], dtype=np.float64)
            return res

        def decode_batch(self, syndromes):
            # Pre-batch return shape; the decorator should reject this.
            return [self.decode(s) for s in syndromes]

    decoder = qec.get_decoder("python_bad_batch_byod", H)
    with pytest.raises(TypeError, match="must return a BatchDecoderResult"):
        decoder.decode_batch([np.zeros(H.shape[0]), np.ones(H.shape[0])])


def test_decoder_plugin_initialization():
    decoder = qec.get_decoder('single_error_lut_example', H)
    assert decoder is not None
    assert hasattr(decoder, 'decode')


def test_decoder_plugin_initialization_with_double_vec():
    vec = np.array([1, 2, 3], dtype=np.float64)
    decoder = qec.get_decoder('single_error_lut_example', H, vec=vec)
    assert decoder is not None
    assert hasattr(decoder, 'decode')


def test_decoder_plugin_initialization_with_float_vec():
    vec = np.array([1, 2, 3], dtype=np.float32)
    decoder = qec.get_decoder('single_error_lut_example', H, vec=vec)
    assert decoder is not None
    assert hasattr(decoder, 'decode')


def test_decoder_plugin_initialization_with_uint8_vec():
    vec = np.array([1, 2, 3], dtype=np.uint8)
    decoder = qec.get_decoder('single_error_lut_example', H, vec=vec)
    assert decoder is not None
    assert hasattr(decoder, 'decode')


def test_decoder_plugin_initialization_with_int32_vec():
    vec = np.array([1, 2, 3], dtype=np.int32)
    decoder = qec.get_decoder('single_error_lut_example', H, vec=vec)
    assert decoder is not None
    assert hasattr(decoder, 'decode')


def test_decoder_plugin_initialization_with_int16_vec():
    vec = np.array([1, 2, 3], dtype=np.int16)
    with pytest.raises(RuntimeError) as e:
        decoder = qec.get_decoder('single_error_lut_example', H, vec=vec)
    assert "Unsupported array data type" in repr(e)


def test_decoder_plugin_result_structure():
    decoder = qec.get_decoder('single_error_lut_example', H)
    result = decoder.decode(create_test_syndrome())

    assert hasattr(result, 'converged')
    assert hasattr(result, 'result')
    assert isinstance(result.converged, bool)
    assert isinstance(result.result, np.ndarray)


def test_single_error_lut_example_uses_canonical_threshold():
    H = np.array([[0, 1]], dtype=np.uint8)
    decoder = qec.get_decoder('single_error_lut_example', H)
    result = decoder.decode([0.5])

    # For this H, syndrome "1" maps to a correction on qubit 1.
    assert result.converged is True
    assert np.array_equal(result.result, [0.0, 1.0])


def test_decoder_result_values():
    decoder = qec.get_decoder('example_byod', H)
    result = decoder.decode(create_test_syndrome())

    assert result.converged is True
    assert isinstance(result.result, np.ndarray)
    assert np.all((0 <= result.result) & (result.result <= 1))


@pytest.mark.parametrize("matrix_shape,syndrome_size", [((5, 10), 5),
                                                        ((15, 30), 15),
                                                        ((20, 40), 20)])
def test_decoder_different_matrix_sizes(matrix_shape, syndrome_size):
    np.random.seed(42)
    H = np.random.randint(0, 2, matrix_shape).astype(np.uint8)
    syndrome = np.random.random(syndrome_size).tolist()

    decoder = qec.get_decoder('example_byod', H)
    convergence, result, opt = decoder.decode(syndrome)

    assert len(result) == syndrome_size
    assert convergence is True
    assert isinstance(result, np.ndarray)
    assert np.all((0 <= result) & (result <= 1))


# FIXME add this back
# def test_decoder_error_handling():
#     H = Tensor(create_test_matrix())
#     decoder = qec.get_decoder('example_byod', H)

#     # Test with incorrect syndrome size
#     with pytest.raises(ValueError):
#         wrong_syndrome = np.random.random(15).tolist()  # Wrong size
#         decoder.decode(wrong_syndrome)

#     # Test with invalid syndrome type
#     with pytest.raises(TypeError):
#         wrong_type_syndrome = "invalid"
#         decoder.decode(wrong_type_syndrome)


def test_decoder_reproducibility():
    decoder = qec.get_decoder('example_byod', H)

    np.random.seed(42)
    convergence1, result1, opt1 = decoder.decode(create_test_syndrome())

    np.random.seed(42)
    convergence2, result2, opt2 = decoder.decode(create_test_syndrome())

    np.testing.assert_array_equal(result1, result2)
    assert convergence1 == convergence2


def test_pass_weights():
    error_probability = 0.1
    weights = np.ones(H.shape[1]) * np.log(
        (1 - error_probability) / error_probability)
    decoder = qec.get_decoder('example_byod', H, weights=weights)
    # Test is that no error is thrown


def test_sort_pcm_columns_non_decreasing_column_weight():
    # Create a test parity-check matrix with random binary values.
    # yapf: disable
    H = np.array([[0, 1, 0, 0, 1, 0, 0, 0, 1],
                  [1, 0, 0, 1, 1, 0, 0, 0, 0],
                  [0, 0, 1, 0, 1, 0, 1, 0, 0],
                  [0, 0, 0, 1, 1, 0, 0, 1, 0],
                  [0, 0, 0, 0, 1, 1, 1, 1, 1]],
                 dtype=np.uint8)
    # yapf: enable

    H_calculated = qec.sort_pcm_columns(H)

    # yapf: disable
    H_expected = np.array(
        [[1, 1, 1, 0, 0, 0, 0, 0, 0],
         [0, 0, 1, 1, 1, 0, 0, 0, 0],
         [0, 0, 1, 0, 0, 1, 1, 0, 0],
         [0, 0, 1, 0, 1, 0, 0, 1, 0],
         [0, 1, 1, 0, 0, 0, 1, 1, 1]],
        dtype=np.uint8)
    # yapf: enable

    assert np.array_equal(H_calculated, H_expected)

    col_order = qec.get_sorted_pcm_column_indices(H)
    expected_order = [1, 8, 4, 0, 3, 2, 6, 7, 5]
    assert col_order == expected_order

    # Now check that reordering the columns of H yields H_expected
    H_reordered = qec.reorder_pcm_columns(H, col_order)
    assert np.array_equal(H_reordered, H_expected)


def test_sort_pcm_columns_invalid_input():
    # Test that passing a non-2D array raises an error.
    H_invalid = np.array([0, 1, 0, 1], dtype=np.uint8)
    with pytest.raises(RuntimeError):
        qec.get_sorted_pcm_column_indices(H_invalid)


def test_generate_random_pcm_rejects_too_large_dense_allocation():
    """Dense random PCM rejects sizes above the dense API limit (~400e6 entries)."""
    with pytest.raises((ValueError, RuntimeError), match="generate_random_pcm"):
        qec.generate_random_pcm(n_rounds=1,
                                n_errs_per_round=25000,
                                n_syndromes_per_round=20000,
                                weight=1,
                                seed=1)


def test_gen_random_pcm():
    pcm = qec.generate_random_pcm(n_rounds=10,
                                  n_errs_per_round=20,
                                  n_syndromes_per_round=10,
                                  weight=3,
                                  seed=13)
    is_sorted = qec.pcm_is_sorted(pcm)
    assert is_sorted is False
    pcm = qec.sort_pcm_columns(pcm)
    is_sorted = qec.pcm_is_sorted(pcm)
    assert is_sorted is True
    print('')
    qec.dump_pcm(pcm)
    print('')
    assert pcm.shape == (100, 200)


def test_generate_random_pcm_sparse_matches_dense():
    """Sparse generator matches dense generate_random_pcm for the same RNG seed."""
    seed = 99
    weight = 3
    pcm_d = qec.generate_random_pcm(4, 5, 3, weight, seed=seed)
    sp = qec.generate_random_pcm_sparse(4, 5, 3, weight, seed=seed)
    assert sp["layout"] == "nested_csc"
    assert sp["num_rows"] == pcm_d.shape[0]
    assert sp["num_cols"] == pcm_d.shape[1]
    pcm_s = np.zeros((sp["num_rows"], sp["num_cols"]), dtype=np.uint8)
    for j, rows in enumerate(sp["nested"]):
        for ri in rows:
            pcm_s[int(ri), j] = 1
    assert np.array_equal(pcm_d, pcm_s)


def test_get_pcm_for_rounds():
    pcm = qec.generate_random_pcm(n_rounds=10,
                                  n_errs_per_round=20,
                                  n_syndromes_per_round=10,
                                  weight=3,
                                  seed=13)
    pcm = qec.sort_pcm_columns(pcm)
    pcm_for_rounds, first_column, last_column = qec.get_pcm_for_rounds(
        pcm, 10, 0, 1)
    assert pcm_for_rounds.shape == (20, 30)
    print('')
    qec.dump_pcm(pcm_for_rounds)
    print('')


def test_shuffle_pcm_columns():
    pcm = qec.generate_random_pcm(n_rounds=10,
                                  n_errs_per_round=20,
                                  n_syndromes_per_round=10,
                                  weight=3,
                                  seed=13)
    sorted_pcm = qec.sort_pcm_columns(pcm)
    shuffled_pcm = qec.shuffle_pcm_columns(sorted_pcm, seed=13)

    # They should not be equal here
    assert not np.array_equal(sorted_pcm, shuffled_pcm)

    # They should be equal after sorting
    assert np.array_equal(qec.sort_pcm_columns(shuffled_pcm), sorted_pcm)


def test_simplify_pcm():
    syndromes_per_round = 10
    pcm = qec.generate_random_pcm(
        n_rounds=10,
        n_errs_per_round=30,
        n_syndromes_per_round=syndromes_per_round,
        weight=1,  # force some duplicate columns for this test
        seed=13)
    weights = np.ones(pcm.shape[1]) * 0.01
    new_pcm, new_weights = qec.simplify_pcm(pcm, weights, syndromes_per_round)
    # qec.dump_pcm(new_pcm)
    print(new_pcm.shape)
    assert new_pcm.shape[0] == pcm.shape[0]
    assert new_pcm.shape[1] < pcm.shape[1]  # we expect fewer columns
    assert new_weights.shape == (new_pcm.shape[1],)

    # Test that the new weights are not all uniform.
    assert not np.allclose(new_weights, new_weights[0])


def test_version():
    decoder = qec.get_decoder('example_byod', H)
    assert "CUDA-Q QEC Base Decoder" in decoder.get_version()


def test_single_error_lut_opt_results():
    # Test with invalid opt_results
    invalid_args = {"opt_results": {"invalid_type": True}}
    with pytest.raises(RuntimeError) as e:
        decoder = qec.get_decoder("single_error_lut", H, **invalid_args)
        decoder.decode(create_test_syndrome())
    assert "Requested result types not available" in str(e.value)

    # Test with valid opt_results
    valid_args = {
        "opt_results": {
            "error_probability": True,
            "syndrome_weight": True,
            "decoding_time": False
        }
    }
    decoder = qec.get_decoder("single_error_lut", H, **valid_args)
    result = decoder.decode(create_test_syndrome())

    # Verify opt_results
    assert result.opt_results is not None
    assert "error_probability" in result.opt_results
    assert "syndrome_weight" in result.opt_results
    assert "decoding_time" not in result.opt_results  # Was set to False

    batch_result = decoder.decode_batch(
        [create_test_syndrome(), create_test_syndrome()])
    assert isinstance(batch_result.result, np.ndarray)
    assert batch_result.result.shape == (2, H.shape[1])
    assert isinstance(batch_result.converged, np.ndarray)
    assert batch_result.converged.shape == (2,)
    assert len(batch_result.opt_results) == 2
    for opt_results in batch_result.opt_results:
        assert opt_results is not None
        assert "error_probability" in opt_results
        assert "syndrome_weight" in opt_results
        assert "decoding_time" not in opt_results


def test_decoder_pymatching_results():
    pcm = qec.generate_random_pcm(n_rounds=2,
                                  n_errs_per_round=10,
                                  n_syndromes_per_round=5,
                                  weight=2,
                                  seed=7)
    pcm, _ = qec.simplify_pcm(pcm, np.ones(pcm.shape[1]), 10)
    # Pick 3 random columns from the PCM and XOR them together to get the
    # syndrome.
    columns = np.random.choice(pcm.shape[1], 3, replace=False)
    syndrome = np.sum(pcm[:, columns], axis=1) % 2
    decoder = qec.get_decoder('pymatching', pcm)
    result = decoder.decode(syndrome)
    assert result.converged is True
    assert isinstance(result.result, np.ndarray)
    assert np.all((0 <= result.result) & (result.result <= 1))
    actual_errors = np.zeros(pcm.shape[1], dtype=np.uint8)
    actual_errors[columns] = 1
    assert np.array_equal(result.result, actual_errors)


# --- Sparse matrix (sparse_binary_matrix) tests ---


def dense_to_nested_csc(H):
    """Convert dense binary matrix (numpy) to sparse dict with nested_csc layout."""
    rows, cols = H.shape
    nested = []
    for j in range(cols):
        nested.append([i for i in range(rows) if H[i, j] != 0])
    return {
        "layout": "nested_csc",
        "num_rows": int(rows),
        "num_cols": int(cols),
        "nested": nested,
    }


def dense_to_nested_csr(H):
    """Convert dense binary matrix (numpy) to sparse dict with nested_csr layout."""
    rows, cols = H.shape
    nested = []
    for i in range(rows):
        nested.append([j for j in range(cols) if H[i, j] != 0])
    return {
        "layout": "nested_csr",
        "num_rows": int(rows),
        "num_cols": int(cols),
        "nested": nested,
    }


def scipy_sparse_to_nested_csc(sp):
    """Convert scipy.sparse (csc or csr) to our sparse dict (nested_csc). Binary only."""
    sp = sp.tocsc()
    num_rows, num_cols = sp.shape
    nested = []
    for j in range(num_cols):
        start, end = sp.indptr[j], sp.indptr[j + 1]
        nested.append(sp.indices[start:end].tolist())
    return {
        "layout": "nested_csc",
        "num_rows": int(num_rows),
        "num_cols": int(num_cols),
        "nested": nested,
    }


def scipy_sparse_to_nested_csr(sp):
    """Convert scipy.sparse (csr or csc) to our sparse dict (nested_csr). Binary only."""
    sp = sp.tocsr()
    num_rows, num_cols = sp.shape
    nested = []
    for i in range(num_rows):
        start, end = sp.indptr[i], sp.indptr[i + 1]
        nested.append(sp.indices[start:end].tolist())
    return {
        "layout": "nested_csr",
        "num_rows": int(num_rows),
        "num_cols": int(num_cols),
        "nested": nested,
    }


def test_get_decoder_sparse_nested_csc():
    """get_decoder with sparse H as dict (nested_csc) produces a working decoder."""
    H_dense = create_test_matrix()
    H_sparse = dense_to_nested_csc(H_dense)
    decoder = qec.get_decoder("example_byod", H_sparse)
    assert decoder is not None
    assert hasattr(decoder, "decode")
    syndrome = create_test_syndrome()
    result = decoder.decode(syndrome)
    assert hasattr(result, "converged")
    assert hasattr(result, "result")
    assert len(result.result) == H_dense.shape[0]


def test_get_decoder_sparse_nested_csr():
    """get_decoder with sparse H as dict (nested_csr) produces a working decoder."""
    H_dense = create_test_matrix()
    H_sparse = dense_to_nested_csr(H_dense)
    decoder = qec.get_decoder("example_byod", H_sparse)
    assert decoder is not None
    assert hasattr(decoder, "decode")
    syndrome = create_test_syndrome()
    result = decoder.decode(syndrome)
    assert hasattr(result, "converged")
    assert hasattr(result, "result")
    assert len(result.result) == H_dense.shape[0]


def test_get_decoder_sparse_vs_dense_same_results():
    """Decoder from sparse H (nested_csc) behaves like dense H: same shape and validity."""
    np.random.seed(123)
    H_dense = np.random.randint(0, 2, (8, 16)).astype(np.uint8)
    syndrome = np.random.random(8).tolist()

    dec_dense = qec.get_decoder("example_byod", H_dense)
    dec_sparse = qec.get_decoder("example_byod", dense_to_nested_csc(H_dense))

    r_dense = dec_dense.decode(syndrome)
    r_sparse = dec_sparse.decode(syndrome)

    assert r_dense.converged == r_sparse.converged
    assert len(r_dense.result) == len(r_sparse.result)
    assert all(0 <= x <= 1 for x in r_dense.result)
    assert all(0 <= x <= 1 for x in r_sparse.result)


def test_get_decoder_sparse_nested_csr_same_as_csc():
    """Decoders from nested_csc and nested_csr (same matrix) produce valid results."""
    H_dense = create_test_matrix()
    syndrome = create_test_syndrome()

    dec_csc = qec.get_decoder("example_byod", dense_to_nested_csc(H_dense))
    dec_csr = qec.get_decoder("example_byod", dense_to_nested_csr(H_dense))

    r_csc = dec_csc.decode(syndrome)
    r_csr = dec_csr.decode(syndrome)

    assert r_csc.converged == r_csr.converged
    assert len(r_csc.result) == len(r_csr.result) == H_dense.shape[0]
    assert all(0 <= x <= 1 for x in r_csc.result)
    assert all(0 <= x <= 1 for x in r_csr.result)


def test_get_decoder_sparse_pymatching():
    """Native pymatching decoder accepts sparse H dict and returns valid result."""
    pcm = qec.generate_random_pcm(
        n_rounds=2,
        n_errs_per_round=10,
        n_syndromes_per_round=5,
        weight=2,
        seed=7,
    )
    pcm, _ = qec.simplify_pcm(pcm, np.ones(pcm.shape[1]), 10)
    H_sparse = dense_to_nested_csc(pcm)

    columns = np.random.choice(pcm.shape[1], 3, replace=False)
    syndrome = (np.sum(pcm[:, columns], axis=1) % 2).tolist()

    decoder = qec.get_decoder("pymatching", H_sparse)
    result = decoder.decode(syndrome)
    assert result.converged is True
    assert len(result.result) == pcm.shape[1]
    assert all(isinstance(x, float) for x in result.result)
    assert all(0 <= x <= 1 for x in result.result)


def test_get_decoder_sparse_dict_missing_keys():
    """Sparse dict missing required keys raises RuntimeError."""
    with pytest.raises(RuntimeError) as exc_info:
        qec.get_decoder("example_byod", {"layout": "nested_csc"})
    assert "layout" in str(exc_info.value) or "num_rows" in str(
        exc_info.value) or "nested" in str(exc_info.value)


def test_get_decoder_sparse_dict_invalid_layout():
    """Sparse dict with invalid layout string raises RuntimeError."""
    H_sparse = dense_to_nested_csc(create_test_matrix())
    H_sparse["layout"] = "invalid_layout"
    with pytest.raises(RuntimeError) as exc_info:
        qec.get_decoder("example_byod", H_sparse)
    assert "nested_csc" in str(exc_info.value) or "nested_csr" in str(
        exc_info.value)


def test_get_decoder_sparse_from_scipy():
    """get_decoder accepts sparse H built from scipy.sparse (converted to our dict)."""
    scipy_sparse = pytest.importorskip("scipy.sparse")
    np.random.seed(42)
    H_dense = np.random.randint(0, 2, (6, 12)).astype(np.uint8)
    H_scipy = scipy_sparse.csr_matrix(H_dense)
    H_sparse_dict = scipy_sparse_to_nested_csc(H_scipy)
    decoder = qec.get_decoder("example_byod", H_sparse_dict)
    assert decoder is not None
    syndrome = np.random.random(6).tolist()
    result = decoder.decode(syndrome)
    assert result.converged is True
    assert len(result.result) == H_dense.shape[0]
    assert all(0 <= x <= 1 for x in result.result)


def test_get_decoder_sparse_python_registered_decoder():
    """Python BYOD factories receive sparse dict unchanged (no dense to_dense bridge)."""
    H_dense = create_test_matrix()
    H_sparse = dense_to_nested_csc(H_dense)
    decoder = qec.get_decoder("example_byod", H_sparse)
    assert hasattr(decoder, "H")
    assert isinstance(decoder.H, dict)
    assert decoder.H["layout"] == "nested_csc"
    result = decoder.decode(create_test_syndrome())
    assert result.converged is True
    assert len(result.result) == H_dense.shape[0]


def test_pymatching_rejects_duplicate_sparse_input():
    """PyMatching requires duplicate-free sparse inputs."""
    sparse_dict = {
        "layout": "nested_csc",
        "num_rows": 2,
        "num_cols": 3,
        "nested": [[0, 0, 0], [0, 1], [1]],
    }
    with pytest.raises(ValueError, match="strictly increasing"):
        qec.get_decoder("pymatching", sparse_dict)


def test_generate_random_pcm_signed_weight_rejects_negative():
    """`weight=-1` surfaces the C++ guard, not nanobind's marshalling error."""
    with pytest.raises((ValueError, RuntimeError), match="weight"):
        qec.generate_random_pcm(n_rounds=2,
                                n_errs_per_round=3,
                                n_syndromes_per_round=4,
                                weight=-1,
                                seed=1)
    with pytest.raises((ValueError, RuntimeError), match="weight"):
        qec.generate_random_pcm_sparse(n_rounds=2,
                                       n_errs_per_round=3,
                                       n_syndromes_per_round=4,
                                       weight=-1,
                                       seed=1)


if __name__ == "__main__":
    pytest.main()
