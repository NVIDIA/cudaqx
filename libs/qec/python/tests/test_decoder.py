# ============================================================================ #
# Copyright (c) 2024 NVIDIA Corporation & Affiliates.                          #
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


def test_decoder_api():
    # Test decode_multi
    decoder = qec.get_decoder('example_byod', H)
    result = decoder.decode_multi(
        [create_test_syndrome(), create_test_syndrome()])
    assert len(result) == 2
    for r in result:
        assert hasattr(r, 'converged')
        assert hasattr(r, 'result')
        assert isinstance(r.converged, bool)
        assert isinstance(r.result, list)
        assert len(r.result) == 10

    # Test decode_async
    decoder = qec.get_decoder('example_byod', H)
    result_async = decoder.decode_async(create_test_syndrome())
    assert hasattr(result_async, 'get')
    assert hasattr(result_async, 'ready')

    result = result_async.get()
    assert hasattr(result, 'converged')
    assert hasattr(result, 'result')
    assert isinstance(result.converged, bool)
    assert isinstance(result.result, list)
    assert len(result.result) == 10


def test_decoder_result_structure():
    decoder = qec.get_decoder('example_byod', H)
    result = decoder.decode(create_test_syndrome())

    assert hasattr(result, 'converged')
    assert hasattr(result, 'result')
    assert isinstance(result.converged, bool)
    assert isinstance(result.result, list)
    assert len(result.result) == 10


def test_decoder_plugin_initialization():
    decoder = qec.get_decoder('single_error_lut_example', H)
    assert decoder is not None
    assert hasattr(decoder, 'decode')


def test_decoder_plugin_result_structure():
    decoder = qec.get_decoder('single_error_lut_example', H)
    result = decoder.decode(create_test_syndrome())

    assert hasattr(result, 'converged')
    assert hasattr(result, 'result')
    assert isinstance(result.converged, bool)
    assert isinstance(result.result, list)


def test_decoder_result_values():
    decoder = qec.get_decoder('example_byod', H)
    result = decoder.decode(create_test_syndrome())

    assert result.converged is True
    assert all(isinstance(x, float) for x in result.result)
    assert all(0 <= x <= 1 for x in result.result)


@pytest.mark.parametrize("matrix_shape,syndrome_size", [((5, 10), 5),
                                                        ((15, 30), 15),
                                                        ((20, 40), 20)])
def test_decoder_different_matrix_sizes(matrix_shape, syndrome_size):
    np.random.seed(42)
    H = np.random.randint(0, 2, matrix_shape).astype(np.uint8)
    syndrome = np.random.random(syndrome_size).tolist()

    decoder = qec.get_decoder('example_byod', H)
    convergence, result = decoder.decode(syndrome)

    assert len(result) == syndrome_size
    assert convergence is True
    assert all(isinstance(x, float) for x in result)
    assert all(0 <= x <= 1 for x in result)


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
    convergence1, result1 = decoder.decode(create_test_syndrome())

    np.random.seed(42)
    convergence2, result2 = decoder.decode(create_test_syndrome())

    assert result1 == result2
    assert convergence1 == convergence2


def test_pass_weights():
    error_probability = 0.1
    weights = np.ones(H.shape[1]) * np.log(
        (1 - error_probability) / error_probability)
    decoder = qec.get_decoder('example_byod', H, weights=weights)
    # Test is that no error is thrown


if __name__ == "__main__":
    pytest.main()
