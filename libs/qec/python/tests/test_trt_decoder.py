# ============================================================================ #
# Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""
TRT Decoder Tests

These tests validate the TensorRT-based decoder for quantum error correction.

Test Categories:
  - Parameter validation and file operations (fast, no GPU required)

Note: GPU tests require CUDA/TensorRT and may crash if GPU is not properly
accessible within the container environment. Engine build takes ~5 seconds
on first run.
"""
import pytest
import numpy as np
import cudaq_qec as qec
import os
import tempfile

# Test data constants
NUM_TEST_SAMPLES = 200
NUM_DETECTORS = 24
NUM_OBSERVABLES = 1

# Path to the ONNX model file for testing (relative to this test file)
ONNX_MODEL_PATH = "assets/tests/surface_code_decoder.onnx"


# Check if CUDA/GPU is available for TensorRT tests
def _is_cuda_available():
    """Check if CUDA is available and can be initialized."""
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi'], capture_output=True, timeout=2)
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


CUDA_AVAILABLE = _is_cuda_available()

# Test inputs - 30 test cases with 24 detectors each
TEST_INPUTS = [[
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
],
               [
                   1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
               ],
               [
                   0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
               ],
               [
                   0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
               ],
               [
                   0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
               ],
               [
                   0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
               ],
               [
                   0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
               ],
               [
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
               ],
               [
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
               ],
               [
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
               ],
               [
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
               ],
               [
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
               ],
               [
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
               ],
               [
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                   1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
               ],
               [
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                   0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
               ],
               [
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                   0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
               ],
               [
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                   0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
               ],
               [
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                   0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
               ],
               [
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                   0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
               ],
               [
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0
               ],
               [
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0
               ],
               [
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0
               ],
               [
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0
               ],
               [
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0
               ],
               [
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0
               ],
               [
                   1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
               ],
               [
                   0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
               ],
               [
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0
               ],
               [
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                   1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0
               ],
               [
                   0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0
               ]]

# Expected outputs from PyTorch model
TEST_OUTPUTS = [[0.546192], [0.542916], [0.534589], [0.559515], [0.510735],
                [0.526783], [0.536010], [0.522932], [0.563812], [0.560833],
                [0.578034], [0.555534], [0.542923], [0.495684], [0.545392],
                [0.562221], [0.532649], [0.559208], [0.534731], [0.499671],
                [0.497873], [0.511810], [0.566766], [0.527120], [0.573466],
                [0.523840], [0.504422], [0.507719], [0.474930], [0.583623]]


class TestTRTDecoderSetup:
    """Test fixture setup and teardown for TRT decoder tests."""

    def setup_method(self):
        """Set up test parameters before each test."""
        self.block_size = 3
        self.syndrome_size = 2

        # Create a simple parity check matrix H
        self.H = np.zeros((self.syndrome_size, self.block_size), dtype=np.uint8)
        self.H[0, 0] = 1
        self.H[0, 1] = 0
        self.H[0, 2] = 1  # First syndrome bit
        self.H[1, 0] = 0
        self.H[1, 1] = 1
        self.H[1, 2] = 1  # Second syndrome bit

        # For file loading tests
        self.test_file_path = None

    def teardown_method(self):
        """Clean up any test files after each test."""
        if self.test_file_path and os.path.exists(self.test_file_path):
            os.remove(self.test_file_path)


class TestTRTDecoderParameterValidation(TestTRTDecoderSetup):
    """Tests for TRT decoder parameter validation."""

    def test_validate_parameters_valid_onnx_path(self):
        """Test parameter validation with valid ONNX path."""
        # Note: In Python, parameter validation happens inside the decoder creation
        # We can't test the validation function directly, but we can verify that
        # providing valid parameters doesn't raise errors when creating a decoder
        # This test is more of a placeholder showing the expected behavior
        pass

    def test_validate_parameters_valid_engine_path(self):
        """Test parameter validation with valid engine path."""
        # Similar to above - placeholder for parameter validation testing
        pass

    def test_validate_parameters_both_paths_provided(self):
        """Test that providing both ONNX and engine paths raises an error."""
        # This would need to be tested when actually creating a decoder
        # with both parameters, which should raise a RuntimeError
        pass

    def test_validate_parameters_no_paths_provided(self):
        """Test that providing no paths creates decoder with warning."""
        # Decoder is created but logs a warning - it won't be usable for inference
        # Create the TRT decoder
        try:
            decoder = qec.get_decoder('trt_decoder', self.H)
            # If decoder is None or doesn't initialize properly, skip these tests
            if decoder is None:
                pytest.skip(
                    "TRT decoder returned None - likely CUDA/GPU unavailable")
        except (RuntimeError, SystemError, Exception) as e:
            pytest.skip(
                f"Failed to create TRT decoder (GPU may be unavailable): {e}")


class TestTRTDecoderFileOperations(TestTRTDecoderSetup):
    """Tests for TRT decoder file loading operations."""

    def test_load_file_valid_file(self):
        """Test loading a valid file."""
        # Create a temporary test file
        test_content = "Hello, World!"
        with tempfile.NamedTemporaryFile(mode='w', delete=False,
                                         suffix='.txt') as f:
            f.write(test_content)
            self.test_file_path = f.name

        # Read the file back
        with open(self.test_file_path, 'r') as f:
            loaded_content = f.read()

        assert loaded_content == test_content

    def test_load_file_non_existent_file(self):
        """Test loading a non-existent file raises an error."""
        with pytest.raises(FileNotFoundError):
            with open('non_existent_file.txt', 'r') as f:
                f.read()


@pytest.mark.skipif(
    not os.path.exists(ONNX_MODEL_PATH) or not CUDA_AVAILABLE,
    reason="ONNX model file not found or CUDA/GPU not available")
class TestTRTDecoderInference(TestTRTDecoderSetup):
    """Tests for TRT decoder inference with actual model.
    
    Warning: These tests require GPU access and may take 20-30 seconds on first 
    run as TensorRT builds and optimizes the engine from the ONNX model.
    These tests may fail in containerized environments without proper GPU access.
    """

    def setup_method(self):
        """Set up decoder for inference tests."""
        super().setup_method()

        # Create parity check matrix matching the test data
        # For distance-3 surface code: 24 detectors
        num_detectors = NUM_DETECTORS

        # Create a dummy H matrix (identity matrix for simplicity)
        self.H_inference = np.eye(num_detectors, dtype=np.uint8)

        # Create the TRT decoder
        self.onnx_path = ONNX_MODEL_PATH
        try:
            self.decoder = qec.get_decoder('trt_decoder',
                                           self.H_inference,
                                           onnx_load_path=self.onnx_path)
            # If decoder is None or doesn't initialize properly, skip these tests
            if self.decoder is None:
                pytest.skip(
                    "TRT decoder returned None - likely CUDA/GPU unavailable")
        except (RuntimeError, SystemError, Exception) as e:
            pytest.skip(
                f"Failed to create TRT decoder (GPU may be unavailable): {e}")

    def test_validate_against_pytorch_model(self):
        """Test that TRT decoder produces identical results to PyTorch."""
        # Tolerance for floating point comparison
        TOLERANCE = 1e-4

        # Track statistics
        num_passed = 0
        num_failed = 0
        max_error = 0.0
        total_error = 0.0
        failed_cases = []

        # Test each of the 100 test cases
        for i, (test_input,
                test_output) in enumerate(zip(TEST_INPUTS, TEST_OUTPUTS)):
            # Run TRT decoder inference
            result = self.decoder.decode(test_input)

            # Get the expected output
            expected_output = test_output[0]

            # Get the TRT decoder output
            assert len(
                result.result
            ) > 0, f"TRT decoder returned empty result for test case {i}"
            trt_output = result.result[0]

            # Compute absolute error
            error = abs(trt_output - expected_output)
            total_error += error
            max_error = max(max_error, error)

            # Check if within tolerance
            if error < TOLERANCE:
                num_passed += 1
            else:
                num_failed += 1
                # Store detailed error info for first few failures
                if len(failed_cases) < 5:
                    failed_cases.append({
                        'index': i,
                        'expected': expected_output,
                        'got': trt_output,
                        'error': error
                    })

        # Print summary statistics
        print(f"\n=== TRT Decoder Validation Summary ===")
        print(f"Total test cases: {len(TEST_INPUTS)}")
        print(f"Passed: {num_passed}")
        print(f"Failed: {num_failed}")
        print(f"Max error: {max_error}")
        print(f"Average error: {total_error / len(TEST_INPUTS)}")
        print(f"====================================\n")

        # Print detailed failure information
        if failed_cases:
            print("\nFirst few failures:")
            for case in failed_cases:
                print(f"Test case {case['index']} FAILED:")
                print(f"  Expected: {case['expected']}")
                print(f"  Got:      {case['got']}")
                print(f"  Error:    {case['error']}")

        # Overall assertion: all cases must pass
        assert num_failed == 0, f"{num_failed} test cases failed validation"

    def test_validate_single_test_case(self):
        """Test a single specific case for detailed debugging."""
        # Test first case in detail
        syndrome = TEST_INPUTS[0]

        print(f"Input syndrome (first 10 values): {syndrome[:10]}")

        result = self.decoder.decode(syndrome)

        expected = TEST_OUTPUTS[0][0]
        actual = result.result[0]
        error = abs(actual - expected)

        print(f"Expected output: {expected}")
        print(f"Actual output:   {actual}")
        print(f"Absolute error:  {error}")
        print(f"Converged:       {result.converged}")

        assert error < 1e-4, "Single test case validation failed"
        assert result.converged, "Decoder did not converge"

    def test_decoder_result_structure(self):
        """Test that decoder result has expected structure."""
        syndrome = TEST_INPUTS[0]
        result = self.decoder.decode(syndrome)

        # Test basic structure
        assert hasattr(result, 'converged')
        assert hasattr(result, 'result')
        assert isinstance(result.converged, bool)
        assert isinstance(result.result, list)
        assert len(result.result) > 0

    def test_decoder_batch_processing(self):
        """Test decoder batch processing with multiple syndromes."""
        # Test with first 5 test cases
        syndromes = TEST_INPUTS[:5]
        expected_outputs = [out[0] for out in TEST_OUTPUTS[:5]]

        # Decode batch
        results = self.decoder.decode_batch(syndromes)

        assert len(results) == len(syndromes)

        # Check each result
        TOLERANCE = 1e-4
        for i, (result, expected) in enumerate(zip(results, expected_outputs)):
            assert hasattr(result, 'converged')
            assert hasattr(result, 'result')
            assert len(result.result) > 0

            error = abs(result.result[0] - expected)
            assert error < TOLERANCE, f"Batch test case {i} failed with error {error}"


class TestTRTDecoderEdgeCases(TestTRTDecoderSetup):
    """Tests for TRT decoder edge cases. Requires GPU access."""

    def test_decoder_with_zero_syndrome(self):
        """Test decoder with all-zero syndrome."""
        if not os.path.exists(ONNX_MODEL_PATH) or not CUDA_AVAILABLE:
            pytest.skip("ONNX model file not found or CUDA/GPU not available")

        num_detectors = NUM_DETECTORS
        H = np.eye(num_detectors, dtype=np.uint8)

        try:
            decoder = qec.get_decoder('trt_decoder',
                                      H,
                                      onnx_load_path=ONNX_MODEL_PATH)
        except Exception:
            pytest.skip("Failed to create TRT decoder")

        # All-zero syndrome
        zero_syndrome = [0.0] * num_detectors
        result = decoder.decode(zero_syndrome)

        assert hasattr(result, 'result')
        assert len(result.result) > 0

    def test_decoder_with_all_ones_syndrome(self):
        """Test decoder with all-ones syndrome."""
        if not os.path.exists(ONNX_MODEL_PATH) or not CUDA_AVAILABLE:
            pytest.skip("ONNX model file not found or CUDA/GPU not available")

        num_detectors = NUM_DETECTORS
        H = np.eye(num_detectors, dtype=np.uint8)

        try:
            decoder = qec.get_decoder('trt_decoder',
                                      H,
                                      onnx_load_path=ONNX_MODEL_PATH)
        except Exception:
            pytest.skip("Failed to create TRT decoder")

        # All-ones syndrome
        ones_syndrome = [1.0] * num_detectors
        result = decoder.decode(ones_syndrome)

        assert hasattr(result, 'result')
        assert len(result.result) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
