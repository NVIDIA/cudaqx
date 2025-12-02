# ONNX Model Training and Test Data Generation for TRT Decoder

This document explains how to train a PyTorch model, export it to ONNX format, and generate test data for validating the TensorRT (TRT) decoder implementation.

## Overview

The TRT decoder uses a neural network model to decode quantum error correction syndromes. To ensure the TRT decoder produces identical results to the original PyTorch model, we need:

1. A trained PyTorch model exported to ONNX format
2. Test data with known input syndromes and expected outputs from the PyTorch model

## Prerequisites

Required Python packages:
```bash
pip install torch stim
```

Required files:
- `libs/qec/unittests/train_mlp_decoder.py` - Training script
- `libs/qec/unittests/generate_test_data_for_trt.py` - Test data generation script

## Training Process

### Step 1: Train the PyTorch Model

The training script (`train_mlp_decoder.py`) does the following:

1. **Generates synthetic quantum error correction data** using Stim:
   - Surface code with distance 3, 3 measurement rounds
   - Configurable error probability (e.g., 0.08 = 8% error rate)
   - Produces detector measurements (syndromes) and observable flips (logical errors)

2. **Trains a multi-layer perceptron (MLP)** decoder:
   - Input: 24 detector measurements (syndrome bits)
   - Output: 1 observable (probability of logical error)
   - Architecture: 256 → 128 → 64 → 1 neurons with ReLU activations and dropout
   - Loss function: Binary cross-entropy
   - Optimizer: Adam with learning rate scheduling

3. **Saves the trained model**:
   - `surface_code_decoder_best.pth` - PyTorch weights (best validation accuracy)
   - `surface_code_decoder.onnx` - ONNX export for TensorRT

### Step 2: Run the Training

From the repository root:

```bash
cd libs/qec/unittests
python3 train_mlp_decoder.py
```

**Key parameters** in `train_mlp_decoder.py`:
- `distance = 3` - Surface code distance
- `num_rounds = 3` - Number of measurement rounds
- `num_train_samples = 5000` - Training dataset size
- `num_val_samples = 1000` - Validation dataset size
- `num_test_samples = 1000` - Test dataset size
- `error_prob = 0.08` - Physical error probability (8%)
- `epochs = 1000` - Number of training epochs
- `hidden_dim = 128` - Hidden layer size

**Expected output:**
```
Num data qubits: 2, Num detectors: 24
Sampling 5000 training samples...
Sampling 1000 validation samples...
Sampling 1000 test samples...
Num observables: 1

Training started...
...
Epoch 1000/1000 | Train Loss: 0.5904 | Train Acc: 0.6910 | Val Loss: 0.7576 | Val Acc: 0.5000
Training complete! Best validation accuracy: 0.5180

TEST SET RESULTS:
Test Loss:                    0.6999
Test Accuracy:                0.5100 (51.00%)

Exporting model to ONNX...
ONNX model saved as surface_code_decoder.onnx
```

### Step 3: Generate Test Data

The test data generation script (`generate_test_data_for_trt.py`) does the following:

1. **Loads the trained PyTorch model** from `surface_code_decoder_best.pth`
2. **Generates 200 test samples** with the same Stim configuration
3. **Runs inference** on all test samples to get PyTorch predictions
4. **Creates a C++ header file** (`trt_test_data.h`) with:
   - First 100 test input syndromes
   - Corresponding PyTorch output predictions
   - Constants for array sizes

Run the generation script:

```bash
cd libs/qec/unittests
python3 generate_test_data_for_trt.py
```

**Expected output:**
```
Generating 200 test samples...
Running inference on test samples...
Generating C++ header file...
Test data header file generated: trt_test_data.h
  - Number of test samples: 100
  - Number of detectors: 24
  - Number of observables: 1
```

## Generated Files

### 1. `surface_code_decoder.onnx` (~188 KB)

ONNX format neural network model that can be loaded by TensorRT.

**Model architecture:**
- Input layer: 24 detectors (float32)
- Hidden layers: 256 → 128 → 64 neurons
- Output layer: 1 observable probability (float32)
- Activations: ReLU + Dropout (0.3)
- Output activation: Sigmoid

### 2. `trt_test_data.h` (~15 KB)

C++ header file containing test data arrays:

```cpp
constexpr int NUM_TEST_SAMPLES = 200;
constexpr int NUM_DETECTORS = 24;
constexpr int NUM_OBSERVABLES = 1;

// 100 test input syndromes (24 detectors each)
const std::vector<std::vector<float>> TEST_INPUTS = {
    {0.0, 0.0, 0.0, ..., 0.0},  // Test case 0
    {0.0, 1.0, 0.0, ..., 1.0},  // Test case 1
    ...
};

// 100 corresponding PyTorch predictions
const std::vector<std::vector<float>> TEST_OUTPUTS = {
    {0.458858},  // Prediction for test case 0
    {0.514391},  // Prediction for test case 1
    ...
};
```

## How the Tests Work

The test file `test_trt_decoder.cpp` validates the TRT decoder implementation:

1. **Loads the ONNX model** into TensorRT
2. **For each test case:**
   - Takes an input syndrome from `TEST_INPUTS`
   - Runs TRT inference
   - Compares TRT output to PyTorch output from `TEST_OUTPUTS`
   - Asserts the error is below tolerance (1e-4)

3. **Reports statistics:**
   - Number of passed/failed test cases
   - Maximum error across all cases
   - Average error

**Example test output:**
```
=== TRT Decoder Validation Summary ===
Total test cases: 100
Passed: 100
Failed: 0
Max error: 0.000087
Average error: 0.000023
====================================
```

## Adjusting Training Parameters

To experiment with different configurations:

### Change Error Rate
Edit `train_mlp_decoder.py`:
```python
error_prob = 0.01  # Lower error rate (easier problem, higher accuracy)
error_prob = 0.08  # Higher error rate (harder problem, lower accuracy)
```

### Change Model Size
Edit `train_mlp_decoder.py`:
```python
hidden_dim = 256  # Larger model (more capacity, slower training)
hidden_dim = 64   # Smaller model (less capacity, faster training)
```

### Change Dataset Size
Edit `train_mlp_decoder.py`:
```python
num_train_samples = 10000  # More training data
num_val_samples = 2000     # More validation data
```

### Change Training Duration
Edit `train_mlp_decoder.py`:
```python
epochs = 500   # Shorter training
epochs = 2000  # Longer training (may improve accuracy)
```

## Troubleshooting

### Low Training Accuracy
- **Cause:** High error rate, insufficient training epochs, or model too small
- **Solution:** Reduce `error_prob`, increase `epochs`, or increase `hidden_dim`

### Test Failures in TRT Decoder
- **Cause:** Mismatch between PyTorch and TRT implementations
- **Solution:** Check ONNX export compatibility, verify ONNX opset version (17)

### ONNX Export Warnings
- **Warning:** "Legacy TorchScript-based ONNX export" deprecation warning
- **Solution:** This is informational; the export still works correctly

## Reference

### Stim Circuit Configuration
The quantum error correction circuit is generated using:
```python
circuit = stim.Circuit.generated(
    "surface_code:rotated_memory_x",
    distance=3,
    rounds=3,
    after_clifford_depolarization=error_prob,
    after_reset_flip_probability=error_prob,
    before_measure_flip_probability=error_prob,
    before_round_data_depolarization=error_prob
)
```

This creates a distance-3 rotated surface code with:
- **Data qubits:** 5 (arranged in a cross pattern)
- **Syndrome qubits:** 4 (measure stabilizers)
- **Detectors:** 24 (from 3 rounds of measurements)
- **Observables:** 1 (logical X error)

### Model Training Details
- **Loss function:** Binary cross-entropy (BCE)
- **Optimizer:** Adam with learning rate 5e-4
- **Learning rate schedule:** ReduceLROnPlateau (factor=0.5, patience=20)
- **Batch size:** 128
- **Dropout rate:** 0.3 (during training only)

## Maintenance

When to regenerate test data:
1. **Model architecture changes** (layer sizes, activation functions)
2. **Training hyperparameters change** (learning rate, epochs)
3. **Circuit configuration changes** (distance, rounds, error_prob)
4. **Periodic re-training** to ensure test data freshness

Always regenerate both files together to maintain consistency:
```bash
python3 train_mlp_decoder.py && python3 generate_test_data_for_trt.py
```


