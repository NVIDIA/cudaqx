# Real-Time Decoding Examples

This directory contains documentation examples for CUDA-Q QEC real-time decoding. These are **simplified examples** designed to illustrate the API structure.

## Examples Overview

### Documentation Examples (This Directory)

- **`cpp/real_time_complete.cpp`** - Minimal C++ example showing API usage
- **`python/real_time_complete.py`** - Minimal Python example showing API usage

These examples demonstrate the API structure but require a decoder configuration file to run.

### Complete Working Examples

For **fully working, tested examples** that you can run out of the box, see:

- **`/workspaces/cuda-qx-g/libs/qec/unittests/realtime/app_examples/surface_code-1.cpp`**
  - Complete C++ example with DEM generation, configuration, and execution
  - Includes noise models, state preparation, stabilizer measurements
  - Can save/load decoder configurations
  - Runnable with: `nvq++ surface_code-1.cpp -lcudaq-qec -lcudaq-qec-realtime`

- **`/workspaces/cuda-qx-g/libs/qec/unittests/realtime/app_examples/surface_code_1.py`**
  - Complete Python example with all features
  - Includes DEM generation, multiple decoders, sliding windows
  - Can target simulation (Stim) or hardware (Quantinuum)
  - Runnable with: `python3 surface_code_1.py --distance 3 --num_shots 10 --save_dem config.yaml`

## How to Run the Documentation Examples

### Prerequisites

1. **Decoder Configuration File**: You need a `decoder_config.yaml` file. Generate one using:
   - The complete examples above with `--save_dem` flag
   - Or create manually using the detector error model from your code

2. **Compilation/Environment**: 
   - C++: Link with `-lcudaq-qec -lcudaq-qec-realtime`
   - Python: Ensure `cudaq` and `cudaq_qec` are in your Python path

### C++ Example

```bash
# Generate a decoder config first (using complete example)
cd /workspaces/cuda-qx-g/libs/qec/unittests/realtime/app_examples
nvq++ surface_code-1.cpp -lcudaq-qec -lcudaq-qec-realtime -o surface_code_1
./surface_code_1 --distance 3 --save_dem /path/to/decoder_config.yaml --state_prep prep0

# Then run the documentation example
cd /workspaces/cuda-qx-g/docs/sphinx/examples/qec/cpp
cp /path/to/decoder_config.yaml .
nvq++ real_time_complete.cpp -lcudaq-qec -lcudaq-qec-realtime -o real_time_example
./real_time_example
```

### Python Example

```bash
# Generate a decoder config first (using complete example)
cd /workspaces/cuda-qx-g/libs/qec/unittests/realtime/app_examples
python3 surface_code_1.py --distance 3 --save_dem /path/to/decoder_config.yaml

# Then run the documentation example
cd /workspaces/cuda-qx-g/docs/sphinx/examples/qec/python
cp /path/to/decoder_config.yaml .
python3 real_time_complete.py
```

## Understanding Decoder Configuration

The decoder configuration YAML file contains:
- **Detector error matrix (H)**: Maps error mechanisms to syndromes
- **Observable flips matrix (O)**: Maps error mechanisms to logical observable flips
- **Detector matrix (D)**: Temporal correlation of detectors across rounds
- **Decoder type and parameters**: e.g., `multi_error_lut` with `lut_error_depth`

Example structure:
```yaml
decoders:
  - id: 0
    type: multi_error_lut
    block_size: 72
    syndrome_size: 24
    H_sparse: [0, 1, 4, -1, 2, 5, -1, ...]
    O_sparse: [0, 18, 36, -1]
    D_sparse: [0, 8, -1, 1, 9, -1, ...]
    decoder_custom_args:
      lut_error_depth: 2
```

## Real-Time Decoding Workflow

1. **Characterization Phase** (Host-side, offline):
   - Generate detector error model (DEM) using simulation
   - Create decoder configuration from DEM
   - Save configuration to YAML

2. **Runtime Phase** (Device-side, online):
   - Load decoder configuration
   - Reset decoder state before each circuit
   - Enqueue syndromes as they're measured
   - Get corrections and apply them
   - Finalize decoders at the end

## API Quick Reference

### C++
```cpp
// Host-side configuration
cudaq::qec::decoding::config::configure_decoders_from_file("config.yaml");

// Device-side operations (in __qpu__ kernels)
cudaq::qec::decoding::reset_decoder(decoder_id);
cudaq::qec::decoding::enqueue_syndromes(decoder_id, syndromes);
auto corrections = cudaq::qec::decoding::get_corrections(decoder_id, return_size, reset);

// Cleanup
cudaq::qec::decoding::config::finalize_decoders();
```

### Python
```python
# Host-side configuration
qec.configure_decoders_from_file("config.yaml")

# Device-side operations (in @cudaq.kernel functions)
qec.reset_decoder(decoder_id)
qec.enqueue_syndromes(decoder_id, syndromes)
corrections = qec.get_corrections(decoder_id, return_size, reset)

# Cleanup
qec.finalize_decoders()
```

## Troubleshooting

- **"Cannot find decoder_config.yaml"**: Generate a configuration file using the complete examples
- **"Decoder X not found"**: Ensure `configure_decoders` was called before the circuit
- **Compilation errors**: Link with both `-lcudaq-qec` and `-lcudaq-qec-realtime` (or `-lcudaq-qec-realtime-quantinuum` for hardware)
- **High error rates**: Check that syndromes match the detector error matrix dimensions

## Documentation

For comprehensive documentation, see:
- **Guide**: `/workspaces/cuda-qx-g/docs/sphinx/examples_rst/qec/realtime_decoding.rst`
- **C++ API**: `/workspaces/cuda-qx-g/docs/sphinx/api/qec/cpp_realtime_decoding_api.rst`
- **Python API**: `/workspaces/cuda-qx-g/docs/sphinx/api/qec/python_realtime_decoding_api.rst`

