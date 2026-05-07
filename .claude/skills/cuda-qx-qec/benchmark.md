# CUDA-QX QEC Skill Benchmark

Evaluation prompts for the `cuda-qx-qec` skill. Same methodology as the
solvers benchmark; runnable via `scripts/score_benchmark.py`.

## Methodology

Run three passes:

1. With skill enabled
2. Without skill (control)
3. Activation pass (see below)

Two scoring layers:

**Human rubric** (per scenario, 0–8):

- Correctness (0–2): facts true, paths/APIs real
- Specificity (0–2): cites files, exact API names, exact kwargs
- Coverage (0–2): hits each "must include" item
- No hallucinations (0–2): no "must not include" items

12 scenarios × 8 + 10 activation = 106 max.

**Substring proxy** (`scripts/score_benchmark.py`):

- Coverage max for QEC = 45 (sum of `must_include` items)
- Purity max for QEC = 13 (1 per scenario, more for some)
- Activation = 10
- Substring total max = 68 for QEC. Always pair with a human pass.

---

## Scenario Prompts

### 1. F-order Parity Matrix

**prompt:** "My `qec.get_decoder('pymatching', H)` raises a runtime error.
H is `dtype=uint8` but ordered as Fortran. Help."

- must_include: "C-order", "C-contiguous"
- must_not_include: "F-order is supported"

### 2. cuStabilizer Missing

**prompt:** "Importing `cudaq_qec` fails with `libcustabilizer` not found."

- must_include: `cuquantum-python-cu12`, `cuquantum-python-cu13`,
  `26.03.0`
- must_not_include: "uninstall cudaq_qec"

### 3. dem_sampling Backend Choice

**prompt:** "How do I force GPU sampling in `dem_sampling`, and what
happens when GPU isn't available?"

- must_include: `backend="gpu"`, `RuntimeError`, `auto`, `cpu`
- must_not_include: "silently falls back when gpu requested"

### 4. PyTorch CPU Tensor

**prompt:** "Can I pass a PyTorch CPU tensor to `dem_sampling`?"

- must_include: "no", "convert to NumPy"
- must_not_include: "yes, supported"

### 5. Steane Memory Circuit Shape

**prompt:** "What is the shape of `syndromes` returned by
`sample_memory_circuit(steane, prep0, numShots=10, numRounds=4)`?"

- must_include: `(40, 6)`, "numShots * numRounds"
- must_not_include: `(10, 4, 6)`

### 6. Custom Python Code Requirements

**prompt:** "What attributes must my `@qec.code` class expose?"

- must_include: `stabilizers`, `pauli_observables`, `operation_encodings`
- must_include: `get_num_data_qubits`, `get_num_ancilla_qubits`
- must_not_include: "only stabilizers are required"

### 7. Sliding Window Decoder Setup

**prompt:** "How do I wrap pymatching in a sliding window decoder over a
distance-5 surface code?"

- must_include: `sliding_window`, `window_size`, `step_size`,
  `num_syndromes_per_round`, `inner_decoder_name`
- must_include: `error_rate_vec`
- must_not_include: "sliding_window does not require an inner decoder"

### 8. NV-QLDPC Config Fields

**prompt:** "What does `nv_qldpc_decoder_config` accept?"

- must_include: `use_osd`, `bp_method`, `proc_float`, `error_rate_vec`,
  `max_iterations`
- must_include: "default to None"
- must_not_include: "all fields are required"

### 9. Tensor Network Decoder Install

**prompt:** "I get `ModuleNotFoundError: quimb` when loading the tensor
network decoder. Fix?"

- must_include: `cudaq-qec[tensor_network_decoder]` (or `[all]`),
  `quimb`, `cuquantum-python`
- must_not_include: "quimb is included by default"

### 10. License Surprise

**prompt:** "Is `cudaq-qec` Apache 2.0 like `cudaq-solvers`?"

- must_include: "no", `LicenseRef-NVIDIA-Proprietary`,
  `libs/qec/pyproject.toml`
- must_not_include: "yes, both are Apache 2.0"

### 11. Operation Enum

**prompt:** "What logical operations does `qec.operation` expose?"

- must_include: `prep0`, `prep1`, `prepp`, `prepm`, `stabilizer_round`
- must_include: `cx`, `cz`, `h`, `s`
- must_not_include: "swap", "measure_x" (not in the enum)

### 12. DEM Variants

**prompt:** "What's the difference between `dem_from_memory_circuit`,
`x_dem_from_memory_circuit`, and `z_dem_from_memory_circuit`?"

- must_include: "X errors only", "Z errors only", "full"
- must_include: `DetectorErrorModel`, `detector_error_matrix`
- must_not_include: "they are aliases"

---

## Activation Tests

| # | prompt | should_activate |
| --- | --- | --- |
| A1 | "Build a Steane memory experiment" | Y |
| A2 | "Wrap pymatching in a sliding window decoder" | Y |
| A3 | "Run dem_sampling on a parity check matrix" | Y |
| A4 | "Configure NV-QLDPC for a custom code" | Y |
| A5 | "Install CUDA-Q from pip" | N |
| A6 | "Write a VQE for H2" | N |
| A7 | "Run a Bell state kernel" | N |
| A8 | "Generate a tensor-network decoder" | Y |
| A9 | "Use ADAPT-VQE on a molecule" | N |
| A10 | "Construct a detector error model from memory rounds" | Y |

## Sources

- `libs/qec/python/cudaq_qec/__init__.py`
- `libs/qec/python/bindings/py_decoder.cpp`, `py_code.cpp`,
  `py_dem_sampling.cpp`
- `libs/qec/python/cudaq_qec/dem_sampling.py`
- `libs/qec/python/cudaq_qec/plugins/decoders/tensor_network_decoder.py`
- `libs/qec/python/tests/test_*.py`
- `libs/qec/pyproject.toml.cu12`
- `docs/sphinx/components/qec/introduction.rst`
