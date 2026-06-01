# One-page concepts glossary

For users who hit unfamiliar terms in the examples or skill prose.
Keep these definitions short; the goal is "enough to keep reading",
not a textbook.

## Quantum error correction

- **Physical qubit** — a real hardware qubit; noisy.
- **Logical qubit** — an abstract qubit encoded into many physical
  qubits via a QEC **code**. Less noisy than its physical pieces,
  provided the code, decoder, and noise level meet the threshold
  theorem's requirements.
- **Stabilizer** — a Pauli operator measured to detect errors without
  collapsing the encoded logical state. A QEC code is defined by its
  stabilizer generators.
- **Syndrome** — the binary string of stabilizer-measurement outcomes.
  A nonzero syndrome means an error happened.
- **Parity-check matrix (H)** — for stabilizer codes, the binary
  matrix whose rows encode the stabilizers' supports.
- **Decoder** — the classical algorithm that converts a syndrome into
  a correction.
- **Logical error rate (LER)** — the probability that, after
  correction, the logical state still ends up wrong.
- **Threshold** — the physical error rate below which the LER
  *decreases* as you grow the code. Below threshold, scaling up
  helps; above threshold, scaling up hurts.
- **Pseudo-threshold** — the cross-over distance, computed by
  intersecting LER curves for two adjacent code distances.
- **DEM (Detector Error Model)** — Stim's circuit-level description
  of all the error mechanisms in a noisy stabilizer-extraction
  circuit, including their probabilities and which detectors /
  observables they flip. The decoder works against this matrix, not
  the bare code parity.

## Decoders in this repo

- **`single_error_lut`** — exhaustive single-fault lookup. Tiny codes
  only. Real-time eligible.
- **`multi_error_lut`** — same idea up to `lut_error_depth` faults.
  Real-time eligible.
- **`tensor_network_decoder`** — exact maximum-likelihood decoding for
  small codes. Python only. The accuracy ceiling against which other
  decoders are measured.
- **`trt_decoder`** — TensorRT inference of a trained neural network.
  Needs an ONNX → engine build step.
- **`sliding_window`** — wraps an inner decoder; processes long
  syndrome streams incrementally for latency.
- **`nv-qldpc-decoder`** — production GPU decoder for QLDPC/surface
  codes. Closed-source plugin.

## Codes in this repo

- **Repetition code** — bit-flip-only; simplest code. Distance d
  protects against ⌊(d-1)/2⌋ bit-flips.
- **Steane code** — `[[7,1,3]]`: 7 physical qubits encode 1 logical,
  distance 3. CSS code; can correct any single-qubit Pauli error.
- **Surface code** — the standard 2D topological code. Distance
  scales with the patch size. CSS code.

## Variational quantum algorithms

- **Ansatz** — a parameterized quantum circuit; a guess at the
  ground-state shape.
- **Hamiltonian** — the operator whose ground-state energy you want
  to find (chemistry) or whose minimum encodes a problem solution
  (optimization).
- **Jordan-Wigner / Bravyi-Kitaev** — two ways to map fermionic
  operators (electrons) to qubit operators (Pauli strings).
- **VQE** — variational quantum eigensolver. Optimizer + ansatz +
  Hamiltonian → energy.
- **ADAPT-VQE** — iteratively grows the ansatz one operator at a
  time, picked from an operator pool by gradient.
- **QAOA** — VQE-shaped algorithm for combinatorial optimization.
- **GQE** — Generative quantum eigensolver: a transformer learns to
  *propose* operator sequences for ADAPT-like growth.
- **Operator pool** — the menu of operators ADAPT (or GQE) can add to
  the ansatz. UCCSD, UCCGSD, UPCCGSD, CEO, spin-complement-GSD.
- **HF / MP2 / CCSD / CASCI / CASSCF / FCI** — classical electronic
  structure baselines, in roughly increasing accuracy and cost. FCI
  is exact in the chosen basis; HF is mean-field. Pick the classical
  baseline to validate VQE against.

## CUDA-Q targets

- **`stim`** — fast stabilizer simulator. **The right target for
  almost all QEC kernel workflows.** Cannot simulate non-Clifford
  gates.
- **`nvidia`** — default statevector simulator. Fine for VQE-sized
  problems, does not scale to QEC.
- **`quantinuum`** — Quantinuum's emulator or hardware. Used for
  real-time decoding on actual H-class hardware.

## Environment variables you will see

- `CUDAQ_DEFAULT_SIMULATOR=stim` — set before `import cudaq_qec` to
  default kernels to stim.
- `OMP_NUM_THREADS=1` — set before `import cudaq_solvers` for
  reproducible PySCF coefficient signs.
- `CUDAQ_INSTALL_PREFIX`, `CUDAQX_INSTALL_PREFIX`, `CUDAQ_DIR`,
  `PYTHONPATH` — install paths; see `cudaq-build`.
- `CUDAQ_QEC_DEBUG_DECODER=1` — log real-time decoder uploads.
