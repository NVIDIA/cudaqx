# Decode (the 80% workflow): code-capacity + circuit-level + decoder choice

How to pick a decoder and run the two most common decoding workflows.
For custom code/decoder definitions see `extend.md`. For sliding window,
real-time, DEM sampling, and predecoders see `realtime.md`. For "the
LER looks wrong" see `decode-triage.md`.

## Decoder Selection

Walk this list top to bottom and stop at the first match.

1. Need an exact maximum-likelihood baseline for a small code? Use
   `tensor_network_decoder`. Python only; install with
   `pip install cudaq-qec[tensor-network-decoder]`. Requires Python 3.11+.
2. Have a trained TensorRT model to plug in? Use `trt_decoder`. Install
   with `pip install cudaq-qec[trt-decoder]`.
3. Need to decode many syndrome rounds with a latency budget? Use
   `sliding_window`, wrapping `nv-qldpc-decoder` as the inner decoder.
4. Have a GPU and a QLDPC or surface code at production scale? Use
   `nv-qldpc-decoder` (closed-source plugin). Wrap the
   `qec.get_decoder("nv-qldpc-decoder", H)` call in `try` so the code
   degrades gracefully when the plugin is not installed. Real-time
   eligible.
5. Tiny code, want a smoke-test baseline? Use `single_error_lut` (no
   parameters) or `multi_error_lut` (set `lut_error_depth`). Both are
   real-time eligible.

Real-time eligibility is owned by `cudaq-qec-realtime` — see its
SKILL.md for the current eligible set and what "eligible" means.

## Code-Capacity Experiment

Decode random bit-flips on a code's parity-check matrix. No circuit, no
kernel.

**Templates**

- Python: `docs/sphinx/examples/qec/python/code_capacity_noise.py`. Read
  this first.
- C++: `docs/sphinx/examples/qec/cpp/code_capacity_noise.cpp`.
- Tensor-network reference (exact ML baseline):
  `docs/sphinx/examples/qec/python/tensor_network_decoder.py`.

**Steps**

1. `code = qec.get_code("steane")` (or another built-in).
2. `H = code.get_parity_z()`. For CSS codes, use the Z-half for bit-flip
   experiments.
3. `decoder = qec.get_decoder("single_error_lut", H)`. Start with the
   simplest decoder.
4. Generate noise and a syndrome with
   `qec.sample_code_capacity(H, nShots, p)`, or build them by hand with
   `qec.generate_random_bit_flips(H.shape[1], p)` and `H @ data % 2`.
5. For each shot, call `result = decoder.decode(syndrome)`, check
   `result.converged`, threshold `result.result` at 0.5 to get a hard
   prediction, and compare `observable @ prediction % 2` against
   `observable @ data % 2`.

**Self-check**: at `p=0.05` over 100 shots with `single_error_lut` on
Steane, expect a small but nonzero number of logical errors. At `p=0`,
expect zero.

## Circuit-Level Memory Experiment

Simulate the stabilizer-extraction circuit under noise, decode the
syndromes, and report the logical error rate.

**Templates**

- Python: `docs/sphinx/examples/qec/python/circuit_level_noise.py`. Read
  first; it contains the CSS slice from Convention 2.
- C++: `docs/sphinx/examples/qec/cpp/circuit_level_noise.cpp`.
- Pseudo-threshold sweep over distances:
  `docs/sphinx/examples/qec/python/pseudo_threshold.py`.
- Per-qubit / per-gate noise:
  `docs/sphinx/examples/qec/python/custom_repetition_code_fine_grain_noise.py`.
- Hand-built PCM (no `qec.get_code`):
  `docs/sphinx/examples/qec/python/repetition_code_pcm.py`.

**Steps**

1. `cudaq.set_target("stim")`.
2. `code = qec.get_code("surface_code", distance=d)` (or another code).
3. Build the noise model:
   `noise = cudaq.NoiseModel(); noise.add_all_qubit_channel("x", cudaq.Depolarization2(p), 1)`.
4. `statePrep = qec.operation.prep0` for Z-basis. For X-basis, use
   `prepp` and `x_dem_from_memory_circuit`.
5. `dem = qec.z_dem_from_memory_circuit(code, statePrep, nRounds, noise)`.
6. `syndromes, data = qec.sample_memory_circuit(code, statePrep, nShots, nRounds, noise)`.
7. Slice off the X-stabilizer half of `syndromes` (Convention 2).
8. `decoder = qec.get_decoder("single_error_lut", dem.detector_error_matrix)`,
   or use `nv-qldpc-decoder`.
9. `dr = decoder.decode_batch(syndromes)`. Convert each `e.result` into
   a `uint8` hard vector.
10. Predicted observable flips:
    `data_predictions = (dem.observables_flips_matrix @ predictions.T) % 2`.
11. True logical measurements:
    `Lz = code.get_observables_z(); logical = (Lz @ data.T) % 2`.
12. LER = number of shots where `data_predictions XOR logical` is nonzero.

**Self-check**

- LER with decoding is below LER without decoding (which is `sum(logical)`).
- At `p=0`, both numbers are 0.
- If you get the same LER with and without decoding, you almost
  certainly failed step 7 (the slice) or used `code.get_parity_z()`
  instead of `dem.detector_error_matrix` in step 8.
