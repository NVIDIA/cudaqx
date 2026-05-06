# CUDA-Q QEC Reference

Companion to `SKILL.md`. Read `SKILL.md` first; it lists the conventions and
the workflow index that this file expands.

## Contents

- [Decoder Selection](#decoder-selection)
- [`nv-qldpc-decoder` parameters](#nv-qldpc-decoder-parameters)
- [Workflows](#workflows)
  - [W1: Code-Capacity Experiment](#w1-code-capacity-experiment)
  - [W2: Circuit-Level Memory Experiment](#w2-circuit-level-memory-experiment)
  - [W3: Custom Code](#w3-custom-code)
  - [W4: Custom Decoder](#w4-custom-decoder)
  - [W5: Sliding Window Decoder](#w5-sliding-window-decoder)
  - [W6: Real-Time Decoding](#w6-real-time-decoding)
  - [W7: DEM Sampling](#w7-dem-sampling)
  - [W8: Predecoder](#w8-predecoder)
- [Noise Model Patterns](#noise-model-patterns)
- [Self-Check Protocol](#self-check-protocol)
- [Troubleshooting: "LER looks wrong"](#troubleshooting-ler-looks-wrong)

---

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

Real-time eligibility today: `nv-qldpc-decoder`, `single_error_lut`, and
`multi_error_lut`. `tensor_network_decoder`, `trt_decoder`, and
`sliding_window` are not yet real-time.

## `nv-qldpc-decoder` parameters

The `bp_method` choice changes which extra parameters are required. Missing
a required gamma parameter raises on construction.

| `bp_method`   | Algorithm                  | Required extra params                                          |
|---------------|----------------------------|----------------------------------------------------------------|
| `0` (default) | Sum-Product BP             | none                                                           |
| `1`           | Min-Sum BP                 | optional `scale_factor`                                        |
| `2`           | Mem-BP (uniform memory)    | `gamma0`                                                       |
| `3`           | DMem-BP (disordered memory)| `gamma_dist=[min,max]` **or** `explicit_gammas`                |

OSD post-processing: `use_osd=True/False`,
`osd_method` in `{0=Off, 1=OSD-0, 2=Exhaustive, 3=Combination Sweep}`,
`osd_order=k`. Other tuning knobs: `max_iterations`, `use_sparsity`,
`bp_batch_size`, `error_rate_vec`.

For Sequential Relay BP, set `composition=1` together with `bp_method=3`,
`gamma0`, either `gamma_dist` or `explicit_gammas`, and
`srelay_config={'pre_iter': N, 'num_sets': K, 'stopping_criterion': 'FirstConv'|'NConv'|'All'}`
(plus `'stop_nconv': M` when using `'NConv'`). Full walkthrough:
`docs/sphinx/examples/qec/python/nv-qldpc-decoder.py`, function
`demonstrate_bp_methods`.

---

## Workflows

### W1: Code-Capacity Experiment

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

### W2: Circuit-Level Memory Experiment

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
4. `statePrep = qec.operation.prep0` for Z-basis. For X-basis, use `prepp`
   and `x_dem_from_memory_circuit`.
5. `dem = qec.z_dem_from_memory_circuit(code, statePrep, nRounds, noise)`.
6. `syndromes, data = qec.sample_memory_circuit(code, statePrep, nShots, nRounds, noise)`.
7. Slice off the X-stabilizer half of `syndromes` (Convention 2).
8. `decoder = qec.get_decoder("single_error_lut", dem.detector_error_matrix)`,
   or use `nv-qldpc-decoder`.
9. `dr = decoder.decode_batch(syndromes)`. Convert each `e.result` into a
   `uint8` hard vector.
10. Predicted observable flips:
    `data_predictions = (dem.observables_flips_matrix @ predictions.T) % 2`.
11. True logical measurements:
    `Lz = code.get_observables_z(); logical = (Lz @ data.T) % 2`.
12. LER = number of shots where `data_predictions XOR logical` is nonzero.

**Self-check**

- LER with decoding is below LER without decoding (which is `sum(logical)`).
- At `p=0`, both numbers are 0.
- If you get the same LER with and without decoding, you almost certainly
  failed step 7 (the slice) or used `code.get_parity_z()` instead of
  `dem.detector_error_matrix` in step 8.

### W3: Custom Code

Define a new code, in Python for prototyping or C++ for production.

**Python template**: `docs/sphinx/examples/qec/python/my_steane.py` and
`my_steane_test.py`.

**Steps (Python)**

1. Define `@cudaq.kernel` functions for each operation you support
   (`prep0`, `stabilizer`, optionally `x`, `z`, `h`, ...). Each takes a
   `qec.patch` and acts on its `data`, `ancx`, `ancz` views.
2. Define a class decorated with `@qec.code("name")` that inherits from
   `qec.Code`. Set `self.stabilizers` (a list of
   `cudaq.SpinOperator.from_word(...)`), `self.pauli_observables`, and
   `self.operation_encodings` (a mapping from `qec.operation.{...}` to
   your kernels). Override `get_num_data_qubits`,
   `get_num_ancilla_x_qubits`, `get_num_ancilla_z_qubits`,
   `get_num_ancilla_qubits`, `get_num_x_stabilizers`, and
   `get_num_z_stabilizers`.
3. Use it via `qec.get_code("name")` once the module is imported.

**C++**: read the "Implementing a New Code" section of
`docs/sphinx/components/qec/introduction.rst`. Subclass `cudaq::qec::code`,
register kernels in `operation_encodings`, set `m_stabilizers`, then
register the type with `CUDAQ_EXTENSION_CUSTOM_CREATOR_FUNCTION(name, ...)`
and `CUDAQ_REGISTER_TYPE(name)`. Reference implementation:
`libs/qec/include/cudaq/qec/codes/steane.h` together with the matching
`.cpp` in `libs/qec/lib/codes/`.

**Self-check**: `qec.get_code("name").get_stabilizers()` returns the
stabilizers you defined. Running W1 against the new code shows zero
logical errors at `p=0`.

### W4: Custom Decoder

Define a new decoder, in Python or C++.

**Steps (Python)**

1. Decorate the class: `@qec.decoder("my_decoder")`.
2. `__init__(self, H, **kwargs)` must call `qec.Decoder.__init__(self, H)`.
3. `decode(self, syndrome)` returns a `qec.DecoderResult()` with
   `.converged: bool` and `.result: list[float]` of length `block_size`.

**C++**: subclass `cudaq::qec::decoder`. The full virtual surface is in
`libs/qec/include/cudaq/qec/decoder.h`, including `decode_async`,
`decode_batch`, and the realtime API
`set_O_sparse`/`set_D_sparse`/`enqueue_syndrome`/`get_obs_corrections`/
`reset_decoder`. Register the type with
`CUDAQ_EXTENSION_CUSTOM_CREATOR_FUNCTION` and `CUDAQ_REGISTER_TYPE`.

**Self-check**: `qec.get_decoder("my_decoder", H)` returns an instance.
Running W1 with this decoder produces a sensible LER (zero at `p=0`).

### W5: Sliding Window Decoder

Decode many syndrome rounds incrementally, processing one window at a
time instead of waiting for the full sequence. Lower latency at the cost
of a small accuracy loss.

**Required keys** (see the "Sliding Window Decoder" section in
`introduction.rst`):

- `error_rate_vec`: one entry per column of `H`. Use `dem.error_rates`.
- `num_syndromes_per_round`: must be constant every round.
- `window_size` and `step_size`: must satisfy
  `(num_rounds - window_size) % step_size == 0`. `num_rounds` is inferred
  from `H.shape[0]` and `num_syndromes_per_round`.
- `inner_decoder_name` (typically `"nv-qldpc-decoder"`) and
  `inner_decoder_params` (a dict).

The PCM passed to `get_decoder("sliding_window", H, ...)` must be in
sorted form. Check with `qec.pcm_is_sorted`. DEMs from
`*_dem_from_memory_circuit` are already canonicalized; hand-built matrices
may need `qec.simplify_pcm` and/or `qec.sort_pcm_columns`.

**Self-check**: the constructor does not raise, partial syndromes leave
the decoder in an intermediate state, and the LER is no worse than
full-sequence decoding by more than the latency-vs-accuracy tradeoff
allows.

### W6: Real-Time Decoding

Decode inside the quantum kernel during circuit execution.

**Templates**

- Python (minimal end-to-end):
  `docs/sphinx/examples/qec/python/real_time_complete.py`.
- C++ (minimal end-to-end):
  `docs/sphinx/examples/qec/cpp/real_time_complete.cpp`.
- Production-shaped (CLI-driven, save and load DEM):
  - Python: `libs/qec/unittests/realtime/app_examples/surface_code_1.py`
    (note the underscore; only `_1` exists in Python).
  - C++: `libs/qec/unittests/realtime/app_examples/surface_code-1.cpp`,
    `surface_code-2.cpp`, and `surface_code-3.cpp` (note the dashes).
- Predecoder pipeline: see W8.
- Sequential Relay BP:
  `docs/sphinx/examples_rst/qec/realtime_relay_bp.rst`.

**The four phases run in this order:**

```
Phase 1: DEM         dem = qec.z_dem_from_memory_circuit(code, op, num_rounds, noise)
Phase 2: Configure   build qec.decoder_config + qec.multi_decoder_config; write YAML
Phase 3: Load        qec.configure_decoders_from_file("config.yaml")  (BEFORE cudaq.run)
Phase 4: In-kernel   qec.reset_decoder / qec.enqueue_syndromes / qec.get_corrections
                     then qec.finalize_decoders() at the end
```

**`qec.decoder_config` cheat sheet** (Phase 2):

```python
config = qec.decoder_config()
config.id = 0                                         # unique per logical qubit
config.type = "multi_error_lut"                       # or "nv-qldpc-decoder", ...
config.block_size = dem.detector_error_matrix.shape[1]
config.syndrome_size = dem.detector_error_matrix.shape[0]
config.H_sparse = qec.pcm_to_sparse_vec(dem.detector_error_matrix)
config.O_sparse = qec.pcm_to_sparse_vec(dem.observables_flips_matrix)
config.D_sparse = qec.generate_timelike_sparse_detector_matrix(
    num_syndromes_per_round, num_rounds, False)

lut_config = qec.multi_error_lut_config(); lut_config.lut_error_depth = 2
config.set_decoder_custom_args(lut_config)
# Or: qec.nv_qldpc_decoder_config(), qec.trt_decoder_config()

multi = qec.multi_decoder_config(); multi.decoders = [config]
open("config.yaml", "w").write(multi.to_yaml_str(200))
```

**Backend selection** (call `cudaq.set_target` before Phase 3):

| Backend                      | Target call                                                                                              |
|------------------------------|-----------------------------------------------------------------------------------------------------------|
| Local Stim simulation        | `cudaq.set_target("stim")`                                                                                |
| Quantinuum emulation         | `cudaq.set_target("quantinuum", emulate=True, machine="Helios-Fake", extra_payload_provider="decoder")`   |
| Quantinuum hardware (Helios) | `cudaq.set_target("quantinuum", emulate=False, machine="Helios-1", extra_payload_provider="decoder")`     |

`extra_payload_provider="decoder"` is required for both Quantinuum paths.
Without it, the decoder UUID is never injected into the job and the
circuit runs without decoding.

**C++ link flags**:

| Backend          | Add to nvq++ link line                                                                                                  |
|------------------|--------------------------------------------------------------------------------------------------------------------------|
| Stim             | `-lcudaq-qec -lcudaq-qec-realtime-decoding -lcudaq-qec-realtime-decoding-simulation`                                     |
| Quantinuum (any) | `-lcudaq-qec -lcudaq-qec-realtime-decoding -lcudaq-qec-realtime-decoding-quantinuum -Wl,--export-dynamic`                |

**Self-check**

- `configure_decoders_from_file` is called after `set_target` and before
  `cudaq.run`.
- The kernel calls `reset_decoder(id)` once per shot, `enqueue_syndromes`
  after each round, and `get_corrections` exactly once before measuring
  the logical observable.
- `qec.finalize_decoders()` is called at the end.
- For Quantinuum, set `CUDAQ_QEC_DEBUG_DECODER=1` and confirm the upload
  log appears:
  `[info] Initializing realtime decoding library with config file: ...`
  followed by `[info] Done initializing decoder N in T seconds`.

### W7: DEM Sampling

Sample errors and syndromes directly from a Detector Error Model, without
re-running the quantum circuit. Useful for scaled-up decoder benchmarks
and for generating decoder training data.

**Templates**: the Python entry point is
`libs/qec/python/cudaq_qec/dem_sampling.py`. The C++ surface lives in
`libs/qec/include/cudaq/qec/dem_sampling.h`. Tests are in
`libs/qec/python/tests/`; search for `dem_sampling`.

**API**

```python
from cudaq_qec import dem_sampling

# Return order is (syndromes, errors). NOT (errors, syndromes).
syndromes, errors = dem_sampling(
    check_matrix,            # NumPy ndarray or PyTorch CUDA tensor
    num_shots,
    error_probabilities,     # one entry per column of check_matrix
    seed=None,
    backend="auto",          # "auto" | "gpu" (cuStabilizer) | "cpu"
)
```

**Notes**

- The return order is `(syndromes, errors)`. It is easy to bind backwards.
  The function's docstring (`libs/qec/python/cudaq_qec/dem_sampling.py`)
  is authoritative.
- `backend="auto"` selects GPU (cuStabilizer) when available and falls
  back to CPU.
- PyTorch CPU tensors are not accepted. Convert to NumPy first.
- For a typical workflow, build the check matrix from
  `dem.detector_error_matrix` and the probabilities from `dem.error_rates`.

**Self-check**

- `syndromes.shape == (num_shots, check_matrix.shape[0])` (number of checks).
- `errors.shape == (num_shots, check_matrix.shape[1])` (number of error mechanisms).
- With `seed` set, two runs return identical arrays.
- Sanity: `(check_matrix @ errors.T) % 2 == syndromes.T`.

### W8: Predecoder

Run a fast first-pass decoder (typically a TensorRT NN, sometimes
PyMatching) in front of a slower main decoder, dispatching only the hard
cases to the main decoder. Built on the W6 real-time stack.

**Templates**

- PyMatching predecoder + main decoder:
  `docs/sphinx/examples_rst/qec/realtime_predecoder_pymatching.rst`.
- FPGA-based predecoder data injection:
  `docs/sphinx/examples_rst/qec/realtime_predecoder_fpga.rst`.
- Sample test scripts:
  `libs/qec/unittests/realtime/hololink_predecoder_test.sh` and
  `libs/qec/unittests/realtime/predecoder_pipeline_common.{h,cpp}`.

The TRT decoder (`trt_decoder`) is the typical neural front-end (bring
your own TensorRT model). Configure it for real-time with
`qec.trt_decoder_config`; see the W6 cheat sheet.

**Self-check**: same as W6, plus confirm that both stages register their
own decoder IDs in the YAML, and that the kernel routes syndromes through
the predecoder ID first.

---

## Noise Model Patterns

The standard QEC pattern is two-qubit depolarizing on every CX:

```python
noise = cudaq.NoiseModel()
noise.add_all_qubit_channel("x", cudaq.Depolarization2(p), 1)  # 1 control => CX
noise.add_all_qubit_channel("h", cudaq.BitFlipChannel(0.001))  # optional
```

Pass the same `noise` object to both `qec.sample_memory_circuit` and
`qec.*_dem_from_memory_circuit` (Convention 4). For per-gate or per-qubit
control, see
`docs/sphinx/examples/qec/python/custom_repetition_code_fine_grain_noise.py`.

---

## Self-Check Protocol

Walk this checklist before reporting "done" on a QEC task. Fix any failure
and retry.

```
[ ] Target set appropriately:
      cudaq.set_target("stim")          for kernel workflows (W2, W3, W6 sim, W8)
      cudaq.set_target("quantinuum",..) for W6 hardware/emulate
      none                              for pure matrix work (W1, W7)
[ ] If circuit-level: decoded against dem.detector_error_matrix,
    not code.get_parity().
[ ] If CSS prep0/prep1: sliced off the X-stabilizer half of the syndrome.
[ ] Same `noise` object passed to both sample_memory_circuit and the
    DEM helper.
[ ] Code actually executes end-to-end with a small nShots.
[ ] At p=0, the LER is 0.
[ ] At nonzero p, the LER with decoding is <= LER without decoding.
[ ] If real-time: configure_decoders_from_file is called between
    set_target and cudaq.run, and finalize_decoders is called at the end.
[ ] If real-time + Quantinuum: extra_payload_provider="decoder" is set.
[ ] If sliding_window: (num_rounds - window_size) % step_size == 0,
    num_syndromes_per_round is constant, and the PCM is sorted.
[ ] If nv-qldpc-decoder with bp_method=2 or 3: required gamma params
    are present (see `nv-qldpc-decoder` parameters above).
```

When the user reports "the LER looks wrong", the first three boxes catch
roughly 90% of cases.

---

## Troubleshooting: "LER looks wrong"

The most common symptom. Causes ranked by frequency.

1. **Did not slice the X-stabilizer half** for a `prep0`/`prep1` (Z-basis)
   experiment. See Convention 2 in `SKILL.md`. Symptom: LER barely
   improves with decoding, or matches the LER without decoding.
2. **Decoded against `code.get_parity()` instead of
   `dem.detector_error_matrix`** for a circuit-level experiment. Column
   ordering and weights are wrong relative to what `sample_memory_circuit`
   produced, so the LER is high.
3. **Different `noise` argument** passed to `sample_memory_circuit` vs.
   `*_dem_from_memory_circuit`. Use the same `noise` object for both.
4. **Forgot `cudaq.set_target("stim")`** for a kernel workflow
   (W2/W3/W6/W8). The default state-vector simulator chokes on QEC sizes
   long before reporting a useful LER. Pure matrix workflows (W1, W7) do
   not launch a kernel and need no target.
5. **Basis mismatch between prep and observable.** For `prep0`/`prep1`,
   use `code.get_observables_z()`. For `prepp`/`prepm`, use
   `get_observables_x()`.
6. **`p` is at or above the threshold** (around 1% for the surface code).
   Test at `p = 0.001` first.

### Other recurring failures

- `ImportError: ... libcustabilizer ...`: install matching cuQuantum
  (`pip install 'cuquantum-python-cu12>=26.03.0'`, or `-cu13`).
- `ImportError: ... libcudart ...`: install matching
  `nvidia-cuda-runtime-cuXX`.
- `Decoder X not found` at runtime: `qec.configure_decoders_from_file(...)`
  was not called between `set_target` and `cudaq.run`.
- Real-time silently runs without decoding on Quantinuum: missing
  `extra_payload_provider="decoder"`. Confirm by setting
  `CUDAQ_QEC_DEBUG_DECODER=1` and looking for
  `[info] Initializing realtime decoding library with config file: ...`.
- C++ link errors on Quantinuum: missing
  `-lcudaq-qec-realtime-decoding-quantinuum` or `-Wl,--export-dynamic`.
- Dimension mismatch: `num_rounds` differs between DEM generation and
  the circuit, or the X/Z half was not sliced.
- `tensor_network_decoder` errors: Python only, requires Python 3.11+.
  On V100 (SM70), pin `cutensor_cu12` with `pip install cutensor_cu12==2.2`.
- `Helios-1E` does not run GPU decoders. Expected; only `Helios-1` does
  today.
- Quantinuum `--emulate` reports zero LER. Expected; target QIR cannot
  yet express noise. Use Stim for noisy local testing.
