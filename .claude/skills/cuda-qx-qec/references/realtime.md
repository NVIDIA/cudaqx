# Real-time, sliding window, DEM sampling, predecoder

Workflows that decode under a latency budget or that operate directly on
DEMs without re-running the kernel. All four share infrastructure
(in-kernel API, DEM types, real-time plumbing).

## Sliding Window Decoder

Decode many syndrome rounds incrementally, processing one window at a
time instead of waiting for the full sequence. Lower latency at the cost
of a small accuracy loss.

**Required keys** (see "Sliding Window Decoder" in
`docs/sphinx/components/qec/introduction.rst`):

- `error_rate_vec`: one entry per column of `H`. Use `dem.error_rates`.
- `num_syndromes_per_round`: must be constant every round.
- `window_size` and `step_size`: must satisfy
  `(num_rounds - window_size) % step_size == 0`. `num_rounds` is
  inferred from `H.shape[0]` and `num_syndromes_per_round`.
- `inner_decoder_name` (typically `"nv-qldpc-decoder"`) and
  `inner_decoder_params` (a dict).

The PCM passed to `get_decoder("sliding_window", H, ...)` must be in
sorted form. Check with `qec.pcm_is_sorted`. DEMs from
`*_dem_from_memory_circuit` are already canonicalized; hand-built
matrices may need `qec.simplify_pcm` and/or `qec.sort_pcm_columns`.

**Self-check**: the constructor does not raise, partial syndromes leave
the decoder in an intermediate state, and the LER is no worse than
full-sequence decoding by more than the latency-vs-accuracy tradeoff
allows.

## Real-Time Decoding (in-kernel)

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
    `surface_code-2.cpp`, `surface_code-3.cpp` (note the dashes).
- Predecoder pipeline: see "Predecoder" below.
- Sequential Relay BP:
  `docs/sphinx/examples_rst/qec/realtime_relay_bp.rst`.

**The four phases run in this order**:

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

`extra_payload_provider="decoder"` is required for both Quantinuum
paths. Without it, the decoder UUID is never injected into the job and
the circuit runs without decoding.

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
- For Quantinuum, set `CUDAQ_QEC_DEBUG_DECODER=1` and confirm:
  `[info] Initializing realtime decoding library with config file: ...`
  followed by `[info] Done initializing decoder N in T seconds`.

## DEM Sampling

Sample errors and syndromes directly from a Detector Error Model,
without re-running the quantum circuit. Useful for scaled-up decoder
benchmarks and for generating decoder training data.

**Templates**: the Python entry point is
`libs/qec/python/cudaq_qec/dem_sampling.py`. The C++ surface lives in
`libs/qec/include/cudaq/qec/dem_sampling.h`. Tests in
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

- The return order is `(syndromes, errors)`. It is easy to bind
  backwards. The function's docstring
  (`libs/qec/python/cudaq_qec/dem_sampling.py`) is authoritative.
- `backend="auto"` selects GPU (cuStabilizer) when available and falls
  back to CPU.
- PyTorch CPU tensors are not accepted. Convert to NumPy first.
- For a typical workflow, build the check matrix from
  `dem.detector_error_matrix` and the probabilities from
  `dem.error_rates`.

**Self-check**

- `syndromes.shape == (num_shots, check_matrix.shape[0])` (number of checks).
- `errors.shape == (num_shots, check_matrix.shape[1])` (number of error mechanisms).
- With `seed` set, two runs return identical arrays.
- Sanity: `(check_matrix @ errors.T) % 2 == syndromes.T`.

## Predecoder

Run a fast first-pass decoder (typically a TensorRT NN, sometimes
PyMatching) in front of a slower main decoder, dispatching only the
hard cases to the main decoder. Built on the real-time stack above.

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
`qec.trt_decoder_config`; see the cheat sheet above.

**Self-check**: same as Real-Time, plus confirm that both stages
register their own decoder IDs in the YAML, and that the kernel routes
syndromes through the predecoder ID first.
