# In-kernel real-time decoding (simulation and hardware)

The four-phase procedure, every required field, and the pitfalls. Copy
the templates verbatim and modify; do not derive.

## Templates (read first)

- Python minimal: `docs/sphinx/examples/qec/python/real_time_complete.py`
- C++ minimal: `docs/sphinx/examples/qec/cpp/real_time_complete.cpp`
- Production-shaped (CLI, save/load DEM):
  - Python: `libs/qec/unittests/realtime/app_examples/surface_code_1.py`
  - C++: `libs/qec/unittests/realtime/app_examples/surface_code-1.cpp`,
    `surface_code-2.cpp`, `surface_code-3.cpp`

## Phase 1: Generate the DEM

```python
cudaq.set_target("stim")

noise = cudaq.NoiseModel()
noise.add_all_qubit_channel("x", cudaq.Depolarization2(0.01), 1)

dem = qec.z_dem_from_memory_circuit(code, qec.operation.prep0, num_rounds, noise)
```

For non-CSS codes, use `qec.dem_from_memory_circuit`. The DEM gives
you `dem.detector_error_matrix` (the PCM the decoder will work
against) and `dem.observables_flips_matrix`.

## Phase 2: Build `decoder_config` + `multi_decoder_config`

```python
config = qec.decoder_config()
config.id = 0                                       # unique per logical qubit
config.type = "multi_error_lut"                     # or "nv-qldpc-decoder"
config.block_size = dem.detector_error_matrix.shape[1]
config.syndrome_size = dem.detector_error_matrix.shape[0]
config.H_sparse = qec.pcm_to_sparse_vec(dem.detector_error_matrix)
config.O_sparse = qec.pcm_to_sparse_vec(dem.observables_flips_matrix)

# Time-like detector matrix (note: num_rounds_plus_one)
num_syndromes_per_round = 2
num_rounds_plus_one = dem.detector_error_matrix.shape[0] // num_syndromes_per_round + 1
config.D_sparse = qec.generate_timelike_sparse_detector_matrix(
    num_syndromes_per_round, num_rounds_plus_one, False)

# Per-decoder custom args
lut_config = qec.multi_error_lut_config()
lut_config.lut_error_depth = 2
config.set_decoder_custom_args(lut_config)
# Alternatives: qec.nv_qldpc_decoder_config(), qec.trt_decoder_config()

multi = qec.multi_decoder_config()
multi.decoders = [config]

with open("config.yaml", "w") as f:
    f.write(multi.to_yaml_str(200))
```

The `200` argument to `to_yaml_str` is the wrap column; not a count
of entries.

## Phase 3: Load the YAML (before `cudaq.run`)

```python
qec.configure_decoders_from_file("config.yaml")
```

If you call this *after* `cudaq.run`, in-kernel API calls will see a
default decoder and produce nonsense corrections. This is the single
most common bug.

## Phase 4: Use in-kernel API

```python
@cudaq.kernel
def qec_circuit() -> int:
    qec.reset_decoder(0)

    data = cudaq.qvector(3)
    ancz = cudaq.qvector(2)
    ancx = cudaq.qvector(0)
    logical = patch(data, ancx, ancz)

    prep0(logical)
    for _ in range(num_rounds):
        syndromes = measure_stabilizers(logical)
        qec.enqueue_syndromes(0, syndromes, 0)

    corrections = qec.get_corrections(0, 1, False)   # (id, num_obs, blocking)
    if corrections[0]:
        for i in range(3):
            x(data[i])
    return cudaq.to_integer(mz(data))

cudaq.run(qec_circuit, shots_count=10)
qec.finalize_decoders()
```

The `blocking` argument controls whether `get_corrections` waits for
the decoder to finish. Use `False` for pipelined streaming, `True` if
you must apply the correction immediately.

## Per-decoder config types

| Decoder type | Custom-args struct | Notes |
|--------------|---------------------|-------|
| `multi_error_lut` | `qec.multi_error_lut_config()` | `lut_error_depth: int` |
| `single_error_lut` | none (no parameters) | Smallest codes only |
| `nv-qldpc-decoder` | `qec.nv_qldpc_decoder_config()` | many params; see `cudaq-qec-decode/references/decode-triage.md` `nv-qldpc-decoder parameters` table |
| `trt_decoder` | `qec.trt_decoder_config()` | path to `.engine`; see `cudaq-qec-ai-decoders` |

Sliding-window inner decoder configuration lives in
`references/sliding-window.md`.

## Hardware target (Quantinuum)

Swap `cudaq.set_target("stim")` for the Quantinuum target and set
credentials:

```python
cudaq.set_target("quantinuum",
                 emulate=False,
                 credentials=os.environ["CUDAQ_QUANTINUUM_CREDENTIALS"])
```

See `references/hardware-helios.md` for the full hardware story
(emulator vs hardware, machine names, batching, logging).

## Common pitfalls

| Symptom | Likely cause |
|---------|--------------|
| Decoder always returns zeros | `configure_decoders_from_file` called *after* `cudaq.run` |
| Wrong corrections for some logical qubits | duplicate `config.id` values |
| `RuntimeError: decoder size mismatch` | `block_size`/`syndrome_size` swapped, or `D_sparse` round count wrong |
| Hang on first shot | `nv-qldpc-decoder` plugin missing; see triage table |
| Different LER vs offline decoding of same DEM | likely an extra/missing syndrome round; check the `+1` in `num_rounds_plus_one` |

## Self-check

```
[ ] Phase 3 fires before cudaq.run.
[ ] Phase 4 functions live inside @cudaq.kernel.
[ ] config.id matches across all in-kernel calls.
[ ] D_sparse uses num_rounds_plus_one.
[ ] qec.finalize_decoders called at end-of-script.
[ ] At p=0 the decoder returns all-zero corrections.
```
