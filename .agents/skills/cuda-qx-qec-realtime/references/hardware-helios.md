# Real-time decoding on Quantinuum Helios

The hardware target for in-kernel real-time decoding. Helios is the
H-class trapped-ion hardware family from Quantinuum; cudaq-qec
real-time decoding has been validated against the Quantinuum emulator
and hardware via CUDA-Q's `quantinuum` target.

## Authentication

```bash
export CUDAQ_QUANTINUUM_CREDENTIALS=$HOME/.cudaq/quantinuum_credentials.json
```

Without credentials, the `quantinuum` target falls back to emulation
mode if available, or raises a hard error.

## Target selection

```python
import cudaq
# Emulator
cudaq.set_target("quantinuum", emulate=True)
# Hardware
cudaq.set_target("quantinuum", emulate=False, machine="<machine-name>")
```

Replace `<machine-name>` with the machine ID granted to your account.
The emulator is the *first* place to verify a real-time decoding
program — running on hardware should be a no-op switch.

## What changes vs simulation

The four-phase procedure (`references/in-kernel.md`) is **identical** on
hardware:

```
Phase 1: DEM                  — same
Phase 2: decoder_config YAML  — same
Phase 3: configure_decoders_from_file — same
Phase 4: in-kernel API        — same
```

What differs:

1. **Target**: `cudaq.set_target("quantinuum", ...)` instead of
   `cudaq.set_target("stim")`.
2. **Decoder eligibility**: only decoders in the real-time eligible
   set (see `cuda-qx-qec-realtime/SKILL.md` Inputs) work on hardware.
   The decoder must be uploaded to the control system.
3. **Logging**: set `CUDAQ_QEC_DEBUG_DECODER=1` to see what gets
   uploaded. Useful for debugging mismatches between local DEM and
   what the hardware applies.
4. **Shot count and queueing**: hardware shots are slow and queued.
   Budget accordingly; tens of shots for development, thousands for a
   real measurement.

## Decoder upload

When `configure_decoders_from_file` runs against a Quantinuum target,
the decoder configuration (matrices + algorithm choice + parameters)
is uploaded to the control system. The upload happens once per
session. If you change the YAML between shots, you must
re-`configure`.

`CUDAQ_QEC_DEBUG_DECODER=1` prints the uploaded payload to stderr —
read this when corrections look wrong.

## Production templates

The validated end-to-end Python and C++ templates live under:

- Python: `libs/qec/unittests/realtime/app_examples/surface_code_1.py`
- C++:
  - `libs/qec/unittests/realtime/app_examples/surface_code-1.cpp`
  - `libs/qec/unittests/realtime/app_examples/surface_code-2.cpp`
  - `libs/qec/unittests/realtime/app_examples/surface_code-3.cpp`

Copy these verbatim into your project tree and adapt. They cover
CLI argument plumbing, DEM save/load to disk, decoder selection
between `nv-qldpc-decoder` and the LUT decoders, and post-run
analysis.

## Self-check

```
[ ] Verified on emulator (cudaq.set_target("quantinuum", emulate=True)) first.
[ ] CUDAQ_QUANTINUUM_CREDENTIALS points at a valid creds file.
[ ] Decoder is in the real-time eligibility set.
[ ] DEM matches the circuit being submitted (same kernel, same num_rounds).
[ ] CUDAQ_QEC_DEBUG_DECODER output on a smoke run shows the expected payload.
[ ] Post-run LER on emulator matches simulation within statistics.
```

## Common pitfalls

| Symptom | Cause |
|---------|-------|
| `RuntimeError: unsupported decoder for real-time` | chose a non-eligible decoder (`tensor_network_decoder`, `trt_decoder`, `sliding_window`) |
| Emulator passes, hardware errors with "machine name" | check the machine string against your account's machine list |
| All-zero corrections on hardware only | upload happened against the wrong DEM; check `CUDAQ_QEC_DEBUG_DECODER=1` output |
| Stale credentials | re-authenticate via the Quantinuum portal and regenerate the creds file |

## When stuck

1. Reproduce on `emulate=True` first; if that fails, hardware will
   too.
2. Open the matching `surface_code-{1,2,3}.cpp` and diff the YAML
   against your script's output.
3. Set `CUDAQ_QEC_DEBUG_DECODER=1` and compare the uploaded payload
   to what `multi.to_yaml_str` produced locally.
4. For non-Quantinuum hardware (FPGA, custom control system), the
   autonomous_decoder + RPC architecture (`autonomous-decoder.md` +
   `ai-predecoder-pipeline.md`) is the path forward; the
   `quantinuum` target is hardware-vendor-specific.
