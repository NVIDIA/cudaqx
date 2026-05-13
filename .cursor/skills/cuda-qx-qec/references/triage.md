# Triage: "the LER looks wrong" + decoder gotchas

The "something is suspicious" entry point. When the user reports
unexpected logical error rate, weird decoder behavior, or import
failures from the QEC stack, walk this file before guessing.

## `nv-qldpc-decoder` parameters

The `bp_method` choice changes which extra parameters are required.
Missing a required gamma parameter raises on construction.

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

For Sequential Relay BP, set `composition=1` together with
`bp_method=3`, `gamma0`, either `gamma_dist` or `explicit_gammas`, and
`srelay_config={'pre_iter': N, 'num_sets': K, 'stopping_criterion':
'FirstConv'|'NConv'|'All'}` (plus `'stop_nconv': M` when using
`'NConv'`). Full walkthrough:
`docs/sphinx/examples/qec/python/nv-qldpc-decoder.py`, function
`demonstrate_bp_methods`.

## Noise Model Patterns

The standard QEC pattern is two-qubit depolarizing on every CX:

```python
noise = cudaq.NoiseModel()
noise.add_all_qubit_channel("x", cudaq.Depolarization2(p), 1)  # 1 control => CX
noise.add_all_qubit_channel("h", cudaq.BitFlipChannel(0.001))  # optional
```

Pass the same `noise` object to both `qec.sample_memory_circuit` and
`qec.*_dem_from_memory_circuit` (Convention 4). For per-gate or
per-qubit control, see
`docs/sphinx/examples/qec/python/custom_repetition_code_fine_grain_noise.py`.

## Self-Check Protocol

Walk this checklist before reporting "done" on a QEC task. Fix any
failure and retry.

```
[ ] Target set appropriately:
      cudaq.set_target("stim")          for kernel workflows
      cudaq.set_target("quantinuum",..) for real-time hardware/emulate
      none                              for pure matrix work (code-capacity, DEM sampling)
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
    are present (see above).
```

When the user reports "the LER looks wrong", the first three boxes
catch roughly 90% of cases.

## Troubleshooting: "LER looks wrong"

Causes ranked by frequency.

1. **Did not slice the X-stabilizer half** for a `prep0`/`prep1`
   (Z-basis) experiment. See Convention 2 in `SKILL.md`. Symptom: LER
   barely improves with decoding, or matches the LER without decoding.
2. **Decoded against `code.get_parity()` instead of
   `dem.detector_error_matrix`** for a circuit-level experiment. Column
   ordering and weights are wrong relative to what
   `sample_memory_circuit` produced, so the LER is high.
3. **Different `noise` argument** passed to `sample_memory_circuit`
   vs. `*_dem_from_memory_circuit`. Use the same `noise` object for
   both.
4. **Forgot `cudaq.set_target("stim")`** for a kernel workflow. The
   default state-vector simulator chokes on QEC sizes long before
   reporting a useful LER. Pure matrix workflows (code-capacity, DEM
   sampling) do not launch a kernel and need no target.
5. **Basis mismatch between prep and observable.** For
   `prep0`/`prep1`, use `code.get_observables_z()`. For
   `prepp`/`prepm`, use `get_observables_x()`.
6. **`p` is at or above the threshold** (around 1% for the surface
   code). Test at `p = 0.001` first.

## Other recurring failures

- `ImportError: ... libcustabilizer ...`: install matching cuQuantum
  (`pip install 'cuquantum-python-cu12>=26.03.0'`, or `-cu13`).
- `ImportError: ... libcudart ...`: install matching
  `nvidia-cuda-runtime-cuXX`.
- `Decoder X not found` at runtime:
  `qec.configure_decoders_from_file(...)` was not called between
  `set_target` and `cudaq.run`.
- Real-time silently runs without decoding on Quantinuum: missing
  `extra_payload_provider="decoder"`. Confirm by setting
  `CUDAQ_QEC_DEBUG_DECODER=1` and looking for
  `[info] Initializing realtime decoding library with config file: ...`.
- C++ link errors on Quantinuum: missing
  `-lcudaq-qec-realtime-decoding-quantinuum` or `-Wl,--export-dynamic`.
- Dimension mismatch: `num_rounds` differs between DEM generation and
  the circuit, or the X/Z half was not sliced.
- `tensor_network_decoder` errors: Python only, requires Python 3.11+.
  On V100 (SM70), pin `cutensor_cu12` with `pip install
  cutensor_cu12==2.2`.
- `Helios-1E` does not run GPU decoders. Expected; only `Helios-1`
  does today.
- Quantinuum `--emulate` reports zero LER. Expected; target QIR cannot
  yet express noise. Use Stim for noisy local testing.
