# First QEC example, walked through

The smallest end-to-end QEC example, with output interpretation.

## Run it

```bash
python docs/sphinx/examples/qec/python/code_capacity_noise.py
```

For pip-only users, find the same source at
<https://nvidia.github.io/cudaq/examples_rst/qec/code_capacity_noise.html>.

## What the example does

1. Builds the **Steane code** (a 7-qubit CSS code that protects 1
   logical qubit against any single physical error).
2. Extracts its parity-check matrix `H` from the Z-stabilizers.
3. Generates random bit-flip errors with probability `p`.
4. Computes the syndrome `s = H @ error % 2`.
5. Decodes the syndrome with `single_error_lut` — a tiny lookup-table
   decoder that maps each syndrome to the most likely single-qubit
   error.
6. Reports how often the decoded correction agreed with the actual
   logical state ("logical error rate", LER).

## Expected output

At `p=0.05` (5% per-qubit error rate) over a few hundred shots, you
will see something like:

```
Logical error rate: 0.043
```

The exact number bounces with the random seed; the order of magnitude
(a few percent) is the right answer for Steane + `single_error_lut`
at `p=0.05`. At `p=0`, the LER should be exactly `0` — a useful
sanity check.

## What to look at if it doesn't work

| Symptom | Likely cause |
|---------|--------------|
| `ImportError: cudaq_qec` | wheel not installed; see `pip-user.md` |
| `ImportError: libgfortran.so.5` | install system `libgfortran` |
| LER very close to 0.5 | randomness is fine; you may be reading raw `error` vs `correction` instead of LER |
| Wildly different LER across runs | not seeded; that's expected for this example |

## Concept check: what is "code capacity"?

The example uses the **code-capacity** noise model: errors happen
once, on each data qubit independently, with probability `p`. There
is no circuit, no time evolution, no measurement noise. It is the
simplest possible QEC setup and is mainly useful to:

- Verify a code's structure (parity matrix, observables).
- Compare two decoders' raw accuracy.
- Establish a "best-case" bound on decoder performance.

For anything closer to reality, you need the **circuit-level** noise
model, where the noisy stabilizer-extraction circuit itself is
simulated. That's the next example to read after this one.

## Where to go next

| If the user wants to | Read |
|----------------------|------|
| Run circuit-level noise | `docs/sphinx/examples/qec/python/circuit_level_noise.py` and `cudaq-qec-decode` SKILL.md |
| Try a stronger decoder | `cudaq-qec-decode/references/decode.md` "Decoder Selection" |
| Plot LER vs physical error rate | `pseudo_threshold.py` example and `cudaq-benchmarking` SKILL.md |
| Write a new code or decoder | `cudaq-qec-extending` SKILL.md |
| Decode in real time on hardware | `cudaq-qec-realtime` SKILL.md |

At this point the user has a working QEC stack. Hand them off to
`cudaq-qec-decode` for the workflow index.
