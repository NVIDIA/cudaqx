# Sliding-window decoding

Decode many syndrome rounds incrementally with a window that slides
forward in time. Bounded latency at the cost of a small accuracy
loss; the right choice when the syndrome stream is much longer than
the inner decoder's batch capacity.

Sliding window is **not real-time eligible today** — it cannot be
called from inside `@cudaq.kernel`. Use it for offline streaming
decoding or as a step in a hybrid pipeline.

## Constructor

```python
sw = qec.get_decoder("sliding_window", H,
    error_rate_vec=dem.error_rates,
    num_syndromes_per_round=N_s,
    window_size=W,
    step_size=S,
    inner_decoder_name="nv-qldpc-decoder",
    inner_decoder_params={...})
```

All keys are required:

| Key | Meaning |
|-----|---------|
| `error_rate_vec` | one entry per column of `H`. Use `dem.error_rates`. |
| `num_syndromes_per_round` | must be constant every round |
| `window_size` | rounds per window |
| `step_size` | rounds advanced between windows |
| `inner_decoder_name` | typically `"nv-qldpc-decoder"` |
| `inner_decoder_params` | dict forwarded to the inner decoder |

Invariant: `(num_rounds - window_size) % step_size == 0`. `num_rounds`
is inferred from `H.shape[0] / num_syndromes_per_round`.

## PCM must be sorted

The PCM passed to `get_decoder("sliding_window", H, ...)` must be in
sorted form. DEMs from `*_dem_from_memory_circuit` are already
canonicalized; check with `qec.pcm_is_sorted(H)`. Hand-built matrices
may need `qec.simplify_pcm` and/or `qec.sort_pcm_columns` first.

## Usage

```python
sw.decode(syndromes[:window_size * num_syndromes_per_round])
sw.decode(syndromes[step_size * num_syndromes_per_round:
                   (step_size + window_size) * num_syndromes_per_round])
# ... continue until consumed
```

Partial syndromes leave the decoder in an intermediate state; only
fully consumed windows produce committed corrections.

## Choosing `window_size` and `step_size`

- `window_size` = inner decoder's accuracy "horizon"; bigger windows
  see more syndromes and decode better, at higher per-window latency.
- `step_size` = how aggressively to advance. Smaller step_size
  produces more overlap (more compute, more accurate); larger
  step_size is faster but tends to miss errors that straddle window
  boundaries.
- A good starting point: `window_size = 2 * d` rounds for a
  distance-d code, `step_size = window_size / 2`.

## Wrapping an inner decoder

Any decoder supported by `qec.get_decoder` can be wrapped. The most
common pairing is `nv-qldpc-decoder` for QLDPC / surface codes. Pass
its full parameter dict via `inner_decoder_params`:

```python
inner_params = dict(bp_method=3,
                    gamma_dist=[0.1, 0.5],
                    use_osd=True,
                    osd_method=2,
                    osd_order=10,
                    max_iterations=100)

sw = qec.get_decoder("sliding_window", H,
    error_rate_vec=dem.error_rates,
    num_syndromes_per_round=N_s,
    window_size=20,
    step_size=10,
    inner_decoder_name="nv-qldpc-decoder",
    inner_decoder_params=inner_params)
```

## Self-check

```
[ ] qec.pcm_is_sorted(H) returns True
[ ] (num_rounds - window_size) % step_size == 0
[ ] Inner decoder constructed standalone first to confirm parameters
[ ] LER no worse than full-batch decoding by more than the accuracy budget
```

## Where else to look

- Component docs: `docs/sphinx/components/qec/introduction.rst`
  "Sliding Window Decoder" section.
- Source: `libs/qec/lib/decoders/sliding_window.cpp`, `libs/qec/lib/decoders/sliding_window.h`.
- For real-time eligible incremental decoders, see
  `cuda-qx-qec-realtime/references/autonomous-decoder.md`.
