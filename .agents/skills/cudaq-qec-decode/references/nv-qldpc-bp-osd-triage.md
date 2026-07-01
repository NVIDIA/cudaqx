# Public-safe `nv-qldpc-decoder` BP/OSD triage

Use this file when an offline `nv-qldpc-decoder` run produces suspicious
convergence, latency, or logical-error numbers. Keep the discussion at the
public API and decoding-concept level. Do not include proprietary kernel
internals, private benchmark data, or unreleased implementation details in
user-facing output.

## What to inspect first

For every run, collect this dashboard:

```text
code / circuit / DEM identity
noise model and physical error rate
decoder kwargs
number of shots
num_converged
num_logical_errors
avg_num_iter, if requested
avg_time_ms and p99_time_ms, if measured
```

Interpret these as different axes, not one score:

```text
accuracy   = logical-error behavior
robustness = convergence and iteration behavior
speed      = average and tail latency
```

## Metric meanings

`converged` means the decoder found a correction whose predicted syndrome
matches the observed syndrome:

```text
H e_hat = s  mod 2
```

This is a parity-check consistency test, not a proof that the logical
observable is correct.

`num_residuals` means the number of unsatisfied checks:

```text
num_residuals = count rows where (H e_hat)_row != s_row
```

Zero residuals is equivalent to syndrome consistency. A residual trace over
iterations is useful for diagnosing whether BP is improving, oscillating, or
stalled.

`num_iter` is the number of BP message-passing iterations used. Fewer
iterations usually means an easier decode, but it is not a standalone
performance metric because a heavier decoder configuration can reduce
iterations while increasing wall-clock latency.

`num_logical_errors` is the QEC correctness metric. A decoder can converge
and still choose the wrong logical class. For circuit-level or DEM workflows,
compare the predicted observable flip against the sampled observable:

```text
predicted_observable = observables_flips_matrix @ e_hat  mod 2
```

`avg_time_ms` is typical per-shot latency. `p99_time_ms` is tail latency; for
latency-bounded systems, p99 can matter more than the average.

## Parameter groups

Basic backend parameters:

```text
use_sparsity
proc_float = "fp32" | "fp64"
bp_batch_size
max_iterations
iter_per_check
clip_value
repeatable
```

Core BP method:

```text
bp_method = 0  sum-product BP
bp_method = 1  min-sum BP
bp_method = 2  min-sum BP with uniform memory
bp_method = 3  min-sum BP with per-bit/disordered memory
```

Memory and relay parameters:

```text
gamma0
gamma_dist = [min, max]
explicit_gammas
composition = 1
srelay_config = {
    "pre_iter": ...,
    "num_sets": ...,
    "stopping_criterion": "All" | "FirstConv" | "NConv",
    "stop_nconv": ...,  # only for NConv
}
gamma_ensemble_size
bp_seed
```

OSD parameters:

```text
use_osd
osd_method
osd_order
osd_batch_size
```

Optional result parameters:

```text
opt_results = {
    "num_iter": True,
    "bp_llr_history": N,
    "num_residuals_per_iteration": True,
}
```

Only request optional traces when needed; they are diagnostic outputs, not
always part of a production benchmark.

## Concepts to explain safely

For user-facing explanations, use these public-safe definitions:

`gamma` is a memory-control parameter. It changes how strongly BP reuses
previous marginal beliefs relative to the original channel evidence. It is
closer to memory or momentum than to a learning rate.

`gamma0` is a uniform memory value. `gamma_dist` describes a range from which
gamma values may be generated. `explicit_gammas` supplies the gamma values
directly.

`num_sets` is the number of gamma strategies available to a relay-style
configuration.

`gamma_ensemble_size` is parallelism over gamma-conditioned BP attempts for
the same syndrome. It is not the same as `bp_batch_size`.

```text
bp_batch_size        = batch over syndromes / shots
gamma_ensemble_size = batch over gamma strategies for one syndrome
num_sets            = total gamma strategies available
```

`OSD` is an optional post-processing stage. It uses BP soft information to
order variables, then searches for a syndrome-consistent correction using
linear-algebra-style decoding over binary variables. Explain OSD as a fallback
or refinement stage, not as part of the normal BP iteration.

## Diagnostic decision tree

Start with correctness before performance:

```text
LER suspicious?
  -> verify DEM / observable / noise consistency first
  -> check num_converged
  -> inspect num_iter and residual behavior
  -> only then tune BP, gamma, relay, or OSD parameters
```

If `num_converged` is low:

```text
check max_iterations
check bp_method and required gamma parameters
check error_rate_vec length and values
check whether OSD should be enabled
inspect residuals per iteration if available
```

If `num_converged` is high but `num_logical_errors` is high:

```text
verify the observable matrix
verify the DEM was generated from the same circuit/noise model as the samples
verify syndrome slicing for CSS workflows
compare against a simpler baseline on a small number of shots
```

If `avg_num_iter` improves but latency gets worse:

```text
check gamma_ensemble_size and other heavier configurations
compare avg_time_ms and p99_time_ms, not only iterations
confirm timing excludes decoder construction and includes GPU synchronization
```

If batched throughput is worse than expected:

```text
separate bp_batch_size effects from gamma_ensemble_size effects
warm up before timing
compare single-shot latency, batched throughput, and p99 separately
```

## Stop gates

Stop and fix the earlier issue before continuing when:

```text
p=0 is not clean
  -> do not tune BP/gamma/OSD; first fix DEM, observable, syndrome shape,
     target, or workflow wiring.

decoder construction fails
  -> do not analyze LER; first fix decoder availability and required params.

converged is high but logical errors are high
  -> do not claim BP is "working"; first check logical observable consistency.

timing includes construction, DEM generation, or first-use warmup
  -> do not compare latency; remeasure decode-only steady-state timing.
```

## Symptom-to-next-step table

| Symptom | First safe next check |
|---------|-----------------------|
| `Decoder X not found` | Verify plugin/extras are installed and import smoke passes. |
| Construction raises on `bp_method=2/3` | Check required gamma params and `srelay_config`. |
| Low `num_converged` | Inspect `max_iterations`, residual trend, error rates, and OSD setting. |
| High `num_logical_errors`, high convergence | Verify DEM / observable matrix / noise consistency. |
| `avg_num_iter` improves but time worsens | Compare config heaviness, `gamma_ensemble_size`, and synchronized timing. |
| p99 much larger than average | Separate hard syndromes from host/GPU timing artifacts. |

## Minimum repro packet

Ask for sanitized information only:

```text
code/circuit identity:
noise model summary and p:
DEM source:
decoder kwargs:
number of shots:
num_converged / num_shots:
num_logical_errors / num_shots:
avg_num_iter:
avg_time_ms / p99_time_ms:
p=0 result:
baseline decoder result, if available:
```

Do not ask for private data dumps, proprietary logs, internal traces, or
unpublished benchmark artifacts.

## Handoff rules

```text
Issue is offline decoder correctness or LER
  -> stay in cudaq-qec-decode.

Issue is TensorRT / trained model accuracy
  -> use cudaq-qec-ai-decoders.

Issue is in-kernel / hardware / latency-bounded execution
  -> use cudaq-qec-realtime.

Issue is low-level GPU performance after correctness is established
  -> use cudaq-profiling-perf.
```

## Redaction rules

Safe to share:

```text
public API names
decoder kwargs
matrix shapes
aggregate metrics
public docs links
small synthetic examples
```

Do not share:

```text
private source snippets
kernel internals
raw proprietary logs
private benchmark datasets
credentials or hardware addresses
unpublished performance claims
```

## Reporting template

Use this structure in responses:

```text
Summary:
  The run is accuracy-limited / robustness-limited / speed-limited.

Evidence:
  num_logical_errors = ...
  num_converged = ...
  avg_num_iter = ...
  avg_time_ms / p99_time_ms = ...

Likely causes:
  ...

Next checks:
  ...
```

Avoid claiming that a gamma or OSD configuration is universally optimal.
Decoder tuning is code-, noise-, hardware-, and workload-dependent.
