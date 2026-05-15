# Public-safe real-time QEC triage

Use this file when a real-time QEC workflow hangs, returns suspicious
corrections, misses a latency budget, or behaves differently from offline
decoding. Keep user-facing answers at the public API and operational
workflow level: target selection, decoder config, in-kernel call order,
observable metrics, logs, and profiler symptoms. Do not include private
kernel internals, transport details, proprietary implementation notes,
hardware-specific secrets, or unpublished benchmark claims.

## First classify the failure

Place the symptom in one bucket:

```text
configuration does not load
in-kernel calls return wrong / zero corrections
simulation LER differs from offline LER
hardware / emulation behaves differently from simulation
latency or p99 exceeds the budget
resources leak or later runs fail
```

Debug in this order:

```text
correct target -> correct config -> correct in-kernel call order
    -> correctness at p=0 -> correctness at small p -> latency
```

## Run dashboard

Collect:

```text
target name and mode
code, operation, rounds, and shots
noise model for simulation, if any
decoder type and config id values
block_size, syndrome_size, and observable count
whether configure_decoders_from_file ran before cudaq.run
whether finalize_decoders ran after the run
whether in-kernel APIs are inside the kernel
LER or correction summary at p=0
LER or correction summary at small nonzero p
avg latency, p99 latency, and throughput if measured
```

Never include credentials, tokens, private hardware addresses, private
config payloads, or raw proprietary logs in user-facing summaries.

## Correctness triage

If corrections are all zero or clearly wrong:

```text
confirm decoder config was loaded before cudaq.run
confirm config id matches reset_decoder / enqueue_syndromes / get_corrections
confirm block_size and syndrome_size are not swapped
confirm the DEM / H used in config matches the circuit being run
confirm target is stim for local noisy simulation
```

If LER differs from offline decoding:

```text
compare against offline decoding on the same DEM and sampled syndromes
check syndrome round count and any extra round conventions in the template
check observable matrix and number of requested observables
check that the same noise model generated the DEM and samples
```

If p=0 is not clean:

```text
stop performance debugging
verify config dimensions and ids
verify in-kernel call placement
verify target and setup order
```

## Configuration triage

Use this public checklist:

```text
target selected before DEM generation / run
decoder config id is unique per logical stream
H / O / D matrices match the chosen workflow template
decoder custom args match the selected decoder type
configuration file is loaded once before the run
finalize_decoders is called after the run
```

For decoder-specific algorithm tuning, use the offline decode skill first.
Real-time triage should not duplicate every decoder parameter table.

## In-kernel API triage

The in-kernel flow should be structurally simple:

```text
reset decoder
enqueue syndrome data as rounds are measured
request corrections
apply or record corrections according to the workflow
```

Common public-level mistakes:

```text
calling reset/enqueue/get_corrections from Python top level
using the wrong decoder id
requesting the wrong number of observables
using blocking mode inconsistently with the circuit logic
forgetting to finalize after the run
```

## Hardware and emulation triage

Keep this section free of private deployment details. Ask for sanitized
facts only:

```text
target mode: local simulation, emulation, or hardware
machine/backend name if public
whether the decoder payload/provider was enabled
whether debug logging confirms decoder initialization
whether simulation reproduces the issue
```

If the issue does not reproduce in local simulation, isolate hardware
configuration, credentials, payload setup, and provider selection without
printing secrets.

## Latency triage

Separate correctness from speed. Only profile once p=0 and a small-p run
look correct.

For latency:

```text
warm up before measuring
exclude one-time configuration from per-shot latency
record average, median if available, and p99
compare simulation, emulation, and hardware separately
use nsys / NVTX guidance from cudaq-profiling-perf for timeline analysis
```

Public symptoms and likely workflow-level causes:

```text
large one-time delay
  -> configuration or initialization counted in steady-state timing

high p99 but acceptable average
  -> intermittent synchronization, queueing, or host-side stalls

GPU appears idle
  -> host bottleneck or missing overlap

latency improves offline but not real-time
  -> config/load path, batching, or call-order issue rather than decoder math
```

Do not include private timeline screenshots, raw hardware logs, or
implementation-specific dispatch details in user-facing output.

## Stop gates

Stop and fix the earlier issue before continuing when:

```text
p=0 is not clean
  -> do not profile latency; fix target, config, dimensions, ids, or call order.

configure_decoders_from_file did not run before cudaq.run
  -> do not analyze corrections; fix setup order.

in-kernel APIs are called from Python top level
  -> do not debug decoder output; move calls into the kernel.

offline and realtime disagree on the same DEM and same syndromes
  -> do not tune hardware latency; reconcile config, rounds, observables,
     and decoder ids first.

timing includes one-time configuration or warmup
  -> do not compare latency; remeasure steady-state hot path.
```

## Symptom-to-next-step table

| Symptom | First safe next check |
|---------|-----------------------|
| Decoder returns all-zero corrections | Confirm config loaded before run and ids match. |
| Run hangs on first shot | Check decoder availability, target, and setup order. |
| Wrong logical qubit corrected | Check unique config ids and in-kernel id usage. |
| Runtime size mismatch | Check `block_size`, `syndrome_size`, and round count. |
| Simulation works, hardware/emulation fails | Sanitize and compare target/provider setup and initialization logs. |
| Average latency ok, p99 bad | Capture steady-state timeline and look for stalls or queueing. |

## Minimum repro packet

Ask for sanitized information only:

```text
target and mode:
code / operation / rounds / shots:
decoder type:
decoder config id values:
block_size / syndrome_size / num_observables:
whether configure_decoders_from_file ran before cudaq.run:
whether finalize_decoders ran after:
where reset/enqueue/get_corrections are called:
p=0 result:
small-p result:
offline same-DEM comparison:
avg latency / p99 latency / throughput:
sanitized error message:
```

Do not request credentials, private hardware addresses, raw proprietary
logs, full private configs, transport traces, or internal implementation
details.

## Handoff rules

```text
Issue is offline decoder math or LER on a DEM
  -> use cudaq-qec-decode.

Issue is trained model accuracy, ONNX, TensorRT engine build, or offline trt_decoder
  -> use cudaq-qec-ai-decoders.

Issue is in-kernel call order, config load, hardware/emulation, or p99 latency
  -> stay in cudaq-qec-realtime.

Issue is timeline / GPU utilization after correctness is established
  -> use cudaq-profiling-perf.

Issue is adding a new decoder or code
  -> use cudaq-qec-extending.
```

## Redaction rules

Safe to share:

```text
public target names
public decoder type names
matrix shapes
sanitized config field names
aggregate LER and latency metrics
public docs links
```

Do not share:

```text
credentials or tokens
private machine addresses
raw hardware logs
private payloads or config dumps
transport-level details
private source snippets
unpublished benchmark claims
```

## Public-safe response template

Use:

```text
Stage:
  configuration / in-kernel API / correctness / hardware / latency

Evidence:
  sanitized metrics and observed behavior

Most likely causes:
  public workflow-level causes

Next checks:
  p=0 check, same-DEM offline comparison, config id/dimension check,
  warm latency capture, or handoff to profiling skill
```

If the user asks for source-level or proprietary internals, redirect to
public docs, API contracts, and sanitized reproduction steps.
