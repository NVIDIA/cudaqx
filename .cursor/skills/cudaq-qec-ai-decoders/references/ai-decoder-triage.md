# Public-safe AI decoder triage

Use this file when an AI decoder workflow has suspicious accuracy,
export, engine-build, integration, or latency behavior. Keep user-facing
answers at the public workflow level: datasets, shapes, metrics, API
contracts, environment compatibility, and reproducible checks. Do not
include private model weights, private datasets, proprietary kernel
details, unpublished implementation details, or internal benchmark claims.

## First classify the failure

Start by locating the stage that first goes wrong:

```text
data generation -> training -> validation/test -> ONNX export
    -> TensorRT build -> cudaq-qec integration -> deployment benchmark
```

Do not debug TensorRT if the PyTorch test-set LER is already bad. Do not
debug deployment latency if the engine outputs do not match PyTorch.

## Run dashboard

Collect these facts before giving advice:

```text
code family, distance, and rounds
physical error rate used for train / val / test
number of train / val / test samples
input shape and output shape
label definition
baseline decoder and baseline LER
train loss, val loss, val LER, test LER
ONNX opset and static/dynamic axes
TensorRT precision mode and target GPU
engine-vs-PyTorch output tolerance
offline trt_decoder LER, if integrated
latency / throughput measurement method, if benchmarked
```

Interpret results by stage:

```text
training quality     = test LER vs baseline on the same data distribution
export quality       = ONNX / engine outputs match PyTorch
integration quality  = trt_decoder LER matches PyTorch test LER
deployment quality   = latency and throughput meet the target budget
```

## Accuracy triage

If train and validation LER are both poor:

```text
check class imbalance handling
check labels match the intended observable or correction target
check input detector ordering and shape
check physical error rate is not outside the training target
check model capacity and dataset size
```

If train LER is good but validation/test LER is poor:

```text
check train/val/test leakage assumptions
check independent sampling for each split
check overfitting: model too large, data too small, weak regularization
check that test p equals the target deployment p
```

If validation LER is good but deployment LER is poor:

```text
compare PyTorch outputs, ONNX outputs, and TensorRT outputs on the same batch
check input dtype, normalization, transposition, and batch axis
check thresholding or logit-to-bit conversion
check that the deployed engine was built from the expected ONNX file
```

Use LER, not only classification accuracy. A model can report high bitwise
accuracy while still failing the logical observable frequently.

## Export and engine triage

For ONNX export:

```text
model is in eval mode
dummy input shape equals inference shape
dynamic axes are limited to batch unless explicitly required
onnx.checker.check_model passes
printed ONNX graph matches the expected high-level architecture
```

For TensorRT:

```text
engine built on compatible GPU / driver / TensorRT versions
precision mode validated against PyTorch outputs
optimization profile covers the benchmark batch size
engine file is not stale or rebuilt from the wrong ONNX
```

If FP16 accuracy diverges, isolate with FP32 first. Treat INT8 as a
separate calibrated workflow, not a default optimization.

## Integration triage

Before blaming the model, verify the decoder contract:

```text
trt_decoder plugin is installed and selected
engine_path points to the intended engine
H / DEM matches the data used to train the model
input layout matches the engine contract
output interpretation matches the label target
offline trt_decoder LER matches PyTorch test-set LER within sampling noise
```

If integrating with a real-time pipeline, switch to the realtime skill and
read its triage file. Training/export issues belong here; in-kernel
deployment and latency-budget issues belong to `cudaq-qec-realtime`.

## Latency triage

Separate these cases:

```text
training is slow
engine inference is slow
cudaq-qec integration is slow
real-time deployment is slow
```

For training slowness:

```text
profile dataloading vs GPU compute
avoid materializing very large datasets as one monolithic tensor
increase batch size only after confirming memory headroom
```

For inference slowness:

```text
warm up before timing
time only inference/decode calls, not construction or engine build
use the deployment batch size when benchmarking
confirm GPU work is synchronized before reading wall-clock numbers
```

For real-time slowness, delegate to `cudaq-qec-realtime` and
`cudaq-profiling-perf`.

## Stop gates

Stop and fix the earlier stage before continuing when:

```text
PyTorch test LER is poor
  -> do not debug ONNX or TensorRT yet; fix data, labels, imbalance,
     model capacity, or train/test mismatch.

ONNX output does not match PyTorch on the same batch
  -> do not build or benchmark a TensorRT engine yet.

TensorRT output does not match PyTorch / ONNX within expected tolerance
  -> do not integrate into cudaq-qec yet; isolate precision or export issues.

offline trt_decoder LER does not match PyTorch test LER
  -> do not deploy real-time yet; fix input/output layout or engine wiring.

latency timing includes training, export, engine build, or decoder construction
  -> do not compare performance; remeasure inference/decode-only timing.
```

## Symptom-to-next-step table

| Symptom | First safe next check |
|---------|-----------------------|
| Model predicts mostly zeros | Check class imbalance and label prevalence. |
| Train improves, validation does not | Check data leakage assumptions, overfitting, and p mismatch. |
| Test LER much worse than validation LER | Confirm independent test set and same target distribution. |
| ONNX export succeeds but output differs | Check eval mode, shapes, dynamic axes, and unsupported ops. |
| TensorRT engine loads but accuracy collapses | Compare same-batch engine outputs against PyTorch. |
| Engine works offline but realtime is slow | Handoff to realtime/profiling after confirming correctness. |

## Minimum repro packet

Ask for sanitized information only:

```text
code/distance/rounds:
train/val/test physical error rates:
train/val/test sample counts:
input shape:
output shape and label meaning:
baseline decoder and baseline LER:
train loss / val loss / val LER / test LER:
ONNX opset and dynamic axes:
TensorRT precision and target GPU:
PyTorch-vs-engine tolerance on same batch:
offline trt_decoder LER:
latency measurement scope:
```

Do not ask for private datasets, model weights, engine binaries, internal
logs, proprietary traces, or unpublished benchmark artifacts.

## Handoff rules

```text
Issue is data generation, training, ONNX, TensorRT build, or offline trt_decoder
  -> stay in cudaq-qec-ai-decoders.

Issue is in-kernel deployment, hardware, ring-buffer behavior, or p99 latency
  -> use cudaq-qec-realtime.

Issue is offline non-AI decoder comparison or LER sanity checking
  -> use cudaq-qec-decode or cudaq-benchmarking.

Issue is low-level GPU timeline or kernel performance after correctness is established
  -> use cudaq-profiling-perf.
```

## Redaction rules

Safe to share:

```text
model family name
input/output shapes
aggregate LER/loss/latency
public API names and command shapes
sanitized config fields
```

Do not share:

```text
private training data
model weights
engine binaries or contents
private benchmark traces
private source snippets
credentials, paths with secrets, or hardware addresses
unpublished performance claims
```

## Public-safe response template

Use this shape when responding:

```text
Stage:
  data / training / export / engine / integration / deployment

Evidence:
  key metrics and shapes

Most likely causes:
  public workflow-level causes only

Next checks:
  small reproducible checks, same batch where possible
```

Avoid sharing private dataset examples, model weights, engine contents,
internal implementation details, or unpublished performance numbers.
