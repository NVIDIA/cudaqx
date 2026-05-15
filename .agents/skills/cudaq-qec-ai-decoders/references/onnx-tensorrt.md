# ONNX export and TensorRT engine build

The two steps between "trained PyTorch model" and "decoder cudaq can
run". Most failures are in the export step; spend time there.

## ONNX export

```python
import torch

model.eval()
dummy = torch.zeros(1, num_detectors, dtype=torch.float32, device="cuda")
torch.onnx.export(
    model, dummy, "my_decoder.onnx",
    input_names=["detectors"],
    output_names=["logits"],
    dynamic_axes={"detectors": {0: "batch"},
                  "logits":    {0: "batch"}},
    opset_version=17,           # 17+ recommended
    do_constant_folding=True,
)
```

Static everywhere except the batch dimension. Variable detector
count → multiple engines, not one engine with extra dynamic axes.

**Sanity check after export**:

```python
import onnx
m = onnx.load("my_decoder.onnx")
onnx.checker.check_model(m)
print(onnx.helper.printable_graph(m.graph))
```

The printed graph should match your PyTorch architecture. Common
catch: a `LayerNorm` or `BatchNorm1d` exports with a different shape
than you expect — re-export with the layer in `eval` mode.

## TensorRT engine build

The bundled script:

```bash
python scripts/build_engine_from_onnx.py \
    --onnx my_decoder.onnx \
    --engine my_decoder.engine \
    --fp16 \
    --max-batch-size 1024
```

What the script does
(see [scripts/build_engine_from_onnx.py](scripts/build_engine_from_onnx.py)):

1. Loads the ONNX with `tensorrt.OnnxParser`.
2. Sets workspace size to 1 GB (`1 << 30`).
3. Sets FP16 (or INT8) flag when supported by the platform.
4. Creates an optimization profile with min/opt/max batch sizes.
5. Builds and serializes the engine to disk.

For decoders, FP16 is essentially free in accuracy and ~2× faster.
INT8 needs a calibration dataset (`tensorrt.IInt8EntropyCalibrator2`)
and rarely pays off for QEC inference.

## Validating the engine

Compare engine outputs to PyTorch outputs on a held-out batch:

```python
import tensorrt as trt
import numpy as np
import torch

# Load engine
with open("my_decoder.engine", "rb") as f, trt.Runtime(trt.Logger()) as rt:
    engine = rt.deserialize_cuda_engine(f.read())

# Run engine on a sample batch and compare to PyTorch
# (Allocate input/output buffers, execute_async_v3, copy results)
# ... TensorRT inference boilerplate ...
# Tolerance: ~1e-2 for FP16, ~1e-5 for FP32. Anything worse means the
# export changed the graph (operator fusion, precision mode mismatch).
```

Any mean-abs-diff above the FP16 tolerance means the engine and the
PyTorch model are computing different things. Common culprits:

- An operator with no FP16 kernel that silently fell back to FP32 in
  some places but not others.
- A `LayerNorm` that ONNX expanded into multiple ops, and TensorRT
  fused some of them.
- A custom op that round-tripped through the ONNX `Reshape` op with
  the wrong shape.

## Caching and portability

Engine files are **GPU + driver specific**. Plan to cache by:

```
<model-hash>_<gpu-arch>_<driver-version>.engine
```

A `.engine` that loads on RTX 6000 will not load on H100 with a
different driver. The hybrid AI predecoder pipeline does this caching
in disk; mimic that pattern in your own deployment.

## Common engine-build pitfalls

| Symptom | Likely cause |
|---------|--------------|
| `ONNX parse error: unsupported op` | opset too old; re-export with opset 17+ |
| Build succeeds but engine outputs all zeros | FP16 underflow; try `--fp32` to isolate |
| Build very slow | tactic source disabled; default is fine, but custom builds can disable cuDNN |
| Engine refuses to load on a different machine | GPU arch or driver mismatch; rebuild on the target |
| Engine 10x slower than expected | dynamic shapes used everywhere; restrict dynamic axes to batch only |

## Self-check

```
[ ] onnx.checker.check_model passes
[ ] Engine outputs match PyTorch outputs within FP16 tolerance on a held-out batch
[ ] Engine file is loadable on the deployment machine
[ ] Engine size is reasonable (typically MB, not GB; if GB, dynamic shapes wide open)
```

## Where next

- Wire the engine into `trt_decoder`: `trt-decoder-integration.md`.
- Drop into the predecoder + MWPM pipeline: `hybrid-deployment.md`.
- Benchmark accuracy + latency: `cudaq-benchmarking` SKILL.md.
