# PyTorch profiler for ML decoder training

For training pipelines (`cudaq-qec-ai-decoders`) the bottleneck is
usually dataloading, optimizer, or a small slow layer. PyTorch's
built-in profiler is the right entry point; `nsys` is overkill for
Python-side work.

## Capture

```python
from torch.profiler import profile, schedule, ProfilerActivity, tensorboard_trace_handler

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=schedule(wait=1, warmup=2, active=5, repeat=1),
    on_trace_ready=tensorboard_trace_handler("./tb_logs"),
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    for step, batch in enumerate(train_loader):
        train_step(batch)
        prof.step()
```

`wait=1, warmup=2, active=5` skips the first iteration (init), warms
up the next two, and profiles the next five. Adjust to match your
batch cadence.

## View the trace

TensorBoard:

```bash
pip install torch-tb-profiler
tensorboard --logdir ./tb_logs
```

Or open the JSON in `chrome://tracing`.

## What to look at first

Three things, in order:

1. **Top operators by self-time.** If `aten::linear` or
   `aten::conv2d` dominates, the model is compute-bound — small wins
   matter (FP16 / TF32 / channels_last).
2. **DataLoader stalls.** Bars in the profile that show CPU activity
   while GPU is idle. If you see this, raise `num_workers`, use
   `pin_memory=True`, switch to a faster image / numpy decoder.
3. **CPU↔GPU transfers.** Each `.to(device)` in your inner loop is a
   sync (unless `non_blocking=True`). The profile shows them as
   "Memcpy HtoD" / "DtoH".

## Common training pathologies

| Pattern | Diagnosis |
|---------|-----------|
| GPU utilization 30%, CPU at 100% on 1 core | dataloader single-threaded; raise `num_workers` |
| Long backward pass | optimizer-step or zero_grad is fragmenting allocator |
| Repeated `aten::copy_` calls | implicit device transfers inside the training loop |
| `aten::layer_norm` showing up large | small batch size; consider `LayerNorm` fusion or larger batch |

## For ML decoder training specifically

- The MLP is tiny; you are almost certainly dataloader-bound.
- Stim sampling is single-threaded; pre-generate the dataset to disk
  and stream from a `Dataset`.
- For a generated-on-the-fly setup, run Stim in a separate process
  via `torch.utils.data.DataLoader(num_workers=N)`.

## When to switch to nsys

- When you suspect a CUDA kernel issue (not a Python issue) — `nsys`
  + `ncu` give better answers.
- When the model exports to ONNX/TensorRT and you need to profile
  the engine, not PyTorch — `nsys` only.
- For multi-process training (DDP), `nsys` per rank is the standard
  approach.

## Self-check

```
[ ] Profile captured for >= 5 active steps after warmup.
[ ] DataLoader stalls inspected (and resolved if present).
[ ] CPU↔GPU transfers inside the loop minimized.
[ ] Top-3 operators identified with self-time numbers.
[ ] After optimization, re-profile and confirm the improvement.
```

## Where next

- TensorRT engine bottlenecks: `cudaq-qec-ai-decoders/references/onnx-tensorrt.md`.
- Whole-pipeline GPU work: `nsys-realtime.md`.
- Single-kernel deep-dive: `ncu-kernels.md`.
