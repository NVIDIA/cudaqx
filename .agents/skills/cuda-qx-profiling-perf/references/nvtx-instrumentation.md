# Adding NVTX ranges to your own code

NVTX (NVIDIA Tools Extension) marks regions on the `nsys` timeline.
Use it to label your decoder's hot path so the profiler tells you
"this range took 5 ms" instead of an opaque kernel name.

## Header

```cpp
#include "cudaq/qec/realtime/nvtx_helpers.h"
```

This wraps NVTX with sensible defaults. The realtime QEC stack uses
this header throughout; mimic that style for your own code.

For code outside cudaqx, use the upstream header directly:

```cpp
#include <nvtx3/nvToolsExt.h>
```

## Push / pop ranges

```cpp
nvtxRangePush("decode_kernel");
launch_my_decode_kernel(...);
nvtxRangePop();
```

Or, scope-bound (RAII):

```cpp
{
  nvtxScopedRange r("decode_kernel");
  launch_my_decode_kernel(...);
}
```

The cudaqx helpers expose a similar RAII type; check
`nvtx_helpers.h`.

## Range colors and categories

For complex pipelines (predecoder + MWPM + dispatch), color-code:

```cpp
nvtxEventAttributes_t attr = {};
attr.version = NVTX_VERSION;
attr.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
attr.colorType = NVTX_COLOR_ARGB;
attr.color = 0xFF00CC00;   // green for predecoder
attr.messageType = NVTX_MESSAGE_TYPE_ASCII;
attr.message.ascii = "predecoder";
nvtxRangePushEx(&attr);
// ...
nvtxRangePop();
```

The realtime pipeline already uses category-coded ranges. Reuse the
same color scheme so timelines from different runs are comparable.

## Python NVTX

For Python code (training, benchmarking driver scripts):

```python
import torch.cuda.nvtx as nvtx

nvtx.range_push("dataloader")
batch = next(iter(loader))
nvtx.range_pop()

nvtx.range_push("forward")
out = model(batch)
nvtx.range_pop()
```

PyTorch's `torch.cuda.nvtx` works seamlessly with `nsys`.

## Where to put ranges

Good targets:

- Each major pipeline stage (predecoder, dispatch, MWPM).
- Each iteration of an outer loop (training step, shot loop).
- Each kernel that you suspect is slow (with the kernel name).

Bad targets:

- Setup / initialization (clutters the timeline).
- Every line of code (NVTX has overhead at very fine granularity).
- Inside very tight inner loops (overhead dominates).

## Verifying the ranges show up

Re-run with `nsys` (`nsys-realtime.md`). The NVTX row should show
your ranges, named exactly as you pushed them. If they're missing:

1. Confirm you linked against NVTX (CMake target).
2. Confirm the binary was built with NVTX support (some Release
   builds strip it).
3. Confirm you ran `nsys --trace=nvtx,...`.

## Common pitfalls

| Symptom | Cause |
|---------|-------|
| Range "leaks" into the next iteration | `nvtxRangePush` without matching `Pop`; use RAII helpers |
| Range names show up as garbage | string lifetime issue; pass a literal or store the string |
| NVTX overhead visible in profile | too fine-grained; coarsen |
| Color-code looks wrong | ARGB encoding; alpha is the high byte |

## Self-check

```
[ ] Major pipeline stages have ranges with descriptive names.
[ ] Push and Pop balanced everywhere (RAII preferred).
[ ] Ranges visible on the nsys timeline.
[ ] Range overhead < 1% of profiled work.
```

## Where next

- Profile the instrumented code: `nsys-realtime.md`.
- Per-kernel deep-dive on a suspicious range: `ncu-kernels.md`.
