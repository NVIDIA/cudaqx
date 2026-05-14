# Training a neural decoder

The PyTorch side of the pipeline: build a dataset from Stim, define a
model, train, validate, hold out a test set. The example is an MLP
for a distance-3 surface code; extend it for bigger codes by scaling
network width and dataset size.

## Authoritative template

`docs/sphinx/examples/qec/python/train_mlp_decoder.py`. Walk it before
writing anything from scratch.

## Dataset generation (Stim)

```python
import stim

circuit = stim.Circuit.generated(
    "surface_code:rotated_memory_x",
    distance=d,
    rounds=T,
    after_clifford_depolarization=p,
    after_reset_flip_probability=p,
    before_measure_flip_probability=p,
    before_round_data_depolarization=p,
)

sampler = circuit.compile_detector_sampler()
detectors, observables = sampler.sample(num_samples, separate_observables=True)
```

`detectors` shape `(N, num_detectors)` is the input; `observables`
shape `(N, num_observables)` is the label. For a single-logical
memory experiment, `num_observables == 1`.

Use **independent** sampler calls for train / val / test. Sharing the
sampler across splits will yield correlated samples and inflated
validation accuracy.

## Model design

For surface codes up to d≈13, an MLP with 2-3 hidden layers and
`hidden_dim = 128…512` is a strong baseline. Architecture choices
that *matter* in practice:

- **Inputs as uint8 → float32**: convert with
  `torch.from_numpy(detectors).float()`. uint8 directly into the
  first linear is silently scaled by PyTorch's `Embedding`-like
  conventions in some versions.
- **`pos_weight` in `BCEWithLogitsLoss`**: typical syndrome-class
  imbalance is 50:1 to 1000:1. Compute as `neg_count / pos_count`
  over the training set.
- **Dropout 0.1-0.3**: helps with overfitting on small QEC datasets.

For d>13, consider a transformer or a graph neural network that
respects the surface-code lattice geometry. Both are research-grade;
no off-the-shelf template in this repo.

## Training loop checklist

```
[ ] Train / val / test split with independent samplers (no leakage)
[ ] BCEWithLogitsLoss with computed pos_weight
[ ] Track val loss and val LER (thresholded predictions vs labels)
[ ] Early stop on val LER, not val loss
[ ] Save best checkpoint by val LER
[ ] After training, evaluate on test set ONCE, never again
```

## Common pitfalls

| Symptom | Likely cause |
|---------|--------------|
| Val LER and train LER both close to baseline (i.e. no learning) | class imbalance not handled |
| Test LER >> val LER | leakage between val and test (shared sampler) |
| Loss diverges | learning rate too high for the imbalanced loss |
| Val LER plateaus immediately, train LER too | dataset too small for the model |
| Memory blowup | dataset materialized as a single tensor; use a `DataLoader` over chunks |

## Reproducibility

Set seeds at the top:

```python
import random, numpy as np, torch
random.seed(0); np.random.seed(0); torch.manual_seed(0)
torch.use_deterministic_algorithms(True, warn_only=True)
```

PyTorch is not bit-reproducible across CUDA versions or GPU
generations; reproducibility means *on the same hardware*. Pin the
Stim seed too:

```python
import stim
# Note: stim.Circuit.compile_detector_sampler() takes a seed argument
sampler = circuit.compile_detector_sampler(seed=42)
```

## When to scale up

- Hit chemical-accuracy-equivalent LER (better than `multi_error_lut`
  with `lut_error_depth=2`)? Scale to d=5, then d=7.
- Hitting overfitting? More data first, larger model second.
- Hitting compute wall? Profile (`cuda-qx-profiling-perf`); usually
  dataloading.

## What to delegate

Once you have a trained `.pt` and a val LER that beats the
`single_error_lut` baseline at the same `p`:

1. Export to ONNX → `references/onnx-tensorrt.md`.
2. Validate the engine against PyTorch outputs (numeric tolerance).
3. Integrate into `trt_decoder` → `references/trt-decoder-integration.md`.
