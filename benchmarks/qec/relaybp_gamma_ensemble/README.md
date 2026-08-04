# Relay-BP gamma-ensemble benchmark

Reproduces the data and figures in
[Improving Relay BP Decoding With Gamma Ensembles](../../../docs/sphinx/examples_rst/qec/nv_qldpc_gamma_ensemble_user_guide.rst).

The circuits in `assets/benchmarks/` come from
[Relay-BP](https://github.com/trmue/relay) (Apache-2.0) and have been modified:
the detectors insensitive to Z-stabilizer errors are stripped. They are stored
via [Git LFS](https://git-lfs.com); run `git lfs pull` if the files appear as
pointer stubs rather than valid stim circuits.

## Requirements

- A GPU.
- `cudaq-qec`. The `nv-qldpc-decoder` plugin is proprietary and ships in the
  released wheel; a build of CUDA-QX from public source does not include it.
- `stim`, `numpy`, `matplotlib`.

## Running

```bash
export QEC_DATA_ROOT=<output directory>

CUDA_VISIBLE_DEVICES=<idle gpu> python3 -u run_sweep.py
python3 plot_sweep.py
```

Pin the run to an idle GPU: another process on the same device distorts latency data.

## Environment variables

| Variable | Default | Effect |
| --- | --- | --- |
| `QEC_DATA_ROOT` | `report_data` | Where `report_data.npz` and `figures/` are written. Both scripts must agree |
| `SHOTS` | `150000` | Shots per configuration
| `CIRCUIT_DIR` | `../../../assets/benchmarks` | Directory holding the Z-only stim circuits |

## Output

`run_sweep.py` writes `$QEC_DATA_ROOT/report_data.npz` (~23 MB at 150,000
shots), holding the per-shot latency, iteration count and
logical-error flag for every code and ensemble size.

`plot_sweep.py` reads that file and writes to `$QEC_DATA_ROOT/figures/`:

- `relaybp_gamma_ensemble_perf.png`
- `relaybp_latency_percentiles.png`
- `relaybp_hard_deadline_ler.png`
- `relaybp_ler_multiplier.png`

It also prints the numeric values reported in `nv_qldpc_gamma_ensemble_user_guide.rst`.