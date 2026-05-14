# Docker / container path

The published container is the most reliable way to try cudaqx on a
Mac, a locked-down workstation, or a fresh Linux box. Everything
(CUDA-Q, CUDA-QX, examples, notebooks) is preinstalled.

## Pull and run

```bash
docker pull ghcr.io/nvidia/cudaqx
docker run --gpus all -it ghcr.io/nvidia/cudaqx
```

On Mac (no NVIDIA GPU), omit `--gpus all`:

```bash
docker run -it ghcr.io/nvidia/cudaqx
```

CPU-only mode disables `nv-qldpc-decoder` and `trt_decoder`. Everything
else still works for small examples.

## What's inside

| Path | What |
|------|------|
| `/workspaces/cudaqx/` | full source checkout |
| `/workspaces/cudaqx/docs/sphinx/examples/` | curated examples |
| `/usr/local/cudaq/` | CUDA-Q install root |
| `/home/cudaq/` | user home |
| `python3` | preconfigured with `cudaq_qec`, `cudaq_solvers` importable |

The dev container is `ghcr.io/nvidia/cudaqx-dev:latest-<arch>-<cuda>`
(see `cuda-qx-build`); the user-facing release container is
`ghcr.io/nvidia/cudaqx`.

## Run the first example

```bash
docker run --gpus all -it ghcr.io/nvidia/cudaqx
# inside the container
cd /workspaces/cudaqx
python3 docs/sphinx/examples/qec/python/code_capacity_noise.py
```

Then delegate to `cuda-qx-qec-decode` SKILL.md.

## Jupyter notebook in the container

The container does not auto-launch Jupyter. To use notebooks:

```bash
docker run --gpus all -it -p 8888:8888 ghcr.io/nvidia/cudaqx
# inside container
pip install jupyterlab
jupyter lab --ip=0.0.0.0 --no-browser --allow-root
```

Open the printed URL in your host browser.

## VSCode dev container (source contributors)

Source contributors should use the `cudaqx-dev` image via VSCode's
"Reopen in Container" feature. See `cuda-qx-build` SKILL.md for the
full devcontainer setup.

## Mac caveats

- No `--gpus all`. GPU-only decoders fail at runtime, not import.
- Docker Desktop must be running before `docker pull`.
- Volume mounts are slow; for serious work clone the repo *inside* the
  container or use a named volume.
- ARM64 (Apple Silicon) images exist for the dev container; the user
  release image `ghcr.io/nvidia/cudaqx` is x86_64-only at time of
  writing — Docker Desktop will run it under emulation (slowly).

## Common container pitfalls

| Symptom | Cause | Fix |
|---------|-------|-----|
| `docker: Error response from daemon: ... could not select device driver "nvidia"` | NVIDIA Container Toolkit not installed on host | install `nvidia-container-toolkit` |
| GPU decoders error inside the container | host driver too old for the container's CUDA | upgrade host NVIDIA driver, or use the cu12 container instead of cu13 |
| Files in the mounted volume have wrong owner | container user vs host UID mismatch | `docker run -u $(id -u):$(id -g)` |
| Out-of-memory during GQE training | container default memory cap too low | `docker run --memory=32g --shm-size=8g` |

When in doubt, drop into the container and check
`scripts/doctor.sh` for a structured environment dump.
