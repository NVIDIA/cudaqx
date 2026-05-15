# Pip install path

`pip install cudaq-qec` and `pip install cudaq-solvers` are the
fastest way to try CUDA-Q Libraries. Both ship Python wheels for Linux
x86_64/aarch64 and require Python 3.11+.

## Choose the right packages

```bash
# Either library, or both
pip install cudaq-qec
pip install cudaq-solvers
pip install cudaq-qec cudaq-solvers

# Optional extras
pip install cudaq-qec[tensor-network-decoder]   # exact ML baseline decoder
pip install cudaq-qec[trt-decoder]              # TensorRT neural decoder
pip install cudaq-solvers[gqe]                  # transformer-driven solver
```

The metapackages `cudaq-qec` / `cudaq-solvers` resolve to a
`-cu12` or `-cu13` wheel based on which `cuda-quantum-cu1x` package
is already installed.

**Sanity check after install:**

```bash
python -c "import cudaq_qec as qec; print(qec.__version__)"
python -c "import cudaq_solvers as solvers; print(solvers.__version__)"
```

If either import fails with `ImportError: libgfortran.so.5: cannot
open shared object`, install `libgfortran`:

```bash
sudo apt-get install gfortran          # Debian/Ubuntu
sudo dnf install libgfortran           # Fedora/RHEL
```

`libgfortran` is needed because the built-in `cobyla` and `lbfgs`
optimizers link against it. The wheel does not bundle it.

## CUDA 12 vs CUDA 13

CUDA-Q Libraries ships separate wheels for CUDA 12 (`-cu12`) and CUDA 13
(`-cu13`). They are **not interchangeable**. To check which you
have:

```bash
pip list | grep -E 'cuda-quantum|cudaq-(qec|solvers)'
```

A mix like `cuda-quantum-cu12` with `cudaq-qec-cu13` is a silent ABI
mismatch — uninstall both and reinstall the matching pair.

For NVIDIA Blackwell GPUs (B100/B200), use CUDA 12.8+ or CUDA 13.
PyTorch (needed for `tensor-network-decoder`, `gqe`, AI decoders)
should match: <https://pytorch.org/get-started/locally/>.

## CPU-only mode

You can run cudaq on a CPU-only system. What works and what doesn't:

| Workflow | CPU only | GPU required |
|----------|----------|--------------|
| Code-capacity QEC (`code_capacity_noise.py`) | yes | no |
| Circuit-level QEC with stim simulator | yes | no |
| `single_error_lut`, `multi_error_lut` decoders | yes | no |
| `nv-qldpc-decoder` | no | **yes** |
| `tensor_network_decoder` (small codes) | yes | preferred |
| `trt_decoder` (TensorRT) | no | **yes** |
| H2 VQE, small molecules | yes | no |
| GQE (transformer training) | no | **yes** (multi-GPU best) |

Set `CUDAQ_DEFAULT_SIMULATOR=stim` before `import cudaq_qec` to use
the CPU stim simulator. Default `nvidia` (statevector) does not scale
to QEC sizes.

## Run your first example

The wheels do not ship the `docs/sphinx/examples/` directory, so you
need to either clone the repo (sparsely) or grab the example from the
published docs. Easiest sparse clone:

```bash
git clone --depth 1 https://github.com/NVIDIA/cudaq.git
cd cudaq
python docs/sphinx/examples/qec/python/code_capacity_noise.py
```

That's the smallest QEC example. It uses `single_error_lut` on the
Steane code and prints a logical error rate. Expected output: a
small but nonzero LER at `p=0.05`, and exactly zero at `p=0`.

If the example runs, delegate to `cudaq-qec-decode` for deeper QEC work
or `cudaq-solvers-algorithms` for variational algorithms.

## Common pip-install pitfalls

| Symptom | Cause | Fix |
|---------|-------|-----|
| `ImportError: libgfortran.so.5` | system `libgfortran` missing | install via package manager |
| `RuntimeError: PySCF server hangs` | port 8000 occupied from a crashed run | `lsof -n -i :8000 && kill -9 <pid>` |
| `ABI mismatch / undefined symbol` | mixed `-cu12` and `-cu13` packages | uninstall all `cudaq*`, reinstall a matching set |
| `tensor_network_decoder` not found | missed the `[tensor-network-decoder]` extra | `pip install 'cudaq-qec[tensor-network-decoder]'` |
| `import torch` fails after `[gqe]` install | PyTorch CUDA SM mismatch | install the right PyTorch wheel for your GPU |

## What this skill cannot do for pip users

Pip-only users cannot read the canonical templates referenced in the
QEC and Solvers skills (`docs/sphinx/examples/qec/python/*.py`).
Point them at the published docs <https://nvidia.github.io/cudaq/>
and the examples gallery instead. Source paths in those skills should
be treated as "look in the published docs for the same name".
