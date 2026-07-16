# Realtime decoding demo

Drive the delivered `decoding_server` from two syndrome sources — a real FPGA,
or a lowered QPU kernel — both decoding through the **same prebuilt server**.

## What this example is

- **Deliverables** (installed, *not* built here): `decoding_server`, the FPGA
  playback tool `hololink_fpga_syndrome_playback`, the QEC + realtime libraries,
  and the decoder plugins.
- **The example** (the only thing you compile): one source,
  `surface_code_realtime_decoding.cpp`, built two ways by `CMakeLists.txt`
  against the **installed SDK** (installed headers/libs only):
  - `surface_code_realtime_decoding` — the **generator** (`--target stim`):
    writes the decoder config and, for the FPGA source, the syndrome file.
  - `surface_code_realtime_decoding-cqr` — the **lowered kernel**
    (`-frealtime-lowering`): the live syndrome source that streams to the server
    over UDP.

## Build (once)

```bash
cmake -S . -B build \
  -DCUDAQ_INSTALL_DIR=<cuda-quantum install prefix> \
  -DCUDAQX_INSTALL_DIR=<cuda-qx install prefix>
cmake --build build
# -> build/surface_code_realtime_decoding      (generator)
# -> build/surface_code_realtime_decoding-cqr  (lowered kernel)
```

In a CUDA-QX container `CUDAQ_INSTALL_DIR` defaults to `/usr/local/cudaq` (or
`$CUDA_QUANTUM_PATH`); point `CUDAQX_INSTALL_DIR` at where the CUDA-QX SDK is
installed. If the realtime libraries live in a separate prefix, add
`-DCUDAQ_REALTIME_DIR=<realtime prefix>`. The lowered kernel links the realtime
dispatch archive (relocatable CUDA device code), so the build needs a CUDA
toolchain; override the device-link architecture with
`-DCMAKE_CUDA_ARCHITECTURES=90` (default `80`) to match your GPU.

## Run

`run_realtime_decoding.sh` resolves the deliverables from `--install-prefix`
(`$PREFIX/bin`, `$PREFIX/lib`) and the two example binaries from
`--example-build-dir` (default `./build`).

### QPU-kernel source (software, UDP, no NIC)

```bash
./run_realtime_decoding.sh --source qpu-kernel --decoder pymatching        --install-prefix <prefix>
./run_realtime_decoding.sh --source qpu-kernel --decoder trt_decoder       --install-prefix <prefix>
./run_realtime_decoding.sh --source qpu-kernel --decoder nv-qldpc-decoder --gpu 0 --install-prefix <prefix>
```

The lowered kernel runs the surface-code memory experiment and streams each
shot's syndromes to the server over UDP; the server decodes and returns
corrections. No NIC, no FPGA, no network setup.

PASS/FAIL uses the same criteria as the in-tree surface_code-4 tests: the run
must complete without decoder errors, the residual logical-error count must
stay at or under `num_shots/50` (a decoder that is connected but decoding
wrong produces far more), the kernel must report server-owned decoders, and
the server must have dispatched at least `num_shots * (num_rounds + 3)` RPCs.

### FPGA source (real FPGA; needs a ConnectX NIC)

```bash
./run_realtime_decoding.sh --source fpga --decoder pymatching \
    --setup-network --device <nic> --bridge-ip <host-ip> --fpga-ip <fpga-ip> \
    --install-prefix <prefix>
#   --decoder trt_decoder / --decoder nv-qldpc-decoder --gpu 0 likewise
```

The delivered playback tool streams pre-generated syndromes over RoCE from the
FPGA into the server's RDMA RX ring. `--spacing` (default 10 µs) paces the
playback so it does not overrun the FPGA's fixed 64-slot ring. There is **no
emulator** in this example — `--source fpga` requires a real FPGA. (Emulator
testing lives in the unittests `hsb_fpga_decoding_server_test.sh`.)

## Decoders

| decoder | qpu-kernel (UDP) | fpga (real FPGA) | extra requirement |
|---|---|---|---|
| `pymatching` | CPU, no hardware | NIC | none |
| `trt_decoder` | GPU (server-side TensorRT) | NIC + GPU | python `onnx` + GPU |
| `nv-qldpc-decoder` | GPU (host-call path) | NIC + GPU (device path) | plugin + `--gpu` |

- **`pymatching`** — CPU matching decoder; nothing extra.
- **`trt_decoder`** — a TensorRT predecoder feeding a PyMatching global decoder.
  TensorRT inference runs server-side on a GPU (both sources). The script
  generates a tiny identity-predecoder ONNX at runtime (needs the python
  `onnx` module); override with `--onnx <model>`.
- **`nv-qldpc-decoder`** — GPU relay-BP. Needs the prebuilt plugin
  (auto-found in the install prefix, else pass `--nv-qldpc-plugin <path.so>`)
  and a GPU selected with `--gpu <id>`. If the plugin is unavailable the script
  exits `77` (skip).

See `./run_realtime_decoding.sh --help` for the full option list.
