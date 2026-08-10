# Decoding-server hardware CI (NVQLink lab)

CI-like testing of the QEC decoding server on real hardware — ConnectX RDMA
NICs, an FPGA syndrome source, and CC >= 9.0 GPUs — none of which normal
GitHub CI has (its GPU runners are A100/L4, so every `device_graph` /
CUDA-graph path is unreachable there).

`run_hw_ci.sh` takes a cudaqx commit, clones it, builds the dev image locally
(`../dev.Dockerfile`, layer-cached), builds cudaq-realtime + cudaqx + the
`realtime_decoding_demo` binaries inside the container
(`container_build.sh`), and runs the test lanes. Every lane ends the summary
as `PASS`, `FAIL`, or `SKIP(reason)`; `--strict` turns skips into failures.

```
./run_hw_ci.sh --sha <commit> --roce-pair rxe --hf-token-file ~/.hf_token
./run_hw_ci.sh --list          # show the lane set
```

## Lanes

* **examples tier** — the shipped `examples/qec/realtime_decoding_demo`
  driver: all 4 decoders (pymatching, multi_error_lut, nv-qldpc-decoder,
  trt_decoder) over `udp` (baseline), `cpu_roce` two-process (RoCE pair),
  and the FPGA source (`cpu_roce` host dispatch ×4, `device_graph`
  nv-qldpc ×1); plus `ising-prepare`, which downloads the gated Hugging Face
  Ising model and rebuilds the TRT artifact bundle **on every run** — a FAIL
  there means the HF/download/export path regressed, independent of decoders.
* **extra tier** — hardware tests outside the examples directory:
  `DecodingServerTwoProcess` + the `surface_code-1-cqr` two-process app
  ctests over `cpu_roce`; `hsb_fpga_decoding_server_test.sh` over `cpu_roce`
  and `gpu_roce` (the FPGA control-plane / SIF playback coverage); the
  CC >= 9.0-only ctests (`test_realtime_qldpc_graph_decoding`,
  mixed-dispatch, FP8 ONNX); and the `gpu_roce` QLDPC bridge as a
  below-the-server cross-check.
* **opt-in** (`--include-opt-in`) — the `gpu_roce` predecoder bridge; it
  links the experimental `cudaq-realtime-pipeline` library (off by default,
  pending a port to the post-PR4770 graph-launch API), so today it SKIPs.

## One-time host setup

1. Docker + nvidia-container-toolkit; user in the `docker` group;
   `docker login ghcr.io` (to pull `ghcr.io/nvidia/cudaqx-dev`).
2. For SoftRoCE mode (`--roce-pair rxe`): `sudo modprobe rdma_rxe`, persisted
   via `echo rdma_rxe | sudo tee /etc/modules-load.d/rdma_rxe.conf`.
   **Known gap:** the dev image inherits Mellanox OFED's `ibverbs-providers`,
   which ships only the mlx5 userspace provider — no rxe. Until a rxe
   provider matching that libibverbs ABI is built into the image, rxe setup
   fails its `ibv_devinfo` preflight and the two-process cpu_roce lanes SKIP
   with a named reason. Machines with a real loopback-cabled port pair
   (`--roce-pair DEV0,DEV1`) are unaffected.
3. FPGA cabled/flashed and reachable (defaults: NIC 192.168.0.1/24, FPGA
   192.168.0.2). Machines whose single cable is wired as a loopback pair
   instead run with `--no-fpga --roce-pair DEV0,DEV1`.
4. Proprietary artifacts (any subset; missing pieces => named SKIPs), staged
   under `--artifacts-dir` (default `/opt/nvqlink-lab-artifacts`), mounted
   read-only at `/artifacts` in the container:

   ```
   decoder-plugins/libcudaq-qec-nv-qldpc-decoder.so   # build against the SHA under test
   cudevice/libcudaq-qec-realtime-cudevice-proprietary.a
   ```

5. Hugging Face access for the Ising lanes: request access once to the gated
   `nvidia/Ising-Decoder-SurfaceCode-1-Fast` repo, then provide a token via
   `--hf-token-file` or `HF_TOKEN`. The token is passed only to the
   `ising-prepare` lane's `docker exec` — never baked into an image or
   written to a log. No token => the ising/trt lanes SKIP.
6. ~60 GB free disk for image layers and build trees.

## Per-machine invocations

```bash
# GB200 #1 (FPGA on one port, no free port pair -> SoftRoCE for two-process):
./run_hw_ci.sh --sha <commit> --roce-pair rxe --fpga-device rocep1s0f0 \
    --hf-token-file ~/.hf_token

# DGX Spark, single cable in loopback mode (port0 <-> port1, no FPGA):
./run_hw_ci.sh --sha <commit> --no-fpga --roce-pair rocep1s0f0,rocep1s0f1

# DGX Spark, single cable in FPGA mode:
./run_hw_ci.sh --sha <commit> --fpga-device rocep1s0f0
```

The CUDA architecture is auto-detected (`--cuda-arch` to override; GB200 =
100, Spark GB10 = 121). On 64 KiB-page kernels (GB200 `-64k`) the ring slot
size defaults to 512 and the `device_graph` lane rounds up to the
host-page-compatible value; `--page-size` overrides.

## Image / pin lifecycle

The dev image is built locally on every run and never pushed; the Docker
layer cache makes unchanged builds take seconds. Its base
(`ghcr.io/nvidia/cudaqx-dev`) carries CUDA-Q at the `.cudaq_version` pin —
the runner picks the pin-matched base tag for the commit under test and
fails fast on a mismatch. For pre-merge commits that bump `.cudaq_version`
(no published base yet), `--build-base` builds the base locally from the
commit's own `docker/build_env/cudaqx.dev.Dockerfile` (multi-hour the first
time per pin, cached after). `--refresh-base` re-pulls a moved base tag.
