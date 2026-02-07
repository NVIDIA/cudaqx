# QEC Utilities

## Hololink FPGA Syndrome Playback

The `hololink_fpga_syndrome_playback` tool programs canned syndrome data into
FPGA BRAM through the Hololink (Holoscan Sensor Bridge) control plane and
enables playback so the data is streamed into the Hololink-style ring buffer
on the GPU.  It uses only the control plane (UDP register reads/writes) — no
ConnectX NIC, RDMA, or GPU memory is required.

After writing BRAM, the tool always reads it back and verifies the contents
match what was written.

With the `--verify` flag the tool also captures incoming correction responses
from the GPU (via the ILA capture block on the FPGA) and verifies that every
RPCResponse header and correction byte matches the expected values from the
syndromes file.

### Build

This target is optional and disabled by default to avoid CI dependencies on
Holoscan Sensor Bridge.

```bash
cmake -G Ninja -S /workspaces/cudaqx -B /workspaces/cudaqx/build \
      -DCUDAQX_QEC_ENABLE_HOLOLINK_TOOLS=ON \
      -DHOLOSCAN_SENSOR_BRIDGE_SOURCE_DIR=/workspaces/holoscan-sensor-bridge \
      -DHOLOSCAN_SENSOR_BRIDGE_BUILD_DIR=/workspaces/holoscan-sensor-bridge/build-core \
      -DCUDAQ_REALTIME_INCLUDE_DIR=/workspaces/cuda-quantum-gitlab/realtime/include \
      <other args>

cmake --build /workspaces/cudaqx/build --target hololink_fpga_syndrome_playback
```

Notes:

- `HOLOSCAN_SENSOR_BRIDGE_BUILD_DIR` must point to a build that produces
  `libhololink_core.a` (e.g. configure holoscan-sensor-bridge with
  `HOLOLINK_BUILD_ONLY_NATIVE=ON`).
- `CUDAQ_REALTIME_INCLUDE_DIR` must contain
  `cudaq/nvqlink/daemon/dispatcher/dispatch_kernel_launch.h`.

### Run

```bash
./hololink_fpga_syndrome_playback \
    --hololink 192.168.0.2 \
    --data-dir /workspaces/cudaqx/libs/qec/unittests/decoders/realtime/data
```

Requires an FPGA/Hololink device; the tool cannot be exercised without
hardware.

### Options

| Option              | Required | Description                                                                                              |
| ------------------- | -------- | -------------------------------------------------------------------------------------------------------- |
| `--hololink <ip>`   | yes      | FPGA IP address                                                                                          |
| `--data-dir <path>` | yes      | Path to syndrome data directory (contains `config_multi_err_lut.yml` and `syndromes_multi_err_lut.txt`)  |
| `--num-shots <n>`   | no       | Number of shots to play back (default: all shots in the file)                                            |
| `--verify`          | no       | Capture and verify correction responses via ILA (requires mock decoder running on GPU)                   |
| `--qp-number <n>`   | no       | Destination QP number for RDMA (hex or decimal, from bridge tool output)                                 |
| `--rkey <n>`        | no       | Remote key for RDMA (from bridge tool output)                                                            |
| `--buffer-addr <n>` | no       | GPU buffer address for RDMA (hex or decimal, from bridge tool output)                                    |
| `--page-size <n>`   | no       | Ring buffer slot size in bytes (default: 256)                                                            |
| `--num-pages <n>`   | no       | Number of ring buffer slots (default: 64)                                                                |

When `--qp-number`, `--rkey`, and `--buffer-addr` are all provided, the tool
configures the FPGA SIF registers so that playback data is sent via RDMA to
the bridge tool's GPU buffers.  The values come from the bridge tool's
console output (see below).

### Using with the mock decoder bridge

The `hololink_mock_decoder_bridge` tool (on the `hololink_bridge` branch)
sets up a GPU-side RDMA transceiver and the cudaq dispatch kernel with the
mock decoder.  To run an end-to-end test:

1. Start the bridge tool on the GPU host:

```bash
./hololink_mock_decoder_bridge \
    --device rocep1s0f0 \
    --peer-ip 10.0.0.2 \
    --config /path/to/config_multi_err_lut.yml \
    --syndromes /path/to/syndromes_multi_err_lut.txt
```

2. Note the QP, RKEY, and buffer address from its output:

```
  Hololink QP Number: 0x1a
  Hololink RKey: 12345
  Hololink Buffer Addr: 0x7f1234560000
```

3. Run the playback tool, passing those values:

```bash
./hololink_fpga_syndrome_playback \
    --hololink 10.0.0.2 \
    --data-dir /path/to/data \
    --qp-number 0x1a \
    --rkey 12345 \
    --buffer-addr 0x7f1234560000 \
    --page-size 256 \
    --num-pages 64 \
    --verify
```

The playback tool configures the FPGA SIF registers via
`DataChannel::authenticate()` and `DataChannel::configure_roce()`, loads
syndromes into BRAM, and starts playback.  The FPGA sends syndrome data via
RDMA to the GPU, the dispatch kernel invokes the mock decoder, and correction
responses are sent back.  With `--verify`, the ILA captures the responses and
the tool checks them against expected values.

### Verification details

**BRAM readback (always-on):** After writing syndrome data to the playback
BRAM, the tool reads every word back over the control plane and compares
against what was written.  Any mismatches are reported and the tool exits
with a non-zero status.

**Correction verification (`--verify`):** Before starting playback, the tool
arms the ILA capture block at `0x4000_0000` (SIF TX, 512-bit data bus).
After playback begins, it polls the sample-count register (`base + 0x84`)
until the expected number of correction responses have been captured, then
disables capture and reads back the samples.  Each captured 512-bit word is
parsed as an RPCResponse (magic, status, result\_len) followed by the
correction byte, and compared against the expected correction from the
syndromes file.  A per-response and overall pass/fail summary is printed.

Note: `--verify` requires the mock decoder to be running on the GPU so that
correction responses are sent back to the FPGA.

---

## Hololink FPGA Emulator

The `hololink_fpga_emulator` tool is a software replacement for the physical
FPGA in the QEC decode loop.  It implements:

- **Hololink UDP control plane server** — accepts WR\_DWORD, WR\_BLOCK,
  RD\_DWORD, and RD\_BLOCK packets from the playback tool (or any Hololink
  client).
- **Emulated BRAM** — stores syndrome data loaded by the playback tool.
- **RDMA transmit** — sends syndromes to the bridge tool's GPU ring buffer
  via RDMA WRITE WITH IMMEDIATE (using libibverbs, no DOCA required).
- **RDMA receive** — receives correction responses from the bridge tool.
- **ILA capture** — stores corrections in emulated ILA registers for
  readback by the playback tool.

### Build

```bash
cmake --build /workspaces/cudaqx/build --target hololink_fpga_emulator
```

The emulator only requires `libibverbs` (no DOCA, no GPU).

### Three-Tool Emulated Workflow (no FPGA)

```
 Playback Tool           Emulator              Bridge Tool
 (control plane)    (FPGA replacement)       (GPU + mock decoder)
      |                    |                       |
      |  1. load syndromes |                       |
      |  ─── UDP WR ──────>|                       |
      |                    |                       |
      |  2. configure RDMA |                       |
      |  ─── UDP WR ──────>|  3. connect QP        |
      |                    |  ────────────────────> |
      |                    |                       |
      |  4. enable playback|                       |
      |  ─── UDP WR ──────>|  5. RDMA WRITE syndr  |
      |                    |  ────────────────────> |
      |                    |        6. decode       |
      |                    |  <──── RDMA WRITE corr |
      |                    |                       |
      |  7. read ILA       |                       |
      |  ─── UDP RD ──────>|                       |
```

**Startup order:**

1. **Start the emulator.** It prints its QP number and waits for
   configuration.

```bash
./hololink_fpga_emulator \
    --device rocep1s0f0 \
    --port 8193 \
    --bridge-ip 10.0.0.1
```

2. **Start the mock decoder bridge** on the GPU host, using the emulator's
   QP number as `--remote-qp`:

```bash
./hololink_mock_decoder_bridge \
    --device rocep1s0f1 \
    --peer-ip 10.0.0.2 \
    --remote-qp 0x<emulator_qp> \
    --config /path/to/config_multi_err_lut.yml \
    --syndromes /path/to/syndromes_multi_err_lut.txt
```

3. **Start the playback tool** using `--control-port` to bypass BOOTP
   enumeration, and pass the bridge tool's QP/RKEY/buffer:

```bash
./hololink_fpga_syndrome_playback \
    --hololink 10.0.0.2 \
    --control-port 8193 \
    --data-dir /path/to/data \
    --qp-number 0x<bridge_qp> \
    --rkey <bridge_rkey> \
    --buffer-addr 0x<bridge_buffer_addr> \
    --verify
```

### Emulator Options

| Option               | Default     | Description                                      |
| -------------------- | ----------- | ------------------------------------------------ |
| `--device=NAME`      | rocep1s0f0  | InfiniBand device name                           |
| `--ib-port=N`        | 1           | IB port number                                   |
| `--port=N`           | 8193        | UDP control plane port                           |
| `--bridge-ip=ADDR`   | (auto)      | Bridge tool IP for RoCEv2 GID                    |
| `--vp-address=ADDR`  | 0x1000      | VP register base address                         |
| `--hif-address=ADDR` | 0x0800      | HIF register base address                        |
| `--page-size=N`      | 256         | Slot size for correction RX buffer               |

### Playback Tool Emulator Options

When using the playback tool with the emulator, these additional options
are available:

| Option                 | Default | Description                                  |
| ---------------------- | ------- | -------------------------------------------- |
| `--control-port <n>`   | (none)  | UDP port for direct connection (bypasses BOOTP) |
| `--vp-address <n>`     | 0x1000  | VP register base (must match emulator)       |
| `--hif-address <n>`    | 0x0800  | HIF register base (must match emulator)      |

### Real FPGA Workflow

When using a real FPGA instead of the emulator, the workflow is:

1. **Start the bridge tool** with `--remote-qp 2` (the FPGA's hardcoded QP).
2. **Start the playback tool** without `--control-port` (uses BOOTP
   enumeration), passing the bridge tool's QP/RKEY/buffer info.

```bash
./hololink_fpga_syndrome_playback \
    --hololink 192.168.0.2 \
    --data-dir /path/to/data \
    --qp-number 0x<bridge_qp> \
    --rkey <bridge_rkey> \
    --buffer-addr 0x<bridge_buffer_addr> \
    --verify
```

---

## Orchestration Script

The `hololink_mock_decoder_test.sh` script automates the full end-to-end
QEC decode loop test, including building tools, configuring the network,
and orchestrating the multi-tool pipeline.

### Quick Start

```bash
# Emulated test (no FPGA): build everything, configure network, run test
./hololink_mock_decoder_test.sh --emulate --build --setup-network

# Real FPGA test: build, configure network, run
./hololink_mock_decoder_test.sh --build --setup-network --fpga-ip 192.168.0.2

# Run only (tools already built, network already configured)
./hololink_mock_decoder_test.sh --emulate

# Build only (no test run)
./hololink_mock_decoder_test.sh --build --no-run
```

### What the Script Does

1. **`--build`** builds three projects in order:
   - `cuda-quantum/realtime` (libcudaq-realtime, libcudaq-realtime-dispatch)
   - hololink (libhololink\_core, libgpu\_roce\_transceiver, and dependencies)
   - cudaqx tools (hololink\_fpga\_emulator, hololink\_mock\_decoder\_bridge,
     hololink\_fpga\_syndrome\_playback)

2. **`--setup-network`** configures ConnectX interfaces:
   - Brings links up, sets MTU, assigns IPs
   - Configures RoCEv2 mode, DSCP trust, disables adaptive RX coalescing
   - In emulate mode: sets up two ports (bridge + emulator)
   - In FPGA mode: sets up one port (bridge only)

3. **Run** (default unless `--no-run`):
   - Starts tools in the correct order, parses QP/RKEY/buffer from stdout
   - Passes RDMA parameters between tools automatically
   - Runs correction verification by default (`--no-verify` to skip)
   - Prints `QEC DECODE LOOP: PASS` or `QEC DECODE LOOP: FAIL`
   - Exits 0 on pass, 1 on fail

### Options

| Option               | Default                              | Description                                      |
| -------------------- | ------------------------------------ | ------------------------------------------------ |
| `--emulate`          | off (FPGA mode)                      | Use FPGA emulator instead of real FPGA           |
| `--build`            | off                                  | Build all tools before running                   |
| `--setup-network`    | off                                  | Configure ConnectX interfaces                    |
| `--no-run`           | off                                  | Skip running the test                            |
| `--no-verify`        | off (verify is ON)                   | Skip ILA correction verification                 |
| `--hololink-dir`     | /workspaces/cuda-qx/hololink        | Hololink source directory                        |
| `--cuda-quantum-dir` | /workspaces/cuda-quantum             | cuda-quantum source directory                    |
| `--cudaqx-dir`       | /workspaces/cudaqx                   | cudaqx source directory                          |
| `--device`           | auto-detect                          | ConnectX IB device name                          |
| `--bridge-ip`        | 10.0.0.1                             | Bridge tool IP address                           |
| `--emulator-ip`      | 10.0.0.2                             | Emulator IP address                              |
| `--fpga-ip`          | 192.168.0.2                          | FPGA IP (non-emulate mode)                       |
| `--mtu`              | 4096                                 | MTU size                                         |
| `--data-dir`         | cudaqx/.../decoders/realtime/data    | Syndrome data directory                          |
| `--gpu`              | 0                                    | GPU device ID                                    |
| `--timeout`          | 60                                   | Timeout in seconds                               |
| `--num-shots`        | all                                  | Limit number of shots                            |
| `--jobs`             | nproc                                | Parallel build jobs                              |
