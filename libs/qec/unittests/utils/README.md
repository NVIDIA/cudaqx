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
