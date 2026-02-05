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
