# QEC Utilities

## Hololink FPGA syndrome playback

The `hololink_fpga_syndrome_playback` tool programs canned syndrome data into
FPGA BRAM through the Hololink (Holoscan sensor bridge) control plane and
enables playback so the data is streamed into the Hololink-style ring buffer
on the GPU.

### Build

This target is optional and disabled by default to avoid CI dependencies on
Holoscan sensor bridge.

```
cmake -G Ninja -S /workspaces/cudaqx -B /workspaces/cudaqx/build \
      -DCUDAQX_QEC_ENABLE_HOLOLINK_TOOLS=ON \
      -DHOLOSCAN_SENSOR_BRIDGE_SOURCE_DIR=/workspaces/holoscan-sensor-bridge \
      -DHOLOSCAN_SENSOR_BRIDGE_BUILD_DIR=/workspaces/holoscan-sensor-bridge/build-core \
      -DCUDAQ_REALTIME_INCLUDE_DIR=/workspaces/cuda-quantum-gitlab/realtime/include \
      <other args>

cmake --build /workspaces/cudaqx/build --target hololink_fpga_syndrome_playback
```

Notes:
- `HOLOSCAN_SENSOR_BRIDGE_BUILD_DIR` must point to a build that produces `libhololink_core.a`
  (e.g. configure holoscan-sensor-bridge with `HOLOLINK_BUILD_ONLY_NATIVE=ON`).
- `CUDAQ_REALTIME_INCLUDE_DIR` must contain
  `cudaq/nvqlink/daemon/dispatcher/dispatch_kernel_launch.h`.

### Run

```
./hololink_fpga_syndrome_playback --hololink 192.168.0.2 \
  --data-dir libs/qec/unittests/decoders/realtime/data
```

Requires an FPGA/hololink device; the tool cannot be exercised without hardware.

Common options (defaults in parentheses):
- `--frame-size <bytes>`: payload size per window, multiple of 64 (auto-sized to RPC payload, aligned to 64)
- `--window-number <n>`: number of windows to program into BRAM (all shots)
- `--timer-value <ticks>`: raw FPGA timer value (computed)
- `--timer-spacing-us <us>` and `--board RFSoC|Other`: timer calculation helper (10 us, RFSoC)
- `--uuid <uuid>` and `--total-sensors/--total-dataplanes/--sifs-per-sensor`:
  custom enumeration strategy (none, 1/1/2)
- `--mtu <bytes>`: suggest MTU in enumeration metadata (unset)
- `--skip-reset`: skip Hololink reset (false)
- `--ptp-sync`: wait for PTP synchronization (false)
