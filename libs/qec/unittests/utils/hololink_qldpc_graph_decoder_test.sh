#!/bin/bash
# ============================================================================ #
# Copyright (c) 2025-2026 NVIDIA Corporation & Affiliates.                    #
# All rights reserved.                                                        #
#                                                                             #
# This source code and the accompanying materials are made available under    #
# the terms of the Apache License 2.0 which accompanies this distribution.    #
# ============================================================================ #
#
# hololink_qldpc_graph_decoder_test.sh
#
# Orchestration script for end-to-end QLDPC BP decode loop testing over
# Hololink RDMA/RoCE.  Uses CPU-launched CUDA graph dispatch (HOST_LOOP)
# with Relay BP.
#
# Modes:
#   Default (FPGA):   bridge + playback  (requires real FPGA)
#   --emulate:        emulator + bridge + playback  (no FPGA needed)
#
# Actions (can be combined):
#   --build            Build all required tools
#   --setup-network    Configure ConnectX interfaces
#   (run is implicit unless only --build / --setup-network are given)
#
# Examples:
#   # Full emulated test: build, configure network, run
#   ./hololink_qldpc_graph_decoder_test.sh --emulate --build --setup-network
#
#   # Just run (tools already built, network already set up)
#   ./hololink_qldpc_graph_decoder_test.sh --emulate
#
#   # Build only
#   ./hololink_qldpc_graph_decoder_test.sh --build --no-run
#
set -euo pipefail

# ============================================================================
# Defaults
# ============================================================================

EMULATE=false
DO_BUILD=false
DO_SETUP_NETWORK=false
DO_RUN=true
VERIFY=true

# Directory defaults
HSB_DIR="/workspaces/holoscan-sensor-bridge"
CUDA_QUANTUM_DIR="/workspaces/cuda-quantum"
CUDA_QX_DIR="/workspaces/cudaqx"
DATA_DIR=""  # auto-detected if empty

# Network defaults
IB_DEVICE=""           # auto-detect
BRIDGE_IP="10.0.0.1"
EMULATOR_IP="10.0.0.2"
FPGA_IP="192.168.0.2"
MTU=4096

# Run defaults
GPU_ID=0
TIMEOUT=60
NUM_SHOTS=""
PAGE_SIZE=384
NUM_PAGES=128
SPACING=""
CONTROL_PORT=8193

# Build parallelism
JOBS=$(nproc 2>/dev/null || echo 8)

# ============================================================================
# Argument Parsing
# ============================================================================

print_usage() {
    cat <<'EOF'
Usage: hololink_qldpc_graph_decoder_test.sh [options]

Orchestration script for QLDPC BP decoder end-to-end testing over
Hololink RDMA/RoCE with CPU-launched CUDA graph dispatch (HOST_LOOP).

Modes:
  --emulate              Use FPGA emulator (3-tool mode, no FPGA needed)
                         Default: FPGA mode (2-tool, requires real FPGA)

Actions:
  --build                Build all required tools before running
  --setup-network        Configure ConnectX network interfaces
  --no-run               Skip running the test (useful with --build)

Build options:
  --hsb-dir DIR          holoscan-sensor-bridge source directory
                         (default: /workspaces/holoscan-sensor-bridge)
  --cuda-quantum-dir DIR cuda-quantum source directory
                         (default: /workspaces/cuda-quantum)
  --cuda-qx-dir DIR      cuda-qx source directory
                         (default: /workspaces/cuda-qx)
  --jobs N               Parallel build jobs (default: nproc)

Network options:
  --device DEV           ConnectX IB device name (default: auto-detect)
  --bridge-ip ADDR       Bridge tool IP (default: 10.0.0.1)
  --emulator-ip ADDR     Emulator IP (default: 10.0.0.2)
  --fpga-ip ADDR         FPGA IP for non-emulate mode (default: 192.168.0.2)
  --mtu N                MTU size (default: 4096)

Run options:
  --data-dir DIR         Syndrome data directory (default: auto-detect)
  --gpu N                GPU device ID (default: 0)
  --timeout N            Timeout in seconds (default: 60)
  --no-verify            Skip correction verification
  --num-shots N          Limit number of shots
  --page-size N          Ring buffer slot size in bytes (default: 384)
  --num-pages N          Number of ring buffer slots (default: 128)
  --spacing N            Inter-shot spacing in microseconds (default: 10)
  --control-port N       UDP control port for emulator (default: 8193)

  --help, -h             Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --emulate)          EMULATE=true ;;
        --build)            DO_BUILD=true ;;
        --setup-network)    DO_SETUP_NETWORK=true ;;
        --no-run)           DO_RUN=false ;;
        --no-verify)        VERIFY=false ;;
        --hsb-dir)          HSB_DIR="$2"; shift ;;
        --cuda-quantum-dir) CUDA_QUANTUM_DIR="$2"; shift ;;
        --cuda-qx-dir)      CUDA_QX_DIR="$2"; shift ;;
        --jobs)             JOBS="$2"; shift ;;
        --device)           IB_DEVICE="$2"; shift ;;
        --bridge-ip)        BRIDGE_IP="$2"; shift ;;
        --emulator-ip)      EMULATOR_IP="$2"; shift ;;
        --fpga-ip)          FPGA_IP="$2"; shift ;;
        --mtu)              MTU="$2"; shift ;;
        --data-dir)         DATA_DIR="$2"; shift ;;
        --gpu)              GPU_ID="$2"; shift ;;
        --timeout)          TIMEOUT="$2"; shift ;;
        --num-shots)        NUM_SHOTS="$2"; shift ;;
        --page-size)        PAGE_SIZE="$2"; shift ;;
        --num-pages)        NUM_PAGES="$2"; shift ;;
        --spacing)          SPACING="$2"; shift ;;
        --control-port)     CONTROL_PORT="$2"; shift ;;
        --help|-h)          print_usage; exit 0 ;;
        *)
            echo "ERROR: Unknown option: $1" >&2
            print_usage >&2
            exit 1
            ;;
    esac
    shift
done

# ============================================================================
# Logging Helpers
# ============================================================================

_log()  { echo "==> $*"; }
_info() { echo "    $*"; }
_err()  { echo "ERROR: $*" >&2; }
_banner() {
    echo ""
    echo "========================================"
    echo "  $*"
    echo "========================================"
    echo ""
}

# ============================================================================
# Cleanup
# ============================================================================

PIDS_TO_KILL=()
TEMP_FILES=()

cleanup() {
    local pid
    for pid in "${PIDS_TO_KILL[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -TERM "$pid" 2>/dev/null || true
            sleep 1
            kill -0 "$pid" 2>/dev/null && kill -KILL "$pid" 2>/dev/null || true
        fi
    done
    for f in "${TEMP_FILES[@]}"; do
        rm -f "$f"
    done
}
trap cleanup EXIT

# ============================================================================
# Network Setup
# ============================================================================

detect_interfaces() {
    if ! command -v ibdev2netdev &>/dev/null; then
        _err "ibdev2netdev not found. Install rdma-core or Mellanox OFED."
        return 1
    fi
    ibdev2netdev
}

ib_to_netdev() {
    local ib_dev="$1"
    local port="${2:-1}"
    ibdev2netdev | awk -v dev="$ib_dev" -v p="$port" \
        '$1 == dev && $3 == p { print $5 }'
}

netdev_to_ib() {
    local iface="$1"
    ibdev2netdev | awk -v iface="$iface" '$5 == iface { print $1 }'
}

setup_port() {
    local iface="$1"
    local ip="$2"
    local mtu="$3"
    local ib_dev

    _info "Configuring $iface: ip=$ip mtu=$mtu"

    local other
    for other in $(ip -o addr show to "${ip}/24" 2>/dev/null | awk '{print $2}' | sort -u); do
        if [[ "$other" != "$iface" ]]; then
            _info "Removing stale ${ip}/24 from $other"
            sudo ip addr del "${ip}/24" dev "$other" 2>/dev/null || true
        fi
    done

    sudo ip link set "$iface" up
    sudo ip link set "$iface" mtu "$mtu"
    sudo ip addr flush dev "$iface"
    sudo ip addr add "${ip}/24" dev "$iface"

    ib_dev=$(netdev_to_ib "$iface")
    if [[ -n "$ib_dev" ]] && command -v rdma &>/dev/null; then
        local port_count
        port_count=$(ls -d "/sys/class/infiniband/${ib_dev}/ports/"* 2>/dev/null | wc -l)
        for p in $(seq 1 "$port_count"); do
            sudo rdma link set "${ib_dev}/${p}" type eth || true
        done
        _info "  RoCEv2 mode configured for $ib_dev"
    fi

    if command -v mlnx_qos &>/dev/null; then
        sudo mlnx_qos -i "$iface" --trust=dscp 2>/dev/null || true
        _info "  DSCP trust mode set"
    fi

    if command -v ethtool &>/dev/null; then
        sudo ethtool -C "$iface" adaptive-rx off rx-usecs 0 2>/dev/null || true
    fi

    _info "  Done: $iface is up at $ip"
}

_add_static_arp() {
    local local_iface="$1"
    local remote_ip="$2"
    local remote_iface="$3"
    local mac
    mac=$(ip link show "$remote_iface" | awk '/ether/ {print $2}')
    if [[ -z "$mac" ]]; then
        _err "Cannot determine MAC address for $remote_iface"
        return 1
    fi
    sudo ip neigh replace "$remote_ip" lladdr "$mac" nud permanent dev "$local_iface"
    _info "  Static ARP: $remote_ip -> $mac on $local_iface"
}

do_setup_network() {
    _log "Setting up ConnectX network"

    if $EMULATE; then
        local interfaces
        interfaces=$(detect_interfaces)

        if [[ -z "$IB_DEVICE" ]]; then
            local iface_bridge iface_emulator
            local first_dev second_dev first_iface second_iface

            first_dev=$(echo "$interfaces" | head -1 | awk '{print $1}')
            first_iface=$(echo "$interfaces" | head -1 | awk '{print $5}')

            second_iface=$(echo "$interfaces" | awk -v d="$first_dev" \
                '$1 == d && $3 == 2 {print $5}')

            if [[ -n "$second_iface" ]]; then
                iface_bridge="$first_iface"
                iface_emulator="$second_iface"
            else
                second_iface=$(echo "$interfaces" | awk 'NR==2 {print $5}')
                if [[ -z "$second_iface" ]]; then
                    _err "Need two ConnectX ports for emulation mode but only found one."
                    return 1
                fi
                iface_bridge="$first_iface"
                iface_emulator="$second_iface"
            fi

            _info "Bridge interface:   $iface_bridge"
            _info "Emulator interface: $iface_emulator"
            setup_port "$iface_bridge" "$BRIDGE_IP" "$MTU"
            setup_port "$iface_emulator" "$EMULATOR_IP" "$MTU"

            BRIDGE_DEVICE=$(netdev_to_ib "$iface_bridge")
            EMULATOR_DEVICE=$(netdev_to_ib "$iface_emulator")

            _add_static_arp "$iface_bridge" "$EMULATOR_IP" "$iface_emulator"
            _add_static_arp "$iface_emulator" "$BRIDGE_IP" "$iface_bridge"
        else
            local iface1 iface2
            iface1=$(ib_to_netdev "$IB_DEVICE" 1)
            iface2=$(ib_to_netdev "$IB_DEVICE" 2)
            if [[ -z "$iface1" || -z "$iface2" ]]; then
                _err "Cannot find two ports on device $IB_DEVICE"
                return 1
            fi
            setup_port "$iface1" "$BRIDGE_IP" "$MTU"
            setup_port "$iface2" "$EMULATOR_IP" "$MTU"
            BRIDGE_DEVICE=$(netdev_to_ib "$iface1")
            EMULATOR_DEVICE=$(netdev_to_ib "$iface2")

            _add_static_arp "$iface1" "$EMULATOR_IP" "$iface2"
            _add_static_arp "$iface2" "$BRIDGE_IP" "$iface1"
        fi

        _info "Waiting 2s for GID tables to populate..."
        sleep 2
    else
        local iface_bridge
        if [[ -n "$IB_DEVICE" ]]; then
            iface_bridge=$(ib_to_netdev "$IB_DEVICE" 1)
        else
            iface_bridge=$(detect_interfaces | head -1 | awk '{print $5}')
        fi

        if [[ -z "$iface_bridge" ]]; then
            _err "Cannot detect ConnectX interface for bridge tool."
            return 1
        fi

        _info "Bridge interface: $iface_bridge"
        setup_port "$iface_bridge" "$BRIDGE_IP" "$MTU"
        BRIDGE_DEVICE=$(netdev_to_ib "$iface_bridge")
    fi
}

# ============================================================================
# Build
# ============================================================================

detect_cuda_arch() {
    local max_arch
    max_arch=$(nvcc --list-gpu-arch 2>/dev/null \
        | grep -oP 'compute_\K[0-9]+' | sort -n | tail -1)
    if [ -n "$max_arch" ]; then
        echo "$max_arch"
    fi
}

do_build() {
    _log "Building all tools (jobs=$JOBS)"

    local cuda_qx_build="${CUDA_QX_DIR}/build"
    local cq_build="${CUDA_QUANTUM_DIR}/realtime/build"
    local hsb_build="${HSB_DIR}/build"

    local cuda_arch
    cuda_arch=$(detect_cuda_arch)
    local cuda_arch_flag=""
    if [ -n "$cuda_arch" ]; then
        cuda_arch_flag="-DCMAKE_CUDA_ARCHITECTURES=$cuda_arch"
        _info "CUDA arch: $cuda_arch"
    fi

    # Detect CUDA compiler/toolkit once and reuse for all stages so mixed
    # environments (CUDA 12.6/13.0, amd64/arm64) stay consistent.
    local cuda_compiler=""
    if [[ -n "${CMAKE_CUDA_COMPILER:-}" ]]; then
        cuda_compiler="${CMAKE_CUDA_COMPILER}"
    elif [[ -x /usr/local/cuda/bin/nvcc ]]; then
        cuda_compiler="/usr/local/cuda/bin/nvcc"
    else
        cuda_compiler="$(command -v nvcc || true)"
    fi
    if [[ -z "$cuda_compiler" || ! -x "$cuda_compiler" ]]; then
        _err "Unable to locate nvcc. Set CMAKE_CUDA_COMPILER or update PATH."
        return 1
    fi

    # Ensure nvcc is on PATH so detect_cuda_arch() and cmake check_language(CUDA) work.
    local cuda_bin_dir
    cuda_bin_dir="$(dirname "$cuda_compiler")"
    case ":$PATH:" in
        *":$cuda_bin_dir:"*) ;;
        *) export PATH="$cuda_bin_dir:$PATH" ;;
    esac

    if [[ -z "$cuda_arch" ]]; then
        cuda_arch=$(detect_cuda_arch)
        if [[ -n "$cuda_arch" ]]; then
            cuda_arch_flag="-DCMAKE_CUDA_ARCHITECTURES=$cuda_arch"
            _info "CUDA arch (re-detected): $cuda_arch"
        fi
    fi

    local cuda_toolkit_root
    cuda_toolkit_root="$(cd "$(dirname "$cuda_compiler")/.." && pwd -P)"
    _info "CUDA compiler: $cuda_compiler"
    _info "CUDA toolkit:  $cuda_toolkit_root"

    # ---- Stage 1: cuda-quantum/realtime ----
    _banner "Stage 1/3: Building cuda-quantum/realtime"
    local cq_src="${CUDA_QUANTUM_DIR}/realtime"
    if [[ ! -d "$cq_src" ]]; then
        _err "cuda-quantum realtime source not found at $cq_src"
        return 1
    fi

    cmake -G Ninja -S "$cq_src" -B "$cq_build" \
        -DCMAKE_BUILD_TYPE=Release \
        $cuda_arch_flag \
        2>&1 | tail -5
    cmake --build "$cq_build" -j "$JOBS" 2>&1 | tail -5
    _info "cuda-quantum/realtime built: $cq_build/lib/"

    # ---- Stage 2: holoscan-sensor-bridge (Hololink) ----
    _banner "Stage 2/3: Building holoscan-sensor-bridge"
    if [[ ! -d "$HSB_DIR" ]]; then
        _err "holoscan-sensor-bridge source not found at $HSB_DIR"
        return 1
    fi

    local target_arch="amd64"
    if [[ "$(uname -m)" == "aarch64" ]]; then
        target_arch="arm64"
    fi

    # Holoscan SDK requires CMake >= 3.30.4; find a suitable binary.
    local hsb_cmake="cmake"
    if [[ -x /tmp/cmake-3.31.6-linux-aarch64/bin/cmake ]]; then
        hsb_cmake="/tmp/cmake-3.31.6-linux-aarch64/bin/cmake"
    elif [[ -x /usr/local/cmake-3.31/bin/cmake ]]; then
        hsb_cmake="/usr/local/cmake-3.31/bin/cmake"
    fi

    local hsb_common_args=(
        -G Ninja
        -S "$HSB_DIR"
        -B "$hsb_build"
        -DCMAKE_BUILD_TYPE=Release
        -DCMAKE_CUDA_COMPILER="$cuda_compiler"
        -DCUDAToolkit_ROOT="$cuda_toolkit_root"
        $cuda_arch_flag
        -DTARGET_ARCH="$target_arch"
        -DHOLOLINK_BUILD_ONLY_NATIVE=OFF
        -DHOLOLINK_BUILD_PYTHON=OFF
        -DHOLOLINK_BUILD_TESTS=OFF
        -DHOLOLINK_BUILD_TOOLS=OFF
        -DHOLOLINK_BUILD_EXAMPLES=OFF
        -DHOLOLINK_BUILD_EMULATOR=OFF
    )
    # Stage 2 needs deterministic cache state; stale cache can re-enable tools.
    if "$hsb_cmake" --help 2>/dev/null | grep -q -- "--fresh"; then
        "$hsb_cmake" --fresh "${hsb_common_args[@]}" 2>&1 | tail -5
    else
        rm -f "$hsb_build/CMakeCache.txt"
        rm -rf "$hsb_build/CMakeFiles"
        "$hsb_cmake" "${hsb_common_args[@]}" 2>&1 | tail -5
    fi

    if [[ -f "$hsb_build/CMakeCache.txt" ]]; then
        _info "Hololink cache options:"
        grep -E '^HOLOLINK_BUILD_(TOOLS|EXAMPLES|TESTS|PYTHON|EMULATOR|ONLY_NATIVE):BOOL=' \
            "$hsb_build/CMakeCache.txt" | sed 's/^/      /' || true
    fi

    "$hsb_cmake" --build "$hsb_build" -j "$JOBS" \
        --target gpu_roce_transceiver hololink_core 2>&1 | tail -5
    _info "holoscan-sensor-bridge built: $hsb_build/"

    # Reconfigure cuda-quantum/realtime with hololink tools (for emulator).
    # Force GPU_ROCE_TRANSCEIVER_LIB to prevent stale cache from a different
    # hololink repo (e.g. switching between holoscan-sensor-bridge and hololink).
    local hsb_gpu_roce_lib="${hsb_build}/src/hololink/operators/gpu_roce_transceiver/libgpu_roce_transceiver.a"
    cmake -G Ninja -S "$cq_src" -B "$cq_build" \
        -DCMAKE_BUILD_TYPE=Release \
        $cuda_arch_flag \
        -DCUDAQ_REALTIME_ENABLE_HOLOLINK_TOOLS=ON \
        -DHOLOSCAN_SENSOR_BRIDGE_SOURCE_DIR="$HSB_DIR" \
        -DHOLOSCAN_SENSOR_BRIDGE_BUILD_DIR="$hsb_build" \
        -DGPU_ROCE_TRANSCEIVER_LIB="$hsb_gpu_roce_lib" \
        2>&1 | tail -5
    # Rebuild bridge-hololink .so (has hololink .a baked in) and emulator.
    cmake --build "$cq_build" -j "$JOBS" \
        --target cudaq-realtime-bridge-hololink hololink_fpga_emulator 2>&1 | tail -5

    # ---- Stage 3: cuda-qx QLDPC graph bridge + playback ----
    _banner "Stage 3/3: Building cuda-qx QLDPC graph bridge + playback"
    if [[ ! -d "$CUDA_QX_DIR" ]]; then
        _err "cuda-qx source not found at $CUDA_QX_DIR"
        return 1
    fi

    # Clear stale cmake cache entries (find_library caches NOTFOUND permanently)
    rm -f "$cuda_qx_build/CMakeCache.txt"

    cmake -G Ninja -S "$CUDA_QX_DIR" -B "$cuda_qx_build" \
        $cuda_arch_flag \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_COMPILER="$cuda_compiler" \
        -DCUDAToolkit_ROOT="$cuda_toolkit_root" \
        -DCUDAQX_QEC_ENABLE_HOLOLINK_TOOLS=ON \
        -DHOLOSCAN_SENSOR_BRIDGE_SOURCE_DIR="$HSB_DIR" \
        -DHOLOSCAN_SENSOR_BRIDGE_BUILD_DIR="$hsb_build" \
        -DGPU_ROCE_TRANSCEIVER_LIB="$hsb_gpu_roce_lib" \
        -DCUDAQ_REALTIME_ROOT="${CUDA_QUANTUM_DIR}/realtime" \
        -DCUDAQ_REALTIME_INCLUDE_DIR="${CUDA_QUANTUM_DIR}/realtime/include" \
        -DCUDAQ_REALTIME_LIBRARY="${cq_build}/lib/libcudaq-realtime.so" \
        -DCUDAQ_REALTIME_DISPATCH_LIBRARY="${cq_build}/lib/libcudaq-realtime-dispatch.a" \
        -DCUDAQ_REALTIME_HOST_DISPATCH_LIBRARY="${cq_build}/lib/libcudaq-realtime-host-dispatch.a" \
        -DCUDAQ_REALTIME_BRIDGE_HOLOLINK_LIBRARY="${cq_build}/lib/libcudaq-realtime-bridge-hololink.so" \
        -DCUDAQ_INSTALL_PREFIX="${CUDAQ_INSTALL_PREFIX:-/usr/local/cudaq}" \
        2>&1 | tail -5
    cmake --build "$cuda_qx_build" -j "$JOBS" \
        --target hololink_qldpc_graph_decoder_bridge \
                 hololink_fpga_syndrome_playback \
        2>&1 | tail -5
    _info "cuda-qx tools built: $cuda_qx_build/libs/qec/unittests/utils/"

    _banner "Build complete"
}

# ============================================================================
# Tool Path Resolution
# ============================================================================

resolve_paths() {
    local cuda_qx_utils="${CUDA_QX_DIR}/build/libs/qec/unittests/utils"
    local cq_build_dir="${CUDA_QUANTUM_DIR}/realtime/build/unittests"

    BRIDGE_BIN="${cuda_qx_utils}/hololink_qldpc_graph_decoder_bridge"
    PLAYBACK_BIN="${cuda_qx_utils}/hololink_fpga_syndrome_playback"
    EMULATOR_BIN="${cq_build_dir}/utils/hololink_fpga_emulator"

    if [[ -z "$DATA_DIR" ]]; then
        DATA_DIR="${CUDA_QX_DIR}/libs/qec/unittests/realtime/qec_roce_decode_test/data"
    fi

    CONFIG_FILE="${DATA_DIR}/config_nv_qldpc_relay.yml"
    SYNDROMES_FILE="${DATA_DIR}/syndromes_nv_qldpc_relay.txt"

    if [[ ! -x "$BRIDGE_BIN" ]]; then
        _err "Bridge binary not found: $BRIDGE_BIN"
        _err "Run with --build to build the tools first."
        return 1
    fi
    if [[ ! -x "$PLAYBACK_BIN" ]]; then
        _err "Playback binary not found: $PLAYBACK_BIN"
        _err "Run with --build to build the tools first."
        return 1
    fi
    if $EMULATE && [[ ! -x "$EMULATOR_BIN" ]]; then
        _err "Emulator binary not found: $EMULATOR_BIN"
        return 1
    fi
    if [[ ! -f "$CONFIG_FILE" ]]; then
        _err "Config file not found: $CONFIG_FILE"
        return 1
    fi
    if [[ ! -f "$SYNDROMES_FILE" ]]; then
        _err "Syndromes file not found: $SYNDROMES_FILE"
        return 1
    fi

    if [ -z "${BRIDGE_DEVICE:-}" ] && [ -n "${IB_DEVICE:-}" ]; then
        BRIDGE_DEVICE="$IB_DEVICE"
    fi
    : "${BRIDGE_DEVICE:=rocep1s0f0}"
    if $EMULATE; then
        : "${EMULATOR_DEVICE:=rocep1s0f1}"
    fi
}

# ============================================================================
# Output Parsing Helpers
# ============================================================================

wait_for_pattern() {
    local logfile="$1"
    local pattern="$2"
    local timeout_sec="$3"
    local pid_to_check="${4:-}"

    local poll_ms=500
    local waited_ms=0
    local timeout_ms=$((timeout_sec * 1000))
    while (( waited_ms < timeout_ms )); do
        if [[ -n "$pid_to_check" ]] && ! kill -0 "$pid_to_check" 2>/dev/null; then
            _err "Process $pid_to_check died unexpectedly"
            return 1
        fi
        local match
        match=$(grep -m1 "$pattern" "$logfile" 2>/dev/null || true)
        if [[ -n "$match" ]]; then
            echo "$match"
            return 0
        fi
        sleep 0.5
        waited_ms=$((waited_ms + poll_ms))
    done
    _err "Timeout waiting for pattern: $pattern"
    return 1
}

extract_hex() {
    local line="$1"
    echo "$line" | grep -oP '0x[0-9a-fA-F]+' | head -1
}

extract_decimal() {
    local line="$1"
    echo "$line" | awk -F': ' '{print $NF}' | tr -d ' '
}

# ============================================================================
# Run: Emulated Mode (3 tools)
# ============================================================================

run_emulated() {
    _banner "QLDPC BP Decode Loop Test (Emulated FPGA, Graph Launch)"

    local emu_log bridge_log
    emu_log=$(mktemp /tmp/qldpc_graph_emulator.XXXXXX.log)
    bridge_log=$(mktemp /tmp/qldpc_graph_bridge.XXXXXX.log)
    TEMP_FILES+=("$emu_log" "$bridge_log")

    # ---- 1. Start emulator ----
    _log "Starting FPGA emulator on port $CONTROL_PORT"
    "$EMULATOR_BIN" \
        --device="$EMULATOR_DEVICE" \
        --port="$CONTROL_PORT" \
        --bridge-ip="$BRIDGE_IP" \
        --page-size="$PAGE_SIZE" \
        > >(tee "$emu_log") 2>&1 &
    local emu_pid=$!
    PIDS_TO_KILL+=("$emu_pid")
    _info "Emulator PID: $emu_pid"

    local emu_qp_line
    emu_qp_line=$(wait_for_pattern "$emu_log" "Emulator QP:" 30 "$emu_pid") || {
        _err "Failed to get emulator QP number"
        return 1
    }
    local emu_qp
    emu_qp=$(extract_hex "$emu_qp_line")
    _info "Emulator QP: $emu_qp"

    # ---- 2. Start QLDPC graph bridge tool ----
    _log "Starting QLDPC BP graph decoder bridge (remote-qp=$emu_qp)"

    local bridge_ld_path
    bridge_ld_path="${CUDA_QUANTUM_DIR}/realtime/build/lib:${CUDA_QX_DIR}/build/lib"

    CUDA_MODULE_LOADING=EAGER \
    LD_LIBRARY_PATH="${bridge_ld_path}:${LD_LIBRARY_PATH:-}" \
    "$BRIDGE_BIN" \
        --device="$BRIDGE_DEVICE" \
        --peer-ip="$EMULATOR_IP" \
        --remote-qp="$emu_qp" \
        --gpu="$GPU_ID" \
        --config="$CONFIG_FILE" \
        --timeout="$TIMEOUT" \
        --page-size="$PAGE_SIZE" \
        --num-pages="$NUM_PAGES" \
        > >(tee "$bridge_log") 2>&1 &
    local bridge_pid=$!
    PIDS_TO_KILL+=("$bridge_pid")
    _info "Bridge PID: $bridge_pid"

    wait_for_pattern "$bridge_log" "Bridge Ready" 60 "$bridge_pid" >/dev/null || {
        _err "Bridge did not become ready"
        _err "--- Bridge log ---"
        cat "$bridge_log" >&2
        return 1
    }

    local qp_line rkey_line addr_line
    qp_line=$(wait_for_pattern "$bridge_log" "QP Number:" 5 "$bridge_pid") || return 1
    rkey_line=$(wait_for_pattern "$bridge_log" "RKey:" 5 "$bridge_pid") || return 1
    addr_line=$(wait_for_pattern "$bridge_log" "Buffer Addr:" 5 "$bridge_pid") || return 1

    local bridge_qp bridge_rkey bridge_addr
    bridge_qp=$(extract_hex "$qp_line")
    bridge_rkey=$(extract_decimal "$rkey_line")
    bridge_addr=$(extract_hex "$addr_line")

    _info "Bridge QP:     $bridge_qp"
    _info "Bridge RKey:   $bridge_rkey"
    _info "Bridge Buffer: $bridge_addr"

    # ---- 3. Start playback tool ----
    _log "Starting syndrome playback (control-port=$CONTROL_PORT)"
    local playback_args=(
        --hololink "$EMULATOR_IP"
        --control-port "$CONTROL_PORT"
        --config "$CONFIG_FILE"
        --syndromes "$SYNDROMES_FILE"
        --function-name nv_qldpc_decode
        --qp-number "$bridge_qp"
        --rkey "$bridge_rkey"
        --buffer-addr "$bridge_addr"
        --page-size "$PAGE_SIZE"
        --num-pages "$NUM_PAGES"
    )
    if $VERIFY; then
        playback_args+=(--verify)
    fi
    if [[ -n "$NUM_SHOTS" ]]; then
        playback_args+=(--num-shots "$NUM_SHOTS")
    fi
    if [[ -n "$SPACING" ]]; then
        playback_args+=(--spacing "$SPACING")
    fi

    local playback_rc=0
    "$PLAYBACK_BIN" "${playback_args[@]}" || playback_rc=$?

    return $playback_rc
}

# ============================================================================
# Run: FPGA Mode (2 tools)
# ============================================================================

run_fpga() {
    _banner "QLDPC BP Decode Loop Test (Real FPGA, Graph Launch)"

    local bridge_log
    bridge_log=$(mktemp /tmp/qldpc_graph_bridge.XXXXXX.log)
    TEMP_FILES+=("$bridge_log")

    # ---- 1. Start QLDPC graph bridge tool ----
    _log "Starting QLDPC BP graph decoder bridge (remote-qp=0x2, fpga-ip=$FPGA_IP)"

    local bridge_ld_path
    bridge_ld_path="${CUDA_QUANTUM_DIR}/realtime/build/lib:${CUDA_QX_DIR}/build/lib"

    CUDA_MODULE_LOADING=EAGER \
    LD_LIBRARY_PATH="${bridge_ld_path}:${LD_LIBRARY_PATH:-}" \
    "$BRIDGE_BIN" \
        --device="$BRIDGE_DEVICE" \
        --peer-ip="$FPGA_IP" \
        --remote-qp=0x2 \
        --gpu="$GPU_ID" \
        --config="$CONFIG_FILE" \
        --timeout="$TIMEOUT" \
        --page-size="$PAGE_SIZE" \
        --num-pages="$NUM_PAGES" \
        > >(tee "$bridge_log") 2>&1 &
    local bridge_pid=$!
    PIDS_TO_KILL+=("$bridge_pid")
    _info "Bridge PID: $bridge_pid"

    wait_for_pattern "$bridge_log" "Bridge Ready" 60 "$bridge_pid" >/dev/null || {
        _err "Bridge did not become ready"
        return 1
    }

    local qp_line rkey_line addr_line
    qp_line=$(wait_for_pattern "$bridge_log" "QP Number:" 5 "$bridge_pid") || return 1
    rkey_line=$(wait_for_pattern "$bridge_log" "RKey:" 5 "$bridge_pid") || return 1
    addr_line=$(wait_for_pattern "$bridge_log" "Buffer Addr:" 5 "$bridge_pid") || return 1

    local bridge_qp bridge_rkey bridge_addr
    bridge_qp=$(extract_hex "$qp_line")
    bridge_rkey=$(extract_decimal "$rkey_line")
    bridge_addr=$(extract_hex "$addr_line")

    _info "Bridge QP:     $bridge_qp"
    _info "Bridge RKey:   $bridge_rkey"
    _info "Bridge Buffer: $bridge_addr"

    # ---- 2. Start playback tool ----
    _log "Starting syndrome playback (fpga=$FPGA_IP)"
    local playback_args=(
        --hololink "$FPGA_IP"
        --config "$CONFIG_FILE"
        --syndromes "$SYNDROMES_FILE"
        --function-name nv_qldpc_decode
        --qp-number "$bridge_qp"
        --rkey "$bridge_rkey"
        --buffer-addr "$bridge_addr"
        --page-size "$PAGE_SIZE"
        --num-pages "$NUM_PAGES"
    )
    if $VERIFY; then
        playback_args+=(--verify)
    fi
    if [[ -n "$NUM_SHOTS" ]]; then
        playback_args+=(--num-shots "$NUM_SHOTS")
    fi
    if [[ -n "$SPACING" ]]; then
        playback_args+=(--spacing "$SPACING")
    fi

    local playback_rc=0
    "$PLAYBACK_BIN" "${playback_args[@]}" || playback_rc=$?

    return $playback_rc
}

# ============================================================================
# Main
# ============================================================================

main() {
    _banner "Hololink QLDPC BP Decoder Test (Graph Launch)"

    _info "Decoder: nv-qldpc-decoder (Relay BP, CPU-launched CUDA graph)"
    if $EMULATE; then
        _info "Mode: FPGA Emulation (3-tool)"
    else
        _info "Mode: Real FPGA (2-tool)"
    fi
    echo ""

    # ---- Build ----
    if $DO_BUILD; then
        do_build
    fi

    # ---- Network setup ----
    if $DO_SETUP_NETWORK; then
        do_setup_network
    fi

    # ---- Run ----
    if ! $DO_RUN; then
        _log "Skipping test run (--no-run)"
        return 0
    fi

    resolve_paths

    local rc=0
    if $EMULATE; then
        run_emulated || rc=$?
    else
        run_fpga || rc=$?
    fi

    # ---- Verdict ----
    echo ""
    if [[ $rc -eq 0 ]]; then
        _banner "QLDPC BP DECODE LOOP (GRAPH LAUNCH): PASS"
    else
        _banner "QLDPC BP DECODE LOOP (GRAPH LAUNCH): FAIL"
    fi

    return $rc
}

main
