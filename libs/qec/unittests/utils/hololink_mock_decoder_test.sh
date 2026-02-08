#!/bin/bash
# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         #
# All rights reserved.                                                        #
#                                                                             #
# This source code and the accompanying materials are made available under    #
# the terms of the Apache License 2.0 which accompanies this distribution.   #
# ============================================================================ #
#
# hololink_mock_decoder_test.sh
#
# Orchestration script for end-to-end QEC decode loop testing.
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
#   ./hololink_mock_decoder_test.sh --emulate --build --setup-network
#
#   # Just run with real FPGA (tools already built, network already set up)
#   ./hololink_mock_decoder_test.sh --fpga-ip 192.168.0.2
#
#   # Build only
#   ./hololink_mock_decoder_test.sh --build --no-run
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
HOLOLINK_DIR="/workspaces/cuda-qx/hololink"
CUDA_QUANTUM_DIR="/workspaces/cuda-quantum"
CUDAQX_DIR="/workspaces/cudaqx"
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
# Page size must be >= frame_size (RPCHeader + max(syndrome_size, 256))
# and 128-byte aligned (Hololink PAGE_SIZE=128).  384 safely covers the
# typical frame_size of 268 bytes without requiring adjustment.
PAGE_SIZE=384
NUM_PAGES=64
CONTROL_PORT=8193

# Build parallelism
JOBS=$(nproc 2>/dev/null || echo 8)

# ============================================================================
# Argument Parsing
# ============================================================================

print_usage() {
    cat <<'EOF'
Usage: hololink_mock_decoder_test.sh [options]

Modes:
  --emulate              Use FPGA emulator (3-tool mode, no FPGA needed)
                         Default: FPGA mode (2-tool, requires real FPGA)

Actions:
  --build                Build all required tools before running
  --setup-network        Configure ConnectX network interfaces
  --no-run               Skip running the test (useful with --build)

Build options:
  --hololink-dir DIR     Hololink source directory
                         (default: /workspaces/cuda-qx/hololink)
  --cuda-quantum-dir DIR cuda-quantum source directory
                         (default: /workspaces/cuda-quantum)
  --cudaqx-dir DIR       cudaqx source directory
                         (default: /workspaces/cudaqx)
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
  --no-verify            Skip ILA correction verification (verify is ON by default)
  --num-shots N          Limit number of shots
  --page-size N          Ring buffer slot size in bytes (default: 256)
  --num-pages N          Number of ring buffer slots (default: 64)
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
        --hololink-dir)     HOLOLINK_DIR="$2"; shift ;;
        --cuda-quantum-dir) CUDA_QUANTUM_DIR="$2"; shift ;;
        --cudaqx-dir)       CUDAQX_DIR="$2"; shift ;;
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

# If only --build or --setup-network were requested (and --no-run wasn't
# explicitly given), infer that run is wanted only when neither action flag
# is the *sole* action.
if $DO_BUILD || $DO_SETUP_NETWORK; then
    # If user didn't explicitly pass --no-run, we still run.
    # --no-run is the only way to suppress running.
    :
fi

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
            # Give it a moment to exit gracefully
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

# Detect ConnectX interfaces from ibdev2netdev.
# Returns lines like: "mlx5_0 port 1 ==> enp1s0f0np0 (Up)"
detect_interfaces() {
    if ! command -v ibdev2netdev &>/dev/null; then
        _err "ibdev2netdev not found. Install rdma-core or Mellanox OFED."
        return 1
    fi
    ibdev2netdev
}

# Given an IB device name (e.g. mlx5_0), return the network interface name.
ib_to_netdev() {
    local ib_dev="$1"
    local port="${2:-1}"
    ibdev2netdev | awk -v dev="$ib_dev" -v p="$port" \
        '$1 == dev && $3 == p { print $5 }'
}

# Get the IB device name for a given network interface.
netdev_to_ib() {
    local iface="$1"
    ibdev2netdev | awk -v iface="$iface" '$5 == iface { print $1 }'
}

# Configure a single ConnectX port.
setup_port() {
    local iface="$1"
    local ip="$2"
    local mtu="$3"
    local ib_dev

    _info "Configuring $iface: ip=$ip mtu=$mtu"

    # Bring link up
    sudo ip link set "$iface" up

    # Set MTU
    sudo ip link set "$iface" mtu "$mtu"

    # Flush existing addresses and assign new one
    sudo ip addr flush dev "$iface"
    sudo ip addr add "${ip}/24" dev "$iface"

    # Configure RoCEv2 mode if rdma tool is available
    ib_dev=$(netdev_to_ib "$iface")
    if [[ -n "$ib_dev" ]] && command -v rdma &>/dev/null; then
        # Set default GID type to RoCEv2 on all ports
        local port_count
        port_count=$(ls -d "/sys/class/infiniband/${ib_dev}/ports/"* 2>/dev/null | wc -l)
        for p in $(seq 1 "$port_count"); do
            sudo rdma link set "${ib_dev}/${p}" type eth || true
        done
        _info "  RoCEv2 mode configured for $ib_dev"
    fi

    # Set DSCP trust mode if mlnx_qos is available (for lossless RoCE)
    if command -v mlnx_qos &>/dev/null; then
        sudo mlnx_qos -i "$iface" --trust=dscp 2>/dev/null || true
        _info "  DSCP trust mode set"
    fi

    # Disable adaptive RX coalescing for low latency
    if command -v ethtool &>/dev/null; then
        sudo ethtool -C "$iface" adaptive-rx off rx-usecs 0 2>/dev/null || true
    fi

    _info "  Done: $iface is up at $ip"
}

# Add a static ARP entry: _add_static_arp <local_iface> <remote_ip> <remote_iface>
# Reads the MAC of remote_iface and adds it as a permanent neighbor entry.
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
        # Need two ports: one for bridge, one for emulator.
        # Auto-detect if no device specified.
        local interfaces
        interfaces=$(detect_interfaces)

        if [[ -z "$IB_DEVICE" ]]; then
            # Find the first IB device with two ports, or the first two devices.
            local iface_bridge iface_emulator
            local first_dev second_dev first_iface second_iface

            first_dev=$(echo "$interfaces" | head -1 | awk '{print $1}')
            first_iface=$(echo "$interfaces" | head -1 | awk '{print $5}')

            # Check if same device has port 2
            second_iface=$(echo "$interfaces" | awk -v d="$first_dev" \
                '$1 == d && $3 == 2 {print $5}')

            if [[ -n "$second_iface" ]]; then
                iface_bridge="$first_iface"
                iface_emulator="$second_iface"
            else
                # Use second device's port 1
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

            # Derive IB device names for the tools (ibv_open_device needs the
            # IB device name like roceP2p1s0f0, not the netdev name).
            BRIDGE_DEVICE=$(netdev_to_ib "$iface_bridge")
            EMULATOR_DEVICE=$(netdev_to_ib "$iface_emulator")

            # Add static ARP entries so the two loopback ports can reach each
            # other (normal ARP may be filtered by Linux on same-host ports).
            _add_static_arp "$iface_bridge" "$EMULATOR_IP" "$iface_emulator"
            _add_static_arp "$iface_emulator" "$BRIDGE_IP" "$iface_bridge"
        else
            # User specified a device -- assume port 1 = bridge, port 2 = emulator
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

        # Allow GID tables to populate after IP assignment
        _info "Waiting 2s for GID tables to populate..."
        sleep 2
    else
        # FPGA mode: only one port needed (for the bridge tool).
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

do_build() {
    _log "Building all tools (jobs=$JOBS)"

    local cudaqx_build="${CUDAQX_DIR}/build"
    local cq_build="${CUDA_QUANTUM_DIR}/realtime/build"
    local hl_build="${HOLOLINK_DIR}/build"

    # ---- Stage 1: cuda-quantum/realtime ----
    _banner "Stage 1/3: Building cuda-quantum/realtime"
    local cq_src="${CUDA_QUANTUM_DIR}/realtime"
    if [[ ! -d "$cq_src" ]]; then
        _err "cuda-quantum realtime source not found at $cq_src"
        return 1
    fi

    cmake -G Ninja -S "$cq_src" -B "$cq_build" \
        -DCMAKE_BUILD_TYPE=Release \
        2>&1 | tail -5
    cmake --build "$cq_build" -j "$JOBS" 2>&1 | tail -5
    _info "cuda-quantum/realtime built: $cq_build/lib/"

    # ---- Stage 2: hololink ----
    _banner "Stage 2/3: Building hololink"
    if [[ ! -d "$HOLOLINK_DIR" ]]; then
        _err "Hololink source not found at $HOLOLINK_DIR"
        return 1
    fi

    # Determine target architecture
    local target_arch="amd64"
    if [[ "$(uname -m)" == "aarch64" ]]; then
        target_arch="arm64"
    fi

    cmake -G Ninja -S "$HOLOLINK_DIR" -B "$hl_build" \
        -DCMAKE_BUILD_TYPE=Release \
        -DTARGETARCH="$target_arch" \
        -DHOLOLINK_BUILD_ONLY_NATIVE=OFF \
        -DHOLOLINK_BUILD_PYTHON=OFF \
        -DHOLOLINK_BUILD_TESTS=OFF \
        -DHOLOLINK_BUILD_TOOLS=OFF \
        -DHOLOLINK_BUILD_EXAMPLES=OFF \
        -DHOLOLINK_BUILD_EMULATOR=OFF \
        2>&1 | tail -5
    cmake --build "$hl_build" -j "$JOBS" \
        --target gpu_roce_transceiver hololink_core 2>&1 | tail -5
    _info "hololink built: $hl_build/"

    # ---- Stage 3: cudaqx tools ----
    _banner "Stage 3/3: Building cudaqx tools"
    if [[ ! -d "$CUDAQX_DIR" ]]; then
        _err "cudaqx source not found at $CUDAQX_DIR"
        return 1
    fi

    cmake -G Ninja -S "$CUDAQX_DIR" -B "$cudaqx_build" \
        -DCUDAQX_QEC_ENABLE_HOLOLINK_TOOLS=ON \
        -DHOLOSCAN_SENSOR_BRIDGE_SOURCE_DIR="$HOLOLINK_DIR" \
        -DHOLOSCAN_SENSOR_BRIDGE_BUILD_DIR="$hl_build" \
        -DCUDAQ_REALTIME_INCLUDE_DIR="${CUDA_QUANTUM_DIR}/realtime/include" \
        -DCUDAQ_REALTIME_LIBRARY="${cq_build}/lib/libcudaq-realtime.so" \
        -DCUDAQ_REALTIME_DISPATCH_LIBRARY="${cq_build}/lib/libcudaq-realtime-dispatch.a" \
        -DCUDAQX_QEC_INCLUDE_TESTS=ON \
        2>&1 | tail -5
    cmake --build "$cudaqx_build" -j "$JOBS" \
        --target hololink_fpga_emulator \
                 hololink_fpga_syndrome_playback \
                 hololink_mock_decoder_bridge \
        2>&1 | tail -5
    _info "cudaqx tools built: $cudaqx_build/libs/qec/unittests/utils/"

    _banner "Build complete"
}

# ============================================================================
# Tool Path Resolution
# ============================================================================

resolve_paths() {
    local build_dir="${CUDAQX_DIR}/build/libs/qec/unittests/utils"

    BRIDGE_BIN="${build_dir}/hololink_mock_decoder_bridge"
    PLAYBACK_BIN="${build_dir}/hololink_fpga_syndrome_playback"
    EMULATOR_BIN="${build_dir}/hololink_fpga_emulator"

    # Auto-detect data directory
    if [[ -z "$DATA_DIR" ]]; then
        DATA_DIR="${CUDAQX_DIR}/libs/qec/unittests/decoders/realtime/data"
    fi

    CONFIG_FILE="${DATA_DIR}/config_multi_err_lut.yml"
    SYNDROMES_FILE="${DATA_DIR}/syndromes_multi_err_lut.txt"

    # Verify binaries exist
    if [[ ! -x "$BRIDGE_BIN" ]]; then
        _err "Bridge binary not found: $BRIDGE_BIN"
        _err "Run with --build to build the tools first."
        return 1
    fi
    if [[ ! -x "$PLAYBACK_BIN" ]]; then
        _err "Playback binary not found: $PLAYBACK_BIN"
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

    # Default IB device names (if network setup didn't set them)
    : "${BRIDGE_DEVICE:=rocep1s0f0}"
    if $EMULATE; then
        : "${EMULATOR_DEVICE:=rocep1s0f1}"
    fi
}

# ============================================================================
# Output Parsing Helpers
# ============================================================================

# Wait for a specific pattern in a log file, with timeout.
# Returns the matched line on stdout.
wait_for_pattern() {
    local logfile="$1"
    local pattern="$2"
    local timeout_sec="$3"
    local pid_to_check="${4:-}"   # optional: exit early if this PID dies

    local elapsed=0
    while (( elapsed < timeout_sec )); do
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
        elapsed=$((elapsed + 1))  # approximate (0.5s per iteration)
    done
    _err "Timeout waiting for pattern: $pattern"
    return 1
}

# Extract a hex value from a line like "  QP Number: 0xABC"
extract_hex() {
    local line="$1"
    echo "$line" | grep -oP '0x[0-9a-fA-F]+' | head -1
}

# Extract a decimal value from a line like "  RKey: 12345"
extract_decimal() {
    local line="$1"
    echo "$line" | awk -F': ' '{print $NF}' | tr -d ' '
}

# ============================================================================
# Run: Emulated Mode (3 tools)
# ============================================================================

run_emulated() {
    _banner "QEC Decode Loop Test (Emulated FPGA)"

    local emu_log bridge_log
    emu_log=$(mktemp /tmp/qec_emulator.XXXXXX.log)
    bridge_log=$(mktemp /tmp/qec_bridge.XXXXXX.log)
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

    # Wait for emulator to print its QP number
    local emu_qp_line
    emu_qp_line=$(wait_for_pattern "$emu_log" "Emulator QP:" 30 "$emu_pid") || {
        _err "Failed to get emulator QP number"
        return 1
    }
    local emu_qp
    emu_qp=$(extract_hex "$emu_qp_line")
    _info "Emulator QP: $emu_qp"

    # ---- 2. Start bridge tool ----
    _log "Starting mock decoder bridge (remote-qp=$emu_qp)"
    "$BRIDGE_BIN" \
        --device="$BRIDGE_DEVICE" \
        --peer-ip="$EMULATOR_IP" \
        --remote-qp="$emu_qp" \
        --gpu="$GPU_ID" \
        --config="$CONFIG_FILE" \
        --syndromes="$SYNDROMES_FILE" \
        --timeout="$TIMEOUT" \
        --page-size="$PAGE_SIZE" \
        --num-pages="$NUM_PAGES" \
        > >(tee "$bridge_log") 2>&1 &
    local bridge_pid=$!
    PIDS_TO_KILL+=("$bridge_pid")
    _info "Bridge PID: $bridge_pid"

    # Wait for "Bridge Ready" and extract QP/RKEY/buffer
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

    # ---- 3. Start playback tool ----
    _log "Starting syndrome playback (control-port=$CONTROL_PORT)"
    local playback_args=(
        --hololink "$EMULATOR_IP"
        --control-port "$CONTROL_PORT"
        --data-dir "$DATA_DIR"
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

    local playback_rc=0
    "$PLAYBACK_BIN" "${playback_args[@]}" || playback_rc=$?

    return $playback_rc
}

# ============================================================================
# Run: FPGA Mode (2 tools)
# ============================================================================

run_fpga() {
    _banner "QEC Decode Loop Test (Real FPGA)"

    local bridge_log
    bridge_log=$(mktemp /tmp/qec_bridge.XXXXXX.log)
    TEMP_FILES+=("$bridge_log")

    # ---- 1. Start bridge tool ----
    _log "Starting mock decoder bridge (remote-qp=0x2, fpga-ip=$FPGA_IP)"
    "$BRIDGE_BIN" \
        --device="$BRIDGE_DEVICE" \
        --peer-ip="$FPGA_IP" \
        --remote-qp=0x2 \
        --gpu="$GPU_ID" \
        --config="$CONFIG_FILE" \
        --syndromes="$SYNDROMES_FILE" \
        --timeout="$TIMEOUT" \
        --page-size="$PAGE_SIZE" \
        --num-pages="$NUM_PAGES" \
        > >(tee "$bridge_log") 2>&1 &
    local bridge_pid=$!
    PIDS_TO_KILL+=("$bridge_pid")
    _info "Bridge PID: $bridge_pid"

    # Wait for "Bridge Ready" and extract QP/RKEY/buffer
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
        --data-dir "$DATA_DIR"
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

    local playback_rc=0
    "$PLAYBACK_BIN" "${playback_args[@]}" || playback_rc=$?

    return $playback_rc
}

# ============================================================================
# Main
# ============================================================================

main() {
    _banner "Hololink Mock Decoder Test"

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
        _banner "QEC DECODE LOOP: PASS"
    else
        _banner "QEC DECODE LOOP: FAIL"
    fi

    return $rc
}

main
