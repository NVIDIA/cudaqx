#!/bin/bash
# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
#
# hsb_fpga_decoding_server_test.sh
#
# Generic orchestration script for end-to-end decoder testing over Holoscan-
# Sensor-Bridge (HSB) RDMA/RoCE, with the decode work served by the standalone
# decoding server (decoding_server) on the CPU HOST_CALL path instead of a
# GPU bridge.  Works with any CPU decoder the server's YAML config selects;
# the default profile is pymatching.
#
# Data path:
#   FPGA/emulator --RDMA WRITE--> server cpu_roce rx ring
#   server        --RDMA SEND---> FPGA SIF TX (captured by the ILA)
#
# Division of labor (identical to hololink_qldpc_graph_decoder_test.sh):
#   - decoding_server (--transport=cpu_roce --qp_config=hsb_fpga) owns the
#     RDMA ring and prints its QP / RKey / Buffer Addr handshake.  It performs
#     NO Hololink control-plane traffic.
#   - hololink_fpga_syndrome_playback is the sole FPGA control-plane writer:
#     it programs the SIF RDMA target with the server's handshake values,
#     writes the syndrome frames to BRAM, arms the ILA, enables the player,
#     and verifies the captured RPC responses.
#
# Modes:
#   Default (FPGA):   server + playback  (requires real FPGA)
#   --emulate:        emulator + server + playback  (no FPGA needed)
#
# Actions (can be combined):
#   --build            Build all required tools
#   --setup-network    Configure ConnectX interfaces
#   (run is implicit unless only --build / --setup-network are given)
#
# Examples:
#   # Full emulated test: build, configure network, run
#   ./hsb_fpga_decoding_server_test.sh --emulate --build --setup-network
#
#   # Just run (tools already built, network already set up)
#   ./hsb_fpga_decoding_server_test.sh --emulate
#
#   # Real FPGA
#   ./hsb_fpga_decoding_server_test.sh --setup-network --device rocep1s0f0 \
#       --bridge-ip 192.168.0.1 --fpga-ip 192.168.0.2
set -euo pipefail

# ============================================================================
# Defaults
# ============================================================================

EMULATE=false
DO_BUILD=false
DO_SETUP_NETWORK=false
DO_RUN=true
VERIFY=true

# Directory defaults.  HSB_DIR is used as-is (already on the correct branch);
# CUDA_QUANTUM_DIR should be checked out at the ref in CUDAQX_DIR/.cudaq_version
# (verified with a warning in do_build).
HSB_DIR="/workspaces/holoscan-sensor-bridge"
CUDA_QUANTUM_DIR="/workspaces/cuda-quantum"
CUDAQX_DIR="/workspaces/cudaqx"
DATA_DIR=""  # empty => generate data files (see resolve_data_files)

# Decoder profile.  By default the config + syndromes files are GENERATED
# fresh each run by the surface_code-4-yaml binary into GEN_DIR (they are
# derived artifacts and not checked in); --config/--syndromes or --data-dir
# switch to pre-made files and skip generation.
DECODER="pymatching"
CONFIG_FILE=""
SYNDROMES_FILE=""

# Data-generation parameters (surface-code memory experiment).  The
# generator's RNG seed is fixed, so runs are reproducible.
GEN_DISTANCE=3
GEN_ROUNDS=4
GEN_P_SPAM=0.01
GEN_SHOTS=100

# Network defaults
IB_DEVICE=""           # auto-detect
BRIDGE_IP="10.0.0.1"   # server-side NIC IP (kept the qldpc script's name)
EMULATOR_IP="10.0.0.2"
FPGA_IP="192.168.0.2"
MTU=4096

# Run defaults
TIMEOUT=60
NUM_SHOTS=""
PAGE_SIZE=384
# Ring depth is intentionally NOT configurable: stock HSB posts WQE_NUM=64
# receive/send WQEs, so a deeper ring aliases two slots per WQE and races
# RX/TX.  The server clamps to 64 as well.
NUM_SLOTS=64
# TX SGE bytes for the server's SEND responses.  RPCResponse (24B) + a
# bit-packed correction byte fits well inside 64, keeping every response a
# single 512-bit ILA beat.
FRAME_SIZE=64
SPACING=""
CONTROL_PORT=8193

# Build parallelism
JOBS=$(nproc 2>/dev/null || echo 8)

# ============================================================================
# Argument Parsing
# ============================================================================

print_usage() {
    cat <<'EOF'
Usage: hsb_fpga_decoding_server_test.sh [options]

Generic orchestration script for decoder end-to-end testing over HSB
RDMA/RoCE with the decoding server (decoding_server) on the CPU
HOST_CALL path.  Default decoder profile: pymatching.

Modes:
  --emulate              Use FPGA emulator (3-tool mode, no FPGA needed)
                         Default: FPGA mode (2-tool, requires real FPGA)

Actions:
  --build                Build all required tools before running
  --setup-network        Configure ConnectX network interfaces
  --no-run               Skip running the test (useful with --build)

Decoder options:
  --decoder NAME         Decoder profile (default: pymatching).  By default the
                         config/syndromes files are generated fresh each run by
                         surface_code-4-yaml into CUDAQX_DIR/build/hsb_fpga_test_data
  --config PATH          Use a pre-made decoding-server YAML config (skips generation)
  --syndromes PATH       Use a pre-made syndromes text file (skips generation)
  --data-dir DIR         Use pre-made DIR/config_NAME.yml + DIR/syndromes_NAME.txt
                         (skips generation)

Data-generation options (ignored when --config/--syndromes/--data-dir given):
  --distance N           Surface-code distance (default: 3)
  --num-rounds N         Measurement rounds (default: 4)
  --p-spam F             SPAM error probability (default: 0.01)
  --gen-shots N          Shots to generate in the syndromes file (default: 100)

Build options:
  --hsb-dir DIR          holoscan-sensor-bridge source directory
                         (default: /workspaces/holoscan-sensor-bridge)
  --cuda-quantum-dir DIR cuda-quantum source directory; must match the ref in
                         CUDAQX_DIR/.cudaq_version (default: /workspaces/cuda-quantum)
  --cudaqx-dir DIR       cudaqx source dir that builds the server + playback
                         (default: /workspaces/cudaqx)
  --jobs N               Parallel build jobs (default: nproc)

Network options:
  --device DEV           ConnectX IB device name (default: auto-detect)
  --bridge-ip ADDR       Server-side NIC IP (default: 10.0.0.1)
  --emulator-ip ADDR     Emulator IP (default: 10.0.0.2)
  --fpga-ip ADDR         FPGA IP for non-emulate mode (default: 192.168.0.2)
  --mtu N                MTU size (default: 4096)

Run options:
  --timeout N            Server timeout in seconds (default: 60)
  --no-verify            Skip correction verification
  --num-shots N          Limit number of shots
  --page-size N          Ring buffer slot size in bytes (default: 384)
  --frame-size N         Server TX SGE bytes (default: 64)
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
        --decoder)          DECODER="$2"; shift ;;
        --config)           CONFIG_FILE="$2"; shift ;;
        --syndromes)        SYNDROMES_FILE="$2"; shift ;;
        --data-dir)         DATA_DIR="$2"; shift ;;
        --distance)         GEN_DISTANCE="$2"; shift ;;
        --num-rounds)       GEN_ROUNDS="$2"; shift ;;
        --p-spam)           GEN_P_SPAM="$2"; shift ;;
        --gen-shots)        GEN_SHOTS="$2"; shift ;;
        --hsb-dir)          HSB_DIR="$2"; shift ;;
        --cuda-quantum-dir) CUDA_QUANTUM_DIR="$2"; shift ;;
        --cudaqx-dir)       CUDAQX_DIR="$2"; shift ;;
        --jobs)             JOBS="$2"; shift ;;
        --device)           IB_DEVICE="$2"; shift ;;
        --bridge-ip)        BRIDGE_IP="$2"; shift ;;
        --emulator-ip)      EMULATOR_IP="$2"; shift ;;
        --fpga-ip)          FPGA_IP="$2"; shift ;;
        --mtu)              MTU="$2"; shift ;;
        --timeout)          TIMEOUT="$2"; shift ;;
        --num-shots)        NUM_SHOTS="$2"; shift ;;
        --page-size)        PAGE_SIZE="$2"; shift ;;
        --frame-size)       FRAME_SIZE="$2"; shift ;;
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
# Network Setup (mirrors hololink_qldpc_graph_decoder_test.sh)
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

# Pre-seed a PERMANENT neighbor entry for a real FPGA on the server interface.
# The server's QP connect resolves the FPGA's L2 (MAC) address via
# ibv_create_ah, which consults the kernel neighbor table; without priming it
# the in-call ARP resolution times out even though the link is up.
_seed_fpga_neighbor() {
    local iface="$1" fpga_ip="$2"
    ping -c 3 -W 1 -I "$iface" "$fpga_ip" >/dev/null 2>&1 || true
    local mac
    mac=$(ip neigh show "$fpga_ip" dev "$iface" 2>/dev/null \
          | awk '{for (i = 1; i <= NF; i++) if ($i == "lladdr") print $(i + 1)}' \
          | head -1)
    if [[ -n "$mac" ]]; then
        sudo ip neigh replace "$fpga_ip" lladdr "$mac" nud permanent dev "$iface"
        _info "  Static ARP: $fpga_ip -> $mac on $iface"
    else
        _err "  Could not resolve FPGA MAC for $fpga_ip on $iface."
        _err "  Check the FPGA is cabled to this NIC, powered, and reachable"
        _err "  (ping $fpga_ip); otherwise the server QP connect will time out."
    fi
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

# Convert an IPv4 address to the trailing groups of its IPv4-mapped RoCE v2
# GID, e.g. 10.0.0.1 -> "ffff:0a00:0001".
ipv4_to_gid_suffix() {
    local o1 o2 o3 o4
    IFS='.' read -r o1 o2 o3 o4 <<< "$1"
    printf "ffff:%02x%02x:%02x%02x" "$o1" "$o2" "$o3" "$o4"
}

# Poll until the IPv4-mapped RoCE v2 GID for $ip appears on $ib_dev port 1.
# The CPU RoCE transceiver requires this specific GID; it only exists while
# the netdev has the IPv4 address AND is up, and it populates asynchronously,
# so a blind sleep races the server's GID lookup.
wait_for_roce_v2_gid() {
    local ib_dev="$1" ip="$2" timeout_s="${3:-15}"
    local suffix gids_dir types_dir elapsed=0
    suffix=$(ipv4_to_gid_suffix "$ip")
    gids_dir="/sys/class/infiniband/${ib_dev}/ports/1/gids"
    types_dir="/sys/class/infiniband/${ib_dev}/ports/1/gid_attrs/types"
    if [[ ! -d "$gids_dir" ]]; then
        _info "  (no GID sysfs for $ib_dev; skipping GID wait)"
        return 0
    fi
    while (( elapsed < timeout_s * 10 )); do
        local g idx gid t
        for g in "$gids_dir"/*; do
            idx=$(basename "$g")
            gid=$(cat "$g" 2>/dev/null)
            if [[ "$gid" == *":${suffix}" ]]; then
                t=$(cat "${types_dir}/${idx}" 2>/dev/null)
                if [[ "$t" == *"RoCE v2"* ]]; then
                    _info "  RoCE v2 GID ready: ${ib_dev}[${idx}] ${gid}"
                    return 0
                fi
            fi
        done
        sleep 0.1
        elapsed=$((elapsed + 1))
    done
    _err "Timed out waiting for IPv4 RoCE v2 GID (${suffix}) on ${ib_dev}."
    _err "The server's cpu_roce bring-up will fail its GID lookup.  Verify"
    _err "${ip} is assigned to the server netdev and the interface is up."
    return 1
}

do_setup_network() {
    _log "Setting up ConnectX network"

    if $EMULATE; then
        local interfaces
        interfaces=$(detect_interfaces)

        if [[ -z "$IB_DEVICE" ]]; then
            local iface_bridge iface_emulator
            local first_dev first_iface second_iface

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

            _info "Server interface:   $iface_bridge"
            _info "Emulator interface: $iface_emulator"
            setup_port "$iface_bridge" "$BRIDGE_IP" "$MTU"
            setup_port "$iface_emulator" "$EMULATOR_IP" "$MTU"

            BRIDGE_DEVICE=$(netdev_to_ib "$iface_bridge")
            EMULATOR_DEVICE=$(netdev_to_ib "$iface_emulator")

            _add_static_arp "$iface_bridge" "$EMULATOR_IP" "$iface_emulator"
            _add_static_arp "$iface_emulator" "$BRIDGE_IP" "$iface_bridge"
        else
            # --device accepts either one dual-port device (ports 1+2) or two
            # comma-separated single-port devices ("devA,devB": server on devA,
            # emulator on devB).
            local iface1 iface2
            if [[ "$IB_DEVICE" == *,* ]]; then
                local dev1="${IB_DEVICE%%,*}" dev2="${IB_DEVICE#*,}"
                iface1=$(ib_to_netdev "$dev1" 1)
                iface2=$(ib_to_netdev "$dev2" 1)
                if [[ -z "$iface1" || -z "$iface2" ]]; then
                    _err "Cannot resolve netdevs for devices $dev1 / $dev2"
                    return 1
                fi
            else
                iface1=$(ib_to_netdev "$IB_DEVICE" 1)
                iface2=$(ib_to_netdev "$IB_DEVICE" 2)
                if [[ -z "$iface1" || -z "$iface2" ]]; then
                    _err "Cannot find two ports on device $IB_DEVICE"
                    return 1
                fi
            fi
            setup_port "$iface1" "$BRIDGE_IP" "$MTU"
            setup_port "$iface2" "$EMULATOR_IP" "$MTU"
            BRIDGE_DEVICE=$(netdev_to_ib "$iface1")
            EMULATOR_DEVICE=$(netdev_to_ib "$iface2")

            _add_static_arp "$iface1" "$EMULATOR_IP" "$iface2"
            _add_static_arp "$iface2" "$BRIDGE_IP" "$iface1"
        fi

        # Wait for the server device's IPv4 RoCE v2 GID before proceeding so
        # the server's cpu_roce GID lookup can't race GID-table population.
        wait_for_roce_v2_gid "$BRIDGE_DEVICE" "$BRIDGE_IP" 15 || true
    else
        local iface_bridge
        if [[ -n "$IB_DEVICE" ]]; then
            iface_bridge=$(ib_to_netdev "$IB_DEVICE" 1)
        else
            iface_bridge=$(detect_interfaces | head -1 | awk '{print $5}')
        fi

        if [[ -z "$iface_bridge" ]]; then
            _err "Cannot detect ConnectX interface for the server."
            return 1
        fi

        _info "Server interface: $iface_bridge"
        setup_port "$iface_bridge" "$BRIDGE_IP" "$MTU"
        BRIDGE_DEVICE=$(netdev_to_ib "$iface_bridge")

        wait_for_roce_v2_gid "$BRIDGE_DEVICE" "$BRIDGE_IP" 15 || true

        # Pre-seed the FPGA's neighbor entry so the server QP connect can
        # resolve its MAC immediately.
        _seed_fpga_neighbor "$iface_bridge" "$FPGA_IP"
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

    local cudaqx_build="${CUDAQX_DIR}/build"
    local cq_build="${CUDA_QUANTUM_DIR}/realtime/build"
    local hsb_build="${HSB_DIR}/build"

    # cuda-quantum must be at the ref cudaqx pins (matches cudaqx CI).
    local pinned_ref current_ref
    pinned_ref=$(jq -r '.cudaq.ref' "${CUDAQX_DIR}/.cudaq_version" 2>/dev/null || true)
    current_ref=$(git -C "$CUDA_QUANTUM_DIR" rev-parse HEAD 2>/dev/null || true)
    if [[ -n "$pinned_ref" && -n "$current_ref" && "$pinned_ref" != "$current_ref" ]]; then
        _err "cuda-quantum checkout ($current_ref)"
        _err "does not match the cudaqx pin  ($pinned_ref)"
        _err "from ${CUDAQX_DIR}/.cudaq_version.  Continuing, but the realtime"
        _err "libraries may be ABI-skewed against this cudaqx tree."
    elif [[ -n "$pinned_ref" ]]; then
        _info "cuda-quantum at the cudaqx pin: $pinned_ref"
    fi

    local cuda_arch
    cuda_arch=$(detect_cuda_arch)
    local cuda_arch_flag=""
    if [ -n "$cuda_arch" ]; then
        cuda_arch_flag="-DCMAKE_CUDA_ARCHITECTURES=$cuda_arch"
        _info "CUDA arch: $cuda_arch"
    fi

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

    # ---- Stage 2: holoscan-sensor-bridge (hololink_core for playback,
    # plus the emulator via a cuda-quantum reconfigure) ----
    _banner "Stage 2/3: Building holoscan-sensor-bridge (hololink_core)"
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
    if "$hsb_cmake" --help 2>/dev/null | grep -q -- "--fresh"; then
        "$hsb_cmake" --fresh "${hsb_common_args[@]}" 2>&1 | tail -5
    else
        rm -f "$hsb_build/CMakeCache.txt"
        rm -rf "$hsb_build/CMakeFiles"
        "$hsb_cmake" "${hsb_common_args[@]}" 2>&1 | tail -5
    fi

    "$hsb_cmake" --build "$hsb_build" -j "$JOBS" \
        --target hololink_core 2>&1 | tail -5
    _info "holoscan-sensor-bridge built: $hsb_build/"

    if $EMULATE; then
        # Reconfigure cuda-quantum/realtime with hololink tools (for the
        # emulator binary only; the server itself has no HSB dependency).
        cmake -G Ninja -S "$cq_src" -B "$cq_build" \
            -DCMAKE_BUILD_TYPE=Release \
            $cuda_arch_flag \
            -DCUDAQ_REALTIME_ENABLE_HOLOLINK_TOOLS=ON \
            -DHOLOSCAN_SENSOR_BRIDGE_SOURCE_DIR="$HSB_DIR" \
            -DHOLOSCAN_SENSOR_BRIDGE_BUILD_DIR="$hsb_build" \
            2>&1 | tail -5
        cmake --build "$cq_build" -j "$JOBS" \
            --target hololink_fpga_emulator 2>&1 | tail -5
    fi

    # ---- Stage 3: cudaqx decoding server + playback + decoder plugin ----
    _banner "Stage 3/3: Building cudaqx server + playback"
    if [[ ! -d "$CUDAQX_DIR" ]]; then
        _err "cudaqx source not found at $CUDAQX_DIR"
        return 1
    fi

    # The server's CMake locates the cpu_transport archives via explicit cache
    # entries (its find_library does not search the cuda-quantum build tree).
    cmake -G Ninja -S "$CUDAQX_DIR" -B "$cudaqx_build" \
        $cuda_arch_flag \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_COMPILER="$cuda_compiler" \
        -DCUDAToolkit_ROOT="$cuda_toolkit_root" \
        -DCUDAQX_QEC_ENABLE_HOLOLINK_TOOLS=ON \
        -DCUDAQ_QEC_BUILD_TRT_DECODER=OFF \
        -DHOLOSCAN_SENSOR_BRIDGE_SOURCE_DIR="$HSB_DIR" \
        -DHOLOSCAN_SENSOR_BRIDGE_BUILD_DIR="$hsb_build" \
        -DCUDAQ_REALTIME_ROOT="${CUDA_QUANTUM_DIR}/realtime" \
        -DCUDAQ_REALTIME_INCLUDE_DIR="${CUDA_QUANTUM_DIR}/realtime/include" \
        -DCUDAQ_REALTIME_LIBRARY="${cq_build}/lib/libcudaq-realtime.so" \
        -DCUDAQ_REALTIME_DISPATCH_LIBRARY="${cq_build}/lib/libcudaq-realtime-dispatch.a" \
        -DCUDAQ_REALTIME_HOST_DISPATCH_LIBRARY="${cq_build}/lib/libcudaq-realtime-host-dispatch.a" \
        -DQEC_UDP_TRANSPORT_LIBRARY="${cq_build}/lib/libcudaq-realtime-udp-transport.a" \
        -DQEC_CPU_ROCE_TRANSPORT_LIBRARY="${cq_build}/lib/libcudaq-realtime-cpu-roce-transport.a" \
        -DQEC_UDP_REALTIME_LIBRARY="${cq_build}/lib/libcudaq-realtime.so" \
        -DQEC_HOST_DISPATCH_LIBRARY="${cq_build}/lib/libcudaq-realtime-host-dispatch.a" \
        -DCUDAQ_INSTALL_PREFIX="${CUDAQ_INSTALL_PREFIX:-/usr/local/cudaq}" \
        -DCUDAQ_DIR="${CUDAQ_INSTALL_PREFIX:-/usr/local/cudaq}/lib/cmake/cudaq" \
        2>&1 | tail -5

    cmake --build "$cudaqx_build" -j "$JOBS" \
        --target decoding_server \
                 hololink_fpga_syndrome_playback \
                 cudaq-qec-pymatching \
                 surface_code-4-yaml \
        2>&1 | tail -5
    _info "cudaqx tools built: $cudaqx_build/bin + $cudaqx_build/libs/qec/unittests/utils/"

    _banner "Build complete"
}

# ============================================================================
# Tool Path Resolution
# ============================================================================

# Decide where the config + syndromes files come from.  Pre-made files
# (--config/--syndromes or --data-dir) win and skip generation; otherwise the
# files are generated into GEN_DIR by generate_data_files().
GENERATE_DATA=false
resolve_data_files() {
    GEN_DIR="${CUDAQX_DIR}/build/hsb_fpga_test_data"

    if [[ -n "$DATA_DIR" ]]; then
        # Pre-made profile directory (e.g. checked-in data for some decoder).
        CONFIG_FILE="${CONFIG_FILE:-${DATA_DIR}/config_${DECODER}.yml}"
        SYNDROMES_FILE="${SYNDROMES_FILE:-${DATA_DIR}/syndromes_${DECODER}.txt}"
        return 0
    fi
    if [[ -n "$CONFIG_FILE" && -n "$SYNDROMES_FILE" ]]; then
        return 0
    fi
    if [[ -n "$CONFIG_FILE" || -n "$SYNDROMES_FILE" ]]; then
        _err "--config and --syndromes must be given together (or use --data-dir)."
        return 1
    fi

    # Default: generate both files fresh this run.
    GENERATE_DATA=true
    CONFIG_FILE="${GEN_DIR}/config_${DECODER}.yml"
    SYNDROMES_FILE="${GEN_DIR}/syndromes_${DECODER}.txt"
}

# Generate the decoder config (DEM + decoder_custom_args) and the syndromes
# file with the surface_code-4-yaml memory-experiment binary.  Runs in GEN_DIR
# so the generator's auxiliary outputs land there too.
generate_data_files() {
    GENERATOR_BIN="${CUDAQX_DIR}/build/libs/qec/unittests/realtime/app_examples/surface_code-4-yaml"
    if [[ ! -x "$GENERATOR_BIN" ]]; then
        _err "Data generator not found: $GENERATOR_BIN"
        _err "Run with --build to build the tools first."
        return 1
    fi

    _log "Generating test data (decoder=$DECODER, distance=$GEN_DISTANCE," \
         "rounds=$GEN_ROUNDS, p_spam=$GEN_P_SPAM, shots=$GEN_SHOTS)"
    mkdir -p "$GEN_DIR"

    local gen_ld_path
    gen_ld_path="${CUDA_QUANTUM_DIR}/realtime/build/lib:${CUDAQX_DIR}/build/lib"

    _info "$GENERATOR_BIN --distance $GEN_DISTANCE --num_rounds $GEN_ROUNDS" \
          "--p_spam $GEN_P_SPAM --decoder_type $DECODER --save_dem $(basename "$CONFIG_FILE")"
    (cd "$GEN_DIR" && \
     LD_LIBRARY_PATH="${gen_ld_path}:${LD_LIBRARY_PATH:-}" \
     "$GENERATOR_BIN" \
        --distance "$GEN_DISTANCE" \
        --num_rounds "$GEN_ROUNDS" \
        --p_spam "$GEN_P_SPAM" \
        --decoder_type "$DECODER" \
        --save_dem "$(basename "$CONFIG_FILE")" > gen_config.log 2>&1) || {
        _err "Config generation failed; see ${GEN_DIR}/gen_config.log"
        tail -5 "${GEN_DIR}/gen_config.log" >&2 || true
        return 1
    }

    _info "$GENERATOR_BIN --distance $GEN_DISTANCE --num_rounds $GEN_ROUNDS" \
          "--p_spam $GEN_P_SPAM --num_shots $GEN_SHOTS --yaml $(basename "$CONFIG_FILE")" \
          "--save_syndrome $(basename "$SYNDROMES_FILE")"
    (cd "$GEN_DIR" && \
     LD_LIBRARY_PATH="${gen_ld_path}:${LD_LIBRARY_PATH:-}" \
     "$GENERATOR_BIN" \
        --distance "$GEN_DISTANCE" \
        --num_rounds "$GEN_ROUNDS" \
        --p_spam "$GEN_P_SPAM" \
        --num_shots "$GEN_SHOTS" \
        --yaml "$(basename "$CONFIG_FILE")" \
        --save_syndrome "$(basename "$SYNDROMES_FILE")" > gen_syndromes.log 2>&1) || {
        _err "Syndrome generation failed; see ${GEN_DIR}/gen_syndromes.log"
        tail -5 "${GEN_DIR}/gen_syndromes.log" >&2 || true
        return 1
    }

    _info "Generated: $CONFIG_FILE"
    _info "Generated: $SYNDROMES_FILE"
}

resolve_paths() {
    local cudaqx_utils="${CUDAQX_DIR}/build/libs/qec/unittests/utils"
    local cq_build_dir="${CUDA_QUANTUM_DIR}/realtime/build/unittests"

    SERVER_BIN="${CUDAQX_DIR}/build/bin/decoding_server"
    PLAYBACK_BIN="${cudaqx_utils}/hololink_fpga_syndrome_playback"
    EMULATOR_BIN="${cq_build_dir}/utils/hololink_fpga_emulator"

    if [[ ! -x "$SERVER_BIN" ]]; then
        _err "Decoding server binary not found: $SERVER_BIN"
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
        # Mirror do_setup_network's handling of the comma form ("devA,devB":
        # server on devA, emulator on devB) so runs without --setup-network
        # split it the same way.
        if [[ "$IB_DEVICE" == *,* ]]; then
            BRIDGE_DEVICE="${IB_DEVICE%%,*}"
            EMULATOR_DEVICE="${IB_DEVICE#*,}"
        else
            BRIDGE_DEVICE="$IB_DEVICE"
        fi
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
# Server + Playback (shared by both modes)
# ============================================================================

# Start the decoding server against $1=peer_ip $2=remote_qp; scrape its
# Bridge Ready handshake into SERVER_QP / SERVER_RKEY / SERVER_ADDR.
start_server() {
    local peer_ip="$1" remote_qp="$2" server_log="$3"

    _log "Starting decoding server (decoder=$DECODER, remote-qp=$remote_qp)"

    local server_ld_path
    server_ld_path="${CUDA_QUANTUM_DIR}/realtime/build/lib:${CUDAQX_DIR}/build/lib"

    LD_LIBRARY_PATH="${server_ld_path}:${LD_LIBRARY_PATH:-}" \
    "$SERVER_BIN" \
        --config="$CONFIG_FILE" \
        --transport=cpu_roce \
        --qp_config=hsb_fpga \
        --device="$BRIDGE_DEVICE" \
        --peer-ip="$peer_ip" \
        --remote-qp="$remote_qp" \
        --num-slots="$NUM_SLOTS" \
        --slot-size="$PAGE_SIZE" \
        --frame-size="$FRAME_SIZE" \
        --timeout="$TIMEOUT" \
        > >(tee "$server_log") 2>&1 &
    SERVER_PID=$!
    PIDS_TO_KILL+=("$SERVER_PID")
    _info "Server PID: $SERVER_PID"

    wait_for_pattern "$server_log" "Bridge Ready" 60 "$SERVER_PID" >/dev/null || {
        _err "Decoding server did not become ready"
        _err "--- Server log ---"
        cat "$server_log" >&2
        return 1
    }

    local qp_line rkey_line addr_line
    qp_line=$(wait_for_pattern "$server_log" "QP Number:" 5 "$SERVER_PID") || return 1
    rkey_line=$(wait_for_pattern "$server_log" "RKey:" 5 "$SERVER_PID") || return 1
    addr_line=$(wait_for_pattern "$server_log" "Buffer Addr:" 5 "$SERVER_PID") || return 1

    SERVER_QP=$(extract_hex "$qp_line")
    SERVER_RKEY=$(extract_decimal "$rkey_line")
    SERVER_ADDR=$(extract_hex "$addr_line")

    _info "Server QP:     $SERVER_QP"
    _info "Server RKey:   $SERVER_RKEY"
    _info "Server Buffer: $SERVER_ADDR"
}

# Run playback against $1=hololink_ip; extra args appended from $2...
run_playback() {
    local hololink_ip="$1"; shift

    _log "Starting syndrome playback (hololink=$hololink_ip)"
    local playback_args=(
        --hololink "$hololink_ip"
        --per-round
        --config "$CONFIG_FILE"
        --syndromes "$SYNDROMES_FILE"
        --qp-number "$SERVER_QP"
        --rkey "$SERVER_RKEY"
        --buffer-addr "$SERVER_ADDR"
        --page-size "$PAGE_SIZE"
        "$@"
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
# Run: Emulated Mode (3 tools)
# ============================================================================

run_emulated() {
    _banner "Decoding Server Decode Loop Test (Emulated FPGA, $DECODER)"

    local emu_log server_log
    emu_log=$(mktemp /tmp/hsb_decoding_server_emulator.XXXXXX.log)
    server_log=$(mktemp /tmp/hsb_decoding_server.XXXXXX.log)
    TEMP_FILES+=("$emu_log" "$server_log")

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

    # ---- 2. Start decoding server ----
    start_server "$EMULATOR_IP" "$emu_qp" "$server_log" || return 1

    # ---- 3. Start playback tool ----
    run_playback "$EMULATOR_IP" --control-port "$CONTROL_PORT"
}

# ============================================================================
# Run: FPGA Mode (2 tools)
# ============================================================================

run_fpga() {
    _banner "Decoding Server Decode Loop Test (Real FPGA, $DECODER)"

    local server_log
    server_log=$(mktemp /tmp/hsb_decoding_server.XXXXXX.log)
    TEMP_FILES+=("$server_log")

    # ---- 1. Start decoding server (FPGA data-plane QP is fixed 0x2) ----
    start_server "$FPGA_IP" "0x2" "$server_log" || return 1

    # ---- 2. Start playback tool (BOOTP enumeration; no control port) ----
    run_playback "$FPGA_IP"
}

# ============================================================================
# Main
# ============================================================================

main() {
    _banner "HSB FPGA Decoding Server Test"

    _info "Decoder: $DECODER (decoding server, CPU HOST_CALL path)"
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

    resolve_data_files
    if $GENERATE_DATA; then
        generate_data_files
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
        _banner "DECODING SERVER DECODE LOOP ($DECODER): PASS"
    else
        _banner "DECODING SERVER DECODE LOOP ($DECODER): FAIL"
    fi

    return $rc
}

main
