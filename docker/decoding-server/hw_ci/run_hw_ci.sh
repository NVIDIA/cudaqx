#!/bin/bash
# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under    #
# the terms of the Apache License 2.0 which accompanies this distribution.    #
# ============================================================================ #
#
# Hardware CI runner for the QEC decoding server (NVQLink lab: DGX Spark /
# GB200 with ConnectX NICs and an FPGA syndrome source).
#
# Given a cudaqx commit, this script -- run on the HOST, not in a container:
#   1. clones cudaqx at that commit into the work dir
#   2. builds the dev image (docker/decoding-server/dev.Dockerfile) locally;
#      the Docker layer cache makes unchanged builds near-instant
#   3. fails fast if the image's baked CUDA-Q does not match the commit's
#      .cudaq_version pin
#   4. builds cudaq-realtime + cudaqx + the demo binaries inside the
#      container (hw_ci/container_build.sh)
#   5. runs the test lanes (see --list) and prints a PASS/FAIL/SKIP summary
#
# Lanes report SKIP (exit 77) when an input is absent -- a proprietary
# artifact, the HF token, the FPGA -- so lost coverage is always visible in
# the summary without failing the run; --strict turns skips into failures.
#
# Proprietary artifacts are staged once on the host (see --artifacts-dir);
# the Ising/TRT model bundle is deliberately NOT staged: the ising-prepare
# lane downloads the gated Hugging Face model and rebuilds the bundle inside
# the container on every run, so that path is continuously validated.
set -uo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
SHA=""
# The repo under test is the one this script lives in; a fresh clone of it
# is made at the requested --sha.
REPO_URL="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel 2>/dev/null)"
[[ -n "$REPO_URL" ]] || { echo "error: cannot resolve the containing git repo (is the script inside a clone?)" >&2; exit 1; }
WORKDIR="$HOME/.cache/cudaqx-hw-ci"
ARTIFACTS_DIR="/opt/nvqlink-lab-artifacts"
TIER="all"                    # examples | extra | all
INCLUDE_OPT_IN=false
ONLY_GLOB=""
SKIP_GLOB=""
STRICT=false
LIST_ONLY=false
REFRESH_BASE=false
BASE_IMAGE=""                 # resolved from the pin unless given
BUILD_BASE=false
CUDA_VERSION="13.0"
CUDA_ARCH=""                  # auto-detected unless given (Spark=121, GB200=100)
ROCE_PAIR="rxe"               # rxe | DEV0,DEV1 | (empty via --roce-pair none = skip cpu_roce pair lanes)
HF_TOKEN_FILE=""
HF_TOKEN_PROMPT=false
FPGA_DEVICE="mlx5_4"          # ConnectX IB device facing the FPGA (GB200 lab wiring)
BRIDGE_IP="192.168.0.1"
FPGA_IP="192.168.0.2"
PAGE_SIZE=""                  # default derived from the host page size
KEEP_CONTAINER=false
NO_FPGA=false

print_usage() {
    cat <<EOF
Usage: run_hw_ci.sh --sha SHA [options]

  --sha SHA               cudaqx commit (or branch/tag) to clone, build, test
                          (cloned from the repo containing this script:
                          $REPO_URL)
  --workdir DIR           clone/build/log root (default: $WORKDIR)
  --artifacts-dir DIR     proprietary artifacts, bind-mounted RO at /artifacts
                          (default: $ARTIFACTS_DIR); layout:
                            decoder-plugins/libcudaq-qec-nv-qldpc-decoder.so
                            cudevice/libcudaq-qec-realtime-cudevice-proprietary.a
                          missing pieces => the dependent lanes SKIP

Lane selection:
  --tier T                examples | extra | all (default: all)
  --include-opt-in        also run the opt-in lanes (see --list)
  --only GLOB             run only lanes matching GLOB (e.g. 'examples/fpga/*')
  --skip GLOB             skip lanes matching GLOB
  --list                  print the lane list for the current flags and exit
  --strict                any SKIP fails the run (full-coverage mode)

Image:
  --base-image IMG        override the cudaqx-dev base image
  --refresh-base          docker build --pull (accept a moved base tag;
                          invalidates cached layers built on the old base)
  --build-base            if the pin-matched base tag cannot be pulled, build
                          it locally from the commit's cudaqx.dev.Dockerfile
                          (multi-hour CUDA-Q build the first time per pin)
  --cuda-version V        base image CUDA flavor (default: $CUDA_VERSION)

Hardware:
  --cuda-arch N           CUDA architecture (default: auto via nvidia-smi;
                          DGX Spark GB10=121, GB200=100)
  --roce-pair rxe         SoftRoCE self-loop for the two-process cpu_roce
                          lanes (the default; no free ConnectX port pair
                          needed), OR
  --roce-pair DEV0,DEV1   a real loopback-cabled IB device pair (DGX Spark),
                          OR --roce-pair none => those lanes SKIP
  --fpga-device DEV       ConnectX IB device facing the FPGA
                          (default: $FPGA_DEVICE, the GB200 lab wiring)
  --bridge-ip IP          server-side NIC IP (default $BRIDGE_IP)
  --fpga-ip IP            FPGA IP (default $FPGA_IP)
  --page-size N           RDMA ring slot size (default: 384, or 512 on
                          64 KiB-page hosts; the device_graph lane rounds up
                          to the host-page-compatible value)
  --no-fpga               skip all FPGA lanes (e.g. Spark cabled in loopback
                          mode: the single cable is FPGA XOR loopback)

Misc:
  --hf-token-file FILE    Hugging Face token for the gated Ising model (or
                          set HF_TOKEN); without a token the trt lanes fall
                          back to a staged bundle (see --artifacts-dir) or SKIP
  --hf-token-prompt       read the token from the terminal instead (nothing
                          written to disk; for shared/public accounts)
  --keep-container        leave the container running afterwards (debugging)
  --help, -h              this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --sha)            SHA="$2"; shift ;;
        --workdir)        WORKDIR="$2"; shift ;;
        --artifacts-dir)  ARTIFACTS_DIR="$2"; shift ;;
        --tier)           TIER="$2"; shift ;;
        --include-opt-in) INCLUDE_OPT_IN=true ;;
        --only)           ONLY_GLOB="$2"; shift ;;
        --skip)           SKIP_GLOB="$2"; shift ;;
        --list)           LIST_ONLY=true ;;
        --strict)         STRICT=true ;;
        --base-image)     BASE_IMAGE="$2"; shift ;;
        --refresh-base)   REFRESH_BASE=true ;;
        --build-base)     BUILD_BASE=true ;;
        --cuda-version)   CUDA_VERSION="$2"; shift ;;
        --cuda-arch)      CUDA_ARCH="$2"; shift ;;
        --roce-pair)      ROCE_PAIR="$2"; [[ "$ROCE_PAIR" == none ]] && ROCE_PAIR=""; shift ;;
        --fpga-device)    FPGA_DEVICE="$2"; shift ;;
        --bridge-ip)      BRIDGE_IP="$2"; shift ;;
        --fpga-ip)        FPGA_IP="$2"; shift ;;
        --page-size)      PAGE_SIZE="$2"; shift ;;
        --no-fpga)        NO_FPGA=true ;;
        --hf-token-file)  HF_TOKEN_FILE="$2"; shift ;;
        --hf-token-prompt) HF_TOKEN_PROMPT=true ;;
        --keep-container) KEEP_CONTAINER=true ;;
        --help|-h)        print_usage; exit 0 ;;
        *) echo "ERROR: unknown option: $1" >&2; print_usage >&2; exit 1 ;;
    esac
    shift
done

_info() { echo "[hw-ci] $*"; }
_err()  { echo "[hw-ci] ERROR: $*" >&2; }
_die()  { _err "$*"; exit 1; }

[[ -n "$SHA" || "$LIST_ONLY" == true ]] || { print_usage >&2; _die "--sha is required"; }
case "$TIER" in examples|extra|all) ;; *) _die "--tier must be examples|extra|all" ;; esac

# Interactive token entry: the token lives only in this process's memory (and
# the ising-prepare lane's docker exec env), never on disk -- for runs from
# shared/public accounts.
if [[ "$HF_TOKEN_PROMPT" == true && "$LIST_ONLY" != true ]]; then
    [[ -r /dev/tty ]] || _die "--hf-token-prompt needs a terminal (use --hf-token-file or HF_TOKEN otherwise)"
    read -rs -p "Hugging Face token (input hidden): " HF_TOKEN < /dev/tty; echo
    [[ -n "$HF_TOKEN" ]] || _die "--hf-token-prompt: empty token"
    export HF_TOKEN
fi

# ---------------------------------------------------------------------------
# Lane bookkeeping.  Every lane lands in the summary exactly once as
# PASS / FAIL / SKIP(reason).  Lanes run strictly sequentially: everything
# here shares one GPU, one FPGA, and one RoCE fabric.
# ---------------------------------------------------------------------------
LANE_NAMES=()
LANE_STATUS=()
LANE_DETAIL=()

lane_selected() {
    local name="$1"
    case "$TIER" in
        examples) [[ "$name" == examples/* || "$name" == optin/* ]] || return 1 ;;
        extra)    [[ "$name" == extra/*    || "$name" == optin/* ]] || return 1 ;;
    esac
    [[ "$name" == optin/* && "$INCLUDE_OPT_IN" != true ]] && return 1
    # shellcheck disable=SC2053
    [[ -n "$ONLY_GLOB" && "$name" != $ONLY_GLOB ]] && return 1
    # shellcheck disable=SC2053
    [[ -n "$SKIP_GLOB" && "$name" == $SKIP_GLOB ]] && return 1
    return 0
}

record_lane() {  # name status detail
    LANE_NAMES+=("$1"); LANE_STATUS+=("$2"); LANE_DETAIL+=("$3")
}

skip_lane() {  # name reason
    lane_selected "$1" || return 0
    if [[ "$LIST_ONLY" == true ]]; then echo "  $1  [would SKIP: $2]"; return 0; fi
    _info "SKIP $1: $2"
    record_lane "$1" SKIP "$2"
}

# run_lane NAME CMD -- CMD is a bash command line executed in the container.
# Extra `docker exec` flags (e.g. -e VAR=...) come from LANE_ENV.
LANE_ENV=()
run_lane() {
    # Consume LANE_ENV first thing so a deselected/listed lane can never leak
    # its env flags (or the HF token) into the next lane that runs.
    local env_flags=(${LANE_ENV[@]+"${LANE_ENV[@]}"})
    LANE_ENV=()
    local name="$1"; shift
    lane_selected "$name" || return 0
    if [[ "$LIST_ONLY" == true ]]; then echo "  $name"; return 0; fi
    local log="$LOG_DIR/${name//\//_}.log"
    _info "LANE $name"
    local t0=$SECONDS rc=0
    docker exec ${env_flags[@]+"${env_flags[@]}"} "$CONTAINER" bash -lc "$*" \
        >"$log" 2>&1 || rc=$?
    local dt=$((SECONDS - t0))
    if [[ $rc -eq 0 ]]; then
        record_lane "$name" PASS "${dt}s"
        _info "PASS $name (${dt}s)"
    elif [[ $rc -eq 77 ]]; then
        local reason
        reason=$(grep -Eo 'SKIP[:(].*' "$log" | tail -1)
        record_lane "$name" SKIP "${reason:-exit 77 (see $log)}"
        _info "SKIP $name: ${reason:-exit 77}"
    else
        record_lane "$name" FAIL "rc=$rc  log: $log"
        _err "FAIL $name (rc=$rc)  log: $log"
    fi
}

print_summary() {
    local pass=0 fail=0 skip=0 i
    echo
    echo "================ HW-CI SUMMARY (sha ${SHORT_SHA:-?}) ================"
    for i in "${!LANE_NAMES[@]}"; do
        printf '%-5s %-48s %s\n' "${LANE_STATUS[$i]}" "${LANE_NAMES[$i]}" "${LANE_DETAIL[$i]}"
        case "${LANE_STATUS[$i]}" in
            PASS) ((pass++)) ;; FAIL) ((fail++)) ;; SKIP) ((skip++)) ;;
        esac
    done
    echo "---------------------------------------------------------------"
    echo "$pass passed, $fail failed, $skip skipped"
    [[ $fail -gt 0 ]] && return 1
    if [[ "$STRICT" == true && $skip -gt 0 ]]; then
        echo "--strict: treating $skip skip(s) as failure"
        return 1
    fi
    return 0
}

# ---------------------------------------------------------------------------
# Host-side setup
# ---------------------------------------------------------------------------
detect_cuda_arch() {
    [[ -n "$CUDA_ARCH" ]] && return 0
    local cap
    cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' .')
    [[ -n "$cap" ]] || _die "cannot auto-detect the GPU (nvidia-smi); pass --cuda-arch"
    CUDA_ARCH="$cap"
    _info "CUDA architecture: sm_$CUDA_ARCH (auto-detected)"
}

checkout_sha() {
    SRC="$WORKDIR/src"
    mkdir -p "$WORKDIR"
    if [[ ! -d "$SRC/.git" ]]; then
        _info "Cloning $REPO_URL"
        git clone "$REPO_URL" "$SRC" || _die "clone failed"
    fi
    # --repo is authoritative on every run, not just the first: retarget the
    # cached clone (a stale origin otherwise silently pins every later run
    # to the first-ever --repo).
    git -C "$SRC" remote set-url origin "$REPO_URL"
    if git -C "$SRC" fetch origin "$SHA" 2>/dev/null; then
        # FETCH_HEAD is exactly the requested branch/tag/SHA tip; never
        # resolve "$SHA" locally here or a branch name would silently hit a
        # stale local ref from clone time.
        git -C "$SRC" checkout --detach FETCH_HEAD || _die "cannot check out '$SHA'"
    else
        # Not fetchable by name (e.g. an abbreviated SHA): fetch everything,
        # then the ref must resolve -- otherwise fail loudly instead of
        # falling back to an arbitrary FETCH_HEAD.
        git -C "$SRC" fetch origin || _die "fetch failed"
        git -C "$SRC" checkout --detach "$SHA" 2>/dev/null || _die "cannot check out '$SHA'"
    fi
    SHORT_SHA=$(git -C "$SRC" rev-parse --short=12 HEAD)
    _info "Testing cudaqx @ $SHORT_SHA"
}

platform() {
    case "$(uname -m)" in
        aarch64|arm64) echo arm64 ;;
        x86_64)        echo amd64 ;;
        *) _die "unsupported platform: $(uname -m)" ;;
    esac
}

resolve_base_image() {
    [[ -n "$BASE_IMAGE" ]] && { _info "Base image (override): $BASE_IMAGE"; return 0; }
    local plat shortref candidate fallback
    plat=$(platform)
    shortref=$(jq -r '.cudaq.ref' "$SRC/.cudaq_version" | head -c8)
    candidate="ghcr.io/nvidia/cudaqx-dev:${shortref}-${plat}-cu${CUDA_VERSION}"
    fallback="ghcr.io/nvidia/cudaqx-dev:latest-${plat}-cu${CUDA_VERSION}"
    if docker image inspect "$candidate" >/dev/null 2>&1 || docker pull "$candidate" >/dev/null 2>&1; then
        BASE_IMAGE="$candidate"
    elif [[ "$BUILD_BASE" == true ]]; then
        _info "Pin-matched base $candidate unavailable; building it locally (--build-base)"
        docker build -f "$SRC/docker/build_env/cudaqx.dev.Dockerfile" \
            --build-arg base_image="ghcr.io/nvidia/cuda-quantum-devcontainer:${plat}-cu${CUDA_VERSION%%.*}.${CUDA_VERSION#*.}-gcc12-main" \
            --build-arg cuda_version="$CUDA_VERSION" \
            -t "$candidate" "$SRC" || _die "local base build failed"
        BASE_IMAGE="$candidate"
    elif docker image inspect "$fallback" >/dev/null 2>&1 || docker pull "$fallback" >/dev/null 2>&1; then
        _info "Pin-matched base $candidate unavailable; falling back to $fallback"
        BASE_IMAGE="$fallback"
    else
        _die "no usable base image: tried $candidate and $fallback (see --build-base / --base-image)"
    fi
    _info "Base image: $BASE_IMAGE"
}

build_image() {
    IMAGE="cudaqx-decoding-hwci:$SHORT_SHA"
    local pull_flag=()
    [[ "$REFRESH_BASE" == true ]] && pull_flag=(--pull)
    _info "Building dev image $IMAGE (cached layers reused when unchanged)"
    docker build ${pull_flag[@]+"${pull_flag[@]}"} \
        -f "$SRC/docker/decoding-server/dev.Dockerfile" \
        --build-arg base_image="$BASE_IMAGE" \
        --build-arg cuda_native_arch="$CUDA_ARCH" \
        -t "$IMAGE" "$SRC/docker/decoding-server" || _die "docker build failed"
}

check_cudaq_pin() {
    local image_ref src_ref
    image_ref=$(docker run --rm "$IMAGE" jq -r '.cudaq.ref' /cudaq_version 2>/dev/null)
    src_ref=$(jq -r '.cudaq.ref' "$SRC/.cudaq_version")
    if [[ -z "$image_ref" || "$image_ref" == "null" ]]; then
        _info "WARNING: image has no /cudaq_version; skipping the stale-pin check"
        return 0
    fi
    if [[ "$image_ref" != "$src_ref" ]]; then
        _die "stale base image: it bakes CUDA-Q $image_ref but the commit pins $src_ref.
Wait for build_dev.yaml to publish the new pin's image, pass --base-image, or use --build-base."
    fi
    _info "CUDA-Q pin check OK ($src_ref)"
}

start_container() {
    CONTAINER="hwci-$SHORT_SHA"
    docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
    mkdir -p "$WORKDIR/ccache"
    local artifacts_mount=()
    if [[ -d "$ARTIFACTS_DIR" ]]; then
        artifacts_mount=(-v "$ARTIFACTS_DIR:/artifacts:ro")
    else
        _info "Artifacts dir $ARTIFACTS_DIR absent; proprietary lanes will SKIP"
    fi
    # rxe mode may need to build the OFED-compat rdma_rxe module in-container
    # (see setup_roce_pair), which compiles against the HOST kernel headers
    # and ofa_kernel tree.
    local rxe_mount=()
    if [[ "$ROCE_PAIR" == "rxe" ]]; then
        rxe_mount=(-v /lib/modules:/lib/modules:ro -v /usr/src:/usr/src:ro)
    fi
    # /dev/infiniband is a live bind mount, NOT --device: --device snapshots
    # the char devices at container creation, so the uverbs node of an rxe
    # device created later by setup_roce_pair would never appear in the
    # container and libibverbs would not find the device.  The container is
    # privileged, so no device-cgroup allowance is lost by the switch.
    docker run -d --name "$CONTAINER" \
        --privileged --net=host --gpus all --shm-size=8g \
        --ulimit memlock=-1:-1 \
        -v /dev/infiniband:/dev/infiniband \
        -v "$SRC:/workspaces/cudaqx" \
        -v "$WORKDIR/ccache:/root/.ccache" \
        ${artifacts_mount[@]+"${artifacts_mount[@]}"} \
        ${rxe_mount[@]+"${rxe_mount[@]}"} \
        "$IMAGE" sleep infinity >/dev/null || _die "docker run failed"
}

cleanup() {
    [[ "$KEEP_CONTAINER" == true ]] && return 0
    if [[ -n "${CONTAINER:-}" ]]; then
        teardown_roce_pair 2>/dev/null
        docker rm -f "$CONTAINER" >/dev/null 2>&1
    fi
}
trap cleanup EXIT

in_ctr() { docker exec "$CONTAINER" bash -lc "$*"; }

# ---------------------------------------------------------------------------
# cpu_roce endpoint pair.
#   rxe:        SoftRoCE self-loop on a dummy netdev -- both endpoints share
#               one rxe device/IP (the pattern documented in the in-tree
#               surface_code-1-cqr-two-process-test.sh).  Needs rdma_rxe on
#               the HOST kernel.
#   DEV0,DEV1:  real loopback-cabled ConnectX pair (e.g. DGX Spark):
#               10.0.0.1/24 <-> 10.0.0.2/24 with permanent neighbor entries
#               (same-host IPs otherwise resolve via lo and RDMA CM times out).
# All netlink/rdma calls run inside the privileged --net=host container, so
# they act on the host netns; objects carry the hwci- prefix for teardown.
# ---------------------------------------------------------------------------
ROCE_READY=false
CH_DEV=""; CH_IP=""; DA_DEV=""; DA_IP=""
NET0=""; NET1=""

apply_roce_pair_addrs() {
    # Idempotent; called before every cpu_roce lane as well as at setup:
    # NetworkManager-managed ports drop statically added addresses on each
    # DHCP retry cycle (see the README for the permanent nmcli fix).
    in_ctr "
        set -e
        ip link set $NET0 up; ip link set $NET1 up
        ip addr replace 10.0.0.1/24 dev $NET0
        ip addr replace 10.0.0.2/24 dev $NET1
        mac0=\$(cat /sys/class/net/$NET0/address)
        mac1=\$(cat /sys/class/net/$NET1/address)
        ip neigh replace 10.0.0.2 lladdr \$mac1 nud permanent dev $NET0
        ip neigh replace 10.0.0.1 lladdr \$mac0 nud permanent dev $NET1
    "
}

wait_roce_gids() {
    # The IPv4-mapped RoCE GIDs (::ffff:10.0.0.x) appear asynchronously
    # after the address add, and the transceivers refuse to start without
    # them.  Host sysfs and the --net=host container see the same tables.
    local i
    for i in $(seq 1 20); do
        grep -qs 'ffff:0a00:0001' "/sys/class/infiniband/$CH_DEV/ports/1/gids/"* && \
        grep -qs 'ffff:0a00:0002' "/sys/class/infiniband/$DA_DEV/ports/1/gids/"* && return 0
        sleep 0.5
    done
    return 1
}

setup_roce_pair() {
    [[ -z "$ROCE_PAIR" ]] && return 0
    if [[ "$ROCE_PAIR" == "rxe" ]]; then
        if ! grep -qw rdma_rxe /proc/modules; then
            sudo -n modprobe rdma_rxe 2>/dev/null || true
        fi
        if ! grep -qw rdma_rxe /proc/modules; then
            # DOCA/MLNX-OFED DKMS hosts: the distro rdma_rxe cannot bind to
            # the OFED ib_core (symbol CRC mismatch), so build the staged
            # OFED-compat copy against the host headers mounted at
            # /lib/modules + /usr/src and load it from the privileged
            # container (see rxe-ofed/README.md).
            _info "distro rdma_rxe not loadable; building the OFED-compat module in-container"
            in_ctr "make -C /opt/rxe-ofed/src >/tmp/rxe-ofed-build.log 2>&1 \
                        && insmod /opt/rxe-ofed/src/rdma_rxe.ko" || {
                _err "OFED-compat rdma_rxe build/load failed; last lines of the build log:"
                in_ctr "tail -15 /tmp/rxe-ofed-build.log" >&2 || true
                _err "(full log: docker exec $CONTAINER cat /tmp/rxe-ofed-build.log)"
                return 1
            }
        fi
        in_ctr "
            ip link add hwci-dummy0 type dummy 2>/dev/null || true
            ip addr replace 10.88.0.1/24 dev hwci-dummy0
            ip link set hwci-dummy0 up
            rdma link show hwci_rxe0 >/dev/null 2>&1 || \
                rdma link add hwci_rxe0 type rxe netdev hwci-dummy0
            ibv_devinfo -d hwci_rxe0 >/dev/null
        " || { _err "SoftRoCE setup failed. Note: if 'rdma link' shows the device but ibv_devinfo cannot open it, the image's Mellanox-OFED ibverbs-providers lacks the rxe userspace provider (a known gap; see the README)"; return 1; }
        CH_DEV=hwci_rxe0; CH_IP=10.88.0.1
        DA_DEV=hwci_rxe0; DA_IP=10.88.0.1
    else
        local dev0="${ROCE_PAIR%%,*}" dev1="${ROCE_PAIR##*,}"
        [[ -n "$dev0" && -n "$dev1" && "$dev0" != "$dev1" ]] \
            || { _err "--roce-pair expects rxe or DEV0,DEV1"; return 1; }
        # Resolve ibdev -> netdev on the host via sysfs: the image has no
        # ibdev2netdev, and --net=host keeps the names identical inside the
        # container anyway.
        NET0=$(ls "/sys/class/infiniband/$dev0/device/net" 2>/dev/null | head -1)
        NET1=$(ls "/sys/class/infiniband/$dev1/device/net" 2>/dev/null | head -1)
        [[ -n "$NET0" && -n "$NET1" ]] \
            || { _err "cannot resolve netdevs for $ROCE_PAIR (see /sys/class/infiniband)"; return 1; }
        CH_DEV="$dev0"; CH_IP=10.0.0.1
        DA_DEV="$dev1"; DA_IP=10.0.0.2
        apply_roce_pair_addrs \
            || { _err "RoCE pair setup failed for $ROCE_PAIR"; return 1; }
        wait_roce_gids \
            || { _err "IPv4 RoCE GIDs did not appear on $dev0/$dev1"; return 1; }
    fi
    ROCE_READY=true
    _info "cpu_roce pair ready: channel=$CH_DEV/$CH_IP daemon=$DA_DEV/$DA_IP"
}

teardown_roce_pair() {
    [[ "$ROCE_PAIR" == "rxe" && -n "${CONTAINER:-}" ]] || return 0
    in_ctr "
        rdma link delete hwci_rxe0 2>/dev/null || true
        ip link delete hwci-dummy0 2>/dev/null || true
    " || true
}

roce_env() {  # docker exec env flags for the cpu_roce topology
    # Re-assert the pair right before each lane: on NetworkManager-managed
    # ports the addresses vanish on NM's retry timer, which killed lanes
    # minutes after a successful setup.
    if [[ "$LIST_ONLY" != true && "$ROCE_PAIR" != rxe && "$ROCE_READY" == true ]]; then
        apply_roce_pair_addrs >/dev/null 2>&1 && wait_roce_gids \
            || _info "WARNING: cpu_roce pair re-assert failed; lane may fail"
    fi
    LANE_ENV+=( -e "CUDAQ_CPU_ROCE_TEST_CHANNEL_DEVICE=$CH_DEV"
                -e "CUDAQ_CPU_ROCE_TEST_CHANNEL_IP=$CH_IP"
                -e "CUDAQ_CPU_ROCE_TEST_DAEMON_DEVICE=$DA_DEV"
                -e "CUDAQ_CPU_ROCE_TEST_DAEMON_IP=$DA_IP" )
}

# ---------------------------------------------------------------------------
# Page-size geometry.  cpu_roce host dispatch has no host-page constraint;
# the device_graph ring (64 slots) must total a multiple of the host page
# size, so its value rounds up to the compatible one.
# ---------------------------------------------------------------------------
derive_page_sizes() {
    local host_page; host_page=$(getconf PAGESIZE)
    if [[ -z "$PAGE_SIZE" ]]; then
        PAGE_SIZE=384
        [[ "$host_page" -gt 4096 ]] && PAGE_SIZE=512
    fi
    local stride=$(( host_page / 64 ))
    PAGE_SIZE_DG=$(( (PAGE_SIZE + stride - 1) / stride * stride ))
    _info "page size: $PAGE_SIZE (device_graph ring: $PAGE_SIZE_DG; host page $host_page)"
}

# ---------------------------------------------------------------------------
# Lane definitions
# ---------------------------------------------------------------------------
DEMO=/workspaces/cudaqx/docs/sphinx/examples/qec/realtime_decoding_demo
DEMO_ARGS="--install-prefix /usr/local/cudaqx --cudaq-prefix /usr/local/cudaq \
--realtime-lib-dir /tmp/cudaq-realtime --example-build-dir /workspaces/cudaqx/demo-build"
NV_QLDPC_PLUGIN_HOST="$ARTIFACTS_DIR/decoder-plugins/libcudaq-qec-nv-qldpc-decoder.so"
CUDEVICE_HOST="$ARTIFACTS_DIR/cudevice/libcudaq-qec-realtime-cudevice-proprietary.a"
NV_QLDPC_PLUGIN_CTR=/artifacts/decoder-plugins/libcudaq-qec-nv-qldpc-decoder.so
CUDEVICE_CTR=/artifacts/cudevice/libcudaq-qec-realtime-cudevice-proprietary.a
ISING_BUNDLE=/tmp/ising-bundle
# Optional pre-built bundle for tokenless machines (see --hf-token-file help):
# copied into the container so the trt lanes run; only the HF download/export
# path loses coverage, and the ising-prepare SKIP reason says so.
ISING_STAGED_HOST="$ARTIFACTS_DIR/ising-bundle"
ISING_STAGED_CTR=/artifacts/ising-bundle
CQ_SRC=/tmp/cudaq-realtime-src

have_hf_token() {
    [[ "$LIST_ONLY" == true ]] && return 0   # --list is host-independent
    [[ -n "${HF_TOKEN:-}" ]] && return 0
    [[ -n "$HF_TOKEN_FILE" && -r "$HF_TOKEN_FILE" ]] && return 0
    return 1
}

hf_token() {
    if [[ -n "${HF_TOKEN:-}" ]]; then echo "$HF_TOKEN"; else cat "$HF_TOKEN_FILE"; fi
}

ising_ready() {
    [[ "$LIST_ONLY" == true ]] && return 0
    in_ctr "test -f $ISING_BUNDLE/metadata.txt" 2>/dev/null
}

# ctest lanes: `ctest -R` exits 0 when NOTHING matches, which would
# false-PASS a lane whose test was never registered (configure-time gates).
# Count first and convert "no match" into a named SKIP.
ctest_cmd() {  # regex reason-when-unregistered
    local regex="$1" reason="$2"
    # --timeout only applies to tests without their own TIMEOUT property; it
    # bounds a wedged test at 15 min instead of ctest's 1500 s default.
    echo "cd /workspaces/cudaqx/build && \
n=\$(ctest -N -R '$regex' 2>/dev/null | sed -n 's/^Total Tests: //p'); \
if [ \"\${n:-0}\" -eq 0 ]; then echo 'SKIP: $reason'; exit 77; fi; \
ctest --output-on-failure --timeout 900 -R '$regex'"
}

fpga_dev_flag() { [[ -n "$FPGA_DEVICE" ]] && echo "--device $FPGA_DEVICE"; }

run_examples_tier() {
    local d

    # -- Ising bundle: fresh gated-HF download + export when a token exists;
    # otherwise fall back to a bundle staged in the artifacts dir.  The copy
    # runs outside the lane gate so `--only .../trt_decoder` also benefits.
    if ! have_hf_token && [[ "$LIST_ONLY" != true && -f "$ISING_STAGED_HOST/metadata.txt" ]]; then
        in_ctr "rm -rf $ISING_BUNDLE && cp -r $ISING_STAGED_CTR $ISING_BUNDLE" \
            || _info "WARNING: staged Ising bundle copy failed"
    fi
    if have_hf_token; then
        [[ "$LIST_ONLY" == true ]] || LANE_ENV=( -e "HF_TOKEN=$(hf_token)" )
        run_lane "examples/ising-prepare" "
            set -e
            rm -rf $ISING_BUNDLE
            app=\$(find /workspaces/cudaqx/build -name surface_code-4-yaml -type f -perm -u+x | head -1)
            [ -n \"\$app\" ] || { echo 'surface_code-4-yaml generator not in the build tree'; exit 1; }
            python3 $DEMO/prepare_ising_artifacts.py prepare \
                --app \"\$app\" --artifacts-dir $ISING_BUNDLE --yes"
    elif ising_ready; then
        skip_lane "examples/ising-prepare" "no HF token; trt lanes use the staged bundle from $ISING_STAGED_HOST (HF download/export path NOT exercised)"
    else
        skip_lane "examples/ising-prepare" "no HF token (--hf-token-file/--hf-token-prompt/HF_TOKEN) and no staged bundle at $ISING_STAGED_HOST"
    fi

    # -- qpu-kernel over udp: the no-hardware baseline -----------------------
    for d in pymatching multi_error_lut nv-qldpc-decoder trt_decoder; do
        local name="examples/qpu-kernel/udp/$d" extra=""
        case "$d" in
            nv-qldpc-decoder)
                [[ -f "$NV_QLDPC_PLUGIN_HOST" || "$LIST_ONLY" == true ]] \
                    || { skip_lane "$name" "missing $NV_QLDPC_PLUGIN_HOST"; continue; }
                extra="--nv-qldpc-plugin $NV_QLDPC_PLUGIN_CTR --gpu 0" ;;
            trt_decoder)
                ising_ready || { skip_lane "$name" "no Ising bundle (ising-prepare failed/skipped)"; continue; }
                extra="--ising-artifacts-dir $ISING_BUNDLE" ;;
        esac
        run_lane "$name" "bash $DEMO/run_realtime_decoding.sh --source qpu-kernel --decoder $d $DEMO_ARGS $extra"
    done

    # -- qpu-kernel over cpu_roce: real RDMA verbs on the RoCE pair ----------
    # No --setup-network: the runner configured the pair itself (the demo's
    # helper resolves ports via ibdev2netdev, which cannot see rxe devices).
    for d in pymatching multi_error_lut nv-qldpc-decoder trt_decoder; do
        local name="examples/qpu-kernel/cpu_roce/$d" extra=""
        [[ "$ROCE_READY" == true || "$LIST_ONLY" == true ]] \
            || { skip_lane "$name" "no cpu_roce pair (--roce-pair not set / setup failed)"; continue; }
        case "$d" in
            nv-qldpc-decoder)
                [[ -f "$NV_QLDPC_PLUGIN_HOST" || "$LIST_ONLY" == true ]] \
                    || { skip_lane "$name" "missing $NV_QLDPC_PLUGIN_HOST"; continue; }
                extra="--nv-qldpc-plugin $NV_QLDPC_PLUGIN_CTR --gpu 0" ;;
            trt_decoder)
                ising_ready || { skip_lane "$name" "no Ising bundle (ising-prepare failed/skipped)"; continue; }
                extra="--ising-artifacts-dir $ISING_BUNDLE" ;;
        esac
        roce_env
        run_lane "$name" "bash $DEMO/run_realtime_decoding.sh --source qpu-kernel --wire cpu_roce --decoder $d $DEMO_ARGS $extra"
    done

    # -- FPGA source, cpu_roce wire, host dispatch ---------------------------
    for d in pymatching multi_error_lut nv-qldpc-decoder trt_decoder; do
        local name="examples/fpga/cpu_roce/$d" extra=""
        [[ "$NO_FPGA" == true ]] && { skip_lane "$name" "--no-fpga"; continue; }
        case "$d" in
            nv-qldpc-decoder)
                [[ -f "$NV_QLDPC_PLUGIN_HOST" || "$LIST_ONLY" == true ]] \
                    || { skip_lane "$name" "missing $NV_QLDPC_PLUGIN_HOST"; continue; }
                extra="--nv-qldpc-plugin $NV_QLDPC_PLUGIN_CTR --gpu 0" ;;
            trt_decoder)
                ising_ready || { skip_lane "$name" "no Ising bundle (ising-prepare failed/skipped)"; continue; }
                extra="--ising-artifacts-dir $ISING_BUNDLE" ;;
        esac
        run_lane "$name" "bash $DEMO/run_realtime_decoding.sh --source fpga --dispatch host --decoder $d \
            --setup-network $(fpga_dev_flag) --bridge-ip $BRIDGE_IP --fpga-ip $FPGA_IP \
            --page-size $PAGE_SIZE $DEMO_ARGS $extra"
    done

    # -- FPGA source, device_graph dispatch (nv-qldpc only) ------------------
    local name="examples/fpga/device-graph/nv-qldpc-decoder"
    if [[ "$NO_FPGA" == true ]]; then
        skip_lane "$name" "--no-fpga"
    elif [[ ! -f "$NV_QLDPC_PLUGIN_HOST" && "$LIST_ONLY" != true ]]; then
        skip_lane "$name" "missing $NV_QLDPC_PLUGIN_HOST"
    elif [[ ! -f "$CUDEVICE_HOST" && "$LIST_ONLY" != true ]]; then
        skip_lane "$name" "missing $CUDEVICE_HOST (device_graph dispatch not built)"
    else
        run_lane "$name" "bash $DEMO/run_realtime_decoding.sh --source fpga --decoder nv-qldpc-decoder \
            --setup-network $(fpga_dev_flag) --bridge-ip $BRIDGE_IP --fpga-ip $FPGA_IP \
            --page-size $PAGE_SIZE_DG --nv-qldpc-plugin $NV_QLDPC_PLUGIN_CTR --gpu 0 $DEMO_ARGS"
    fi
}

run_extra_tier() {
    local hsb=/workspaces/cudaqx/libs/qec/unittests/utils/hsb_fpga_decoding_server_test.sh

    # -- two-process device_call channel over real RDMA verbs ----------------
    if [[ "$ROCE_READY" == true || "$LIST_ONLY" == true ]]; then
        LANE_ENV=( -e "QEC_DECODING_SERVER_TRANSPORT=cpu_roce" ); roce_env
        run_lane "extra/ctest/two-process-cpu-roce" \
            "$(ctest_cmd 'DecodingServerTwoProcess' 'DecodingServerTwoProcess tests not registered')"
        LANE_ENV=( -e "QEC_DECODING_SERVER_TRANSPORT=cpu_roce" ); roce_env
        run_lane "extra/ctest/app-two-process-cpu-roce" \
            "$(ctest_cmd 'app_examples.surface_code-1-cqr-two-process' 'two-process app tests not registered')"
    else
        skip_lane "extra/ctest/two-process-cpu-roce" "no cpu_roce pair (--roce-pair not set / setup failed)"
        skip_lane "extra/ctest/app-two-process-cpu-roce" "no cpu_roce pair (--roce-pair not set / setup failed)"
    fi

    # -- decoding_server over the FPGA (HSB control plane + SIF playback) ----
    if [[ "$NO_FPGA" == true ]]; then
        skip_lane "extra/hsb-fpga-server/cpu_roce" "--no-fpga"
        skip_lane "extra/hsb-fpga-server/gpu_roce" "--no-fpga"
    else
        run_lane "extra/hsb-fpga-server/cpu_roce" \
            "bash $hsb --setup-network $(fpga_dev_flag) --bridge-ip $BRIDGE_IP --fpga-ip $FPGA_IP \
             --page-size $PAGE_SIZE --cuda-quantum-dir $CQ_SRC"
        if [[ ( -f "$NV_QLDPC_PLUGIN_HOST" && -f "$CUDEVICE_HOST" ) || "$LIST_ONLY" == true ]]; then
            run_lane "extra/hsb-fpga-server/gpu_roce" \
                "bash $hsb --setup-network $(fpga_dev_flag) --bridge-ip $BRIDGE_IP --fpga-ip $FPGA_IP \
                 --page-size $PAGE_SIZE_DG --transport gpu_roce --decoder nv-qldpc-decoder \
                 --nv-qldpc-plugin $NV_QLDPC_PLUGIN_CTR --cuda-quantum-dir $CQ_SRC"
        elif [[ ! -f "$NV_QLDPC_PLUGIN_HOST" ]]; then
            skip_lane "extra/hsb-fpga-server/gpu_roce" "missing $NV_QLDPC_PLUGIN_HOST"
        else
            skip_lane "extra/hsb-fpga-server/gpu_roce" "missing $CUDEVICE_HOST (device_graph dispatch not built)"
        fi
    fi

    # -- GB200-class GPU ctests unreachable in normal CI ----------------------
    run_lane "extra/ctest/qldpc-graph" \
        "$(ctest_cmd '^test_realtime_qldpc_graph_decoding$' 'not registered (needs nv-qldpc plugin + cudevice archive at configure)')"
    run_lane "extra/ctest/mixed-dispatch" \
        "$(ctest_cmd 'app_examples.surface_code-4-yaml-mixed-dispatch' 'not registered (needs cudevice archive at configure)')"

    # -- gpu_roce bridge cross-check (same data plane, no server layers) ------
    local gbridge=/workspaces/cudaqx/libs/qec/unittests/utils/gpu_roce_qldpc_graph_decoder_test.sh
    if [[ "$NO_FPGA" == true ]]; then
        skip_lane "extra/gpu-roce-qldpc-bridge" "--no-fpga"
    elif [[ ( ! -f "$NV_QLDPC_PLUGIN_HOST" || ! -f "$CUDEVICE_HOST" ) && "$LIST_ONLY" != true ]]; then
        skip_lane "extra/gpu-roce-qldpc-bridge" "missing nv-qldpc plugin and/or cudevice archive"
    else
        # --spacing 100: at the playback tool's default 10 us inter-shot
        # spacing the ILA verification deterministically undercounts
        # (194/500 on the Spark); 100 us matches the demo lanes' pacing.
        run_lane "extra/gpu-roce-qldpc-bridge" "
            set -e
            bridge=\$(find /workspaces/cudaqx/build -name gpu_roce_qldpc_graph_decoder_bridge -type f | head -1)
            [ -n \"\$bridge\" ] || { echo 'SKIP: bridge executable not built'; exit 77; }
            bash $gbridge --setup-network $(fpga_dev_flag) --bridge-ip $BRIDGE_IP --fpga-ip $FPGA_IP \
                --page-size $PAGE_SIZE_DG --spacing 100 \
                --cuda-qx-dir /workspaces/cudaqx --cuda-quantum-dir $CQ_SRC \
                --hsb-dir /opt/holoscan-sensor-bridge \
                --proprietary-archive $CUDEVICE_CTR --nv-qldpc-plugin $NV_QLDPC_PLUGIN_CTR"
    fi
}

run_optin_tier() {
    # The predecoder bridge links the experimental cudaq-realtime-pipeline
    # library (CUDAQX_QEC_ENABLE_REALTIME_PIPELINE, OFF by default and
    # pending a port to the post-PR4770 graph-launch API), so today this
    # lane documents the coverage gap rather than exercising it.
    if [[ "$NO_FPGA" == true ]]; then
        skip_lane "optin/gpu-roce-predecoder" "--no-fpga"
        return 0
    fi
    run_lane "optin/gpu-roce-predecoder" "
        bridge=\$(find /workspaces/cudaqx/build -name gpu_roce_predecoder_bridge -type f | head -1)
        [ -n \"\$bridge\" ] || { echo 'SKIP: bridge not built (experimental realtime_pipeline)'; exit 77; }
        bash /workspaces/cudaqx/libs/qec/unittests/realtime/gpu_roce_predecoder_test.sh \
            --setup-network $(fpga_dev_flag) --bridge-ip $BRIDGE_IP --fpga-ip $FPGA_IP"
}

run_all_lanes() {
    run_examples_tier
    run_extra_tier
    run_optin_tier
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if [[ "$LIST_ONLY" == true ]]; then
    ROCE_READY=true   # listing shows the full lane set, not this host's skips
    PAGE_SIZE=${PAGE_SIZE:-384}; PAGE_SIZE_DG=$PAGE_SIZE
    echo "Lanes for --tier $TIER$([[ "$INCLUDE_OPT_IN" == true ]] && echo ' --include-opt-in'):"
    run_all_lanes
    exit 0
fi

detect_cuda_arch
derive_page_sizes
checkout_sha
LOG_DIR="$WORKDIR/logs/$SHORT_SHA"
mkdir -p "$LOG_DIR"
resolve_base_image
build_image
check_cudaq_pin
start_container
setup_roce_pair || _info "continuing without a cpu_roce pair"

_info "Image ready; compiling the commit under test inside the container:"
_info "  cudaq-realtime + cudaqx + demo binaries (log: $LOG_DIR/container_build.log)"
build_log="$LOG_DIR/container_build.log"
if ! in_ctr "bash /workspaces/cudaqx/docker/decoding-server/hw_ci/container_build.sh \
        --cuda-arch $CUDA_ARCH" >"$build_log" 2>&1; then
    tail -40 "$build_log" >&2
    _die "in-container source build failed; full log: $build_log"
fi
_info "Source build complete (cudaq-realtime + cudaqx + demo)"

run_all_lanes
print_summary
