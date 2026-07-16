#!/bin/bash
# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
#
# run_realtime_decoding.sh
#
# Drive the delivered decoding_server from one of two syndrome sources, both
# decoding through the SAME prebuilt server:
#
#   --source qpu-kernel   The lowered QEC kernel supplies syndromes itself over
#                         UDP, in software (no NIC).  The server decodes on the
#                         CPU host-call path (pymatching/trt) or the GPU
#                         (nv-qldpc).  This is the portable, hardware-free mode.
#
#   --source fpga         The delivered hololink_fpga_syndrome_playback tool
#                         streams pre-generated syndromes over RoCE from a REAL
#                         FPGA into the server's RDMA ring (needs a ConnectX NIC
#                         cabled to the FPGA).  pymatching/trt decode on the CPU
#                         (cpu_roce); nv-qldpc on the GPU device-graph scheduler
#                         (gpu_roce).  There is NO emulator here -- emulator
#                         testing lives in the unittests hsb_fpga_decoding_server_test.sh.
#
# DELIVERABLES (consumed prebuilt from --install-prefix, never built here):
#   decoding_server, hololink_fpga_syndrome_playback, the QEC + realtime libs,
#   and the decoder plugins.
# EXAMPLE BINARIES (the only things the user compiles; from --example-build-dir):
#   surface_code_realtime_decoding       generator: writes config + syndromes
#   surface_code_realtime_decoding-cqr   lowered kernel: the live UDP source
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============================================================================
# Defaults
# ============================================================================

SOURCE="qpu-kernel"          # qpu-kernel | fpga
DECODER="pymatching"         # pymatching | trt_decoder | nv-qldpc-decoder

# Where the deliverables live (bin/ + lib/).  Required unless the per-artifact
# overrides below are all given.  No dev-tree fallback -- shipped code resolves
# from one install prefix.
INSTALL_PREFIX="${CUDAQX_INSTALL_DIR:-}"
# CUDA-Q install (its runtime libs + realtime libs are needed on the load path).
CUDAQ_PREFIX="${CUDA_QUANTUM_PATH:-/usr/local/cudaq}"
# Optional separate realtime lib dir (udp transport / dispatch), if not colocated.
REALTIME_LIB_DIR="${CUDAQ_REALTIME_DIR:-}"

# The example's own build directory (holds the two compiled binaries).
EXAMPLE_BUILD_DIR="${SCRIPT_DIR}/build"

# Per-artifact overrides (power users); empty => resolved from the prefixes.
SERVER_BIN=""
PLAYBACK_BIN=""
GENERATOR_BIN=""
KERNEL_BIN=""
NV_QLDPC_PLUGIN="${CUDAQ_QEC_NV_QLDPC_PLUGIN:-}"

# trt_decoder identity ONNX: AUTO generates it at run time (needs python onnx).
ONNX_PATH="AUTO"
ONNX_FILE=""

# Surface-code experiment parameters (RNG seed is fixed => reproducible).
GEN_DISTANCE=3
GEN_ROUNDS=4
GEN_P_SPAM=0.01

# Shot counts differ by mode: the FPGA plays back a fixed set through a 64-slot
# RDMA RX ring; the qpu-kernel self-paces via its blocking get_corrections and
# has no ring to overrun, so it runs more shots by default.  Both overridable.
FPGA_SHOTS=85
QPU_KERNEL_SHOTS=200
NUM_SHOTS=""                 # explicit override

# FPGA pacing: inter-shot spacing (us) keeps playback from overrunning the
# FPGA's fixed 64-slot RX ring.  The qpu-kernel path needs none.
SPACING="10"

# FPGA transport is derived from the decoder unless set: pymatching/trt ->
# cpu_roce (CPU host-call), nv-qldpc -> gpu_roce (GPU device-graph scheduler).
TRANSPORT=""
GPU_ID=0

# Network (fpga source only)
DO_SETUP_NETWORK=false
IB_DEVICE=""                 # auto-detect first Up ConnectX
BRIDGE_IP="10.0.0.1"
FPGA_IP="192.168.0.2"
MTU=4096
PAGE_SIZE=384
# RX ring depth for both transports, bounded by the HSB QP's 64 receive WQEs
# (the FPGA writes frame rid to slot rid % NUM_SLOTS; more slots than WQEs drops
# frames under load). The server clamps cpu_roce --num-slots to this; the
# gpu_roce ring is capped to it in run_fpga.
NUM_SLOTS=64
FRAME_SIZE=64
# Server lifetime failsafe (seconds). decoding_server's --timeout is a TOTAL
# runtime cap (not inactivity): the server exits once elapsed time exceeds it,
# even mid-run. 300 matches the in-tree surface_code-4 external-server tests;
# the default 60 would kill long runs (large --num-shots, slow decoders).
TIMEOUT=300
VERIFY=true

# ============================================================================
# Argument parsing
# ============================================================================

print_usage() {
    cat <<'EOF'
Usage: run_realtime_decoding.sh --source {qpu-kernel|fpga} [options]

Sources:
  --source qpu-kernel     Lowered kernel streams syndromes over UDP (no NIC).
  --source fpga           Delivered playback streams from a real FPGA over RoCE.

Common:
  --decoder NAME          pymatching (default) | trt_decoder | nv-qldpc-decoder
  --install-prefix DIR    Deliverables prefix (decoding_server, playback, libs,
                          plugins in DIR/bin and DIR/lib).  Required (or give the
                          per-artifact overrides).
  --cudaq-prefix DIR      CUDA-Q install (default: $CUDA_QUANTUM_PATH or
                          /usr/local/cudaq)
  --realtime-lib-dir DIR  Extra realtime lib dir (udp transport / dispatch)
  --example-build-dir DIR The example's build dir with the two binaries
                          (default: <script dir>/build)
  --onnx PATH             trt_decoder ONNX model (default AUTO: generate identity
                          predecoder at run time; needs python 'onnx')
  --nv-qldpc-plugin PATH  Prebuilt libcudaq-qec-nv-qldpc-decoder.so (or set
                          CUDAQ_QEC_NV_QLDPC_PLUGIN); required for nv-qldpc
  --distance N            Surface-code distance (default 3)
  --num-rounds N          Measurement rounds (default 4)
  --p-spam F              SPAM error probability (default 0.01)
  --num-shots N           Shots (default: 85 fpga / 200 qpu-kernel)
  --gpu N                 GPU id for nv-qldpc (default 0)

Per-artifact overrides:
  --server PATH  --playback PATH  --generator PATH  --kernel PATH

FPGA-only:
  --setup-network         Configure the ConnectX interface first (needs sudo)
  --device DEV            ConnectX IB device (default: auto-detect)
  --bridge-ip ADDR        Server-side NIC IP (default 10.0.0.1)
  --fpga-ip ADDR          FPGA IP (default 192.168.0.2)
  --mtu N                 MTU (default 4096)
  --spacing US            Inter-shot spacing us (default 10; keeps the RX ring
                          from overrunning)
  --transport T           cpu_roce | gpu_roce (default: derived from decoder)
  --no-verify             Skip playback correction verification

  --help, -h              Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --source)           SOURCE="$2"; shift ;;
        --decoder)          DECODER="$2"; shift ;;
        --install-prefix)   INSTALL_PREFIX="$2"; shift ;;
        --cudaq-prefix)     CUDAQ_PREFIX="$2"; shift ;;
        --realtime-lib-dir) REALTIME_LIB_DIR="$2"; shift ;;
        --example-build-dir) EXAMPLE_BUILD_DIR="$2"; shift ;;
        --onnx)             ONNX_PATH="$2"; shift ;;
        --nv-qldpc-plugin)  NV_QLDPC_PLUGIN="$2"; shift ;;
        --distance)         GEN_DISTANCE="$2"; shift ;;
        --num-rounds)       GEN_ROUNDS="$2"; shift ;;
        --p-spam)           GEN_P_SPAM="$2"; shift ;;
        --num-shots)        NUM_SHOTS="$2"; shift ;;
        --gpu)              GPU_ID="$2"; shift ;;
        --server)           SERVER_BIN="$2"; shift ;;
        --playback)         PLAYBACK_BIN="$2"; shift ;;
        --generator)        GENERATOR_BIN="$2"; shift ;;
        --kernel)           KERNEL_BIN="$2"; shift ;;
        --setup-network)    DO_SETUP_NETWORK=true ;;
        --device)           IB_DEVICE="$2"; shift ;;
        --bridge-ip)        BRIDGE_IP="$2"; shift ;;
        --fpga-ip)          FPGA_IP="$2"; shift ;;
        --mtu)              MTU="$2"; shift ;;
        --spacing)          SPACING="$2"; shift ;;
        --transport)        TRANSPORT="$2"; shift ;;
        --no-verify)        VERIFY=false ;;
        --help|-h)          print_usage; exit 0 ;;
        *) echo "ERROR: unknown option: $1" >&2; print_usage >&2; exit 1 ;;
    esac
    shift
done

if [[ "$SOURCE" != "qpu-kernel" && "$SOURCE" != "fpga" ]]; then
    echo "ERROR: --source must be qpu-kernel or fpga (got '$SOURCE')" >&2; exit 1
fi

_log()  { echo "==> $*"; }
_info() { echo "    $*"; }
_err()  { echo "ERROR: $*" >&2; }
_banner() { echo; echo "========================================";
            echo "  $*"; echo "========================================"; echo; }

# ============================================================================
# Path resolution (install-prefix only; per-artifact overrides win)
# ============================================================================

resolve_paths() {
    # Canonicalize user-supplied paths first: generate_data runs the generator
    # from a temp working dir, and library paths are exported to subprocesses,
    # so a relative path that validates here would break there. Only rewrite
    # values that resolve to an existing path -- sentinels (e.g. the AUTO onnx
    # default) and missing paths stay as-is for the normal validation below.
    local _v _abs
    for _v in INSTALL_PREFIX CUDAQ_PREFIX REALTIME_LIB_DIR EXAMPLE_BUILD_DIR \
              SERVER_BIN PLAYBACK_BIN GENERATOR_BIN KERNEL_BIN ONNX_PATH NV_QLDPC_PLUGIN; do
        if [[ -n "${!_v}" ]]; then
            if _abs=$(readlink -f -- "${!_v}" 2>/dev/null) && [[ -e "$_abs" ]]; then
                printf -v "$_v" '%s' "$_abs"
            fi
        fi
    done

    if [[ -z "$SERVER_BIN"    && -n "$INSTALL_PREFIX" ]]; then SERVER_BIN="$INSTALL_PREFIX/bin/decoding_server"; fi
    if [[ -z "$PLAYBACK_BIN"  && -n "$INSTALL_PREFIX" ]]; then PLAYBACK_BIN="$INSTALL_PREFIX/bin/hololink_fpga_syndrome_playback"; fi
    if [[ -z "$GENERATOR_BIN" ]]; then GENERATOR_BIN="$EXAMPLE_BUILD_DIR/surface_code_realtime_decoding"; fi
    if [[ -z "$KERNEL_BIN"    ]]; then KERNEL_BIN="$EXAMPLE_BUILD_DIR/surface_code_realtime_decoding-cqr"; fi

    if [[ -z "$SERVER_BIN" ]]; then
        _err "No decoding_server: pass --install-prefix DIR (or --server PATH)."; exit 1
    fi
    if [[ ! -x "$SERVER_BIN" ]]; then _err "decoding_server not found/executable: $SERVER_BIN"; exit 1; fi
    if [[ ! -x "$GENERATOR_BIN" ]]; then
        _err "generator not found: $GENERATOR_BIN"
        _err "Build the example first (cmake --build <dir>) or pass --generator PATH."; exit 1
    fi
    if [[ "$SOURCE" == "qpu-kernel" && ! -x "$KERNEL_BIN" ]]; then
        _err "lowered kernel not found: $KERNEL_BIN (build the example, or --kernel PATH)"; exit 1
    fi
    if [[ "$SOURCE" == "fpga" && ! -x "$PLAYBACK_BIN" ]]; then
        _err "playback tool not found: $PLAYBACK_BIN (a hololink-enabled deliverable)"; exit 1
    fi

    # Load path: deliverable libs + plugins, plus the CUDA-Q runtime/realtime.
    local p=""
    [[ -n "$INSTALL_PREFIX"   ]] && p="$INSTALL_PREFIX/lib:$INSTALL_PREFIX/lib/decoder-plugins"
    [[ -n "$CUDAQ_PREFIX"     ]] && p="$p:$CUDAQ_PREFIX/lib:$CUDAQ_PREFIX/lib/plugins"
    [[ -n "$REALTIME_LIB_DIR" ]] && p="$p:$REALTIME_LIB_DIR/lib:$REALTIME_LIB_DIR"
    export LD_LIBRARY_PATH="${p}:${LD_LIBRARY_PATH:-}"

    # nv-qldpc plugin must be dlopen-able from the deliverable plugins dir.
    if [[ "$DECODER" == "nv-qldpc-decoder" ]]; then
        if [[ -z "$NV_QLDPC_PLUGIN" && -n "$INSTALL_PREFIX" \
              && -f "$INSTALL_PREFIX/lib/decoder-plugins/libcudaq-qec-nv-qldpc-decoder.so" ]]; then
            NV_QLDPC_PLUGIN="$INSTALL_PREFIX/lib/decoder-plugins/libcudaq-qec-nv-qldpc-decoder.so"
        fi
        if [[ -z "$NV_QLDPC_PLUGIN" || ! -f "$NV_QLDPC_PLUGIN" ]]; then
            _err "nv-qldpc-decoder plugin not available (pass --nv-qldpc-plugin or set"
            _err "CUDAQ_QEC_NV_QLDPC_PLUGIN).  Skipping this decoder."
            exit 77   # standard "skip" code
        fi
        local link="$INSTALL_PREFIX/lib/decoder-plugins/$(basename "$NV_QLDPC_PLUGIN")"
        if [[ -n "$INSTALL_PREFIX" && ! -e "$link" ]]; then
            # Fail fast: the server discovers plugins in this directory, so a
            # silently failed link surfaces minutes later as an opaque
            # "server did not construct an 'nv-qldpc-decoder' session".
            if ! mkdir -p "$INSTALL_PREFIX/lib/decoder-plugins" 2>/dev/null \
               || ! ln -sf "$NV_QLDPC_PLUGIN" "$link" 2>/dev/null; then
                _err "cannot place the nv-qldpc plugin in $INSTALL_PREFIX/lib/decoder-plugins"
                _err "(is the install prefix writable?)"; exit 1
            fi
        fi
    fi
}

# ============================================================================
# Config + syndrome generation (uses the example's generator binary)
# ============================================================================

GEN_DIR=""
CONFIG_FILE=""
SYNDROMES_FILE=""

generate_identity_onnx() {
    if [[ "$ONNX_PATH" != "AUTO" ]]; then
        [[ -f "$ONNX_PATH" ]] || { _err "--onnx not found: $ONNX_PATH"; return 1; }
        ONNX_FILE="$ONNX_PATH"; return 0
    fi
    if ! python3 -c "import onnx" 2>/dev/null; then
        _err "python3 'onnx' module required for the identity ONNX (pip install onnx),"
        _err "or pass --onnx PATH."; return 1
    fi
    ONNX_FILE="$GEN_DIR/trt_identity_predecoder.onnx"
    local syndrome_size=$(((GEN_DISTANCE * GEN_DISTANCE - 1) * GEN_ROUNDS))
    _info "Generating identity ONNX: $ONNX_FILE (syndrome_size=$syndrome_size)"
    python3 - "$ONNX_FILE" "$syndrome_size" <<'PY'
import sys
import onnx
from onnx import TensorProto, helper
output_path = sys.argv[1]
syndrome_size = int(sys.argv[2])
input_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, syndrome_size])
output_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, syndrome_size + 1])
zero = helper.make_node("Constant", [], ["pre_l"],
    value=helper.make_tensor("zero", TensorProto.FLOAT, [1, 1], [0.0]))
concat = helper.make_node("Concat", ["pre_l", "input"], ["output"], axis=1)
graph = helper.make_graph([zero, concat], "trt_identity_predecoder", [input_info], [output_info])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 19)])
model.ir_version = 9
onnx.checker.check_model(model)
onnx.save(model, output_path)
PY
    [[ -f "$ONNX_FILE" ]] || { _err "identity ONNX generation failed"; return 1; }
}

generate_data() {
    GEN_DIR="$(mktemp -d /tmp/realtime_decoding_demo.XXXXXX)"
    CONFIG_FILE="$GEN_DIR/config_${DECODER}.yml"
    SYNDROMES_FILE="$GEN_DIR/syndromes_${DECODER}.txt"

    local gen_extra=()
    if [[ "$DECODER" == "trt_decoder" ]]; then
        generate_identity_onnx || exit 1
        gen_extra+=(--onnx_path "$ONNX_FILE")
    fi
    if [[ "$DECODER" == "nv-qldpc-decoder" ]]; then
        gen_extra+=(--use-relay-bp)
    fi

    _log "Generating decoder config (decoder=$DECODER, d=$GEN_DISTANCE, rounds=$GEN_ROUNDS)"
    ( cd "$GEN_DIR" && "$GENERATOR_BIN" \
        --distance "$GEN_DISTANCE" --num_rounds "$GEN_ROUNDS" --p_spam "$GEN_P_SPAM" \
        --decoder_type "$DECODER" ${gen_extra[@]+"${gen_extra[@]}"} \
        --save_dem "$CONFIG_FILE" ) >"$GEN_DIR/gen_config.log" 2>&1 || {
        _err "config generation failed; see $GEN_DIR/gen_config.log"; tail -5 "$GEN_DIR/gen_config.log" >&2; exit 1; }

    # nv-qldpc runs on a GPU: pin the decoder to --gpu via cuda_device_id (the
    # host worker / device-graph scheduler both honor it). The fpga/gpu_roce
    # device path additionally switches the transport. The generator omits these
    # optional fields, so inject them under the decoder's `type:` line.
    # pymatching/trt need neither.
    if [[ "$DECODER" == "nv-qldpc-decoder" ]]; then
        local inject_tr=""
        if [[ "$SOURCE" == "fpga" && "$TRANSPORT" == "gpu_roce" ]]; then
            inject_tr="    transport:       gpu_roce"
        fi
        awk -v gpu="$GPU_ID" -v tr="$inject_tr" '{ print }
             /^[[:space:]]*type:/ && !d { if (tr != "") print tr;
                                          print "    cuda_device_id:  " gpu; d=1 }' \
            "$CONFIG_FILE" > "$CONFIG_FILE.tmp" && mv "$CONFIG_FILE.tmp" "$CONFIG_FILE"
        # Verify the injection landed: a YAML formatting change in the
        # generator would otherwise silently drop --gpu / the transport
        # override (pinning nv-qldpc to GPU 0, or mismatching the server's
        # transport).
        if ! grep -qE "^[[:space:]]*cuda_device_id:[[:space:]]*${GPU_ID}\$" "$CONFIG_FILE"; then
            _err "config injection failed: cuda_device_id not found in $CONFIG_FILE"; exit 1
        fi
        if [[ -n "$inject_tr" ]] && \
           ! grep -qE "^[[:space:]]*transport:[[:space:]]*gpu_roce\$" "$CONFIG_FILE"; then
            _err "config injection failed: transport not found in $CONFIG_FILE"; exit 1
        fi
    fi

    # The FPGA source replays a pre-generated syndrome file; the qpu-kernel
    # source generates syndromes live inside the kernel, so it needs none.
    if [[ "$SOURCE" == "fpga" ]]; then
        local shots="${NUM_SHOTS:-$FPGA_SHOTS}"
        _log "Generating $shots syndromes for playback"
        ( cd "$GEN_DIR" && "$GENERATOR_BIN" \
            --distance "$GEN_DISTANCE" --num_rounds "$GEN_ROUNDS" --p_spam "$GEN_P_SPAM" \
            --num_shots "$shots" --yaml "$CONFIG_FILE" \
            --save_syndrome "$SYNDROMES_FILE" ) >"$GEN_DIR/gen_syndromes.log" 2>&1 || {
            _err "syndrome generation failed; see $GEN_DIR/gen_syndromes.log"; tail -5 "$GEN_DIR/gen_syndromes.log" >&2; exit 1; }
    fi
    _info "config:    $CONFIG_FILE"
    if [[ "$SOURCE" == "fpga" ]]; then _info "syndromes: $SYNDROMES_FILE"; fi
}

# ============================================================================
# Cleanup
# ============================================================================
PIDS=()
cleanup() {
    for pid in "${PIDS[@]:-}"; do
        # `|| true` on the TERM: a pid reaped between the kill -0 and the kill
        # -TERM would otherwise abort this EXIT trap under set -e AND override
        # the script's exit status (a PASS run would exit nonzero).
        [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null && { kill -TERM "$pid" 2>/dev/null || true; sleep 0.5;
            kill -0 "$pid" 2>/dev/null && kill -KILL "$pid" 2>/dev/null || true; }
    done
    [[ -n "$GEN_DIR" && -d "$GEN_DIR" ]] && rm -rf "$GEN_DIR" || true
}
trap cleanup EXIT

wait_for_pattern() {
    local logfile="$1" pattern="$2" timeout_sec="$3" pid="${4:-}"
    local waited=0
    while (( waited < timeout_sec * 10 )); do
        [[ -n "$pid" ]] && ! kill -0 "$pid" 2>/dev/null && { _err "process $pid died"; return 1; }
        local m; m=$(grep -m1 "$pattern" "$logfile" 2>/dev/null || true)
        [[ -n "$m" ]] && { echo "$m"; return 0; }
        sleep 0.1; waited=$((waited + 1))
    done
    _err "timeout waiting for: $pattern"; return 1
}

# ============================================================================
# qpu-kernel pass/fail criteria -- the same checks the in-tree surface_code-4
# ctest driver applies (unittests/realtime/app_examples/surface_code-4-yaml-
# test.sh), so this example fails on the same evidence as the existing tests:
# hard decoder-failure messages, the corrections-line completion proof, a
# residual logical-error ceiling of num_shots/50 (min 1), and external-server
# evidence (server-owned decoders + a dispatch-count floor).
# ============================================================================
verify_qpu_kernel_output() {
    local kernel_log="$1" server_log="$2" shots="$3"
    local rc=0

    _log "Checking realtime output (surface_code-4 test criteria)"

    # A non-graphlike DEM handed to pymatching surfaces as "Invalid column in H".
    if grep -q "Invalid column in H" "$kernel_log"; then
        _err "found 'Invalid column in H' (decoder received a non-graphlike DEM)"; rc=1
    fi
    # Hard decoder-init / dispatch failures.
    if grep -q "terminate called" "$kernel_log"; then
        _err "found 'terminate called' (the app aborted)"; rc=1
    fi
    if grep -q "Decoder 0 not found" "$kernel_log"; then
        _err "found 'Decoder 0 not found' (decoder was not registered)"; rc=1
    fi
    if grep -q "Error initializing decoders" "$kernel_log"; then
        _err "found 'Error initializing decoders'"; rc=1
    fi

    # This line proves the realtime decoding path actually ran to completion.
    if ! grep -q "Number of corrections decoder found:" "$kernel_log"; then
        _err "missing 'Number of corrections decoder found:' line (decoding did not complete)"; rc=1
    fi

    # Residual logical errors must stay under the ceiling: a decoder that is
    # wired up but decoding wrong produces ~random logicals, far above it.
    local max_non_zero=$((shots / 50))
    (( max_non_zero < 1 )) && max_non_zero=1
    local non_zero
    non_zero=$(grep "Number of non-zero values measured :" "$kernel_log" \
        | awk -F': ' '{print $2}' | tr -d '[:space:]')
    if ! [[ "$non_zero" =~ ^[0-9]+$ ]]; then
        _err "'Number of non-zero values measured' is not a number (got '$non_zero')"; rc=1
    elif (( non_zero > max_non_zero )); then
        _err "residual logical errors ($non_zero) exceed ceiling ($max_non_zero) -- decoder appears wired-but-wrong"; rc=1
    else
        _info "residual logical errors: $non_zero (ceiling $max_non_zero) -- OK"
    fi

    # External-server evidence: the kernel must report server-owned decoders
    # (no silent local-decoder fallback), and the server must have dispatched
    # at least shots * (rounds + 3) RPCs -- rounds+1 enqueues + get_corrections
    # + reset per shot.
    if ! grep -q "External decoding server owns all configured decoder instances" "$kernel_log"; then
        _err "kernel did not report server-owned decoders"; rc=1
    fi
    local dispatches min_dispatches=$((shots * (GEN_ROUNDS + 3)))
    dispatches=$(sed -n 's/^QEC_DECODING_SERVER_DISPATCHED count=\([0-9][0-9]*\)$/\1/p' \
        "$server_log" | tail -n1)
    if ! [[ "$dispatches" =~ ^[0-9]+$ ]] || (( dispatches < min_dispatches )); then
        _err "server dispatch count '${dispatches:-<missing>}' is below $min_dispatches"; rc=1
    else
        _info "server evidence: dispatches=$dispatches (>= $min_dispatches) -- OK"
    fi

    return $rc
}

# ============================================================================
# qpu-kernel source: lowered kernel drives the server over UDP
# ============================================================================
run_qpu_kernel() {
    _banner "Realtime decoding: qpu-kernel source (UDP), decoder=$DECODER"

    local server_log; server_log="$GEN_DIR/server.log"
    _log "Starting decoding_server on UDP"
    # The GPU for nv-qldpc is selected via cuda_device_id in the config (injected
    # in generate_data), not a server flag.
    "$SERVER_BIN" --config="$CONFIG_FILE" --transport=udp --port=0 \
        --timeout="$TIMEOUT" >"$server_log" 2>&1 &
    local spid=$!; PIDS+=("$spid")
    wait_for_pattern "$server_log" "QEC_DECODING_SERVER_READY" 60 "$spid" >/dev/null || {
        cat "$server_log" >&2; return 1; }
    local port; port=$(grep -oE 'port=[0-9]+' "$server_log" | head -1 | cut -d= -f2)
    _info "server ready on UDP port $port (pid $spid)"

    local shots="${NUM_SHOTS:-$QPU_KERNEL_SHOTS}"
    local kernel_log; kernel_log="$GEN_DIR/kernel.log"
    _log "Running lowered kernel: $shots shots"
    local rc=0
    CUDAQ_QEC_REALTIME_MODE=external_server QEC_DECODING_SERVER_PORT="$port" \
        "$KERNEL_BIN" --yaml "$CONFIG_FILE" \
        --distance "$GEN_DISTANCE" --num_rounds "$GEN_ROUNDS" \
        --p_spam "$GEN_P_SPAM" --num_shots "$shots" 2>&1 | tee "$kernel_log" || rc=$?
    # Stop the server and wait for it to exit so its shutdown stats
    # (QEC_DECODING_SERVER_DISPATCHED ...) are flushed to the log.
    kill -TERM "$spid" 2>/dev/null || true
    wait "$spid" 2>/dev/null || true
    if (( rc != 0 )); then
        _err "realtime phase exited with non-zero status ($rc)"
    fi
    local vrc=0
    verify_qpu_kernel_output "$kernel_log" "$server_log" "$shots" || vrc=$?
    (( rc != 0 )) && return $rc
    return $vrc
}

# ============================================================================
# fpga source: delivered playback streams syndromes over RoCE from a real FPGA
# (network helpers ported from the unittests hsb_fpga_decoding_server_test.sh)
# ============================================================================
extract_hex()     { echo "$1" | grep -oP '0x[0-9a-fA-F]+' | head -1; }
extract_decimal() { echo "$1" | awk -F': ' '{print $NF}' | tr -d ' '; }

ib_to_netdev() { ibdev2netdev | awk -v d="$1" -v p="${2:-1}" '$1==d && $3==p {print $5}'; }
netdev_to_ib() { ibdev2netdev | awk -v i="$1" '$5==i {print $1}'; }

ipv4_to_gid_suffix() {
    local o1 o2 o3 o4; IFS='.' read -r o1 o2 o3 o4 <<< "$1"
    printf "ffff:%02x%02x:%02x%02x" "$o1" "$o2" "$o3" "$o4"
}
wait_for_roce_v2_gid() {
    local ib_dev="$1" ip="$2" timeout_s="${3:-15}" suffix gids types elapsed=0
    suffix=$(ipv4_to_gid_suffix "$ip")
    gids="/sys/class/infiniband/${ib_dev}/ports/1/gids"
    types="/sys/class/infiniband/${ib_dev}/ports/1/gid_attrs/types"
    [[ -d "$gids" ]] || return 0
    while (( elapsed < timeout_s * 10 )); do
        local g idx gid t
        for g in "$gids"/*; do
            idx=$(basename "$g"); gid=$(cat "$g" 2>/dev/null)
            if [[ "$gid" == *":${suffix}" ]]; then
                t=$(cat "${types}/${idx}" 2>/dev/null)
                [[ "$t" == *"RoCE v2"* ]] && { _info "RoCE v2 GID ready: ${ib_dev}[${idx}]"; return 0; }
            fi
        done
        sleep 0.1; elapsed=$((elapsed + 1))
    done
    _err "timed out waiting for RoCE v2 GID (${suffix}) on ${ib_dev}"; return 1
}
setup_port() {
    local iface="$1" ip="$2" mtu="$3" ib_dev
    _info "Configuring $iface: ip=$ip mtu=$mtu"
    sudo ip link set "$iface" up
    sudo ip link set "$iface" mtu "$mtu"
    sudo ip addr flush dev "$iface"
    sudo ip addr add "${ip}/24" dev "$iface"
    ib_dev=$(netdev_to_ib "$iface")
    if [[ -n "$ib_dev" ]] && command -v rdma &>/dev/null; then
        local pc; pc=$(ls -d "/sys/class/infiniband/${ib_dev}/ports/"* 2>/dev/null | wc -l)
        for p in $(seq 1 "$pc"); do sudo rdma link set "${ib_dev}/${p}" type eth || true; done
    fi
    command -v mlnx_qos &>/dev/null && sudo mlnx_qos -i "$iface" --trust=dscp 2>/dev/null || true
    command -v ethtool  &>/dev/null && sudo ethtool -C "$iface" adaptive-rx off rx-usecs 0 2>/dev/null || true
}
seed_fpga_neighbor() {
    local iface="$1" fpga_ip="$2" mac
    ping -c 3 -W 1 -I "$iface" "$fpga_ip" >/dev/null 2>&1 || true
    mac=$(ip neigh show "$fpga_ip" dev "$iface" 2>/dev/null \
          | awk '{for(i=1;i<=NF;i++) if($i=="lladdr") print $(i+1)}' | head -1)
    if [[ -n "$mac" ]]; then
        sudo ip neigh replace "$fpga_ip" lladdr "$mac" nud permanent dev "$iface"
        _info "static ARP: $fpga_ip -> $mac on $iface"
    else
        _err "could not resolve FPGA MAC for $fpga_ip on $iface (cabled? powered? ping $fpga_ip)"
    fi
}

setup_network() {
    _log "Setting up ConnectX network for the FPGA"
    local iface
    if [[ -n "$IB_DEVICE" ]]; then iface=$(ib_to_netdev "$IB_DEVICE" 1)
    else iface=$(ibdev2netdev | awk '/\(Up\)/{print $5; exit}'); fi
    [[ -n "$iface" ]] || { _err "cannot detect a ConnectX interface"; return 1; }
    _info "server interface: $iface"
    setup_port "$iface" "$BRIDGE_IP" "$MTU"
    BRIDGE_DEVICE=$(netdev_to_ib "$iface")
    wait_for_roce_v2_gid "$BRIDGE_DEVICE" "$BRIDGE_IP" 15 || true
    seed_fpga_neighbor "$iface" "$FPGA_IP"
}

start_roce_server() {
    local peer_ip="$1" remote_qp="$2" server_log="$3"
    _log "Starting decoding_server (transport=$TRANSPORT, remote-qp=$remote_qp)"
    local ready
    if [[ "$TRANSPORT" == "gpu_roce" ]]; then
        CUDA_MODULE_LOADING=EAGER \
        HOLOLINK_DEVICE="$BRIDGE_DEVICE" HOLOLINK_PEER_IP="$peer_ip" \
        HOLOLINK_REMOTE_QP="$((remote_qp))" HOLOLINK_FRAME_SIZE="$PAGE_SIZE" \
        HOLOLINK_NUM_PAGES="$GPU_ROCE_NUM_PAGES" \
        "$SERVER_BIN" --config="$CONFIG_FILE" --transport=gpu_roce --timeout="$TIMEOUT" \
            > >(tee "$server_log") 2>&1 &
        ready="QEC_DECODING_SERVER_READY gpu_roce"
    else
        "$SERVER_BIN" --config="$CONFIG_FILE" --transport=cpu_roce --qp_config=hsb_fpga \
            --device="$BRIDGE_DEVICE" --peer-ip="$peer_ip" --remote-qp="$remote_qp" \
            --num-slots="$NUM_SLOTS" --slot-size="$PAGE_SIZE" --frame-size="$FRAME_SIZE" \
            --timeout="$TIMEOUT" > >(tee "$server_log") 2>&1 &
        ready="Bridge Ready"
    fi
    local spid=$!; PIDS+=("$spid"); SERVER_PID="$spid"
    wait_for_pattern "$server_log" "$ready" 60 "$spid" >/dev/null || { cat "$server_log" >&2; return 1; }
    wait_for_pattern "$server_log" "decoder 0 type: ${DECODER}" 5 "$spid" >/dev/null || {
        _err "server did not construct a '${DECODER}' session"; cat "$server_log" >&2; return 1; }
    local qp rk ad
    qp=$(wait_for_pattern "$server_log" "QP Number:"   5 "$spid") || return 1
    rk=$(wait_for_pattern "$server_log" "RKey:"        5 "$spid") || return 1
    ad=$(wait_for_pattern "$server_log" "Buffer Addr:" 5 "$spid") || return 1
    SERVER_QP=$(extract_hex "$qp"); SERVER_RKEY=$(extract_decimal "$rk"); SERVER_ADDR=$(extract_hex "$ad")
    _info "server QP=$SERVER_QP RKey=$SERVER_RKEY Buffer=$SERVER_ADDR"
}

run_fpga() {
    _banner "Realtime decoding: fpga source (RoCE), decoder=$DECODER"
    # (TRANSPORT was derived from the decoder in main(), before generate_data.)
    #
    # The HSB QP has a fixed 64-entry receive WQE depth and the receiver
    # pre-posts one WQE per ring slot, so a ring with more slots than WQEs
    # leaves slots un-posted and drops frames under load. Both transports run
    # a WQE-depth ring: cpu_roce via NUM_SLOTS (which the server also clamps),
    # gpu_roce via GPU_ROCE_NUM_PAGES here.
    local HSB_WQE_DEPTH=64
    local GPU_ROCE_NUM_PAGES="$HSB_WQE_DEPTH"
    if (( NUM_SLOTS > HSB_WQE_DEPTH )); then
        _err "NUM_SLOTS=$NUM_SLOTS exceeds the HSB WQE depth ($HSB_WQE_DEPTH); clamping"
        NUM_SLOTS="$HSB_WQE_DEPTH"
    fi
    if [[ "$TRANSPORT" == "gpu_roce" ]]; then
        # DOCA requires the gpu_roce ring allocation (num_pages * page size) to
        # be a multiple of the HOST page size. 64 x 384 B = 24 KiB satisfies
        # 4K/8K-page kernels; on 16K/64K-page hosts (some aarch64 configs) the
        # server would reject the ring at startup, so fail fast with the
        # constraint spelled out.
        local host_page; host_page=$(getconf PAGESIZE)
        if (( (GPU_ROCE_NUM_PAGES * PAGE_SIZE) % host_page != 0 )); then
            _err "gpu_roce ring ($GPU_ROCE_NUM_PAGES slots x $PAGE_SIZE B) is not a multiple of this host's page size ($host_page B)."
            _err "The HSB frame stride must be a multiple of $(( host_page / HSB_WQE_DEPTH )) B on this host; see the unittests"
            _err "hsb_fpga_decoding_server_test.sh (--page-size) for a tunable-geometry run."
            exit 1
        fi
    fi
    : "${BRIDGE_DEVICE:=${IB_DEVICE:-rocep1s0f0}}"

    local server_log="$GEN_DIR/server.log"
    start_roce_server "$FPGA_IP" "0x2" "$server_log" || return 1

    _log "Streaming syndromes from the FPGA via playback (spacing=${SPACING}us)"
    # The FPGA writes syndrome frame rid to RDMA slot (rid % num-pages), so the
    # playback ring modulus MUST equal the server's RX ring depth. Otherwise
    # frames past one ring land outside the server's registered memory region
    # and are silently lost, and the server starves after exactly one ring
    # (num_slots). The cpu_roce server ring is NUM_SLOTS; the gpu_roce server
    # ring is GPU_ROCE_NUM_PAGES.
    local pb_pages="$NUM_SLOTS"
    if [[ "$TRANSPORT" == "gpu_roce" ]]; then pb_pages="$GPU_ROCE_NUM_PAGES"; fi
    local args=( --hololink "$FPGA_IP" --per-round --config "$CONFIG_FILE"
        --syndromes "$SYNDROMES_FILE" --qp-number "$SERVER_QP" --rkey "$SERVER_RKEY"
        --buffer-addr "$SERVER_ADDR" --page-size "$PAGE_SIZE" --num-pages "$pb_pages" )
    $VERIFY && args+=(--verify)
    [[ -n "$NUM_SHOTS" ]] && args+=(--num-shots "$NUM_SHOTS")
    [[ -n "$SPACING"   ]] && args+=(--spacing "$SPACING")
    local rc=0
    "$PLAYBACK_BIN" "${args[@]}" || rc=$?
    kill -TERM "$SERVER_PID" 2>/dev/null || true; sleep 0.3
    return $rc
}

# ============================================================================
# Main
# ============================================================================
main() {
    _banner "Realtime Decoding Demo"
    _info "source=$SOURCE  decoder=$DECODER"

    resolve_paths
    # Derive the FPGA transport before generating the config: nv-qldpc runs on
    # the gpu_roce device-graph path, and generate_data injects that transport
    # (+ cuda_device_id) into the config. pymatching/trt use cpu_roce.
    if [[ "$SOURCE" == "fpga" && -z "$TRANSPORT" ]]; then
        case "$DECODER" in
            nv-qldpc-decoder) TRANSPORT="gpu_roce" ;;
            *)                TRANSPORT="cpu_roce" ;;
        esac
    fi
    if [[ "$SOURCE" == "fpga" ]] && $DO_SETUP_NETWORK; then setup_network; fi
    generate_data

    local rc=0
    if [[ "$SOURCE" == "qpu-kernel" ]]; then run_qpu_kernel || rc=$?
    else run_fpga || rc=$?; fi

    echo
    if [[ $rc -eq 0 ]]; then _banner "REALTIME DECODING ($SOURCE / $DECODER): PASS"
    else _banner "REALTIME DECODING ($SOURCE / $DECODER): FAIL"; fi
    return $rc
}
main
