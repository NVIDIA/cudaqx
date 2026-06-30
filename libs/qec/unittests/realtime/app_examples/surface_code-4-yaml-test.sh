# ============================================================================ #
# Copyright (c) 2024 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Driver for the surface_code-4-yaml realtime example. It exercises the two
# phases of the app:
#   Phase 1 (generation): --save_dem <cfg> --decoder_type <type> writes a YAML
#                         decoder config.
#   Phase 2 (realtime):   --yaml <cfg> loads that config and decodes; the
#                         decoder is read FROM the file (so --decoder_type is
#                         NOT passed here -- the app rejects --yaml +
#                         --decoder_type).
#
# The driver is decoder-agnostic. For trt_decoder, pass an ONNX path as arg 7
# or pass AUTO to generate a small [pre_L=0, residual=identity] model sized for
# the requested distance/window. Additional app args (for example
# --use-relay-bp) may follow arg 7.

set -euo pipefail

# Expected args:
#  $1 exe            Path to the surface_code-4-yaml executable
#  $2 distance       Surface code distance (D)
#  $3 num_rounds     Number of measurement rounds (R, >=D, multiple of D)
#  $4 decoder_window Decoder window size (W, >=D, multiple of D, <=R)
#  $5 decoder_type   Decoder to generate (optional, defaults to pymatching)
#  $6 num_shots      Number of shots (optional, defaults to 200)
#  $7 onnx_path      ONNX path for trt_decoder, or AUTO to generate one
#  $8... extra args  Extra app args to pass to generation/realtime phases

if [[ $# -lt 4 ]]; then
  echo "Error: Expected at least 4 arguments (got $#)"
  echo "Usage: $0 <exe> <distance> <num_rounds> <decoder_window> [decoder_type=pymatching] [num_shots=200]"
  exit 1
fi

EXE=$1
DISTANCE=$2
NUM_ROUNDS=$3
DECODER_WINDOW=$4
DECODER_TYPE=${5:-pymatching}
NUM_SHOTS=${6:-200}
ONNX_PATH=${7:-}
EXTRA_APP_ARGS=()
if [[ $# -ge 8 ]]; then
  EXTRA_APP_ARGS=("${@:8}")
fi

export CUDAQ_DEFAULT_SIMULATOR=stim
export CUDAQ_QEC_REALTIME_MODE=${CUDAQ_QEC_REALTIME_MODE:-inproc_rpc}

P_SPAM=0.01

# Residual logical-error ceiling. A correctly-wired decoder should leave very
# few residual logical errors; a wired-but-wrong decoder (e.g. one that loads
# but does not actually correct) leaves many. num_shots/4 is a generous ceiling
# that still catches a broken decoder.
MAX_NON_ZERO=$((NUM_SHOTS / 4))

# Create an isolated working directory and (by default) clean it up on exit.
WORKDIR=$(mktemp -d)
cleanup() {
  if [[ -z "${KEEP_LOG_FILES:-}" ]]; then
    rm -rf "$WORKDIR"
  else
    echo "KEEP_LOG_FILES set; leaving work dir: $WORKDIR"
  fi
}
trap cleanup EXIT

CONFIG_FILE=$WORKDIR/config.yml
REALTIME_LOG=$WORKDIR/realtime.log

if [[ "$DECODER_TYPE" == "trt_decoder" && "$ONNX_PATH" == "AUTO" ]]; then
  ONNX_PATH=$WORKDIR/trt_identity_predecoder.onnx
  SYNDROME_SIZE=$(((DISTANCE * DISTANCE - 1) * DECODER_WINDOW))
  PYTHON_BIN=${PYTHON:-python3}
  "$PYTHON_BIN" - "$ONNX_PATH" "$SYNDROME_SIZE" <<'PY'
import sys

import onnx
from onnx import TensorProto, helper

output_path = sys.argv[1]
syndrome_size = int(sys.argv[2])

input_info = helper.make_tensor_value_info(
    "input", TensorProto.FLOAT, [1, syndrome_size])
output_info = helper.make_tensor_value_info(
    "output", TensorProto.FLOAT, [1, syndrome_size + 1])
zero = helper.make_node(
    "Constant",
    [],
    ["pre_l"],
    value=helper.make_tensor("zero", TensorProto.FLOAT, [1, 1], [0.0]),
)
concat = helper.make_node("Concat", ["pre_l", "input"], ["output"], axis=1)
graph = helper.make_graph(
    [zero, concat], "trt_identity_predecoder", [input_info], [output_info])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 19)])
model.ir_version = 10
onnx.checker.check_model(model)
onnx.save(model, output_path)
PY
fi

echo "=============================================================="
echo "surface_code-4-yaml test"
echo "  exe            = $EXE"
echo "  distance       = $DISTANCE"
echo "  num_rounds     = $NUM_ROUNDS"
echo "  decoder_window = $DECODER_WINDOW"
echo "  decoder_type   = $DECODER_TYPE"
echo "  num_shots      = $NUM_SHOTS"
echo "  realtime mode  = $CUDAQ_QEC_REALTIME_MODE"
if [[ -n "$ONNX_PATH" ]]; then
  echo "  onnx_path      = $ONNX_PATH"
fi
if [[ ${#EXTRA_APP_ARGS[@]} -gt 0 ]]; then
  echo "  extra args     = ${EXTRA_APP_ARGS[*]}"
fi
echo "  max non-zero   = $MAX_NON_ZERO"
echo "=============================================================="

return_code=0

# -------------------------------------------------------------------------- #
# Phase 1: generation -- write the YAML decoder config.
# -------------------------------------------------------------------------- #
echo ""
echo "=== Phase 1: generate config (--save_dem, --decoder_type $DECODER_TYPE) ==="
GEN_ARGS=(
  --distance "$DISTANCE" \
  --num_rounds "$NUM_ROUNDS" \
  --decoder_window "$DECODER_WINDOW" \
  --num_shots "$NUM_SHOTS" \
  --p_spam "$P_SPAM" \
  --decoder_type "$DECODER_TYPE" \
  --save_dem "$CONFIG_FILE"
)
if [[ -n "$ONNX_PATH" ]]; then
  GEN_ARGS+=(--onnx_path "$ONNX_PATH")
fi
if [[ ${#EXTRA_APP_ARGS[@]} -gt 0 ]]; then
  GEN_ARGS+=("${EXTRA_APP_ARGS[@]}")
fi
"$EXE" "${GEN_ARGS[@]}"

# Assert the config file was created and is non-empty.
if [[ ! -s "$CONFIG_FILE" ]]; then
  echo "FAIL: config file '$CONFIG_FILE' was not created or is empty"
  exit 1
fi
echo "Config file generated: $CONFIG_FILE ($(stat -c %s "$CONFIG_FILE") bytes)"

# -------------------------------------------------------------------------- #
# Phase 2: realtime -- load the YAML config and decode.
# The decoder is read from the file, so do NOT pass --decoder_type here.
# -------------------------------------------------------------------------- #
echo ""
echo "=== Phase 2: realtime decode (--yaml $CONFIG_FILE) ==="
# Use a pipefail-safe tee so a crash in the app still surfaces a non-zero status
# while we keep the full log for assertions below.
set +e
REALTIME_ARGS=(
  --distance "$DISTANCE" \
  --num_rounds "$NUM_ROUNDS" \
  --decoder_window "$DECODER_WINDOW" \
  --num_shots "$NUM_SHOTS" \
  --p_spam "$P_SPAM" \
  --yaml "$CONFIG_FILE"
)
if [[ ${#EXTRA_APP_ARGS[@]} -gt 0 ]]; then
  REALTIME_ARGS+=("${EXTRA_APP_ARGS[@]}")
fi
"$EXE" "${REALTIME_ARGS[@]}" 2>&1 | tee "$REALTIME_LOG"
app_status=${PIPESTATUS[0]}
set -e

if [[ "$app_status" -ne 0 ]]; then
  echo "FAIL: realtime phase exited with non-zero status ($app_status)"
  return_code=1
fi

# -------------------------------------------------------------------------- #
# Assertions on the realtime log.
# -------------------------------------------------------------------------- #
echo ""
echo "=== Checking realtime output ==="

# A non-graphlike DEM handed to pymatching surfaces as "Invalid column in H".
if grep -q "Invalid column in H" "$REALTIME_LOG"; then
  echo "FAIL: found 'Invalid column in H' (decoder received a non-graphlike DEM)"
  return_code=1
fi

# Hard decoder-init / dispatch failures.
if grep -q "terminate called" "$REALTIME_LOG"; then
  echo "FAIL: found 'terminate called' (the app aborted)"
  return_code=1
fi
if grep -q "Decoder 0 not found" "$REALTIME_LOG"; then
  echo "FAIL: found 'Decoder 0 not found' (decoder was not registered)"
  return_code=1
fi
if grep -q "Error initializing decoders" "$REALTIME_LOG"; then
  echo "FAIL: found 'Error initializing decoders'"
  return_code=1
fi

# A "Number of corrections decoder found:" line MUST be present -- it proves the
# realtime decoding path actually ran to completion.
if ! grep -q "Number of corrections decoder found:" "$REALTIME_LOG"; then
  echo "FAIL: missing 'Number of corrections decoder found:' line (decoding did not complete)"
  return_code=1
fi

# Pull out the residual logical-error count and sanity check it.
num_non_zero_values=$(grep "Number of non-zero values measured :" "$REALTIME_LOG" | awk -F': ' '{print $2}' | tr -d '[:space:]')

if ! [[ "$num_non_zero_values" =~ ^[0-9]+$ ]]; then
  echo "FAIL: 'Number of non-zero values measured' is not a number (got '$num_non_zero_values')"
  return_code=1
elif [[ "$num_non_zero_values" -gt "$MAX_NON_ZERO" ]]; then
  echo "FAIL: residual logical errors ($num_non_zero_values) exceed ceiling ($MAX_NON_ZERO) -- decoder appears wired-but-wrong"
  return_code=1
else
  echo "Residual logical errors: $num_non_zero_values (ceiling $MAX_NON_ZERO) -- OK"
fi

echo ""
if [[ "$return_code" -eq 0 ]]; then
  echo "PASS: surface_code-4-yaml ($DECODER_TYPE, d=$DISTANCE) realtime decode succeeded"
else
  echo "FAIL: surface_code-4-yaml ($DECODER_TYPE, d=$DISTANCE) test failed"
fi

exit $return_code
