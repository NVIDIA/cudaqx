# ============================================================================ #
# Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Abort on failure
set -e
set -o pipefail

return_code=0
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# Expected args:
#  ${CMAKE_CURRENT_BINARY_DIR}/surface_code-1-local
#  ${CMAKE_CURRENT_BINARY_DIR}/surface_code-1-local{-quantinuum-emulate}
#  distance
#  number_of_non_zero_values_threshold
#  number_of_corrections_decoder_threshold
#  Path to server executable
#  num_rounds
#  decoder_window
#  Path to libcudaq-qec-realtime-decoding-quantinuum-private.so
#  decoder_type (optional, defaults to multi_error_lut)
#  sw_window_size (optional, for sliding_window decoder, defaults to decoder_window)
#  sw_step_size (optional, for sliding_window decoder, defaults to 1)
#  inproc_rpc (optional, pass inproc_rpc/1/true to route through HOST_CALL)

# Check that at least 9 arguments are provided.
if [[ $# -lt 9 ]]; then
  echo "Error: Expected at least 9 arguments (got $#)"
  exit 1
fi

EXE_PATH1=$1
EXE_PATH2=$2
DISTANCE=$3
number_of_non_zero_values_threshold=$4
number_of_corrections_decoder_threshold=$5
SERVER_EXECUTABLE=$6
NUM_ROUNDS=$7 
DECODER_WINDOW=$8
LIB_DIR=$9
DECODER_TYPE=${10:-multi_error_lut}
SW_WINDOW_SIZE=${11:-$DECODER_WINDOW}
SW_STEP_SIZE=${12:-1}
INPROC_RPC=${13:-}
EXTRA_CLI_ARGS=${EXTRA_CLI_ARGS:-}

REALTIME_ENV=()
if [[ "$INPROC_RPC" == "inproc_rpc" || "$INPROC_RPC" == "1" ||
      "$INPROC_RPC" == "true" || "$INPROC_RPC" == "TRUE" ||
      "$INPROC_RPC" == "on" || "$INPROC_RPC" == "ON" ]]; then
  REALTIME_ENV=(env CUDAQ_QEC_REALTIME_MODE=inproc_rpc)
fi

if [[ "$DECODER_TYPE" == "trt_decoder" ]] && ! nvidia-smi -L >/dev/null 2>&1; then
  echo "Skipping trt_decoder surface-code test: no CUDA GPU visible."
  exit 0
fi

export CUDAQ_DEFAULT_SIMULATOR=stim

ERROR_RATE=0.01
NUM_SHOTS=1000

# Get timestamp suffix in YYYY-MM-DD-HH-MM-SS, with random number appended using /dev/urandom.
timestamp=$(date +%Y-%m-%d-%H-%M-%S)
RNG_SUFFIX=$(od -An -N4 -i /dev/urandom | tr -d ' ')
# Remove any negative sign.
RNG_SUFFIX=$(echo $RNG_SUFFIX | sed 's/-//g')
FULL_SUFFIX=$timestamp-$RNG_SUFFIX

CONFIG_FILE=config-${FULL_SUFFIX}.yml
TRT_ONNX_FILE=
TRT_ONNX_ARGS=()
if [[ "$DECODER_TYPE" == "trt_decoder" ]]; then
  TRT_ONNX_FILE=predecoder-${FULL_SUFFIX}.onnx
  TRT_ONNX_ARGS=(--onnx_path "$TRT_ONNX_FILE")
fi

# Generate the config file using the first executable.
$EXE_PATH1 --distance $DISTANCE --num_rounds $NUM_ROUNDS --num_shots $NUM_SHOTS --save_dem $CONFIG_FILE --decoder_window $DECODER_WINDOW --decoder_type $DECODER_TYPE --sw_window_size $SW_WINDOW_SIZE --sw_step_size $SW_STEP_SIZE "${TRT_ONNX_ARGS[@]}" $EXTRA_CLI_ARGS | tee save_dem-$FULL_SUFFIX.log

if [[ "$DECODER_TYPE" == "trt_decoder" ]]; then
  python3 "$SCRIPT_DIR/../generate_trt_split_predecoder_onnx.py" \
    --config "$CONFIG_FILE" --out "$TRT_ONNX_FILE" \
    | tee generate_onnx-$FULL_SUFFIX.log
fi

export CUDAQ_DUMP_JIT_IR=${CUDAQ_DUMP_JIT_IR:-0}

# This is a temporary workaround to allow the Quantinuum tests to run with the
# current compiler. Without this environment variable, one is likely to
# encounter errors like:
# invalid instruction found in adaptive QIR profile:   %0 = alloca [8 x i1], align 1
# Disable once
# https://gitlab-master.nvidia.com/cuda-quantum/cuda-quantum/-/merge_requests/24
# is merged.
# export QIR_ALLOW_ALL_INSTRUCTIONS=1



# Use the config file using the second executable.
echo Running "${REALTIME_ENV[@]}" $EXE_PATH2 --distance $DISTANCE --num_shots $NUM_SHOTS --load_dem $CONFIG_FILE --num_rounds $NUM_ROUNDS --decoder_window $DECODER_WINDOW --decoder_type $DECODER_TYPE --sw_window_size $SW_WINDOW_SIZE --sw_step_size $SW_STEP_SIZE "${TRT_ONNX_ARGS[@]}" $EXTRA_CLI_ARGS
"${REALTIME_ENV[@]}" $EXE_PATH2 --distance $DISTANCE --num_shots $NUM_SHOTS --load_dem $CONFIG_FILE --num_rounds $NUM_ROUNDS --decoder_window $DECODER_WINDOW --decoder_type $DECODER_TYPE --sw_window_size $SW_WINDOW_SIZE --sw_step_size $SW_STEP_SIZE "${TRT_ONNX_ARGS[@]}" $EXTRA_CLI_ARGS |& tee load_dem-$FULL_SUFFIX.log

# If CUDAQ_DUMP_JIT_IR is "1", then extract the QIR from the
# load_dem-$FULL_SUFFIX.log file and place it in qir-$FULL_SUFFIX.ll.
if [[ "${CUDAQ_DUMP_JIT_IR}" == "1" ]]; then
  # The QIR starts with "ModuleID" and ends with "backwards_branching"
  QIR=$(sed -n '/ModuleID/,/backwards_branching/p' load_dem-$FULL_SUFFIX.log)
  echo "Writing QIR to qir-$FULL_SUFFIX.ll"
  echo "$QIR" > qir-$FULL_SUFFIX.ll
fi


# Look for results like this in the output:
# Number of non-zero values measured : 2
# Number of corrections decoder found: 48

# Make sure that the first value is < 10, and the second value is > 10.

num_non_zero_values=$(grep "Number of non-zero values measured :" load_dem-$FULL_SUFFIX.log | awk -F': ' '{print $2}')
num_corrections_decoder=$(grep "Number of corrections decoder found:" load_dem-$FULL_SUFFIX.log | awk -F': ' '{print $2}')

# Make sure that the value is a number.
if ! [[ "$num_non_zero_values" =~ ^[0-9]+$ ]]; then
  echo "Error: Number of non-zero values measured is not a number"
  return_code=1
fi

if ! [[ "$num_corrections_decoder" =~ ^[0-9]+$ ]]; then
  echo "Error: Number of corrections decoder found is not a number"
  return_code=1
fi

if [[ "$num_non_zero_values" -gt $number_of_non_zero_values_threshold ]]; then
  echo "Error: Number of non-zero values measured is greater than $number_of_non_zero_values_threshold (unexpected)"
  return_code=1
fi

if [[ "$num_corrections_decoder" -lt $number_of_corrections_decoder_threshold ]]; then
  echo "Error: Number of corrections decoder found is less than $number_of_corrections_decoder_threshold (unexpected)"
  return_code=1
fi

echo "Test completed for distance $DISTANCE with return code $return_code"

# ============================================================================ #
# Test --save_syndrome and --load_syndrome functionality
# ============================================================================ #
echo ""
echo "=== Testing syndrome save/load functionality ==="

SYNDROME_FILE=syndromes-${FULL_SUFFIX}.txt
SYNDROME_NUM_SHOTS=10  # Use fewer shots for syndrome test

# Step 1: Run simulation with --save_syndrome to capture syndrome data
# Use local executable for syndrome capture (works with any platform)
echo "Step 1: Saving syndromes to $SYNDROME_FILE"
"${REALTIME_ENV[@]}" $EXE_PATH1 --distance $DISTANCE --num_shots $SYNDROME_NUM_SHOTS --load_dem $CONFIG_FILE --num_rounds $NUM_ROUNDS --decoder_window $DECODER_WINDOW --decoder_type $DECODER_TYPE --sw_window_size $SW_WINDOW_SIZE --sw_step_size $SW_STEP_SIZE "${TRT_ONNX_ARGS[@]}" $EXTRA_CLI_ARGS --save_syndrome $SYNDROME_FILE |& tee save_syndrome-$FULL_SUFFIX.log

# Check that the syndrome file was created
if [[ ! -f "$SYNDROME_FILE" ]]; then
  echo "Error: Syndrome file was not created"
  return_code=1
else
  echo "Syndrome file created successfully"
  
  # Check that the file contains expected markers
  if grep -q "SHOT_START" $SYNDROME_FILE && grep -q "CORRECTIONS_START" $SYNDROME_FILE; then
    echo "Syndrome file contains expected markers"
  else
    echo "Error: Syndrome file missing expected markers"
    return_code=1
  fi
  
  # Step 2: Replay syndromes with --load_syndrome
  # Use local executable for replay (doesn't need quantum simulation)
  echo "Step 2: Replaying syndromes from $SYNDROME_FILE"
  "${REALTIME_ENV[@]}" $EXE_PATH1 --distance $DISTANCE --num_shots $SYNDROME_NUM_SHOTS --load_dem $CONFIG_FILE --num_rounds $NUM_ROUNDS --decoder_window $DECODER_WINDOW --decoder_type $DECODER_TYPE --sw_window_size $SW_WINDOW_SIZE --sw_step_size $SW_STEP_SIZE "${TRT_ONNX_ARGS[@]}" $EXTRA_CLI_ARGS --load_syndrome $SYNDROME_FILE |& tee load_syndrome-$FULL_SUFFIX.log
  
  # Check for successful replay
  if grep -q "Replay complete" load_syndrome-$FULL_SUFFIX.log; then
    echo "Syndrome replay completed successfully"
    
    # Check if corrections matched (if verification was performed)
    if grep -q "SUCCESS: All corrections match" load_syndrome-$FULL_SUFFIX.log; then
      echo "Correction verification: PASSED"
    elif grep -q "mismatched" load_syndrome-$FULL_SUFFIX.log; then
      echo "Error: Corrections did not match during replay"
      return_code=1
    fi
  else
    echo "Error: Syndrome replay did not complete"
    return_code=1
  fi
fi

echo "Syndrome save/load test completed"

# Remove the log files and the config file, unless an environment variable is set.
if [[ -z "${KEEP_LOG_FILES}" ]]; then
  rm -f load_dem-$FULL_SUFFIX.log save_dem-$FULL_SUFFIX.log server-$FULL_SUFFIX.log $CONFIG_FILE
  rm -f generate_onnx-$FULL_SUFFIX.log $TRT_ONNX_FILE
  rm -f save_syndrome-$FULL_SUFFIX.log load_syndrome-$FULL_SUFFIX.log $SYNDROME_FILE
fi

exit $return_code
