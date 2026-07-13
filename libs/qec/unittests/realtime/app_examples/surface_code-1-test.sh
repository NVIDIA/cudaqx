# ============================================================================ #
# Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Abort on failure
set -e

return_code=0

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
EXTRA_CLI_ARGS=${EXTRA_CLI_ARGS:-}

# The inproc_rpc realtime path is served by the device-graph scheduler, which
# uses device-side graph launch -- compute capability 9.0+ (Hopper) only; the
# dispatch kernel's TRIGGER_GRAPH interception is compiled out below sm_90 and
# the enqueue RPC surfaces the raw sentinel as a non-zero status.  Skip on
# older GPUs (e.g. A100/sm_80 CI runners), matching the skip in
# test_realtime_qldpc_graph_decoding.  Exit code 77 pairs with the test's
# SKIP_RETURN_CODE property.
if [[ "${CUDAQ_QEC_REALTIME_MODE:-}" == "inproc_rpc" ]]; then
  compute_cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1)
  if [[ -n "$compute_cap" && "${compute_cap%%.*}" -lt 9 ]]; then
    echo "SKIP: CUDAQ_QEC_REALTIME_MODE=inproc_rpc requires device-side graph" \
         "launch (compute capability 9.0+); found ${compute_cap}"
    exit 77
  fi
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

# Generate the config file using the first executable.
$EXE_PATH1 --distance $DISTANCE --num_rounds $NUM_ROUNDS --num_shots $NUM_SHOTS --save_dem $CONFIG_FILE --decoder_window $DECODER_WINDOW --decoder_type $DECODER_TYPE --sw_window_size $SW_WINDOW_SIZE --sw_step_size $SW_STEP_SIZE $EXTRA_CLI_ARGS | tee save_dem-$FULL_SUFFIX.log

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
echo Running $EXE_PATH2 --distance $DISTANCE --num_shots $NUM_SHOTS --load_dem $CONFIG_FILE --num_rounds $NUM_ROUNDS --decoder_window $DECODER_WINDOW --decoder_type $DECODER_TYPE --sw_window_size $SW_WINDOW_SIZE --sw_step_size $SW_STEP_SIZE $EXTRA_CLI_ARGS
$EXE_PATH2 --distance $DISTANCE --num_shots $NUM_SHOTS --load_dem $CONFIG_FILE --num_rounds $NUM_ROUNDS --decoder_window $DECODER_WINDOW --decoder_type $DECODER_TYPE --sw_window_size $SW_WINDOW_SIZE --sw_step_size $SW_STEP_SIZE $EXTRA_CLI_ARGS |& tee load_dem-$FULL_SUFFIX.log

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

# For the cqr host-dispatch variant, verify the device_calls actually crossed
# the cudaq-realtime ring to the in-process decoding server (the count is 0 if
# they silently bypassed to a direct trampoline).
if [[ "${CUDAQ_DEVICE_CALL_CHANNEL:-}" == "host_dispatch" ]]; then
  cqr_dispatch_count=$(grep "CQR service dispatch count:" load_dem-$FULL_SUFFIX.log | awk -F': ' '{print $2}')
  if ! [[ "$cqr_dispatch_count" =~ ^[0-9]+$ ]] || [[ "$cqr_dispatch_count" -eq 0 ]]; then
    echo "Error: CQR service dispatch count is missing or zero; device_calls did not traverse host dispatch"
    return_code=1
  else
    echo "CQR service dispatch count check passed ($cqr_dispatch_count dispatches)"
  fi
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
$EXE_PATH1 --distance $DISTANCE --num_shots $SYNDROME_NUM_SHOTS --load_dem $CONFIG_FILE --num_rounds $NUM_ROUNDS --decoder_window $DECODER_WINDOW --decoder_type $DECODER_TYPE --sw_window_size $SW_WINDOW_SIZE --sw_step_size $SW_STEP_SIZE $EXTRA_CLI_ARGS --save_syndrome $SYNDROME_FILE |& tee save_syndrome-$FULL_SUFFIX.log

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
  $EXE_PATH1 --distance $DISTANCE --num_shots $SYNDROME_NUM_SHOTS --load_dem $CONFIG_FILE --num_rounds $NUM_ROUNDS --decoder_window $DECODER_WINDOW --decoder_type $DECODER_TYPE --sw_window_size $SW_WINDOW_SIZE --sw_step_size $SW_STEP_SIZE $EXTRA_CLI_ARGS --load_syndrome $SYNDROME_FILE |& tee load_syndrome-$FULL_SUFFIX.log
  
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
  rm -f save_syndrome-$FULL_SUFFIX.log load_syndrome-$FULL_SUFFIX.log $SYNDROME_FILE
fi

exit $return_code
