# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Two-process surface-code test: the surface_code-1-cqr application (simulated
# QPU + cudaq-realtime device_call channel) in one process, the standard
# decoding server (decoding_server) in the other. The server is configured from
# daemon YAML: a top-level server.transports block belongs to the daemon and the
# decoder block is the same YAML the app's --save_dem pass produces.
#
# The wire between the two processes is selected by QEC_DECODING_SERVER_TRANSPORT:
#   udp (default)  UDP loopback; runs anywhere.
#   cpu_roce       CPU RoCE RDMA channel; needs an RDMA device (real ConnectX
#                  or SoftRoCE/rdma_rxe) and the same topology env vars as
#                  CUDA-Q's CpuRoceChannelTester:
#                    CUDAQ_CPU_ROCE_TEST_CHANNEL_DEVICE / _CHANNEL_IP
#                    CUDAQ_CPU_ROCE_TEST_DAEMON_DEVICE  / _DAEMON_IP
#                  (e.g. a SoftRoCE self-loop: both = rxe_cudaq0 / 10.88.0.1)
#
# Set QEC_TWO_PROCESS_RECONFIG=1 to continue after the basic decode and exercise
# SIGHUP live decoder reconfiguration in the same long-lived server process.
#
# Expected args:
#   1: path to surface_code-1-cqr executable
#   2: distance
#   3: number_of_non_zero_values_threshold
#   4: number_of_corrections_decoder_threshold
#   5: path to decoding_server executable
#   6: num_rounds
#   7: decoder_window
#   8: decoder_type (optional, defaults to multi_error_lut)

set -euo pipefail

if [[ $# -lt 7 ]]; then
  echo "Error: Expected at least 7 arguments (got $#)" >&2
  exit 1
fi

EXE_PATH=$1
DISTANCE=$2
number_of_non_zero_values_threshold=$3
number_of_corrections_decoder_threshold=$4
SERVER_PATH=$5
NUM_ROUNDS=$6
DECODER_WINDOW=$7
DECODER_TYPE=${8:-multi_error_lut}

if [[ ! -x "$EXE_PATH" ]]; then
  echo "Error: surface_code-1-cqr executable not found or not executable: $EXE_PATH" >&2
  exit 1
fi
if [[ ! -x "$SERVER_PATH" ]]; then
  echo "Error: decoding_server executable not found or not executable: $SERVER_PATH" >&2
  exit 1
fi

export CUDAQ_DEFAULT_SIMULATOR=${CUDAQ_DEFAULT_SIMULATOR:-stim}

NUM_SHOTS=${QEC_TWO_PROCESS_NUM_SHOTS:-1000}
RECONFIG_ENABLED=${QEC_TWO_PROCESS_RECONFIG:-0}
RECONFIG_REQUESTS=${QEC_RECONFIG_REQUESTS:-2}
RECONFIG_NUM_SHOTS=${QEC_RECONFIG_NUM_SHOTS:-100}
SERVER_TIMEOUT=${QEC_TWO_PROCESS_SERVER_TIMEOUT:-300}
STATS_INTERVAL=${QEC_RECONFIG_STATS_INTERVAL:-0}
TRANSPORT=${QEC_DECODING_SERVER_TRANSPORT:-udp}
VERBOSE=${QEC_TWO_PROCESS_VERBOSE:-${QEC_RECONFIG_VERBOSE:-}}

if [[ "$TRANSPORT" != "udp" && "$TRANSPORT" != "cpu_roce" ]]; then
  echo "Error: surface_code-1-cqr two-process HOST_CALL test supports udp and cpu_roce only (got '$TRANSPORT')" >&2
  echo "       gpu_roce uses the standalone server's device-graph/Hololink path." >&2
  exit 1
fi
if ! [[ "$NUM_SHOTS" =~ ^[0-9]+$ ]] || [[ "$NUM_SHOTS" -lt 1 ]]; then
  echo "Error: QEC_TWO_PROCESS_NUM_SHOTS must be a positive integer" >&2
  exit 1
fi
if ! [[ "$RECONFIG_REQUESTS" =~ ^[0-9]+$ ]] || [[ "$RECONFIG_REQUESTS" -lt 1 ]]; then
  echo "Error: QEC_RECONFIG_REQUESTS must be a positive integer" >&2
  exit 1
fi
if ! [[ "$RECONFIG_NUM_SHOTS" =~ ^[0-9]+$ ]] || [[ "$RECONFIG_NUM_SHOTS" -lt 1 ]]; then
  echo "Error: QEC_RECONFIG_NUM_SHOTS must be a positive integer" >&2
  exit 1
fi
if ! [[ "$STATS_INTERVAL" =~ ^[0-9]+$ ]]; then
  echo "Error: QEC_RECONFIG_STATS_INTERVAL must be a non-negative integer" >&2
  exit 1
fi

WORKDIR=${QEC_TWO_PROCESS_WORKDIR:-$(mktemp -d /tmp/qec-2proc.XXXXXX)}
CONFIG_FILE=$WORKDIR/multi_error_lut.yml
SLIDING_CONFIG=$WORKDIR/sliding_window.yml
LIVE_CONFIG=$WORKDIR/live.yml
SERVER_LOG=$WORKDIR/server.log
APP_LOG=$WORKDIR/client.log
SAVE_MULTI_LOG=$WORKDIR/save_multi_error_lut.log
SAVE_SLIDING_LOG=$WORKDIR/save_sliding_window.log
ENDPOINT_FILE=${QEC_DECODING_SERVER_ENDPOINT_FILE:-$WORKDIR/endpoint.env}
SERVER_PID=""
CLIENT_RUNS=0
TOTAL_CLIENT_SHOTS=0
FAILURE_REPORTED=0

cleanup() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill -TERM "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  if [[ -z "${KEEP_LOG_FILES:-}" && -z "${QEC_TWO_PROCESS_WORKDIR:-}" ]]; then
    rm -rf "$WORKDIR"
  else
    echo "Kept two-process logs in $WORKDIR"
  fi
}

print_log_excerpt() {
  local label=$1
  local path=$2
  local lines=${3:-80}

  [[ -f "$path" ]] || return 0
  echo "----- $label ($path, last $lines lines) -----" >&2
  tail -n "$lines" "$path" >&2 || true
}

dump_failure_logs() {
  if [[ "$FAILURE_REPORTED" == "1" ]]; then
    return 0
  fi
  FAILURE_REPORTED=1
  echo "==> two-process test failed; saved logs are in $WORKDIR" >&2
  print_log_excerpt "server log" "$SERVER_LOG" 140
  print_log_excerpt "client log" "$APP_LOG" 100
  print_log_excerpt "multi_error_lut generation log" "$SAVE_MULTI_LOG" 50
  print_log_excerpt "sliding_window generation log" "$SAVE_SLIDING_LOG" 50
}

fail() {
  echo "Error: $*" >&2
  dump_failure_logs
  exit 1
}

maybe_print_full_log() {
  local label=$1
  local path=$2

  if [[ -n "$VERBOSE" && -f "$path" ]]; then
    echo "----- $label ($path) -----"
    cat "$path"
  fi
}

trap cleanup EXIT
trap dump_failure_logs ERR

wait_for_log() {
  local pattern=$1
  local timeout_s=${2:-15}
  local waited=0
  while (( waited < timeout_s * 10 )); do
    if grep -Eq "$pattern" "$SERVER_LOG" 2>/dev/null; then
      return 0
    fi
    sleep 0.1
    waited=$((waited + 1))
  done
  fail "timed out waiting for server log pattern: $pattern"
}

write_server_transport_yaml() {
  echo "server:"
  echo "  transports:"
  echo "    - type: $TRANSPORT"
  echo "      port: 0"
  if [[ "$TRANSPORT" == "cpu_roce" ]]; then
    echo "      device: ${CUDAQ_CPU_ROCE_TEST_DAEMON_DEVICE:-mlx5_0}"
    echo "      local_ip: ${CUDAQ_CPU_ROCE_TEST_DAEMON_IP:-10.0.0.2}"
  fi
}

write_live_config() {
  local decoder_config=$1
  {
    write_server_transport_yaml
    cat "$decoder_config"
  } >"$LIVE_CONFIG"
}

write_invalid_live_config() {
  {
    write_server_transport_yaml
    echo "decoders: []"
  } >"$LIVE_CONFIG"
}

generate_config() {
  local decoder_type=$1
  local output=$2
  local log=$3

  echo "==> generating $decoder_type config: $output"
  if ! "$EXE_PATH" --distance "$DISTANCE" --num_rounds "$NUM_ROUNDS" \
    --num_shots 10 --save_dem "$output" \
    --decoder_window "$DECODER_WINDOW" --decoder_type "$decoder_type" \
    >"$log" 2>&1; then
    fail "failed to generate $decoder_type config"
  fi
  maybe_print_full_log "$decoder_type generation log" "$log"
}

run_client() {
  local decoder_type=$1
  local config_file=$2
  local label=$3
  local shots=$4

  echo "==> client run: $label decoder=$decoder_type shots=$shots"
  {
    echo "----- $label decoder=$decoder_type shots=$shots endpoint=$ENDPOINT_FILE -----"
    QEC_DECODING_SERVER_ENDPOINT_FILE=$ENDPOINT_FILE \
      QEC_DECODING_SERVER_TRANSPORT=$TRANSPORT \
      "$EXE_PATH" --distance "$DISTANCE" --num_shots "$shots" \
        --load_dem "$config_file" --num_rounds "$NUM_ROUNDS" \
        --decoder_window "$DECODER_WINDOW" --decoder_type "$decoder_type"
  } >>"$APP_LOG" 2>&1 || fail "client run failed: $label"
  maybe_print_full_log "client log" "$APP_LOG"

  local inproc_count
  inproc_count=$(grep "CQR service dispatch count:" "$APP_LOG" \
    | tail -1 | awk -F': ' '{print $2}' || true)
  if [[ "$inproc_count" != "0" ]]; then
    fail "expected in-process CQR dispatch count 0 after $label, got '$inproc_count'"
  fi

  CLIENT_RUNS=$((CLIENT_RUNS + 1))
  TOTAL_CLIENT_SHOTS=$((TOTAL_CLIENT_SHOTS + shots))
}

check_latest_client_counts() {
  local num_non_zero_values
  local num_corrections_decoder

  num_non_zero_values=$(grep "Number of non-zero values measured :" "$APP_LOG" \
    | tail -1 | awk -F': ' '{print $2}' || true)
  num_corrections_decoder=$(grep "Number of corrections decoder found:" "$APP_LOG" \
    | tail -1 | awk -F': ' '{print $2}' || true)

  if ! [[ "$num_non_zero_values" =~ ^[0-9]+$ ]]; then
    fail "Number of non-zero values measured is not a number"
  fi
  if ! [[ "$num_corrections_decoder" =~ ^[0-9]+$ ]]; then
    fail "Number of corrections decoder found is not a number"
  fi
  if [[ "$num_non_zero_values" -gt "$number_of_non_zero_values_threshold" ]]; then
    fail "Number of non-zero values measured is greater than $number_of_non_zero_values_threshold (unexpected)"
  fi
  if [[ "$num_corrections_decoder" -lt "$number_of_corrections_decoder_threshold" ]]; then
    fail "Number of corrections decoder found is less than $number_of_corrections_decoder_threshold (unexpected)"
  fi
}

apply_config() {
  local config_file=$1
  local expected_pattern=$2

  write_live_config "$config_file"
  kill -HUP "$SERVER_PID"
  wait_for_log "$expected_pattern" 20
}

server_dispatch_count() {
  sed -n 's/^QEC_DECODING_SERVER_DISPATCHED count=\([0-9][0-9]*\)$/\1/p' \
    "$SERVER_LOG" | tail -1
}

echo "==> workdir: $WORKDIR"
generate_config "$DECODER_TYPE" "$CONFIG_FILE" "$SAVE_MULTI_LOG"
write_live_config "$CONFIG_FILE"

# CI guard only; production daemons omit --timeout and run until signalled.
SERVER_ARGS=(--config="$LIVE_CONFIG" --timeout="$SERVER_TIMEOUT")
if [[ "$STATS_INTERVAL" -gt 0 ]]; then
  SERVER_ARGS+=(--stats-interval="$STATS_INTERVAL")
fi

echo "==> starting decoding_server using YAML server.transports=$TRANSPORT"
QEC_DECODING_SERVER_ENDPOINT_FILE=$ENDPOINT_FILE \
  "$SERVER_PATH" "${SERVER_ARGS[@]}" >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!

wait_for_log "QEC_DECODING_SERVER_READY" 30
SERVER_PORT=$(sed -n 's/.*QEC_DECODING_SERVER_READY port=\([0-9][0-9]*\).*/\1/p' "$SERVER_LOG" | head -1)
echo "==> server ready: pid=$SERVER_PID port=${SERVER_PORT:-n/a} endpoint=$ENDPOINT_FILE"

run_client "$DECODER_TYPE" "$CONFIG_FILE" "basic" "$NUM_SHOTS"
check_latest_client_counts

if [[ "$RECONFIG_ENABLED" == "1" || "$RECONFIG_ENABLED" == "ON" || \
      "$RECONFIG_ENABLED" == "TRUE" || "$RECONFIG_ENABLED" == "true" ]]; then
  if [[ "$DECODER_TYPE" != "multi_error_lut" ]]; then
    fail "QEC_TWO_PROCESS_RECONFIG requires decoder_type multi_error_lut (got '$DECODER_TYPE')"
  fi

  generate_config sliding_window "$SLIDING_CONFIG" "$SAVE_SLIDING_LOG"

  for request in $(seq 1 "$RECONFIG_REQUESTS"); do
    run_client multi_error_lut "$CONFIG_FILE" "pre-reconfigure/$request" \
      "$RECONFIG_NUM_SHOTS"
  done

  apply_config "$SLIDING_CONFIG" \
    "QEC_DECODING_SERVER_CONFIG_APPLIED.*type0=sliding_window"
  for request in $(seq 1 "$RECONFIG_REQUESTS"); do
    run_client sliding_window "$SLIDING_CONFIG" "post-reconfigure/$request" \
      "$RECONFIG_NUM_SHOTS"
  done

  write_invalid_live_config
  kill -HUP "$SERVER_PID"
  wait_for_log "QEC_DECODING_SERVER_CONFIG_FAILED.*serving_old" 20
  run_client sliding_window "$SLIDING_CONFIG" "after-rejected-config" \
    "$RECONFIG_NUM_SHOTS"

  if [[ "$STATS_INTERVAL" -gt 0 ]]; then
    echo "==> waiting for daemon-owned periodic live stats"
    wait_for_log "QEC_DECODING_SERVER_STATS.*dispatched=" 20
  fi
fi

kill -TERM "$SERVER_PID"
wait "$SERVER_PID" 2>/dev/null || true
SERVER_PID=""
trap - EXIT

server_count=$(server_dispatch_count)
min_server_dispatches=$((TOTAL_CLIENT_SHOTS * 3))
if ! [[ "$server_count" =~ ^[0-9]+$ ]] || \
   [[ "$server_count" -lt "$min_server_dispatches" ]]; then
  dump_failure_logs
  echo "Error: server dispatch count '$server_count' is missing or below $min_server_dispatches; device_calls did not cross the $TRANSPORT wire" >&2
  exit 1
fi

echo "Server dispatch count check passed ($server_count dispatches over $TRANSPORT)"
echo "Two-process test completed for distance $DISTANCE; client_runs=$CLIENT_RUNS shots=$TOTAL_CLIENT_SHOTS"
grep -E 'QEC_DECODING_SERVER_(READY|CONFIG_APPLIED|CONFIG_FAILED|STATS|DISPATCHED|MAX_CONCURRENT_DECODERS)' "$SERVER_LOG" || true

cleanup
exit 0
