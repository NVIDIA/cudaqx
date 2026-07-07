# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Two-process surface-code test: the surface_code-1-cqr application (simulated
# QPU + cudaq-realtime device_call channel) in one process, the standard
# decoding server (qec_decoding_daemon) in the other. The daemon is configured
# from the same YAML the app's --save_dem pass produces, so the decoder setup
# crosses the process boundary as configuration, not code.
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
# Expected args:
#   1: path to surface_code-1-cqr executable
#   2: distance
#   3: number_of_non_zero_values_threshold
#   4: number_of_corrections_decoder_threshold
#   5: path to qec_decoding_daemon executable
#   6: num_rounds
#   7: decoder_window
#   8: decoder_type (optional, defaults to multi_error_lut)

set -e

return_code=0

if [[ $# -lt 7 ]]; then
  echo "Error: Expected at least 7 arguments (got $#)"
  exit 1
fi

EXE_PATH=$1
DISTANCE=$2
number_of_non_zero_values_threshold=$3
number_of_corrections_decoder_threshold=$4
DAEMON_PATH=$5
NUM_ROUNDS=$6
DECODER_WINDOW=$7
DECODER_TYPE=${8:-multi_error_lut}

export CUDAQ_DEFAULT_SIMULATOR=stim

NUM_SHOTS=1000

timestamp=$(date +%Y-%m-%d-%H-%M-%S)
RNG_SUFFIX=$(od -An -N4 -i /dev/urandom | tr -d ' ' | sed 's/-//g')
FULL_SUFFIX=$timestamp-$RNG_SUFFIX

CONFIG_FILE=config-2proc-${FULL_SUFFIX}.yml
DAEMON_LOG=daemon-2proc-${FULL_SUFFIX}.log
APP_LOG=load_dem-2proc-${FULL_SUFFIX}.log

# [1] Generate the decoder config (no realtime channel needed for this pass).
$EXE_PATH --distance $DISTANCE --num_rounds $NUM_ROUNDS --num_shots $NUM_SHOTS \
  --save_dem $CONFIG_FILE --decoder_window $DECODER_WINDOW \
  --decoder_type $DECODER_TYPE | tee save_dem-2proc-$FULL_SUFFIX.log

# [2] Start the decoding server on an ephemeral port with that config.
# For udp the READY port is the UDP data port; for cpu_roce it is the TCP
# rendezvous port (the RDMA wire itself is negotiated via QP/rkey exchange).
TRANSPORT=${QEC_DECODING_SERVER_TRANSPORT:-udp}
DAEMON_ARGS=(--config=$CONFIG_FILE --transport=$TRANSPORT --port=0 --timeout=300)
if [[ "$TRANSPORT" == "cpu_roce" ]]; then
  DAEMON_ARGS+=(--device=${CUDAQ_CPU_ROCE_TEST_DAEMON_DEVICE:-mlx5_0})
  DAEMON_ARGS+=(--local-ip=${CUDAQ_CPU_ROCE_TEST_DAEMON_IP:-10.0.0.2})
fi
$DAEMON_PATH "${DAEMON_ARGS[@]}" \
  > $DAEMON_LOG 2>&1 &
DAEMON_PID=$!
cleanup() {
  kill -TERM $DAEMON_PID 2>/dev/null || true
  wait $DAEMON_PID 2>/dev/null || true
}
trap cleanup EXIT

# Wait for the READY line and parse the port.
DAEMON_PORT=""
for _ in $(seq 1 100); do
  DAEMON_PORT=$(grep -m1 "QEC_DECODING_DAEMON_READY" $DAEMON_LOG 2>/dev/null \
    | sed -n 's/.*port=\([0-9]\+\).*/\1/p')
  [[ -n "$DAEMON_PORT" ]] && break
  sleep 0.1
done
if [[ -z "$DAEMON_PORT" ]]; then
  echo "Error: daemon did not print QEC_DECODING_DAEMON_READY"
  cat $DAEMON_LOG
  exit 1
fi
echo "Decoding server ready on $TRANSPORT port $DAEMON_PORT"

# [3] Run the experiment; QEC_DECODING_SERVER_PORT routes every
# cudaq::qec::decoding device_call over the selected channel to the daemon
# (the app reads QEC_DECODING_SERVER_TRANSPORT for the channel type).
QEC_DECODING_SERVER_PORT=$DAEMON_PORT \
  $EXE_PATH --distance $DISTANCE --num_shots $NUM_SHOTS \
  --load_dem $CONFIG_FILE --num_rounds $NUM_ROUNDS \
  --decoder_window $DECODER_WINDOW --decoder_type $DECODER_TYPE \
  |& tee $APP_LOG

# [4] Stop the daemon and collect its dispatch count.
kill -TERM $DAEMON_PID
wait $DAEMON_PID 2>/dev/null || true
trap - EXIT

num_non_zero_values=$(grep "Number of non-zero values measured :" $APP_LOG | awk -F': ' '{print $2}')
num_corrections_decoder=$(grep "Number of corrections decoder found:" $APP_LOG | awk -F': ' '{print $2}')
inproc_dispatch_count=$(grep "CQR service dispatch count:" $APP_LOG | awk -F': ' '{print $2}')
daemon_dispatch_count=$(grep "QEC_DECODING_DAEMON_DISPATCHED" $DAEMON_LOG | sed -n 's/.*count=\([0-9]\+\).*/\1/p')

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

# Two-process self-verification:
#  - the app's in-process service count must be 0 (nothing decoded locally),
#  - the daemon's dispatch count must cover every shot's device_calls
#    (>= 3 per shot: reset_decoder + enqueues + get_corrections).
if [[ "$inproc_dispatch_count" != "0" ]]; then
  echo "Error: expected in-process CQR dispatch count 0 (got '$inproc_dispatch_count'); decode did not stay in the daemon"
  return_code=1
fi
min_daemon_dispatches=$((NUM_SHOTS * 3))
if ! [[ "$daemon_dispatch_count" =~ ^[0-9]+$ ]] || \
   [[ "$daemon_dispatch_count" -lt $min_daemon_dispatches ]]; then
  echo "Error: daemon dispatch count '$daemon_dispatch_count' is missing or below $min_daemon_dispatches; device_calls did not cross the $TRANSPORT wire"
  cat $DAEMON_LOG
  return_code=1
else
  echo "Daemon dispatch count check passed ($daemon_dispatch_count dispatches over $TRANSPORT)"
fi

echo "Two-process test completed for distance $DISTANCE with return code $return_code"

# Clean up log/config files unless instructed to keep them.
if [[ -z "${KEEP_LOG_FILES}" ]]; then
  rm -f $CONFIG_FILE $DAEMON_LOG $APP_LOG save_dem-2proc-$FULL_SUFFIX.log
fi

exit $return_code
