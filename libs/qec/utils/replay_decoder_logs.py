# ============================================================================ #
# Copyright (c) 2025 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# This script allows a user to replay a real-time decoder log file (assuming the
# right instrumentation is enabled). It can be used to compare the online results
# to the offline results and/or replay the data with a different config file.

# How to use this script:
# python3 replay_decoder_logs.py --config config.yml --decoder-log decoder.log

import sys
import argparse
import os
import numpy
import yaml
import cudaq_qec as qec


# ---------------------------------------------------------------------------- #
# Helper function to convert a sparse list to a dense matrix. -1 is the row delimiter.
def sparse_to_dense(sparse_list, num_rows, num_cols, dtype=numpy.uint8):
    mat = numpy.zeros((num_rows, num_cols), dtype=dtype)
    row = 0
    for idx in sparse_list:
        if idx == -1:
            row += 1
        else:
            mat[row, idx] = 1
    return mat


# ---------------------------------------------------------------------------- #
# Traverse the decoder log file looking for decode calls. Note that when a
# decoder is created, a dummy decode call is made to "warm up" the decoder, so
# you may see more decode calls than shots.
def parse_decoder_log(decoder_log_file, log_detectors_sparse, log_errors_sparse,
                      log_observables_dense, decoder_id_list):
    # running id of the last decoder seen (needed since the decoder id is not
    # included in the 1 very verbose decode log message).
    last_decoder_id = -1
    print(f'Parsing decoder log file {decoder_log_file}...')
    enqueue_msg = "Entering enqueue_syndromes_ui64 for decoder id: "  # needed for last_decoder_id
    with open(decoder_log_file, 'r') as f:
        for line in f:
            line = line.strip()
            if enqueue_msg in line:
                # Needed for last_decoder_id.
                last_decoder_id = int(line.split(enqueue_msg)[1].split(" ")[0])
            if "[DecoderStats]" in line:
                line = line.split("[DecoderStats]")[1]
                # print(line)
                if "InputDetectors:" in line:  # this is a decode call
                    line = line.split(" ")
                    for elem in line:
                        if ":" in elem:
                            key, value = elem.split(":")
                            # print(key, value)
                            if key == "InputDetectors":
                                if value == "":
                                    log_detectors_sparse.append([])
                                else:
                                    log_detectors_sparse.append(
                                        [int(x) for x in value.split(",")])
                                if last_decoder_id == -1:
                                    print(
                                        f"Error: last_decoder_id is -1. This is a fatal error processing the log file."
                                    )
                                    exit(1)
                                decoder_id_list.append(last_decoder_id)
                            elif key == "Errors":
                                if value == "":
                                    log_errors_sparse.append([])
                                else:
                                    log_errors_sparse.append(
                                        [int(x) for x in value.split(",")])
                            elif key == "ObservableCorrectionsThisCall":
                                log_observables_dense.append(
                                    [int(x) for x in value.split(",")])


# ---------------------------------------------------------------------------- #
# Parse the decoder config file and create the decoders from it.
def parse_decoder_config(config_file, decoders, O_per_decoder):
    print(f'Creating decoders from config file {args.config}...')
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        for decoder_id, decoder in enumerate(config['decoders']):
            # Loop through each decoder in the config.
            num_rows = decoder['syndrome_size']
            num_cols = decoder['block_size']
            num_observables = decoder['O_sparse'].count(-1)
            # Form H
            H = sparse_to_dense(decoder['H_sparse'], num_rows, num_cols)
            O = sparse_to_dense(decoder['O_sparse'], num_observables, num_cols)
            O_per_decoder.append(O)
            decoder_custom_args = decoder['decoder_custom_args']

            # Change these to primitive types. This is annoying to have to do, but I
            # don't know of a better way to do this.
            decoder_custom_args = dict(decoder_custom_args)
            for key, value in decoder_custom_args.items():
                if type(value) == list:
                    # TODO - update this to be more general if some decoder parameters need
                    # to be a different type.
                    decoder_custom_args[key] = numpy.array(value,
                                                           dtype=numpy.float64)
                elif type(value) == int:
                    decoder_custom_args[key] = int(value)
                elif type(value) == float:
                    decoder_custom_args[key] = float(value)
                elif type(value) == bool:
                    decoder_custom_args[key] = bool(value)
            decoders.append(
                qec.get_decoder(decoder['type'], H, **decoder_custom_args))
            print(f"Decoder {decoder_id} created.")


# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #

parser = argparse.ArgumentParser(description='Replay decoder logs.')
parser.add_argument('--config', type=str, required=True, help='Config file.')
parser.add_argument('--decoder-log',
                    type=str,
                    required=True,
                    help='Decoder log file.')
parser.add_argument('--verbose-on-mismatch',
                    action='store_true',
                    help='Verbose output on mismatches.')
args = parser.parse_args()

# Check if the files exist.
if not os.path.exists(args.config):
    print(f"Config file does not exist: {args.config}")
    exit(1)
if not os.path.exists(args.decoder_log):
    print(f"Decoder log file does not exist: {args.decoder_log}")
    exit(1)

# The length of these lists should be the same as the number of decode calls in
# the log file.
decoder_id_list = []  # Decoder ID of each decode call.
log_detectors_sparse = []  # Detection events seen in the log file.
log_errors_sparse = []  # Errors seen in the log file.
replay_errors_sparse = []  # Errors seen in the replay.
log_observables_dense = []  # Observable flips seen in the log file.
replay_observables_dense = []  # Observable flips calculated in the replay.

decoders = []
O_per_decoder = []

parse_decoder_log(args.decoder_log, log_detectors_sparse, log_errors_sparse,
                  log_observables_dense, decoder_id_list)
parse_decoder_config(args.config, decoders, O_per_decoder)

# Basic error checking
max_decoder_id = max(decoder_id_list)
if len(decoders) < max_decoder_id + 1:
    print(
        f"Error: Decoder list is too short. Expected {max_decoder_id + 1} decoders, but only {len(decoders)} decoders were created."
    )
    exit(1)

# ---------------------------------------------------------------------------- #
# Now loop through the syndromes and compare the results.
decode_call_idx = 0
replay_error_mismatch = 0
replay_observable_mismatch = 0
print(f'Processing {len(log_detectors_sparse)} decode calls.')
for s, o in zip(log_detectors_sparse, log_observables_dense):
    # Create a 1D array of length syndrome size.
    syndrome = numpy.zeros(
        decoders[decoder_id_list[decode_call_idx]].get_syndrome_size(),
        dtype=numpy.uint8)
    for idx in s:
        syndrome[idx] = 1
    result = decoders[decoder_id_list[decode_call_idx]].decode(syndrome)
    dec_err_sparse = [
        i for i in range(len(result.result)) if result.result[i] > 0.5
    ]
    replay_errors_sparse.append(dec_err_sparse)
    mismatch_flag = False
    if dec_err_sparse != log_errors_sparse[decode_call_idx]:
        replay_error_mismatch += 1
        mismatch_flag = True
        if args.verbose_on_mismatch:
            print(
                f"Replay mismatch in error in decode_call_idx {decode_call_idx}"
            )
            print(f"Decoded errors : {dec_err_sparse}")
            print(f"Expected errors: {log_errors_sparse[decode_call_idx]}")
    dec_err_dense = numpy.array(result.result, dtype=numpy.uint8)
    O_replay = (
        O_per_decoder[decoder_id_list[decode_call_idx]] @ dec_err_dense %
        2).astype(numpy.uint8)
    replay_observables_dense.append(O_replay)
    O_log = numpy.array(log_observables_dense[decode_call_idx],
                        dtype=numpy.uint8)
    if (O_replay != O_log).any():
        replay_observable_mismatch += 1
        mismatch_flag = True
        if args.verbose_on_mismatch:
            print(
                f"Replay mismatch in observables in decode_call_idx {decode_call_idx}"
            )
            print(f"Decoded observables : {O_replay}")
            print(f"Expected observables: {O_log}")
    if not args.verbose_on_mismatch:
        if mismatch_flag:
            print('x', end='', flush=True)
        else:
            print('.', end='', flush=True)
    decode_call_idx += 1

print()
print(f"Number of error mismatches during replay: {replay_error_mismatch}")
print(
    f"Number of observable mismatches during replay: {replay_observable_mismatch}"
)
