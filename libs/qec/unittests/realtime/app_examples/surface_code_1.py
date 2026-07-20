# ============================================================================ #
# Copyright (c) 2025 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import os
import atexit
import sys
import argparse
import time
from typing import Callable, List, Tuple
import numpy as np
from collections.abc import Iterable

# Force stim as the default simulator for emulation
os.environ["CUDAQ_DEFAULT_SIMULATOR"] = "stim"

import cudaq
import cudaq_qec as qec
from cudaq_qec import patch

sys.tracebacklimit = 999
PER_SHOT_DEBUG = 0
MOCK_SHIM_DEBUG = 0


def pcm_from_sparse_vec(sparse_vec: Iterable[int], num_rows: int,
                        num_cols: int) -> np.ndarray:
    pcm = np.zeros((num_rows, num_cols), dtype=np.uint8)
    row = 0
    for col in sparse_vec:
        if col < 0:
            row += 1
            continue
        if 0 <= row < num_rows and 0 <= col < num_cols:
            pcm[row, col] = 1
        else:
            raise IndexError(f"Out of bounds: row:{row}, col={col}")
    return pcm


def save_dem_to_file(dem, d_sparse, dem_filename, numSyndromesPerRound,
                     num_logical):
    multi_config = qec.multi_decoder_config()
    decoders = []
    for i in range(num_logical):
        config = qec.decoder_config()
        config.id = i
        config.type = "multi_error_lut"
        config.block_size = dem.num_error_mechanisms()
        config.syndrome_size = dem.num_detectors()
        config.H_sparse = qec.pcm_to_sparse_vec(dem.detector_error_matrix)
        config.O_sparse = qec.pcm_to_sparse_vec(dem.observables_flips_matrix)
        config.D_sparse = d_sparse
        config.decoder_custom_args = {"lut_error_depth": 2}
        decoders.append(config)

    multi_config.decoders = decoders
    config_str = multi_config.to_yaml_str(200)
    print("Generated config:", config_str)
    config_file = open(dem_filename, 'w')
    config_file.write(config_str)
    config_file.close()
    print(f"Saved config to file: {dem_filename}")


def load_dem_from_file(dem_filename: str, dem: qec.DetectorErrorModel,
                       num_logical: int) -> None:
    print(f"load_dem_from_file: Loading dem from file: {dem_filename}")
    with open(dem_filename, 'r') as f:
        dem_str = f.read()

    multi_cfg = qec.multi_decoder_config.from_yaml_str(dem_str)
    if num_logical != len(multi_cfg.decoders):
        print(
            f"ERROR: numLogical [{num_logical}] != config.decoders.size() [{len(multi_cfg.decoders)}]"
        )
        sys.exit(1)

    dec_cfg = multi_cfg.decoders[0]
    multi_error_lut_cfg = dec_cfg.decoder_custom_args

    dem.detector_error_matrix = pcm_from_sparse_vec(dec_cfg.H_sparse,
                                                    dec_cfg.syndrome_size,
                                                    dec_cfg.block_size)

    # Count number of observables as number of -1 separators in O_sparse
    num_observables = sum(1 for x in dec_cfg.O_sparse if x == -1)
    dem.observables_flips_matrix = pcm_from_sparse_vec(dec_cfg.O_sparse,
                                                       num_observables,
                                                       dec_cfg.block_size)

    print(f"Loaded dem from file: {dem_filename}")

    # Configure the decoder
    qec.configure_decoders(multi_cfg)


def get_stab_cnot_schedule(stab_type: str, distance: int) -> List[int]:
    grid = qec.stabilizer_grid(distance)
    if stab_type not in ("X", "Z"):
        raise RuntimeError(
            "get_stab_cnot_schedule: Invalid stabilizer type. Must be 'X' or 'Z'."
        )

    # CNOT pairs ordered by timestep within each stabilizer, so that mid-round
    # ancilla (hook) errors land perpendicular to the logical operators.
    # Stabilizer indices match the sorted parity-matrix rows and hence the
    # ancilla indexing.
    return list(grid.get_cnot_schedule_pairs_x() if stab_type ==
                "X" else grid.get_cnot_schedule_pairs_z())


def debug_print_syndromes(syndrome_x_int: int, syndrome_z_int: int) -> None:
    print(f"syndrome_x_int: {syndrome_x_int}, syndrome_z_int: {syndrome_z_int}")


def debug_print_apply_corrections(correction: int) -> None:
    print(f"Applying correction: {correction}")


def debug_start_shot() -> None:
    print("Starting shot")


# FIXME: this is a temporary kernel to replace the missing `get_operation_one_qubit` implementation, which should return a valid quantum kernel.
@cudaq.kernel
def prep_0(logical: patch) -> None:
    reset(logical.data)


@cudaq.kernel
def logical_cnot(ctrl_data: cudaq.qview, tgt_data: cudaq.qview) -> None:
    for i in range(ctrl_data.size()):
        x(ctrl_data[i], tgt_data[i])


@cudaq.kernel
def spam_error(logical_qubit: patch, p_spam_data: float, p_spam_ancx: float,
               p_spam_ancz: float) -> None:
    for i in range(len(logical_qubit.data)):
        cudaq.apply_noise(cudaq.Depolarization1, p_spam_data,
                          logical_qubit.data[i])
    for i in range(len(logical_qubit.ancx)):
        cudaq.apply_noise(cudaq.Depolarization1, p_spam_ancx,
                          logical_qubit.ancx[i])
    for i in range(len(logical_qubit.ancz)):
        cudaq.apply_noise(cudaq.Depolarization1, p_spam_ancz,
                          logical_qubit.ancz[i])


@cudaq.kernel
def se_z_ft(logical_qubit: patch,
            cnot_sched: List[int]) -> List[cudaq.measure_handle]:
    for i in range(0, len(cnot_sched), 2):
        cx(logical_qubit.data[cnot_sched[i + 1]],
           logical_qubit.ancz[cnot_sched[i]])
    results = mz(logical_qubit.ancz)
    for q in logical_qubit.ancz:
        reset(q)
    return results


@cudaq.kernel
def se_x_ft(logical_qubit: patch,
            cnot_sched: List[int]) -> List[cudaq.measure_handle]:
    h(logical_qubit.ancx)
    for i in range(0, len(cnot_sched), 2):
        cx(logical_qubit.ancx[cnot_sched[i]],
           logical_qubit.data[cnot_sched[i + 1]])
    h(logical_qubit.ancx)
    results = mz(logical_qubit.ancx)
    for q in logical_qubit.ancx:
        reset(q)
    return results


# Runs the stabilizer measurement rounds. When declare_detectors is set (DEM
# generation), every round after the first declares one cross-round detector
# per syndrome bit against the previous round, in the same [Z..., X...] order
# the syndromes are enqueued; the first round is the reference round and
# declares none of its own. All detector bookkeeping (the measure-handle
# lists) is guarded by declare_detectors - a synthesized constant on hardware
# targets - so it folds away entirely for live runs: the adaptive QIR profile
# rejects the memory traffic that live measure-handle lists leave behind.
@cudaq.kernel
def custom_memory_circuit_stabs(
    data: cudaq.qview,
    xstab_anc: cudaq.qview,
    zstab_anc: cudaq.qview,
    num_rounds: int,
    cnot_schedX_flat: List[int],
    cnot_schedZ_flat: List[int],
    enqueue_synd: bool,
    do_errors_after_non_last_rounds: bool,
    p_spam: float,
    logical_qubit_idx: int,
    decoder_window: int,
    manually_inject_errors: bool,
    declare_detectors: bool,
) -> None:
    # Create the logical patch
    logical = patch(data, xstab_anc, zstab_anc)
    # Bool mirror of the current round's syndrome for the enqueue calls,
    # rebuilt in place each round with constant-bound loops (size queries on
    # measure-handle lists, e.g. to_bools, do not survive the full loop
    # unrolling that hardware targets require).
    combined_bools = [False for i in range(len(xstab_anc) + len(zstab_anc))]
    # The previous round's measurement handles (cross-round detectors only).
    have_prev = False
    prev = [
        cudaq.measure_handle() for _ in range(len(xstab_anc) + len(zstab_anc))
    ]

    # Handle the stabilizer lock-in round (numRounds == 1). The syndrome
    # lists are read with constant-bound indexed loops (len of the qviews):
    # iterating the returned lists directly gives loops bounded by the list
    # size, which does not survive the full loop unrolling that hardware
    # targets require.
    if num_rounds == 1:
        syndrome_z = se_z_ft(logical, cnot_schedZ_flat)
        syndrome_x = se_x_ft(logical, cnot_schedX_flat)
        for k in range(len(zstab_anc)):
            combined_bools[k] = bool(syndrome_z[k])
        for k in range(len(xstab_anc)):
            combined_bools[len(zstab_anc) + k] = bool(syndrome_x[k])
        if enqueue_synd:
            qec.enqueue_syndromes_test(logical_qubit_idx, combined_bools, 0)
        return

    # Process rounds window by window for the main measurement rounds
    # This is a plain stationary window implementation. Not a sliding window
    # implementation!
    for window_idx in range(num_rounds // decoder_window):
        # For window_idx > 0, enqueue the last syndrome from previous window first
        if window_idx > 0 and enqueue_synd:
            qec.enqueue_syndromes_test(logical_qubit_idx, combined_bools, 0)

        # Process the current window rounds
        for round_idx in range(window_idx * decoder_window,
                               (window_idx + 1) * decoder_window):
            syndrome_z = se_z_ft(logical, cnot_schedZ_flat)
            syndrome_x = se_x_ft(logical, cnot_schedX_flat)
            for k in range(len(zstab_anc)):
                combined_bools[k] = bool(syndrome_z[k])
            for k in range(len(xstab_anc)):
                combined_bools[len(zstab_anc) + k] = bool(syndrome_x[k])

            if enqueue_synd:
                qec.enqueue_syndromes_test(logical_qubit_idx, combined_bools, 0)
            if declare_detectors:
                combined_syndrome = [
                    cudaq.measure_handle()
                    for _ in range(len(xstab_anc) + len(zstab_anc))
                ]
                for k in range(len(zstab_anc)):
                    combined_syndrome[k] = syndrome_z[k]
                for k in range(len(xstab_anc)):
                    combined_syndrome[len(zstab_anc) + k] = syndrome_x[k]
                if have_prev:
                    for k in range(len(xstab_anc) + len(zstab_anc)):
                        cudaq.detector(prev[k], combined_syndrome[k])
                prev = combined_syndrome
                have_prev = True

            if do_errors_after_non_last_rounds and round_idx < (
                    window_idx + 1) * decoder_window - 1:
                spam_error(logical, p_spam, 0.0, 0.0)
                # Force a single error that should likely be correctable.
                if manually_inject_errors:
                    if (round_idx == 0):
                        x(logical.data[3])


# When declare_detectors is set (only meaningful with num_logical = 1 and
# allow_device_calls = False), the kernel annotates itself for DEM generation
# via cudaq.dem_from_kernel: every stabilizer round declares cross-round
# detectors against the previous round (the lock-in round is the first
# reference), and the Z logical observable is declared over the final data
# measurements at z_obs_indices.
@cudaq.kernel
def demo_circuit_qpu(
    allow_device_calls: bool,
    declare_detectors: bool,
    #state_prep: Callable[[patch], None],
    num_data: int,
    num_ancx: int,
    num_ancz: int,
    num_rounds: int,
    num_logical: int,
    cnot_schedX_flat: List[int],
    cnot_schedZ_flat: List[int],
    p_spam: float,
    apply_corrections: bool,
    decoder_window: int,
    manually_inject_errors: bool,
    z_obs_indices: List[int],
) -> int:
    # if PER_SHOT_DEBUG:
    #     debug_start_shot()

    num_corrections = 0

    # Reset the decoder
    if allow_device_calls:
        for i in range(num_logical):
            qec.reset_decoder(i)

    # Allocate qubits
    data = cudaq.qvector(num_logical * num_data)
    xstab_anc = cudaq.qvector(num_logical * num_ancx)
    zstab_anc = cudaq.qvector(num_logical * num_ancz)

    # State preparation
    for i in range(num_logical):
        sub_data = data[i * num_data:(i + 1) *
                        num_data]  # FIXME: all sub_data are incorrect
        sub_x = xstab_anc[i * num_ancx:(i + 1) * num_ancx]  # same other vectors
        sub_z = zstab_anc[i * num_ancz:(i + 1) * num_ancz]
        logical = patch(sub_data, sub_x, sub_z)
        prep_0(logical)  # FIXME: replace with state_prep(logical)

    if declare_detectors:
        # DEM-generation slice (num_logical == 1, no device calls): a single
        # call covering the stabilizer lock-in round plus one decoder window
        # (num_rounds + 1 rounds, with spam after every non-last round). This
        # reproduces the exact gate and noise sequence of the live path in
        # the else branch - the injected error round there equals the
        # per-round spam - while keeping all measurement handles inside one
        # kernel call, so the lock-in round can serve as the reference round
        # for the first cross-round detectors. Keep the two paths in
        # lockstep.
        sub_data = data[0:num_data]
        sub_x = xstab_anc[0:num_ancx]
        sub_z = zstab_anc[0:num_ancz]
        custom_memory_circuit_stabs(
            sub_data,
            sub_x,
            sub_z,
            num_rounds + 1,
            cnot_schedX_flat,
            cnot_schedZ_flat,
            False,  # enqueue_synd
            True,  # do_errors_after_non_last_rounds
            p_spam,
            0,
            num_rounds + 1,  # decoder_window: a single window
            False,  # manually_inject_errors
            True,  # declare_detectors
        )
    else:
        # One stabilizer round to lock in
        for i in range(num_logical):
            sub_data = data[i * num_data:(i + 1) *
                            num_data]  # FIXME: all sub_data are incorrect
            sub_x = xstab_anc[i * num_ancx:(i + 1) *
                              num_ancx]  # same other vectors
            sub_z = zstab_anc[i * num_ancz:(i + 1) * num_ancz]
            custom_memory_circuit_stabs(
                sub_data,
                sub_x,
                sub_z,
                1,
                cnot_schedX_flat,
                cnot_schedZ_flat,
                allow_device_calls,
                False,
                p_spam,
                i,
                decoder_window,
                manually_inject_errors,
                False,  # declare_detectors
            )

        # Inject errors
        for i in range(num_logical):
            sub_data = data[i * num_data:(i + 1) *
                            num_data]  # FIXME: all sub_data are incorrect
            sub_x = xstab_anc[i * num_ancx:(i + 1) *
                              num_ancx]  # same other vectors
            sub_z = zstab_anc[i * num_ancz:(i + 1) * num_ancz]
            logical = patch(sub_data, sub_x, sub_z)
            spam_error(logical, p_spam, 0.0, 0.0)

        # Do stabilizer rounds
        for i in range(num_logical):
            sub_data = data[i * num_data:(i + 1) *
                            num_data]  # FIXME: all sub_data are incorrect
            sub_x = xstab_anc[i * num_ancx:(i + 1) *
                              num_ancx]  # same other vectors
            sub_z = zstab_anc[i * num_ancz:(i + 1) * num_ancz]
            custom_memory_circuit_stabs(
                sub_data,
                sub_x,
                sub_z,
                num_rounds,
                cnot_schedX_flat,
                cnot_schedZ_flat,
                allow_device_calls,
                True,
                p_spam,
                i,
                decoder_window,
                manually_inject_errors,
                False,  # declare_detectors
            )

    # Only apply corrections after processing all windows
    if allow_device_calls and apply_corrections:
        for i in range(num_logical):
            sub_data = data[i * num_data:(i + 1) *
                            num_data]  # FIXME: all sub_data are incorrect
            sub_x = xstab_anc[i * num_ancx:(i + 1) *
                              num_ancx]  # same other vectors
            sub_z = zstab_anc[i * num_ancz:(i + 1) * num_ancz]
            corrections = qec.get_corrections(i, 1, False)
            if corrections[0] != 0:
                num_corrections += 1
                # Transversal correction
                x(sub_data)
                #if PER_SHOT_DEBUG:
                #    debug_print_apply_corrections(corrections[0])

    # Note: this only works up to 64 bits, so a single logical qubit with distance 7.
    ret = 0
    for i in range(num_logical):
        if i > 0:
            ret = ret << num_data
        sub_data = data[i * num_data:(i + 1) * num_data]
        sub_meas = mz(sub_data)
        if declare_detectors and i == 0:
            zlog = [cudaq.measure_handle() for _ in range(len(z_obs_indices))]
            for k in range(len(z_obs_indices)):
                zlog[k] = sub_meas[z_obs_indices[k]]
            cudaq.logical_observable(zlog, observable_index=0)
        # Pack the measured bits branch-free (bit j = data qubit j, the same
        # LSB-first order as cudaq.to_integer). Routing the measurement
        # results through a call (to_bools/to_integer) or a branch would tag
        # this kernel with qubitMeasurementFeedback, which
        # cudaq.dem_from_kernel rejects.
        for j in range(num_data):
            bitval = sub_meas[j]
            ret = ret | (bitval << j)

    # The remaining bits are allocated to the number of corrections.
    ret = ret | (num_corrections << (num_data * num_logical))
    return ret


def demo_circuit_host(code_obj: qec.code,
                      distance: int,
                      p_spam: float,
                      state_prep_op: qec.operation,
                      num_shots: int,
                      num_rounds: int,
                      num_logical: int,
                      dem_filename: str,
                      save_dem: bool,
                      load_dem: bool,
                      decoder_window: int,
                      target_name: str = "stim",
                      emulate: bool = True,
                      machine_name: str = "",
                      project_id: str = "",
                      max_cost: int = 0,
                      max_qubits: int = 0):
    if not code_obj.contains_operation(state_prep_op):
        raise RuntimeError(
            f"sample_memory_circuit_error - requested state prep kernel not found."
        )

    # prep = code_obj.get_operation_one_qubit(state_prep_op) # FIXME: fix this
    prep = prep_0
    if not code_obj.contains_operation(qec.operation.stabilizer_round):
        raise RuntimeError(
            f"demo_circuit_host error - no stabilizer round kernel for this code."
        )

    num_data = code_obj.get_num_data_qubits()
    num_ancx = code_obj.get_num_ancilla_x_qubits()
    num_ancz = code_obj.get_num_ancilla_z_qubits()
    print("num data " + str(num_data))
    print("num_ancx " + str(num_ancx))
    print("num_ancz " + str(num_ancz))

    cnot_schedX_flat = get_stab_cnot_schedule('X', distance)
    cnot_schedZ_flat = get_stab_cnot_schedule('Z', distance)

    print("cnot_schedX_flat: ", end="")
    for i in range(0, len(cnot_schedX_flat), 2):
        print(f"{cnot_schedX_flat[i]} {cnot_schedX_flat[i+1]}, ", end="")
    print()

    print("cnot_schedZ_flat: ", end="")
    for i in range(0, len(cnot_schedZ_flat), 2):
        print(f"{cnot_schedZ_flat[i]} {cnot_schedZ_flat[i+1]}, ", end="")
    print()

    noise = cudaq.NoiseModel()

    # The Z logical observable's data-qubit support (row 0 of the Z
    # observables matrix); demo_circuit_qpu declares the matching
    # logical_observable over the final data measurements when generating the
    # DEM.
    obs_matrix = code_obj.get_observables_z()
    z_obs_indices = [
        int(col) for col in range(obs_matrix.shape[1]) if obs_matrix[0, col]
    ]

    # Build or load DEM
    dem = qec.DetectorErrorModel()

    if load_dem:
        print(f"Loading DEM from {dem_filename}")
        load_dem_from_file(dem_filename, dem, num_logical)
    else:
        print(f"Preparing DEM to save to {dem_filename}")
        # Always use stim to build the DEM
        cudaq.set_target("stim")
        if p_spam == 0.0:
            raise RuntimeError("Cannot build a DEM with p_spam = 0.0.")
        # Analyze the same demo_circuit_qpu kernel the shots run, with
        # declare_detectors so it annotates its detectors and observable.
        # Always use numLogical = 1, and decoder_window rounds instead of
        # numRounds: the decoder consumes one window at a time.
        dem_text, m2d, m2o = cudaq.dem_from_kernel(
            demo_circuit_qpu,
            False,  # allow_device_calls
            True,  # declare_detectors
            num_data,
            num_ancx,
            num_ancz,
            decoder_window,  # Use decoder_window instead of numRounds for DEM generation
            1,  # numLogical
            cnot_schedX_flat,
            cnot_schedZ_flat,
            p_spam,
            False,  # applyCorrections
            decoder_window,
            False,  # manuallyInjectErrors
            z_obs_indices,
            noise_model=noise,
            return_measurement_matrices=True)
        dem = qec.dem_from_stim_text(dem_text)

        numSyndromesPerRound = distance * distance - 1
        if dem.num_detectors() != decoder_window * numSyndromesPerRound:
            raise RuntimeError(
                "Number of detectors [" + str(dem.num_detectors()) +
                "] is not equal to decoder_window * numSyndromesPerRound [" +
                str(decoder_window * numSyndromesPerRound) + "]")
        print("numSyndromesPerRound:", numSyndromesPerRound)
        dem.canonicalize_for_rounds(numSyndromesPerRound,
                                    remove_zero_syndrome_errors=True)

        # The runtime detector matrix comes straight from the analysis'
        # measurements-to-detectors map: row d lists the (chronological, and
        # thus enqueue-ordered) measurement indices whose XOR forms detector
        # d.
        m2d = m2d.tocsr()
        d_sparse = []
        for r in range(m2d.shape[0]):
            row = m2d.indices[m2d.indptr[r]:m2d.indptr[r + 1]]
            d_sparse.extend(sorted(int(c) for c in row))
            d_sparse.append(-1)

        print("dem.detector_error_matrix:")
        print(dem.detector_error_matrix)
        print("dem.observables_flips_matrix:")
        print(dem.observables_flips_matrix)
        save_dem_to_file(dem, d_sparse, dem_filename, numSyndromesPerRound,
                         num_logical)
        return

    # Actual run
    extra_target_kwargs = {}
    if target_name == "quantinuum":
        if machine_name == "":
            raise RuntimeError(
                "demo_circuit_host: machine_name must be set when target_name is quantinuum."
            )
        if not emulate:
            if project_id == "":
                raise RuntimeError(
                    "demo_circuit_host: project_id must be set when target_name is quantinuum and emulate is false (remote execution)."
                )
            extra_target_kwargs["project"] = project_id

            # If not syntax checker, max_cost and max_qubits are also required
            if not machine_name.endswith("SC"):
                if max_cost <= 0:
                    raise RuntimeError(
                        "demo_circuit_host: max_cost must be set to a positive integer when running on Quantinuum QPU/Emulator."
                    )
                extra_target_kwargs["max_cost"] = max_cost
                if max_qubits <= 0:
                    raise RuntimeError(
                        "demo_circuit_host: max_qubits must be set to a positive integer when running on Quantinuum QPU/Emulator."
                    )
                extra_target_kwargs["max_qubits"] = max_qubits

    cudaq.set_target(target_name,
                     emulate=emulate,
                     machine=machine_name,
                     extra_payload_provider="decoder",
                     **extra_target_kwargs)
    print("target: " + cudaq.get_target().name)

    num_syndromes_per_round = distance * distance - 1
    if dem.detector_error_matrix.shape[0] % num_syndromes_per_round != 0:
        raise RuntimeError(
            f"Num syndromes per round {num_syndromes_per_round} is not a divisor of the number of syndrome measurements {dem.detector_error_matrix.shape[0]}."
        )

    num_rounds_synd = dem.detector_error_matrix.shape[
        0] // num_syndromes_per_round
    if num_rounds_synd != decoder_window:
        raise RuntimeError(
            f"Num rounds of syndrome data [{num_rounds_synd}] is not equal to the decoder window [{decoder_window}]."
        )

    print("Calling cudaq.run ...")
    manually_inject_errors = target_name == "quantinuum" and emulate
    print("manually_inject_errors: " + str(manually_inject_errors))
    is_remote_qpu = target_name == "quantinuum" and not emulate
    # Run shots
    run_result = cudaq.run(
        demo_circuit_qpu,
        True,  # allow_device_calls
        False,  # declare_detectors
        # prep_0,
        num_data,
        num_ancx,
        num_ancz,
        num_rounds,
        num_logical,
        cnot_schedX_flat,
        cnot_schedZ_flat,
        p_spam,
        True,
        decoder_window,
        manually_inject_errors,
        z_obs_indices,
        shots_count=num_shots,
        noise_model=cudaq.NoiseModel() if not is_remote_qpu else None)

    print("Done with cudaq.run!")
    # print(f"Result: {len(run_result)}")

    num_non_zero = 0
    num_corrections = 0
    print("Result size: " + str(len(run_result)))
    for i, word in enumerate(run_result):
        print(f"Measured word: {word}")
        num_corrections += (word >> (num_data * num_logical))
        for j in range(num_logical):
            result_vec = np.zeros(num_data, dtype=np.uint8)
            for l in range(j * num_data, (j + 1) * num_data):
                result_vec[l - j * num_data] = 1 if (word &
                                                     (1 << l)) != 0 else 0
            # Calculate the logical observable for each logical qubit
            logical_result = ((obs_matrix @ result_vec) % 2)[0]
            print(
                f"Logical result [shot = {i}] for logical qubit {j}: {logical_result}"
            )
            if logical_result != 0:
                num_non_zero += 1

    print(f"Number of non-zero values measured: {num_non_zero}")
    print(f"Number of corrections decoder found: {num_corrections}")
    # Return the results in a dictionary
    results = {"num_non_zero": num_non_zero, "num_corrections": num_corrections}
    return results


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description="Surface code Sample App 1")
    parser.add_argument("--distance", type=int, default=5)
    parser.add_argument("--num_shots", type=int, default=10)
    parser.add_argument("--p_spam", type=float, default=0.01)
    parser.add_argument("--num_logical", type=int, default=1)
    parser.add_argument("--num_rounds",
                        type=int,
                        default=-1,
                        help="defaults to distance if not set")
    parser.add_argument("--decoder_window",
                        type=int,
                        default=-1,
                        help="defaults to distance if not set")
    parser.add_argument("--save_dem",
                        type=str,
                        default=None,
                        help="path to save DEM YAML")
    parser.add_argument("--load_dem",
                        type=str,
                        default=None,
                        help="path to load DEM YAML")
    parser.add_argument("--target",
                        type=str,
                        default="stim",
                        help="Name of the target to use. Default is stim.")
    parser.add_argument(
        "--machine_name",
        type=str,
        default="",
        help="Name of the machine to use when target is quantinuum.")
    parser.add_argument(
        "--emulate",
        default=True,
        help="Set to use emulation when running on a real QPU target.")
    parser.add_argument("--seed",
                        type=int,
                        default=None,
                        help="Random seed to use.")
    parser.add_argument(
        "--project_id",
        type=str,
        default="",
        help="Project ID to use when running on a real QPU target.")
    parser.add_argument("--max_cost",
                        type=int,
                        default=-1,
                        help="Max cost for Quantinuum target.")
    parser.add_argument("--max_qubits",
                        type=int,
                        default=-1,
                        help="Max qubits for Quantinuum target.")

    args = parser.parse_args(argv)

    save_dem = args.save_dem is not None
    load_dem = args.load_dem is not None
    dem_filename = args.save_dem if save_dem else (args.load_dem or "")
    target_name = args.target
    machine_name = args.machine_name
    emulate = args.emulate
    if isinstance(emulate, str):
        if emulate.lower() in ("true", "1", "yes"):
            emulate = True
        elif emulate.lower() in ("false", "0", "no"):
            emulate = False
        else:
            raise RuntimeError(
                f"Invalid value for emulate: {args.emulate}. Must be a boolean."
            )
    seed = args.seed
    project_id = args.project_id
    max_cost = args.max_cost
    max_qubits = args.max_qubits

    if target_name == "quantinuum" and machine_name == "":
        if not emulate:
            raise RuntimeError(
                "Error: machine_name must be set when target is quantinuum.")

        machine_name = "Helios-LocalE"  # Dummy default for emulation (to activate Helios code generation)

    distance = args.distance
    num_rounds = args.num_rounds if args.num_rounds != -1 else distance
    decoder_window = args.decoder_window if args.decoder_window != -1 else distance

    if num_rounds < distance or (num_rounds % distance) != 0:
        print(
            f"Error: num_rounds {num_rounds} must be >= distance {distance} and a multiple of distance"
        )
        return 1

    if decoder_window < distance or (decoder_window % distance) != 0:
        print(
            f"Error: decoder_window {decoder_window} must be >= distance {distance} and a multiple of distance"
        )
        return 1

    if decoder_window > num_rounds:
        print(
            f"Error: decoder_window {decoder_window} must be <= num_rounds {num_rounds}"
        )
        return 1

    if (num_rounds % decoder_window) != 0:
        print(
            f"Error: num_rounds {num_rounds} must be a multiple of decoder_window {decoder_window}"
        )
        return 1

    if args.num_logical * distance * distance > 64:
        print(
            f"Error: num_logical {args.num_logical} * distance^2 {distance*distance} >= 64 is not supported."
        )
        return 1

    print(
        f"Running with p_spam = {args.p_spam}, distance = {distance}, num_logical = {args.num_logical}, num_rounds = {num_rounds}, decoder_window = {decoder_window}, num_shots = {args.num_shots}"
    )

    code_obj = qec.get_code("surface_code", distance=distance)

    if not load_dem and not save_dem:
        print(
            "No DEM load or save file specified. Construct a local DEM and run."
        )
        # Create a temporary DEM file name, use time stamp to avoid collisions if multiple instances of this app are run.
        dem_filename = f"temp_dem_{format(time.time())}.yaml"

        # Add call back to delete the temp file at exit
        atexit.register(os.remove, dem_filename)

        save_dem = True
        load_dem = False
        # Create DEM:
        print(f"Preparing DEM to save to {dem_filename}")
        demo_circuit_host(
            code_obj,
            distance,
            args.p_spam,
            qec.operation.prep0,
            args.num_shots,
            num_rounds,
            args.num_logical,
            dem_filename,
            save_dem,
            load_dem,
            decoder_window,
        )

        # Set to load the DEM we just created for the actual run
        load_dem = True
        save_dem = False

    # Main run
    if seed is not None:
        print(f"Setting random seed to {seed}")
        cudaq.set_random_seed(seed)
    demo_circuit_host(
        code_obj,
        distance,
        args.p_spam,
        qec.operation.prep0,
        args.num_shots,
        num_rounds,
        args.num_logical,
        dem_filename,
        save_dem,
        load_dem,
        decoder_window,
        target_name,
        emulate,
        machine_name,
        project_id,
        max_cost,
        max_qubits,
    )

    qec.finalize_decoders()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
