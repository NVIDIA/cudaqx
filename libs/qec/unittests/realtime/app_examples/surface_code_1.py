# ============================================================================ #
# Copyright (c) 2025 NVIDIA Corporation & Affiliates.                          #
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


def sorted_stabilizer_ops_inplace_numpy(ops: List[cudaq.Operator]) -> None:
    if not ops:
        return []

    words = np.array([term.get_pauli_word() for term in ops], dtype=str)

    z_idx = np.char.find(words, "Z")
    x_idx = np.char.find(words, "X")

    # Group 0 if Z exists, else 1
    group = np.where(z_idx >= 0, 0, 1)
    # Index: first Z if exists else first X (big sentinel if none)
    idx = np.where(z_idx >= 0, z_idx, np.where(x_idx >= 0, x_idx, 10**9))

    order = np.lexsort((idx, group)).tolist()
    ops[:] = [ops[i] for i in order]


def save_dem_to_file(dem, dem_filename, numSyndromesPerRound, num_logical):
    multi_config = qec.multi_decoder_config()
    decoders = []
    for i in range(num_logical):
        # We actually send 1 additional round in this example, so add 1.
        numRounds = dem.num_detectors() // numSyndromesPerRound + 1
        config = qec.decoder_config()
        config.id = i
        config.type = "nv-qldpc-decoder"
        config.block_size = dem.num_error_mechanisms()
        config.syndrome_size = dem.num_detectors()
        config.num_syndromes_per_round = numSyndromesPerRound
        config.H_sparse = qec.pcm_to_sparse_vec(dem.detector_error_matrix)
        config.O_sparse = qec.pcm_to_sparse_vec(dem.observables_flips_matrix)
        config.D_sparse = qec.generate_timelike_sparse_detector_matrix(
            numSyndromesPerRound, numRounds, False)
        nv = qec.nv_qldpc_decoder_config()
        nv.use_sparsity = True
        nv.error_rate_vec = dem.error_rates
        nv.use_osd = True
        nv.max_iterations = 50
        nv.osd_order = 60
        nv.osd_method = 3
        config.set_decoder_custom_args(nv)
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
    with open(dem_filename, 'rb') as f:
        dem_str = f.read()

    multi_cfg = qec.multi_decoder_config.from_yaml_str(dem_str)
    if num_logical != len(multi_cfg.decoders):
        print(
            f"ERROR: numLogical [{num_logical}] != config.decoders.size() [{len(multi_cfg.decoders)}]"
        )
        sys.exit(1)

    dec_cfg = multi_cfg.decoders[0]
    nv_qldpc_cfg = dec_cfg.decoder_custom_args

    dem.detector_error_matrix = pcm_from_sparse_vec(dec_cfg.H_sparse,
                                                    dec_cfg.syndrome_size,
                                                    dec_cfg.block_size)

    # Count number of observables as number of -1 separators in O_sparse
    num_observables = sum(1 for x in dec_cfg.O_sparse if x == -1)
    dem.observables_flips_matrix = pcm_from_sparse_vec(dec_cfg.O_sparse,
                                                       num_observables,
                                                       dec_cfg.block_size)

    dem.error_rates = nv_qldpc_cfg.error_rate_vec
    print(f"Loaded dem from file: {dem_filename}")

    # Configure the decoder
    qec.configure_decoders(multi_cfg)


def get_stab_cnot_schedule(stab_type: str, distance: int) -> List[int]:
    grid = qec.stabilizer_grid(distance)
    if stab_type not in ("X", "Z"):
        raise RuntimeError(
            "get_stab_cnot_schedule: Invalid stabilizer type. Must be 'X' or 'Z'."
        )

    stabs = grid.get_spin_op_stabilizers()
    sorted_stabilizer_ops_inplace_numpy(stabs)

    stab_idx = 0
    cnot_schedule: List[int] = []

    for stab in stabs:
        word = stab.get_pauli_word(distance * distance)
        if stab_type not in word:
            continue
        for d, ch in enumerate(word):
            if ch == stab_type:
                cnot_schedule.extend([stab_idx, d])
        stab_idx += 1

    return cnot_schedule


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
def se_z_ft(logical_qubit: patch, cnot_sched: List[int]) -> List[bool]:
    for i in range(0, len(cnot_sched), 2):
        cx(logical_qubit.data[cnot_sched[i + 1]],
           logical_qubit.ancz[cnot_sched[i]])
    results = mz(logical_qubit.ancz)
    for q in logical_qubit.ancz:
        reset(q)
    return results


@cudaq.kernel
def se_x_ft(logical_qubit: patch, cnot_sched: List[int]) -> List[bool]:
    h(logical_qubit.ancx)
    for i in range(0, len(cnot_sched), 2):
        cx(logical_qubit.ancx[cnot_sched[i]],
           logical_qubit.data[cnot_sched[i + 1]])
    h(logical_qubit.ancx)
    results = mz(logical_qubit.ancx)
    for q in logical_qubit.ancx:
        reset(q)
    return results


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
) -> None:
    # Create the logical patch
    logical = patch(data, xstab_anc, zstab_anc)
    combined_syndrome = [False for i in range(len(xstab_anc) + len(zstab_anc))]
    # Handle the stabilizer lock-in round (numRounds == 1)
    if num_rounds == 1:
        syndrome_z = se_z_ft(logical, cnot_schedZ_flat)
        syndrome_x = se_x_ft(logical, cnot_schedX_flat)
        i = 0
        for s in syndrome_z:
            combined_syndrome[i] = s
            i += 1
        for s in syndrome_x:
            combined_syndrome[i] = s
            i += 1
        if enqueue_synd:
            qec.enqueue_syndromes(logical_qubit_idx, combined_syndrome, 0)
        return

    # Process rounds window by window for the main measurement rounds
    # This is a plain stationary window implementation. Not a sliding window
    # implementation!
    for window_idx in range(num_rounds // decoder_window):
        # For window_idx > 0, enqueue the last syndrome from previous window first
        if window_idx > 0 and enqueue_synd:
            qec.enqueue_syndromes(logical_qubit_idx, combined_syndrome, 0)

        # Process the current window rounds
        for round_idx in range(window_idx * decoder_window,
                               (window_idx + 1) * decoder_window):
            syndrome_z = se_z_ft(logical, cnot_schedZ_flat)
            syndrome_x = se_x_ft(logical, cnot_schedX_flat)
            i = 0
            for s in syndrome_z:
                combined_syndrome[i] = s
                i += 1
            for s in syndrome_x:
                combined_syndrome[i] = s
                i += 1

            if enqueue_synd:
                qec.enqueue_syndromes(logical_qubit_idx, combined_syndrome, 0)

            if do_errors_after_non_last_rounds and round_idx < (
                    window_idx + 1) * decoder_window - 1:
                spam_error(logical, p_spam, 0.0, 0.0)
                # Force a single error that should likely be correctable.
                if manually_inject_errors:
                    if (round_idx == 0):
                        x(logical.data[3])


@cudaq.kernel
def demo_circuit_qpu(
    allow_device_calls: bool,
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

    # One stabilizer round to lock in
    for i in range(num_logical):
        sub_data = data[i * num_data:(i + 1) *
                        num_data]  # FIXME: all sub_data are incorrect
        sub_x = xstab_anc[i * num_ancx:(i + 1) * num_ancx]  # same other vectors
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
        )

    # Inject errors
    for i in range(num_logical):
        sub_data = data[i * num_data:(i + 1) *
                        num_data]  # FIXME: all sub_data are incorrect
        sub_x = xstab_anc[i * num_ancx:(i + 1) * num_ancx]  # same other vectors
        sub_z = zstab_anc[i * num_ancz:(i + 1) * num_ancz]
        logical = patch(sub_data, sub_x, sub_z)
        spam_error(logical, p_spam, 0.0, 0.0)

    # Do stabilizer rounds
    for i in range(num_logical):
        sub_data = data[i * num_data:(i + 1) *
                        num_data]  # FIXME: all sub_data are incorrect
        sub_x = xstab_anc[i * num_ancx:(i + 1) * num_ancx]  # same other vectors
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
        ret |= cudaq.to_integer(sub_meas)

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
                      machine_name: str = ""):
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

    # Build or load DEM (MSM path)
    dem = qec.DetectorErrorModel()

    if load_dem:
        print(f"Loading DEM from {dem_filename}")
        load_dem_from_file(dem_filename, dem, num_logical)
    else:
        print(f"Preparing DEM to save to {dem_filename}")
        # Always use stim to build the DEM
        cudaq.set_target("stim")
        cudaq.set_noise(noise)
        if p_spam == 0.0:
            raise RuntimeError(
                "Cannot build a DEM with p_spam = 0.0 (cannot get the MSM).")
        # Always use numLogical = 1 for the MSM
        (msm_as_strings, msm_dimensions, msm_probabilities, msm_prob_err_id
        ) = qec.compute_msm(
            lambda: demo_circuit_qpu(
                False,
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
            ),
            True)

        print("MSM result obtained.")
        # print(f"MSM dimensions: {msm_dimensions}")
        # print(f"MSM probabilities: {msm_probabilities}")
        # print(f"MSM probability error ID: {msm_prob_err_id}")

        # Populate error rates and error IDs
        dem.error_rates = msm_probabilities
        dem.error_ids = msm_prob_err_id
        mzTable = qec.construct_mz_table(msm_as_strings)
        print("mzTable:", mzTable)
        # Subtract the number of data qubits to get the number of syndrome measurements.
        totalNumSyndromes = mzTable.shape[0] - distance * distance
        numNoiseMechs = mzTable.shape[1]
        numSyndromesPerRound = distance * distance - 1
        if (totalNumSyndromes % numSyndromesPerRound != 0):
            raise RuntimeError("Num syndromes per round is not a divisor of "
                               "the number of syndrome measurements")

        numRoundsOfSyndromData = totalNumSyndromes // numSyndromesPerRound
        if (numRoundsOfSyndromData != decoder_window +
                1):  # Use decoder_window instead of numRounds
            raise RuntimeError("Num rounds of syndrome data [" +
                               str(numRoundsOfSyndromData) +
                               "] is not equal to the decoder_window + 1[" +
                               str(decoder_window + 1) + "]")
        detector_error_matrix = np.zeros(
            (decoder_window * numSyndromesPerRound, numNoiseMechs),
            dtype=np.uint8)
        # There should be (decoder_window + 1) rounds of data in MSM.
        # TODO: [feature] Good candidate. Auto-generating the detector error
        # matrix. Currently, we need to manually construct the detector error
        # matrix by copying the measurements from the MSM.
        for round in range(
                decoder_window):  # Use decoder_window instead of numRounds
            for syndrome in range(numSyndromesPerRound):
                for noise_mech in range(numNoiseMechs):
                    detector_error_matrix[
                        round * numSyndromesPerRound + syndrome,
                        noise_mech] = mzTable[
                            (round + 0) * numSyndromesPerRound + syndrome,
                            noise_mech] ^ mzTable[
                                (round + 1) * numSyndromesPerRound + syndrome,
                                noise_mech]
        dem.detector_error_matrix = detector_error_matrix
        print("detector_error_matrix:", dem.detector_error_matrix)

        first_data_row = (
            decoder_window +
            1) * numSyndromesPerRound  # Use decoder_window instead of numRounds
        numSyndromesPerRound = distance * distance - 1
        msm_obs = np.zeros((mzTable.shape[0] - first_data_row, numNoiseMechs),
                           dtype=np.uint8)
        for row in range(first_data_row, mzTable.shape[0]):
            for col in range(numNoiseMechs):
                msm_obs[row - first_data_row, col] = mzTable[row, col]

        print("msm_obs:", msm_obs)

        # Populate dem.observables_flips_matrix by converting the physical data
        # qubit measurements to logical observables.
        obs_matrix = code_obj.get_observables_z()
        print("obs_matrix:", obs_matrix)
        dem.observables_flips_matrix = (obs_matrix @ msm_obs) % 2
        print("numSyndromesPerRound:", numSyndromesPerRound)
        dem.canonicalize_for_rounds(numSyndromesPerRound)

        print("dem.detector_error_matrix:")
        print(dem.detector_error_matrix)
        print("dem.observables_flips_matrix:")
        print(dem.observables_flips_matrix)
        save_dem_to_file(dem, dem_filename, numSyndromesPerRound, num_logical)
        return

    # Actual run
    if target_name == "quantinuum":
        if machine_name == "":
            raise RuntimeError(
                "demo_circuit_host: machine_name must be set when target_name is quantinuum."
            )
    cudaq.set_target(target_name,
                     emulate=emulate,
                     machine=machine_name,
                     extra_payload_provider="decoder")
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
    # Run shots
    run_result = cudaq.run(
        demo_circuit_qpu,
        True,
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
        shots_count=num_shots,
        noise_model=cudaq.NoiseModel())

    print("Done with cudaq.run!")
    # print(f"Result: {len(run_result)}")

    obs_matrix = code_obj.get_observables_z()
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
    args = parser.parse_args(argv)

    save_dem = args.save_dem is not None
    load_dem = args.load_dem is not None
    dem_filename = args.save_dem if save_dem else (args.load_dem or "")
    target_name = args.target
    machine_name = args.machine_name
    emulate = args.emulate
    seed = args.seed

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
    )

    qec.finalize_decoders()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
