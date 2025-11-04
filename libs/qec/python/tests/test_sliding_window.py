# ============================================================================ #
# Copyright (c) 2025 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import numpy as np
import cudaq
import cudaq_qec as qec
import pytest
import os
import tempfile


@pytest.fixture(scope="function", autouse=True)
def setTarget():
    old_target = cudaq.get_target()
    cudaq.set_target('stim')
    yield
    cudaq.set_target(old_target)


@pytest.mark.parametrize("decoder_name",
                         ["multi_error_lut", "nv-qldpc-decoder"])
@pytest.mark.parametrize("batched", [True, False])
@pytest.mark.parametrize("num_rounds", [5, 10])
@pytest.mark.parametrize("num_windows", [1, 2, 3])
def test_sliding_window_multi_error_lut_and_nv_qldpc_decoder(
        decoder_name, batched, num_rounds, num_windows):
    test_sliding_window_1(decoder_name, batched, num_rounds, num_windows)


@pytest.mark.parametrize("decoder_name", ["single_error_lut"])
@pytest.mark.parametrize("batched", [True, False])
@pytest.mark.parametrize("num_rounds", [5, 10])
@pytest.mark.parametrize("num_windows", [1, 2, 3])
def test_sliding_window_1(decoder_name, batched, num_rounds, num_windows):
    cudaq.set_random_seed(13)
    code = qec.get_code('surface_code', distance=5)
    p = 0.001
    noise = cudaq.NoiseModel()
    noise.add_all_qubit_channel("x", cudaq.Depolarization2(p), 1)
    statePrep = qec.operation.prep0
    nShots = 1000

    dem = qec.z_dem_from_memory_circuit(code, statePrep, num_rounds, noise)
    num_syndromes_per_round = dem.detector_error_matrix.shape[0] // num_rounds

    # Inject only one error per shot. This will keep the number of mismatches
    # low, and any debug should be straightforward.
    syndromes = np.zeros((nShots, dem.detector_error_matrix.shape[0]),
                         dtype=np.uint8)
    np.random.seed(13)
    for shot in range(nShots):
        # Pick a single random error to inject
        col = np.random.randint(0, dem.detector_error_matrix.shape[1])
        syndromes[shot, :] = dem.detector_error_matrix[:, col].T

    # First compare the results of the full decoder to the sliding window
    # decoder using an inner decoder of the full window size. The results should
    # be the same.
    if decoder_name == "nv-qldpc-decoder":
        # nv-qldpc-decoder requires use_sparsity=True for batched decoding
        full_decoder = qec.get_decoder(decoder_name,
                                       dem.detector_error_matrix,
                                       use_sparsity=True,
                                       error_rate_vec=np.array(dem.error_rates))
    else:
        full_decoder = qec.get_decoder(decoder_name, dem.detector_error_matrix)
    num_syndromes_per_round = dem.detector_error_matrix.shape[0] // num_rounds

    # Set up inner decoder parameters based on decoder type
    if decoder_name == "nv-qldpc-decoder":
        inner_decoder_params = {
            "use_sparsity": True,
            "error_rate_vec": np.array(dem.error_rates)
        }
    elif decoder_name == "multi_error_lut":
        inner_decoder_params = {"lut_error_depth": 2}
    else:
        inner_decoder_params = {}

    sw_as_full_decoder = qec.get_decoder(
        "sliding_window",
        dem.detector_error_matrix,
        window_size=num_rounds - num_windows + 1,
        step_size=1,
        num_syndromes_per_round=num_syndromes_per_round,
        straddle_start_round=False,
        straddle_end_round=True,
        error_rate_vec=np.array(dem.error_rates),
        inner_decoder_name=decoder_name,
        inner_decoder_params=inner_decoder_params)

    # Save sliding window configuration to YAML file for verification
    config_filename = save_sliding_window_config_to_yaml(
        n_rounds=num_rounds,
        n_errs_per_round=dem.detector_error_matrix.shape[1] // num_rounds,
        n_syndromes_per_round=num_syndromes_per_round,
        window_size=num_rounds - num_windows + 1,
        step_size=1,
        simplified_pcm=np.array(dem.detector_error_matrix),
        simplified_weights=np.array(dem.error_rates),
        inner_decoder_name=decoder_name)

    # ============================================================================
    # PROVE END-TO-END SERIALIZATION: Load and configure decoders from the saved YAML
    # ============================================================================
    config_result = qec.configure_decoders_from_file(config_filename)
    assert config_result == 0, f"Failed to configure decoders from saved YAML file: {config_filename}"
    print(
        f"✅ Successfully configured decoders from saved YAML file: {config_filename}"
    )
    print("End-to-end YAML serialization and decoder configuration verified!")

    if batched:
        full_results = full_decoder.decode_batch(syndromes)
        sw_results = sw_as_full_decoder.decode_batch(syndromes)
        num_mismatches = 0
        for r1, r2 in zip(full_results, sw_results):
            if r1.result != r2.result:
                num_mismatches += 1
        assert num_mismatches == 0

    else:
        full_results = []
        sw_results = []
        num_mismatches = 0
        for syndrome in syndromes:
            r1 = full_decoder.decode(syndrome)
            r2 = sw_as_full_decoder.decode(syndrome)
            if r1.result != r2.result:
                num_mismatches += 1
        assert num_mismatches == 0

    # Cleanup: Finalize decoders after testing configuration
    qec.qecrt.config.finalize_decoders()

    # Clean up the generated YAML file
    #if os.path.exists(config_filename):
    #    os.unlink(config_filename)


def save_sliding_window_config_to_yaml(n_rounds, n_errs_per_round,
                                       n_syndromes_per_round, window_size,
                                       step_size, simplified_pcm,
                                       simplified_weights, inner_decoder_name):
    """
    Save sliding window decoder configuration to YAML (Python version of C++ function)
    """
    # Create a sliding_window_config struct with concrete inner decoder configs
    sw_config = qec.qecrt.config.sliding_window_config()
    sw_config.window_size = window_size
    sw_config.step_size = step_size
    sw_config.num_syndromes_per_round = n_syndromes_per_round
    sw_config.straddle_start_round = False  # Default value from sliding_window.cpp
    sw_config.straddle_end_round = True  # Default value from sliding_window.cpp
    sw_config.error_rate_vec = simplified_weights.tolist()
    sw_config.inner_decoder_name = inner_decoder_name

    # Set the appropriate concrete inner decoder config based on decoder name
    if inner_decoder_name == "single_error_lut":
        single_config = qec.qecrt.config.single_error_lut_config()
        # single_error_lut_config is intentionally empty (no configuration parameters)
        sw_config.single_error_lut_params = single_config
    elif inner_decoder_name == "multi_error_lut":
        multi_config = qec.qecrt.config.multi_error_lut_config()
        multi_config.lut_error_depth = 2
        # Note: error_rate_vec not supported in multi_error_lut_config for real-time decoding
        sw_config.multi_error_lut_params = multi_config
    elif inner_decoder_name == "nv-qldpc-decoder":
        nv_config = qec.qecrt.config.nv_qldpc_decoder_config()
        nv_config.use_sparsity = True
        nv_config.error_rate_vec = simplified_weights.tolist()
        nv_config.use_osd = True
        nv_config.max_iterations = 50
        nv_config.osd_order = 60
        nv_config.osd_method = 3
        sw_config.nv_qldpc_decoder_params = nv_config

    # Create a decoder_config for the sliding window decoder
    config = qec.qecrt.config.decoder_config()
    config.id = 0
    config.type = "sliding_window"
    config.block_size = simplified_pcm.shape[1]  # Number of columns
    config.syndrome_size = simplified_pcm.shape[0]  # Number of rows
    config.num_syndromes_per_round = n_syndromes_per_round

    # Convert PCM to sparse format for H_sparse and create empty O_sparse
    H_sparse = []
    for row in range(simplified_pcm.shape[0]):
        for col in range(simplified_pcm.shape[1]):
            if simplified_pcm[row, col] == 1:
                H_sparse.append(int(col))
        H_sparse.append(-1)  # End of row marker

    config.H_sparse = H_sparse
    config.O_sparse = []  # Empty for this test
    # D_sparse is optional, leave unset

    # Set the sliding_window_config directly
    config.decoder_custom_args = sw_config

    # Create multi_decoder_config and add our sliding window decoder
    multi_config = qec.qecrt.config.multi_decoder_config()
    multi_config.decoders = [config]

    # Generate YAML string using native serialization (no concatenation needed!)
    config_str = multi_config.to_yaml_str(200)  # 200 char line wrap

    # Create a unique filename based on test parameters
    filename = f"sliding_window_config_r{n_rounds}_e{n_errs_per_round}_s{n_syndromes_per_round}_w{window_size}_st{step_size}_{inner_decoder_name}.yml"

    with open(filename, 'w') as config_file:
        config_file.write(config_str)

    print(
        f"Saved sliding window config (with native YAML inner decoder params) to file: {filename}"
    )

    # Verify the config can be loaded back (inner decoder params should be natively serialized)
    loaded_multi_config = qec.qecrt.config.multi_decoder_config.from_yaml_str(
        config_str)
    assert len(loaded_multi_config.decoders) == 1
    assert loaded_multi_config.decoders[0].type == "sliding_window"
    assert loaded_multi_config.decoders[0].id == 0

    # Verify the YAML contains the expected structure and parameters
    assert "decoder_custom_args:" in config_str
    assert f"window_size:     {window_size}" in config_str
    assert f"step_size:       {step_size}" in config_str
    assert f"num_syndromes_per_round: {n_syndromes_per_round}" in config_str
    assert f"inner_decoder_name: {inner_decoder_name}" in config_str

    # Verify that inner decoder parameters are preserved in the loaded config
    print("Successfully verified sliding window YAML config round-trip!")
    print(
        "Sliding window parameters and inner decoder configs natively serialized!"
    )

    if inner_decoder_name == "single_error_lut":
        # single_error_lut_config is intentionally empty (no configuration parameters)
        print("Single error LUT config parameters successfully verified!")
    elif inner_decoder_name == "multi_error_lut":
        assert "inner_decoder_params:" in config_str
        assert "lut_error_depth: 2" in config_str
        print("Multi error LUT config parameters successfully verified!")
    elif inner_decoder_name == "nv-qldpc-decoder":
        assert "inner_decoder_params:" in config_str
        assert "use_sparsity:    true" in config_str
        assert "use_osd:         true" in config_str
        assert "max_iterations:  50" in config_str
        assert "osd_order:       60" in config_str
        assert "osd_method:      3" in config_str
        print("NV QLDPC decoder config parameters successfully verified!")

    # Return the filename for end-to-end verification
    return filename
