# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import pytest
import numpy as np
import cudaq_qec as qec
import subprocess


def is_nv_qldpc_decoder_available():
    """
    Helper function to check if the NV-QLDPC decoder is available.
    """
    try:
        # First check to sure a GPU is available.
        result = subprocess.run(["nvidia-smi"], capture_output=True, timeout=2)
        if result.returncode != 0:
            return False
        # Now make sure that the NV-QLDPC decoder is available.
        H_list = [[1, 0, 0, 1, 0, 1, 1], [0, 1, 0, 1, 1, 0, 1],
                  [0, 0, 1, 0, 1, 1, 1]]
        H_np = np.array(H_list, dtype=np.uint8)
        nv_dec_gpu_and_cpu = qec.get_decoder("nv-qldpc-decoder", H_np)
        return True
    except Exception as e:
        return False


def check_decoder_yaml_roundtrip(multi_config):
    """
    Helper function to test that a decoder configuration can be serialized to
    and from YAML.
    """
    # Serialize to YAML
    config_str = multi_config.to_yaml_str(200)

    # Deserialize from YAML
    multi_config_from_yaml = qec.multi_decoder_config.from_yaml_str(config_str)

    # And now serialize the deserialized configuration back to YAML, just for
    # good measure.
    round_trip_config_str = multi_config_from_yaml.to_yaml_str(200)

    # Validate
    match_strings = round_trip_config_str == config_str
    match_configs = multi_config_from_yaml == multi_config

    assert match_strings, "YAML strings don't match after round trip"
    assert match_configs, "Configs don't match after round trip"


def check_decoder_creation(multi_config):
    """
    Helper function to create and finalize a decoder configuration.
    """
    status = qec.configure_decoders(multi_config)
    assert status == 0, f"configure_decoders returned non-zero status: {status}"
    qec.finalize_decoders()


def create_test_empty_decoder_config(decoder_id):
    """
    Helper function to create a sample, skeleton test decoder configuration for
    a single error LUT decoder.
    """
    config = qec.decoder_config()
    config.id = decoder_id
    config.type = "single_error_lut"
    config.block_size = 20
    config.syndrome_size = 10
    config.num_syndromes_per_round = config.syndrome_size

    # Create sparse H matrix representation from a zero matrix
    H = np.zeros((config.syndrome_size, config.block_size), dtype=np.uint8)
    config.H_sparse = qec.pcm_to_sparse_vec(H)

    # Create sparse O matrix representation from a zero matrix
    O = np.zeros((2, config.block_size), dtype=np.uint8)
    config.O_sparse = qec.pcm_to_sparse_vec(O)

    # Generate timelike sparse detector matrix
    config.D_sparse = qec.generate_timelike_sparse_detector_matrix(
        config.num_syndromes_per_round, 2, include_first_round=False)

    return config


def create_test_decoder_config_nv_qldpc(decoder_id):
    """
    Helper function to create a sample, skeleton test decoder configuration for
    the NV-QLDPC decoder.
    """
    config = create_test_empty_decoder_config(decoder_id)
    config.type = "nv-qldpc-decoder"

    # Create NV-QLDPC decoder configuration
    nv_config = qec.nv_qldpc_decoder_config()
    nv_config.use_sparsity = True
    nv_config.max_iterations = 50
    nv_config.use_osd = True
    nv_config.osd_order = 60
    nv_config.osd_method = 3
    nv_config.error_rate_vec = [0.1] * config.block_size

    nv_config.n_threads = 128
    nv_config.bp_batch_size = 1
    nv_config.osd_batch_size = 16
    nv_config.iter_per_check = 2
    nv_config.clip_value = 10.0
    nv_config.bp_method = 3
    nv_config.scale_factor = 1.0
    nv_config.proc_float = "fp64"

    # Relay-BP configuration
    nv_config.gamma0 = 0.0
    nv_config.gamma_dist = [0.1, 0.2]
    nv_config.srelay_config = qec.qecrt.config.srelay_bp_config()
    nv_config.srelay_config.pre_iter = 5
    nv_config.srelay_config.num_sets = 10
    nv_config.srelay_config.stopping_criterion = "NConv"
    nv_config.srelay_config.stop_nconv = 10
    # explicit_gammas must have num_sets rows (10 in this case)
    nv_config.explicit_gammas = [[0.1] * config.block_size for _ in range(10)]
    nv_config.bp_seed = 42
    nv_config.composition = 1

    # Set the custom args
    config.set_decoder_custom_args(nv_config)

    return config


def test_single_decoder():
    """
    Test YAML serialization/deserialization and creation of a single NV-QLDPC decoder.
    """
    if not is_nv_qldpc_decoder_available():
        pytest.skip("NV-QLDPC decoder is not available, skipping test")
    multi_config = qec.multi_decoder_config()
    config = create_test_decoder_config_nv_qldpc(0)
    multi_config.decoders = [config]

    check_decoder_yaml_roundtrip(multi_config)
    check_decoder_creation(multi_config)


def test_multi_decoder():
    """
    Test YAML serialization/deserialization and creation of multiple NV-QLDPC decoders.
    """
    if not is_nv_qldpc_decoder_available():
        pytest.skip("NV-QLDPC decoder is not available, skipping test")
    multi_config = qec.multi_decoder_config()
    config1 = create_test_decoder_config_nv_qldpc(0)
    config2 = create_test_decoder_config_nv_qldpc(1)
    multi_config.decoders = [config1, config2]

    check_decoder_yaml_roundtrip(multi_config)
    check_decoder_creation(multi_config)


def test_multi_lut_decoder():
    """
    Test YAML serialization/deserialization and creation of a multi-error LUT decoder.
    """
    multi_config = qec.multi_decoder_config()
    config = create_test_empty_decoder_config(0)
    config.type = "multi_error_lut"

    lut_config = qec.multi_error_lut_config()
    lut_config.lut_error_depth = 2
    config.set_decoder_custom_args(lut_config)

    multi_config.decoders = [config]

    check_decoder_yaml_roundtrip(multi_config)
    check_decoder_creation(multi_config)


def test_single_lut_decoder():
    """
    Test YAML serialization/deserialization and creation of a single-error LUT decoder.
    """
    multi_config = qec.multi_decoder_config()
    config = create_test_empty_decoder_config(0)
    config.type = "single_error_lut"

    single_lut_config = qec.qecrt.config.single_error_lut_config()
    config.set_decoder_custom_args(single_lut_config)

    multi_config.decoders = [config]

    check_decoder_yaml_roundtrip(multi_config)
    check_decoder_creation(multi_config)


def test_sliding_window_decoder():
    """
    Test YAML serialization/deserialization and creation of a sliding window decoder.
    """
    n_rounds = 4
    n_errs_per_round = 30
    n_syndromes_per_round = 10
    n_cols = n_rounds * n_errs_per_round
    n_rows = n_rounds * n_syndromes_per_round
    weight = 3

    # Generate random PCM
    pcm = qec.generate_random_pcm(n_rounds=n_rounds,
                                  n_errs_per_round=n_errs_per_round,
                                  n_syndromes_per_round=n_syndromes_per_round,
                                  weight=weight,
                                  seed=13)
    pcm = qec.sort_pcm_columns(pcm, n_syndromes_per_round)

    # Top-level decoder config
    multi_config = qec.multi_decoder_config()
    config = create_test_empty_decoder_config(0)
    config.type = "sliding_window"
    config.block_size = n_cols
    config.syndrome_size = n_rows
    config.num_syndromes_per_round = n_syndromes_per_round

    # Convert PCM to sparse representation
    config.H_sparse = qec.pcm_to_sparse_vec(pcm)

    # Create sparse O matrix (2 x n_cols zero matrix)
    O = np.zeros((2, n_cols), dtype=np.uint8)
    config.O_sparse = qec.pcm_to_sparse_vec(O)

    # Reset D_sparse for sliding window (set to None to indicate it's not used)
    config.D_sparse = None

    # Sliding window config
    sw_config = qec.qecrt.config.sliding_window_config()
    sw_config.window_size = 1
    sw_config.step_size = 1
    sw_config.num_syndromes_per_round = n_syndromes_per_round
    sw_config.straddle_start_round = False
    sw_config.straddle_end_round = True
    sw_config.error_rate_vec = [0.1] * config.block_size

    # Inner decoder config
    sw_config.inner_decoder_name = "multi_error_lut"
    sw_config.multi_error_lut_params = qec.multi_error_lut_config()
    sw_config.multi_error_lut_params.lut_error_depth = 2

    config.set_decoder_custom_args(sw_config)

    multi_config.decoders = [config]

    check_decoder_yaml_roundtrip(multi_config)
    check_decoder_creation(multi_config)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
