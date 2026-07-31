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


def is_nv_qldpc_decoder_available():
    """
    Helper function to check if the NV-QLDPC decoder is available.
    """
    try:
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

    # Create sparse H matrix representation from a zero matrix
    H = np.zeros((config.syndrome_size, config.block_size), dtype=np.uint8)
    config.H_sparse = qec.pcm_to_sparse_vec(H)

    # Create sparse O matrix representation from a zero matrix
    O = np.zeros((2, config.block_size), dtype=np.uint8)
    config.O_sparse = qec.pcm_to_sparse_vec(O)

    # Generate timelike sparse detector matrix
    config.D_sparse = qec.generate_timelike_sparse_detector_matrix(
        config.syndrome_size, 2, include_first_round=False)
    return config


def create_test_decoder_config_nv_qldpc(decoder_id):
    """
    Helper function to create a sample, skeleton test decoder configuration for
    the NV-QLDPC decoder.
    """
    config = create_test_empty_decoder_config(decoder_id)
    config.type = "nv-qldpc-decoder"

    # Create NV-QLDPC decoder configuration (a parameter dict; keys are
    # governed by the decoder's registered schema)
    config.decoder_custom_args = {
        "use_sparsity": True,
        "max_iterations": 50,
        "use_osd": True,
        "osd_order": 60,
        "osd_method": 3,
        "error_rate_vec": [0.1] * config.block_size,
        "n_threads": 128,
        "bp_batch_size": 1,
        "osd_batch_size": 16,
        "iter_per_check": 2,
        "clip_value": 10.0,
        "bp_method": 3,
        "scale_factor": 1.0,
        "proc_float": "fp64",
        # Relay-BP configuration
        "gamma0": 0.0,
        "gamma_dist": [0.1, 0.2],
        "srelay_config": {
            "pre_iter": 5,
            "num_sets": 10,
            "stopping_criterion": "NConv",
            "stop_nconv": 10,
        },
        # explicit_gammas must have num_sets rows (10 in this case)
        "explicit_gammas": [[0.1] * config.block_size for _ in range(10)],
        "bp_seed": 42,
        "composition": 1,
    }

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

    config.decoder_custom_args = {"lut_error_depth": 2}

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

    config.decoder_custom_args = {}

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

    # Convert PCM to sparse representation
    config.H_sparse = qec.pcm_to_sparse_vec(pcm)

    # Create sparse O matrix (2 x n_cols zero matrix)
    O = np.zeros((2, n_cols), dtype=np.uint8)
    config.O_sparse = qec.pcm_to_sparse_vec(O)

    config.D_sparse = qec.generate_timelike_sparse_detector_matrix(
        config.syndrome_size, 2, include_first_round=False)

    # Sliding window config. inner_decoder_params is validated against the
    # schema registered under inner_decoder_name.
    config.decoder_custom_args = {
        "window_size": 1,
        "step_size": 1,
        "num_syndromes_per_round": n_syndromes_per_round,
        "straddle_start_round": False,
        "straddle_end_round": True,
        "error_rate_vec": [0.1] * config.block_size,
        "inner_decoder_name": "multi_error_lut",
        "inner_decoder_params": {
            "lut_error_depth": 2
        },
    }

    multi_config.decoders = [config]

    check_decoder_yaml_roundtrip(multi_config)
    check_decoder_creation(multi_config)


def test_sliding_window_boundary_syndromes_roundtrip():
    """
    Test that a sliding_window's num_boundary_syndromes parameter survives a
    YAML round trip. This is serialization-only (the boundary-layout decoding
    behavior is exercised by the direct-decoder tests in test_sliding_window).
    """
    multi_config = qec.multi_decoder_config()
    config = create_test_empty_decoder_config(0)
    config.type = "sliding_window"
    config.block_size = 6
    config.syndrome_size = 4

    H = np.zeros((config.syndrome_size, config.block_size), dtype=np.uint8)
    config.H_sparse = qec.pcm_to_sparse_vec(H)
    O = np.zeros((1, config.block_size), dtype=np.uint8)
    config.O_sparse = qec.pcm_to_sparse_vec(O)
    config.D_sparse = qec.generate_timelike_sparse_detector_matrix(
        config.syndrome_size, 2, include_first_round=False)

    config.decoder_custom_args = {
        "window_size": 1,
        "step_size": 1,
        "num_syndromes_per_round": 2,
        "num_boundary_syndromes": 1,
        "error_rate_vec": [0.1] * config.block_size,
        "inner_decoder_name": "single_error_lut",
    }

    multi_config.decoders = [config]

    check_decoder_yaml_roundtrip(multi_config)


# ---------------------------------------------------------------------------
# Chunk-form configurations
# ---------------------------------------------------------------------------

REP5_CHECKS = 4


def chunk_form_yaml(num_rounds):
    """A d=5 repetition code written as phases rather than flat matrices."""
    return f"""
id: 0
type: single_error_lut
num_rounds: {num_rounds}
dem_chunks:
  init:
    num_faults: 9
    H_mid_sparse: [ 0, 1, 5, -1, 1, 2, 6, -1, 2, 3, 7, -1, 3, 4, 8, -1 ]
    H_out_sparse: [ 5, -1, 6, -1, 7, -1, 8, -1 ]
    O_sparse: [ 0, -1 ]
    error_rates: [ 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02 ]
  bulk:
    num_faults: 9
    H_in_sparse: [ 0, 1, 5, -1, 1, 2, 6, -1, 2, 3, 7, -1, 3, 4, 8, -1 ]
    H_out_sparse: [ 5, -1, 6, -1, 7, -1, 8, -1 ]
    O_sparse: [ 0, -1 ]
    error_rates: [ 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02 ]
  final:
    num_faults: 5
    H_in_sparse: [ 0, 1, -1, 1, 2, -1, 2, 3, -1, 3, 4, -1 ]
    O_sparse: [ 0, -1 ]
    error_rates: [ 0.02, 0.02, 0.02, 0.02, 0.02 ]
"""


def test_chunk_form_parses_with_no_flat_matrices():
    config = qec.decoder_config.from_yaml_str(chunk_form_yaml(5))

    assert config.num_rounds == 5
    assert config.dem_chunks is not None
    assert config.dem_chunks.init.num_faults == 9
    # The flat fields are derived, so they stay unset until expansion.
    assert config.block_size == 0
    assert config.syndrome_size == 0
    assert len(config.H_sparse) == 0


def test_expand_dem_chunks_fills_the_flat_fields():
    config = qec.decoder_config.from_yaml_str(chunk_form_yaml(5))
    closed = qec.qecrt.config.expand_dem_chunks(config)

    assert closed is not None
    assert config.syndrome_size == 5 * REP5_CHECKS
    # 4 init/bulk phases of 9 faults, plus a 5-fault final round.
    assert config.block_size == 41
    assert len(closed.error_rates) == config.block_size
    # One -1 terminator per row, so H and D are as tall as the DEM.
    assert config.H_sparse.count(-1) == config.syndrome_size
    assert config.D_sparse.count(-1) == config.syndrome_size


def test_expand_dem_chunks_is_a_no_op_on_a_flat_config():
    config = qec.decoder_config.from_yaml_str(chunk_form_yaml(3))
    qec.qecrt.config.expand_dem_chunks(config)
    flat_H = list(config.H_sparse)

    # Already flat now, so a second call must leave it alone.
    assert qec.qecrt.config.expand_dem_chunks(config) is None
    assert list(config.H_sparse) == flat_H


def test_same_chunks_serve_different_round_counts():
    sizes = {}
    for rounds in (2, 3, 5, 8):
        config = qec.decoder_config.from_yaml_str(chunk_form_yaml(rounds))
        qec.qecrt.config.expand_dem_chunks(config)
        sizes[rounds] = (config.syndrome_size, config.block_size)

    for rounds, (syndrome_size, block_size) in sizes.items():
        assert syndrome_size == rounds * REP5_CHECKS
        assert block_size == 9 * (rounds - 1) + 5


def test_chunk_form_builds_a_working_decoder():
    config = qec.decoder_config.from_yaml_str(chunk_form_yaml(5))
    multi_config = qec.multi_decoder_config()
    multi_config.decoders = [config]

    assert qec.qecrt.config.configure_decoders(multi_config) == 0
    qec.qecrt.config.finalize_decoders()


def test_num_rounds_is_required_with_dem_chunks():
    without = chunk_form_yaml(5).replace("num_rounds: 5\n", "")
    with pytest.raises(Exception):
        qec.decoder_config.from_yaml_str(without)


def test_num_rounds_is_rejected_without_dem_chunks():
    yaml_str = """
id: 0
type: single_error_lut
num_rounds: 5
block_size: 3
syndrome_size: 2
H_sparse: [ 0, -1, 1, -1 ]
O_sparse: [ 0, -1 ]
D_sparse: [ 0, -1, 0, 1, -1 ]
"""
    with pytest.raises(Exception):
        qec.decoder_config.from_yaml_str(yaml_str)


def test_derived_fields_are_rejected_in_chunk_form():
    with_block_size = chunk_form_yaml(5).replace(
        "num_rounds: 5", "num_rounds: 5\nblock_size: 41")
    with pytest.raises(Exception):
        qec.decoder_config.from_yaml_str(with_block_size)


def test_chunk_form_can_be_built_programmatically():
    config = qec.decoder_config()
    config.id = 0
    config.type = "single_error_lut"
    config.num_rounds = 4

    spec = qec.DemChunksSpec()
    spec.init.num_faults = 9
    spec.init.H_mid_sparse = [
        0, 1, 5, -1, 1, 2, 6, -1, 2, 3, 7, -1, 3, 4, 8, -1
    ]
    spec.init.H_out_sparse = [5, -1, 6, -1, 7, -1, 8, -1]
    spec.init.O_sparse = [0, -1]
    spec.init.error_rates = [0.02] * 9
    spec.bulk.num_faults = 9
    spec.bulk.H_in_sparse = [0, 1, 5, -1, 1, 2, 6, -1, 2, 3, 7, -1, 3, 4, 8, -1]
    spec.bulk.H_out_sparse = [5, -1, 6, -1, 7, -1, 8, -1]
    spec.bulk.O_sparse = [0, -1]
    spec.bulk.error_rates = [0.02] * 9
    spec.final.num_faults = 5
    spec.final.H_in_sparse = [0, 1, -1, 1, 2, -1, 2, 3, -1, 3, 4, -1]
    spec.final.O_sparse = [0, -1]
    spec.final.error_rates = [0.02] * 5
    config.dem_chunks = spec

    qec.qecrt.config.expand_dem_chunks(config)
    assert config.syndrome_size == 4 * REP5_CHECKS


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
