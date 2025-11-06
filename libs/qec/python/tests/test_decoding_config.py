# ============================================================================ #
# Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import pytest
import numpy as np
import math

import cudaq_qec as qec

# nv_qldpc_decoder_config tests


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


FIELDS = {
    "use_sparsity": (bool, True, False),
    "error_rate": (float, 1e-3, 5e-2),
    "error_rate_vec": (list, [0.01, 0.02, 0.03], [0.2, 0.1]),
    "max_iterations": (int, 25, 50),
    "n_threads": (int, 4, 8),
    "use_osd": (bool, False, True),
    "osd_method": (int, 1, 2),
    "osd_order": (int, 7, 3),
    "bp_batch_size": (int, 64, 128),
    "osd_batch_size": (int, 16, 32),
    "iter_per_check": (int, 2, 3),
    "clip_value": (float, 10.0, 7.5),
    "bp_method": (int, 0, 1),
    "scale_factor": (float, 0.5, 1.25),
    "proc_float": (str, "fp32", "fp64"),
}


def test_nv_qldpc_decoder_config_defaults_are_none():
    nv = qec.nv_qldpc_decoder_config()
    for name in FIELDS:
        assert getattr(nv, name) is None, f"Expected {name} to default to None"


@pytest.mark.parametrize("name, meta", list(FIELDS.items()))
def test_nv_qldpc_decoder_config_set_and_get_each_optional(name, meta):
    nv = qec.nv_qldpc_decoder_config()

    py_type, sample_val, alt_val = meta

    # Initially None
    assert getattr(nv, name) is None

    # Set to a valid value and get back
    setattr(nv, name, sample_val)
    got = getattr(nv, name)
    if py_type is float:
        assert isinstance(got, float)
        assert math.isclose(got, float(sample_val), rel_tol=1e-12, abs_tol=0.0)
    elif py_type is list:
        assert isinstance(got, list)
        assert all(isinstance(x, float)
                   for x in got), f"{name} must be a list of float"
        assert got == sample_val
    else:
        assert isinstance(got, py_type)
        assert got == sample_val

    # Change to an alternate valid value
    setattr(nv, name, alt_val)
    got2 = getattr(nv, name)
    if py_type is float:
        assert math.isclose(got2, float(alt_val), rel_tol=1e-12, abs_tol=0.0)
    else:
        assert got2 == alt_val

    # Set value to None
    setattr(nv, name, None)
    assert getattr(nv, name) is None


def test_nv_qldpc_decoder_config_setting_wrong_types_raises_typeerror():
    nv = qec.nv_qldpc_decoder_config()

    with pytest.raises(TypeError):
        nv.max_iterations = "ten"

    with pytest.raises(TypeError):
        nv.use_sparsity = "True"

    with pytest.raises(TypeError):
        nv.error_rate = "0.1"

    with pytest.raises(TypeError):
        nv.error_rate_vec = [0.1, "nope", 0.3]

    with pytest.raises(TypeError):
        nv.error_rate_vec = 3.14


def test_nv_qldpc_decoder_config_error_rate_vec_accepts_python_list_of_float():
    nv = qec.nv_qldpc_decoder_config()

    vals = [0.0, 0.125, 0.25]
    nv.error_rate_vec = vals
    got = nv.error_rate_vec
    assert isinstance(got, list)
    assert all(isinstance(x, float) for x in got)
    assert got == vals


def test_nv_qldpc_decoder_config_toggle_multiple_fields_and_clear():
    nv = qec.nv_qldpc_decoder_config()

    nv.use_sparsity = True
    nv.error_rate = 0.0123
    nv.error_rate_vec = [0.1, 0.2, 0.3]
    nv.max_iterations = 100
    nv.n_threads = 8
    nv.use_osd = True
    nv.osd_method = 2
    nv.osd_order = 4
    nv.bp_batch_size = 32
    nv.osd_batch_size = 16
    nv.iter_per_check = 3
    nv.clip_value = 7.5
    nv.bp_method = 1
    nv.scale_factor = 0.8
    nv.proc_float = "fp64"

    assert nv is not None
    assert nv.use_sparsity is True
    assert math.isclose(nv.error_rate, 0.0123)
    assert nv.error_rate_vec == [0.1, 0.2, 0.3]
    assert nv.max_iterations == 100
    assert nv.n_threads == 8

    nv.use_sparsity = None
    nv.error_rate = None
    nv.error_rate_vec = None
    nv.max_iterations = None
    nv.n_threads = None

    assert nv.use_sparsity is None
    assert nv.error_rate is None
    assert nv.error_rate_vec is None
    assert nv.max_iterations is None
    assert nv.n_threads is None


# multi_error_lut_config tests

FIELDS_MULTI_ERROR_LUT = {
    "lut_error_depth": (int, 1, 3),
}


def test_multi_error_lut_config_defaults_are_none():
    m = qec.multi_error_lut_config()
    for name in FIELDS_MULTI_ERROR_LUT:
        assert getattr(m, name) is None, f"Expected {name} to default to None"


def test_configure_valid_multi_error_lut_decoders():
    nv = qec.multi_error_lut_config()
    nv.lut_error_depth = 2

    dc = qec.decoder_config()
    dc.id = 0
    dc.type = "multi_error_lut"
    dc.block_size = 10
    dc.syndrome_size = 3
    dc.num_syndromes_per_round = 3
    dc.H_sparse = [1, 2, 3, -1, 6, 7, 8, -1, -1]
    dc.set_decoder_custom_args(nv)

    mdc = qec.multi_decoder_config()
    mdc.decoders = [dc]
    ret = qec.configure_decoders(mdc)
    qec.finalize_decoders()
    assert isinstance(ret, int)
    assert ret == 0


# decoder_config tests


def test_decoder_config_yaml_roundtrip_and_custom_args():
    # Build NV config and embed into DecoderConfig via helper
    nv = qec.nv_qldpc_decoder_config()
    nv.use_sparsity = True
    nv.error_rate = 0.01
    nv.max_iterations = 50
    nv.error_rate_vec = [0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.1]

    dc = qec.decoder_config()
    dc.id = 0
    dc.type = "nv-qldpc-decoder"
    dc.block_size = 10
    dc.syndrome_size = 3
    dc.num_syndromes_per_round = 3
    dc.H_sparse = [1, 2, 3, -1, 6, 7, 8, -1, -1]
    dc.set_decoder_custom_args(nv)

    yaml_text = dc.to_yaml_str()
    assert isinstance(yaml_text, str) and len(yaml_text) > 0

    dc2 = qec.decoder_config.from_yaml_str(yaml_text)

    # Basic scalar fields
    assert dc2 is not None
    assert dc2.id == 0
    assert dc2.type == "nv-qldpc-decoder"
    assert dc2.block_size == 10
    assert dc2.syndrome_size == 3
    assert dc2.num_syndromes_per_round == 3

    # Recover NV config from decoder_custom_args (it's already the config object)
    nv2 = dc2.decoder_custom_args
    assert nv2 is not None
    assert nv2.use_sparsity is True
    assert math.isclose(nv2.error_rate, 0.01)
    assert nv2.max_iterations == 50


# multi_decoder_config tests


def test_multi_decoder_config_yaml_roundtrip():
    # Build NV config and embed into DecoderConfig via helper
    nv = qec.nv_qldpc_decoder_config()
    nv.use_sparsity = True
    nv.error_rate = 0.01
    nv.error_rate_vec = [0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.1]
    nv.max_iterations = 50

    d1 = qec.decoder_config()
    d1.id = 0
    d1.type = "nv-qldpc-decoder"
    d1.block_size = 10
    d1.syndrome_size = 3
    d1.num_syndromes_per_round = 3
    d1.H_sparse = [1, 2, 3, -1, 6, 7, 8, -1, -1]
    d1.set_decoder_custom_args(nv)

    lut_config = qec.multi_error_lut_config()
    lut_config.lut_error_depth = 3

    d2 = qec.decoder_config()
    d2.id = 1
    d2.type = "multi_error_lut"
    d2.block_size = 10
    d2.syndrome_size = 3
    d2.num_syndromes_per_round = 3
    d2.H_sparse = [1, 2, 3, -1, 6, 7, 8, -1, -1]
    d2.set_decoder_custom_args(lut_config)

    mdc = qec.multi_decoder_config()
    mdc.decoders = [d1, d2]

    yaml_text = mdc.to_yaml_str()
    assert isinstance(yaml_text, str) and "0" in yaml_text and "1" in yaml_text

    mdc2 = qec.multi_decoder_config.from_yaml_str(yaml_text)
    assert mdc2 is not None
    assert len(mdc2.decoders) == 2
    ids = sorted({md.id for md in mdc2.decoders})
    assert ids == [0, 1]


def test_configure_decoders_from_str_smoke():
    multi_decoder_config = qec.multi_decoder_config()
    yaml_str = multi_decoder_config.to_yaml_str()
    status = qec.configure_decoders_from_str(yaml_str)
    assert isinstance(status, int)
    qec.finalize_decoders()

    nv = qec.nv_qldpc_decoder_config()
    nv.error_rate_vec = [0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.1]

    decoder_config = qec.decoder_config()
    decoder_config.id = 0
    decoder_config.type = "nv-qldpc-decoder"
    decoder_config.block_size = 10
    decoder_config.syndrome_size = 3
    decoder_config.num_syndromes_per_round = 3
    decoder_config.H_sparse = [1, 2, 3, -1, 6, 7, 8, -1, -1]
    decoder_config.set_decoder_custom_args(nv)
    yaml_str = decoder_config.to_yaml_str()
    # Do not instantiate the decoder if it is not available.
    if not is_nv_qldpc_decoder_available():
        return
    status = qec.configure_decoders_from_str(yaml_str)
    assert isinstance(status, int)
    qec.finalize_decoders()


def test_configure_decoders_from_file_smoke(tmp_path):
    path = tmp_path / "decoders.yaml"
    path.write_text(qec.multi_decoder_config().to_yaml_str(), encoding="utf-8")

    status = qec.configure_decoders_from_file(str(path))
    assert isinstance(status, int)
    qec.finalize_decoders()


def test_configure_valid_decoders():
    nv = qec.nv_qldpc_decoder_config()
    nv.use_sparsity = True
    nv.error_rate = 0.01
    nv.error_rate_vec = [0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.1]
    nv.max_iterations = 50

    dc = qec.decoder_config()
    dc.id = 0
    dc.type = "multi_error_lut"
    dc.block_size = 10
    dc.syndrome_size = 3
    dc.num_syndromes_per_round = 3
    dc.H_sparse = [1, 2, 3, -1, 6, 7, 8, -1, -1]
    lut_config = qec.multi_error_lut_config()
    lut_config.lut_error_depth = 2
    dc.set_decoder_custom_args(lut_config)

    mdc = qec.multi_decoder_config()
    mdc.decoders = [dc]
    ret = qec.configure_decoders(mdc)
    qec.finalize_decoders()
    assert isinstance(ret, int)
    assert ret == 0


def test_configure_invalid_decoders():
    nv = qec.nv_qldpc_decoder_config()
    nv.use_sparsity = True
    nv.error_rate = 0.01
    nv.error_rate_vec = [0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.1]
    nv.max_iterations = 50

    decoder_config = qec.decoder_config()
    decoder_config.id = 0
    decoder_config.type = "invalid-decoder"
    decoder_config.block_size = 10
    decoder_config.syndrome_size = 3
    decoder_config.num_syndromes_per_round = 3
    decoder_config.H_sparse = [1, 2, 3, -1, 6, 7, 8, -1, -1]
    decoder_config.set_decoder_custom_args(nv)

    multi_decoder_config = qec.multi_decoder_config()
    multi_decoder_config.decoders = [decoder_config]
    ret = qec.configure_decoders(multi_decoder_config)
    assert isinstance(ret, int)
    assert ret != 0


if __name__ == "__main__":
    pytest.main()
