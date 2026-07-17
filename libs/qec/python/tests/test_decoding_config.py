# ============================================================================ #
# Copyright (c) 2024 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import math

import numpy as np
import pytest

import cudaq_qec as qec

# Decoder custom args are plain dicts. Their keys are governed by the
# parameter schema each decoder registers (see qec.decoder_param_schema).

nv_qldpc_schema_missing = qec.decoder_param_schema("nv-qldpc-decoder") is None


def is_nv_qldpc_decoder_available():
    """
    Helper function to check if the NV-QLDPC decoder is available.
    """
    try:
        H_list = [[1, 0, 0, 1, 0, 1, 1], [0, 1, 0, 1, 1, 0, 1],
                  [0, 0, 1, 0, 1, 1, 1]]
        H_np = np.array(H_list, dtype=np.uint8)
        qec.get_decoder("nv-qldpc-decoder", H_np)
        return True
    except Exception:
        return False


# Schema introspection tests


@pytest.mark.skipif(
    nv_qldpc_schema_missing,
    reason="nv-qldpc-decoder plugin (and its parameter schema) not available")
def test_nv_qldpc_decoder_param_schema_introspection():
    schema = qec.decoder_param_schema("nv-qldpc-decoder")
    assert schema is not None
    by_key = {entry["key"]: entry for entry in schema}
    assert by_key["max_iterations"]["kind"] == "int32"
    assert by_key["error_rate_vec"]["kind"] == "float64_vec"
    assert by_key["srelay_config"]["kind"] == "subschema"
    assert by_key["srelay_config"]["subschema"] == "srelay_bp"


def test_decoder_param_schema_introspection():
    sw_schema = qec.decoder_param_schema("sliding_window")
    assert sw_schema is not None
    by_key = {entry["key"]: entry for entry in sw_schema}
    assert by_key["error_rate_vec"]["required"] is True
    assert by_key["inner_decoder_params"]["kind"] == "discriminated"
    assert by_key["inner_decoder_params"]["discriminator"] == \
        "inner_decoder_name"

    assert qec.decoder_param_schema("no-such-decoder") is None

    names = qec.registered_decoder_schemas()
    assert "pymatching" in names
    assert "multi_error_lut" in names


# decoder_config custom args tests


def test_decoder_custom_args_is_a_dict():
    dc = qec.decoder_config()
    assert dc.decoder_custom_args == {}

    dc.decoder_custom_args = {"lut_error_depth": 2}
    assert dc.decoder_custom_args == {"lut_error_depth": 2}

    dc.set_decoder_custom_args({"lut_error_depth": 3})
    assert dc.decoder_custom_args == {"lut_error_depth": 3}


@pytest.mark.skipif(
    nv_qldpc_schema_missing,
    reason="nv-qldpc-decoder plugin (and its parameter schema) not available")
def test_decoder_config_yaml_roundtrip_and_custom_args():
    dc = qec.decoder_config()
    dc.id = 0
    dc.type = "nv-qldpc-decoder"
    dc.block_size = 10
    dc.syndrome_size = 3
    dc.H_sparse = [1, 2, 3, -1, 6, 7, 8, -1, -1]
    dc.decoder_custom_args = {
        "use_sparsity": True,
        "error_rate": 0.01,
        "max_iterations": 50,
        "error_rate_vec": [0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.1],
        "srelay_config": {
            "pre_iter": 5,
            "stopping_criterion": "NConv",
        },
    }

    yaml_text = dc.to_yaml_str()
    assert isinstance(yaml_text, str) and len(yaml_text) > 0

    dc2 = qec.decoder_config.from_yaml_str(yaml_text)

    # Basic scalar fields
    assert dc2 is not None
    assert dc2.id == 0
    assert dc2.type == "nv-qldpc-decoder"
    assert dc2.block_size == 10
    assert dc2.syndrome_size == 3

    args = dc2.decoder_custom_args
    assert args["use_sparsity"] is True
    assert math.isclose(args["error_rate"], 0.01)
    assert args["max_iterations"] == 50
    assert args["srelay_config"]["pre_iter"] == 5
    assert args["srelay_config"]["stopping_criterion"] == "NConv"


def test_pymatching_config_yaml_roundtrip():
    dc = qec.decoder_config()
    dc.id = 0
    dc.type = "pymatching"
    dc.block_size = 3
    dc.syndrome_size = 3
    dc.H_sparse = [0, -1, 1, -1, 2, -1]
    dc.O_sparse = [0, -1, 1, -1, 2, -1]
    dc.D_sparse = [0, -1, 1, -1, 2, -1]
    dc.decoder_custom_args = {
        "error_rate_vec": [0.1, 0.2, 0.3],
        "merge_strategy": "smallest_weight",
    }

    yaml_text = dc.to_yaml_str()
    assert isinstance(yaml_text, str) and "pymatching" in yaml_text

    dc2 = qec.decoder_config.from_yaml_str(yaml_text)
    assert dc2 is not None
    assert dc2.type == "pymatching"

    args = dc2.decoder_custom_args
    assert list(args["error_rate_vec"]) == [0.1, 0.2, 0.3]
    assert args["merge_strategy"] == "smallest_weight"


def test_unknown_custom_arg_key_is_rejected():
    dc = qec.decoder_config()
    dc.id = 0
    dc.type = "pymatching"
    dc.block_size = 3
    dc.syndrome_size = 3
    dc.H_sparse = [0, -1, 1, -1, 2, -1]
    dc.O_sparse = [0, -1, 1, -1, 2, -1]
    dc.D_sparse = [0, -1, 1, -1, 2, -1]
    dc.decoder_custom_args = {"merge_strategy": "smallest_weight"}
    yaml_text = qec_yaml_for(dc)

    misspelled = yaml_text.replace("merge_strategy", "merge_stratgey")
    with pytest.raises(RuntimeError):
        qec.multi_decoder_config.from_yaml_str(misspelled)


def test_validate_custom_args_checks_dict_built_configs():
    # Dicts assigned to decoder_custom_args never pass through the YAML
    # parser; validate_custom_args applies the same schema checks explicitly.
    dc = qec.decoder_config()
    dc.type = "pymatching"
    dc.decoder_custom_args = {"merge_strategy": "smallest_weight"}
    dc.validate_custom_args()

    dc.decoder_custom_args = {"merge_stratgey": "smallest_weight"}
    with pytest.raises(RuntimeError, match="merge_stratgey"):
        dc.validate_custom_args()

    # Non-empty args for a type with no registered schema are rejected.
    dc.type = "decoder_without_registered_schema"
    dc.decoder_custom_args = {"anything": 1}
    with pytest.raises(RuntimeError, match="no registered parameter schema"):
        dc.validate_custom_args()

    # multi_decoder_config validates every decoder.
    mdc = qec.multi_decoder_config()
    mdc.decoders = [dc]
    with pytest.raises(RuntimeError):
        mdc.validate_custom_args()


@pytest.mark.skipif(
    nv_qldpc_schema_missing,
    reason="nv-qldpc-decoder plugin (and its parameter schema) not available")
def test_custom_args_dict_values_convert_to_schema_types():
    # The setter converts dict values to the canonical types the registered
    # schema declares: Python ints are accepted for f64 params and negative
    # ints for int32 params (the generic conversion stores every int as
    # size_t, which would reject both).
    dc = qec.decoder_config()
    dc.id = 0
    dc.type = "nv-qldpc-decoder"
    dc.block_size = 1
    dc.syndrome_size = 1
    dc.H_sparse = [0, -1]
    dc.O_sparse = [0, -1]
    dc.D_sparse = [0, -1]

    dc.decoder_custom_args = {"clip_value": 2}  # int for an f64 param
    assert "clip_value" in dc.to_yaml_str()

    dc.decoder_custom_args = {"bp_seed": -1}  # negative int32
    assert dc.decoder_custom_args == {"bp_seed": -1}
    round_tripped = qec.multi_decoder_config.from_yaml_str(qec_yaml_for(dc))
    assert round_tripped.decoders[0].decoder_custom_args == {"bp_seed": -1}

    # Mismatched types raise a clear error naming the parameter.
    with pytest.raises(RuntimeError, match="bp_seed"):
        dc.decoder_custom_args = {"bp_seed": "oops"}


def test_decoder_config_from_yaml_str_rejects_malformed_yaml():
    with pytest.raises(RuntimeError, match="Invalid decoder configuration"):
        qec.decoder_config.from_yaml_str("id: [oops")


def test_validate_custom_args_runs_schema_validate_hook():
    # sliding_window registers a validate hook for cross-field constraints
    # (step_size must be between 1 and window_size).
    dc = qec.decoder_config()
    dc.type = "sliding_window"
    dc.decoder_custom_args = {
        "window_size": 4,
        "step_size": 2,
        "error_rate_vec": [0.01, 0.01],
        "inner_decoder_name": "single_error_lut",
    }
    dc.validate_custom_args()

    dc.decoder_custom_args = {
        "window_size": 2,
        "step_size": 4,
        "error_rate_vec": [0.01, 0.01],
        "inner_decoder_name": "single_error_lut",
    }
    with pytest.raises(RuntimeError, match="step_size"):
        dc.validate_custom_args()

    # Missing required key (error_rate_vec).
    dc.decoder_custom_args = {"inner_decoder_name": "single_error_lut"}
    with pytest.raises(RuntimeError, match="error_rate_vec"):
        dc.validate_custom_args()


def test_decoder_config_json_schema_validates_yaml_documents():
    # The exported JSON Schema lets standard third-party tooling validate
    # user-provided configuration YAML without loading this library.
    jsonschema = pytest.importorskip("jsonschema")
    yaml = pytest.importorskip("yaml")
    import json

    schema = json.loads(qec.decoder_config_json_schema())
    jsonschema.Draft202012Validator.check_schema(schema)
    validator = jsonschema.Draft202012Validator(schema)

    # Every registered decoder schema appears in the export.
    assert set(qec.registered_decoder_schemas()) == set(
        schema["$defs"]["decoder_params"])

    # A real emitted configuration validates.
    dc = qec.decoder_config()
    dc.id = 0
    dc.type = "pymatching"
    dc.block_size = 3
    dc.syndrome_size = 3
    dc.H_sparse = [0, -1, 1, -1, 2, -1]
    dc.O_sparse = [0, -1, 1, -1, 2, -1]
    dc.D_sparse = [0, -1, 1, -1, 2, -1]
    dc.decoder_custom_args = {
        "error_rate_vec": [0.1, 0.1, 0.1],
        "merge_strategy": "smallest_weight",
    }
    document = yaml.safe_load(qec_yaml_for(dc))
    validator.validate(document)

    # Unknown custom-arg keys fail validation.
    bad = yaml.safe_load(qec_yaml_for(dc))
    bad["decoders"][0]["decoder_custom_args"]["merge_stratgey"] = \
        bad["decoders"][0]["decoder_custom_args"].pop("merge_strategy")
    with pytest.raises(jsonschema.ValidationError):
        validator.validate(bad)

    # Missing required custom-arg keys fail validation (sliding_window
    # requires error_rate_vec and inner_decoder_name).
    missing = yaml.safe_load(qec_yaml_for(dc))
    missing["decoders"][0]["type"] = "sliding_window"
    missing["decoders"][0]["decoder_custom_args"] = {"window_size": 2}
    with pytest.raises(jsonschema.ValidationError):
        validator.validate(missing)

    # Custom args for a type with no registered schema fail validation.
    unregistered = yaml.safe_load(qec_yaml_for(dc))
    unregistered["decoders"][0]["type"] = "decoder_without_registered_schema"
    with pytest.raises(jsonschema.ValidationError):
        validator.validate(unregistered)

    # Missing envelope fields fail validation.
    no_type = yaml.safe_load(qec_yaml_for(dc))
    del no_type["decoders"][0]["type"]
    with pytest.raises(jsonschema.ValidationError):
        validator.validate(no_type)


def qec_yaml_for(dc):
    mdc = qec.multi_decoder_config()
    mdc.decoders = [dc]
    return mdc.to_yaml_str()


# trt_decoder tests (need the trt_decoder plugin for its parameter schema)

trt_schema_missing = qec.decoder_param_schema("trt_decoder") is None


@pytest.mark.skipif(
    trt_schema_missing,
    reason="trt_decoder plugin (and its parameter schema) not available")
def test_trt_decoder_config_yaml_roundtrip():
    dc = qec.decoder_config()
    dc.id = 0
    dc.type = "trt_decoder"
    dc.block_size = 10
    dc.syndrome_size = 3
    dc.H_sparse = [1, 2, 3, -1, 6, 7, 8, -1, -1]
    dc.decoder_custom_args = {
        "engine_load_path": "/path/to/engine.trt",
        "precision": "fp16",
        "memory_workspace": 1073741824,  # 1GB
    }

    yaml_text = dc.to_yaml_str()
    assert isinstance(yaml_text, str) and len(yaml_text) > 0

    dc2 = qec.decoder_config.from_yaml_str(yaml_text)

    assert dc2 is not None
    assert dc2.id == 0
    assert dc2.type == "trt_decoder"

    args = dc2.decoder_custom_args
    assert args["engine_load_path"] == "/path/to/engine.trt"
    assert args["precision"] == "fp16"
    assert args["memory_workspace"] == 1073741824


@pytest.mark.skipif(
    trt_schema_missing,
    reason="trt_decoder plugin (and its parameter schema) not available")
def test_trt_decoder_chromobius_global_config_yaml_roundtrip():
    dc = qec.decoder_config()
    dc.id = 0
    dc.type = "trt_decoder"
    dc.block_size = 3
    dc.syndrome_size = 3
    dc.H_sparse = [0, -1, 1, -1, 2, -1]
    dc.O_sparse = [0, -1, 1, -1, 2, -1]
    dc.D_sparse = [0, -1, 1, -1, 2, -1]
    dc.decoder_custom_args = {
        "onnx_load_path": "/tmp/predecoder.onnx",
        "global_decoder": "chromobius",
        "global_decoder_params": {
            "ignore_decomposition_failures": True,
            "return_weight": False,
        },
    }

    yaml_text = dc.to_yaml_str()
    assert isinstance(yaml_text, str) and "chromobius" in yaml_text

    dc2 = qec.decoder_config.from_yaml_str(yaml_text)
    assert dc2 is not None
    assert dc2.type == "trt_decoder"

    args = dc2.decoder_custom_args
    assert args["global_decoder"] == "chromobius"
    assert args["global_decoder_params"]["ignore_decomposition_failures"] \
        is True
    assert args["global_decoder_params"]["return_weight"] is False


@pytest.mark.skipif(
    trt_schema_missing,
    reason="trt_decoder plugin (and its parameter schema) not available")
def test_trt_decoder_default_global_params_materialized():
    # A named global decoder with a registered schema gets an empty params
    # section materialized on parse.
    dc = qec.decoder_config()
    dc.id = 0
    dc.type = "trt_decoder"
    dc.block_size = 3
    dc.syndrome_size = 3
    dc.H_sparse = [0, -1, 1, -1, 2, -1]
    dc.O_sparse = [0, -1, 1, -1, 2, -1]
    dc.D_sparse = [0, -1, 1, -1, 2, -1]
    dc.decoder_custom_args = {"global_decoder": "pymatching"}

    mdc2 = qec.multi_decoder_config.from_yaml_str(qec_yaml_for(dc))
    args = mdc2.decoders[0].decoder_custom_args
    assert args["global_decoder_params"] == {}

    # ... and already on FIRST emission (matching the old typed path), so
    # emitted YAML is stable across round trips.
    first = qec_yaml_for(dc)
    assert "global_decoder_params" in first
    assert mdc2.to_yaml_str() == first


@pytest.mark.skipif(
    nv_qldpc_schema_missing,
    reason="nv-qldpc-decoder plugin (and its parameter schema) not available")
def test_validate_custom_args_checks_value_kinds():
    # A dict assigned before `type` is set takes the generic conversion
    # (ints stored as size_t), which an f64 param cannot read back at
    # emission. validate_custom_args must name the offending key instead of
    # letting emission fail with a low-context error later.
    dc = qec.decoder_config()
    dc.decoder_custom_args = {"clip_value": 2}  # type not set yet
    dc.type = "nv-qldpc-decoder"
    with pytest.raises(RuntimeError, match="clip_value"):
        dc.validate_custom_args()

    # Assigned after `type`, the same dict converts to the schema's declared
    # types and validates.
    dc.decoder_custom_args = {"clip_value": 2}
    dc.validate_custom_args()


def test_non_schema_keys_dropped_from_emission_and_decoder_params():
    # A key outside the registered schema cannot round-trip through YAML, so
    # it is warned-and-dropped from the emitted YAML (and from the map local
    # decoders receive) rather than taking effect locally but silently
    # vanishing when the config is serialized for a remote target.
    dc = qec.decoder_config()
    dc.id = 0
    dc.type = "multi_error_lut"
    dc.block_size = 3
    dc.syndrome_size = 3
    dc.H_sparse = [0, -1, 1, -1, 2, -1]
    dc.decoder_custom_args = {"lut_error_depth": 2, "not_a_real_param": 42}

    yaml_text = qec_yaml_for(dc)
    assert "lut_error_depth" in yaml_text
    assert "not_a_real_param" not in yaml_text

    # The stored args are untouched; validate_custom_args still rejects them
    # for callers who want a hard error instead of the warn-and-drop.
    assert dc.decoder_custom_args["not_a_real_param"] == 42
    with pytest.raises(RuntimeError, match="not_a_real_param"):
        dc.validate_custom_args()


@pytest.mark.skipif(
    trt_schema_missing,
    reason="trt_decoder plugin (and its parameter schema) not available")
def test_trt_decoder_rejects_unknown_global_params():
    dc = qec.decoder_config()
    dc.id = 0
    dc.type = "trt_decoder"
    dc.block_size = 3
    dc.syndrome_size = 3
    dc.H_sparse = [0, -1, 1, -1, 2, -1]
    dc.O_sparse = [0, -1, 1, -1, 2, -1]
    dc.D_sparse = [0, -1, 1, -1, 2, -1]
    yaml_text = """
decoders:
  - id: 0
    type: trt_decoder
    block_size: 3
    syndrome_size: 3
    H_sparse: [0, -1, 1, -1, 2, -1]
    O_sparse: [0, -1, 1, -1, 2, -1]
    D_sparse: [0, -1, 1, -1, 2, -1]
    decoder_custom_args:
      global_decoder: my_plugin
      global_decoder_params: {}
"""
    with pytest.raises(RuntimeError):
        qec.multi_decoder_config.from_yaml_str(yaml_text)


# multi_decoder_config tests


@pytest.mark.skipif(
    nv_qldpc_schema_missing,
    reason="nv-qldpc-decoder plugin (and its parameter schema) not available")
def test_multi_decoder_config_yaml_roundtrip():
    d1 = qec.decoder_config()
    d1.id = 0
    d1.type = "nv-qldpc-decoder"
    d1.block_size = 10
    d1.syndrome_size = 3
    d1.H_sparse = [1, 2, 3, -1, 6, 7, 8, -1, -1]
    d1.decoder_custom_args = {
        "use_sparsity": True,
        "error_rate": 0.01,
        "error_rate_vec": [0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.1],
        "max_iterations": 50,
    }

    d2 = qec.decoder_config()
    d2.id = 1
    d2.type = "multi_error_lut"
    d2.block_size = 10
    d2.syndrome_size = 3
    d2.H_sparse = [1, 2, 3, -1, 6, 7, 8, -1, -1]
    d2.decoder_custom_args = {"lut_error_depth": 3}

    mdc = qec.multi_decoder_config()
    mdc.decoders = [d1, d2]

    yaml_text = mdc.to_yaml_str()
    assert isinstance(yaml_text, str) and "0" in yaml_text and "1" in yaml_text

    mdc2 = qec.multi_decoder_config.from_yaml_str(yaml_text)
    assert mdc2 is not None
    assert len(mdc2.decoders) == 2
    ids = sorted({md.id for md in mdc2.decoders})
    assert ids == [0, 1]
    assert mdc2 == mdc


# configure_decoders tests


def test_configure_valid_multi_error_lut_decoders():
    dc = qec.decoder_config()
    dc.id = 0
    dc.type = "multi_error_lut"
    dc.block_size = 10
    dc.syndrome_size = 3
    dc.H_sparse = [1, 2, 3, -1, 6, 7, 8, -1, -1]
    dc.D_sparse = qec.generate_timelike_sparse_detector_matrix(
        dc.syndrome_size, 2, include_first_round=False)
    dc.decoder_custom_args = {"lut_error_depth": 2}

    mdc = qec.multi_decoder_config()
    mdc.decoders = [dc]
    ret = qec.configure_decoders(mdc)
    qec.finalize_decoders()
    assert isinstance(ret, int)
    assert ret == 0


def test_configure_decoders_from_str_smoke():
    multi_decoder_config = qec.multi_decoder_config()
    yaml_str = multi_decoder_config.to_yaml_str()
    status = qec.configure_decoders_from_str(yaml_str)
    assert isinstance(status, int)
    qec.finalize_decoders()

    if nv_qldpc_schema_missing:
        return
    decoder_config = qec.decoder_config()
    decoder_config.id = 0
    decoder_config.type = "nv-qldpc-decoder"
    decoder_config.block_size = 10
    decoder_config.syndrome_size = 3
    decoder_config.H_sparse = [1, 2, 3, -1, 6, 7, 8, -1, -1]
    decoder_config.decoder_custom_args = {
        "error_rate_vec": [0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.1],
    }
    multi_decoder_config = qec.multi_decoder_config()
    multi_decoder_config.decoders = [decoder_config]
    yaml_str = multi_decoder_config.to_yaml_str()
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


def make_pymatching_multi_decoder_config(pm_args, h_sparse=None):
    dc = qec.decoder_config()
    dc.id = 0
    dc.type = "pymatching"
    dc.block_size = 3
    dc.syndrome_size = 3
    dc.H_sparse = h_sparse if h_sparse is not None else [0, -1, 1, -1, 2, -1]
    dc.O_sparse = [0, -1, 1, -1, 2, -1]
    dc.D_sparse = [0, -1, 1, -1, 2, -1]
    dc.decoder_custom_args = pm_args

    mdc = qec.multi_decoder_config()
    mdc.decoders = [dc]
    return mdc


def configure_pymatching_status(pm_args, h_sparse=None):
    try:
        return qec.configure_decoders(
            make_pymatching_multi_decoder_config(pm_args, h_sparse))
    finally:
        qec.finalize_decoders()


def test_configure_valid_pymatching_decoder():
    ret = configure_pymatching_status({
        "error_rate_vec": [0.1, 0.1, 0.1],
        "merge_strategy": "smallest_weight",
    })
    assert isinstance(ret, int)
    assert ret == 0


@pytest.mark.parametrize(
    "error_rate_vec",
    ([0.1, 0.1], [0.0, 0.1, 0.1], [0.1, 0.6, 0.1]),
)
def test_configure_invalid_pymatching_error_rate_vec(error_rate_vec):
    ret = configure_pymatching_status({
        "error_rate_vec": error_rate_vec,
        "merge_strategy": "smallest_weight",
    })
    assert isinstance(ret, int)
    assert ret != 0


def test_configure_invalid_pymatching_merge_strategy():
    ret = configure_pymatching_status({
        "error_rate_vec": [0.1, 0.1, 0.1],
        "merge_strategy": "not-a-strategy",
    })
    assert isinstance(ret, int)
    assert ret != 0


def test_configure_invalid_pymatching_non_graphlike_h_sparse():
    ret = configure_pymatching_status(
        {
            "error_rate_vec": [0.1, 0.1, 0.1],
            "merge_strategy": "smallest_weight",
        },
        h_sparse=[0, -1, 0, -1, 0, -1])
    assert isinstance(ret, int)
    assert ret != 0


def test_configure_invalid_decoders():
    decoder_config = qec.decoder_config()
    decoder_config.id = 0
    decoder_config.type = "invalid-decoder"
    decoder_config.block_size = 10
    decoder_config.syndrome_size = 3
    decoder_config.H_sparse = [1, 2, 3, -1, 6, 7, 8, -1, -1]
    decoder_config.decoder_custom_args = {"max_iterations": 50}

    multi_decoder_config = qec.multi_decoder_config()
    multi_decoder_config.decoders = [decoder_config]
    ret = qec.configure_decoders(multi_decoder_config)
    assert isinstance(ret, int)
    assert ret != 0


if __name__ == "__main__":
    pytest.main()
