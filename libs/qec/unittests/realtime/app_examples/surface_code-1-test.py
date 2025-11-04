# ============================================================================ #
# Copyright (c) 2025 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import os
import pathlib
import time
import pytest

# Force stim as the default simulator for emulation
# Note: this must be done before importing `cudaq`
os.environ["CUDAQ_DEFAULT_SIMULATOR"] = "stim"

import cudaq
import cudaq_qec as qec

from surface_code_1 import demo_circuit_host

CASES = [
    pytest.param(
        {
            "distance": 3,
            "p_spam": 0.01,
            "num_rounds": 12,
            "decoder_window": 6,
            "num_shots": 1000,
            "target": "stim",
            "number_of_non_zero_values_threshold": 40,
            "number_of_corrections_decoder_threshold": 40
        },
        id="d3-local"),
    pytest.param(
        {
            "distance": 5,
            "p_spam": 0.01,
            "num_rounds": 20,
            "decoder_window": 10,
            "num_shots": 1000,
            "target": "stim",
            "number_of_non_zero_values_threshold": 40,
            "number_of_corrections_decoder_threshold": 40
        },
        id="d5-local",
    ),
    pytest.param(
        {
            "distance": 3,
            "p_spam": 0.01,
            "num_rounds": 12,
            "decoder_window": 6,
            "num_shots": 1000,
            "target": "quantinuum",
            "machine_name": "Helios-1Dummy",
            "number_of_non_zero_values_threshold": 0,
            "number_of_corrections_decoder_threshold": 1000
        },
        id="d3-quantinuum-emulate-in-process"),
    pytest.param(
        {
            "distance": 5,
            "p_spam": 0.01,
            "num_rounds": 20,
            "decoder_window": 10,
            "num_shots": 1000,
            "target": "quantinuum",
            "machine_name": "Helios-1Dummy",
            "number_of_non_zero_values_threshold": 0,
            "number_of_corrections_decoder_threshold": 0
        },
        id="d5-quantinuum-emulate-in-process",
    ),
]


# Fixtures
@pytest.fixture(scope="module", params=CASES)
def case(request):
    return request.param


@pytest.fixture(scope="module")
def code_obj(case):
    return qec.get_code("surface_code", distance=case["distance"])


@pytest.fixture(scope="module")
def tmp_case_dir(tmp_path_factory, case):
    return tmp_path_factory.mktemp(f"dem_d{case['distance']}")


@pytest.fixture(scope="module")
def dem_file(case, code_obj, tmp_case_dir):
    dem_path = pathlib.Path(
        tmp_case_dir
    ) / f"temp_dem_d{case['distance']}_{format(time.time())}.yaml"
    print(dem_path)
    demo_circuit_host(
        code_obj=code_obj,
        distance=case["distance"],
        p_spam=case["p_spam"],
        state_prep_op=qec.operation.prep0,
        num_shots=case["num_shots"],
        num_rounds=case["num_rounds"],
        num_logical=1,
        dem_filename=str(dem_path),
        save_dem=True,
        load_dem=False,
        decoder_window=case["decoder_window"],
        target_name="stim",
        emulate=True,
        machine_name="",
    )
    assert dem_path.exists() and dem_path.stat().st_size > 0
    return dem_path


# Tests (parameterized via `case`)


def test_run_from_demo_in_process(case, code_obj, dem_file, capsys):
    result_dict = demo_circuit_host(
        code_obj=code_obj,
        distance=case["distance"],
        p_spam=case["p_spam"],
        state_prep_op=qec.operation.prep0,
        num_shots=case["num_shots"],
        num_rounds=case["num_rounds"],
        num_logical=1,
        dem_filename=str(dem_file),
        save_dem=True,
        load_dem=True,
        decoder_window=case["decoder_window"],
        target_name=case["target"],
        emulate=True,
        machine_name=case["machine_name"] if "machine_name" in case else "",
    )

    qec.finalize_decoders()
    # Check the returned result has expected keys
    print("Result for distance", case["distance"], ":", result_dict)
    assert "num_non_zero" in result_dict
    assert "num_corrections" in result_dict
    # Check conditions
    assert result_dict["num_non_zero"] <= case[
        "number_of_non_zero_values_threshold"]
    assert result_dict["num_corrections"] >= case[
        "number_of_corrections_decoder_threshold"]


def test_build_dem_with_zero_p_spam_raises(case, code_obj, tmp_case_dir):
    bad_dem = pathlib.Path(tmp_case_dir) / f"bad_dem_d{case['distance']}.yaml"
    with pytest.raises(RuntimeError,
                       match="Cannot build a DEM with p_spam = 0.0"):
        demo_circuit_host(
            code_obj=code_obj,
            distance=case["distance"],
            p_spam=0.0,
            state_prep_op=qec.operation.prep0,
            num_shots=1,
            num_rounds=case["num_rounds"],
            num_logical=1,
            dem_filename=str(bad_dem),
            save_dem=True,
            load_dem=False,
            decoder_window=case["decoder_window"],
            target_name="stim",
            emulate=True,
            machine_name="",
        )


def test_quantinuum_requires_machine_name(case, code_obj, dem_file):
    with pytest.raises(
            RuntimeError,
            match="machine_name must be set when target_name is quantinuum"):
        demo_circuit_host(
            code_obj=code_obj,
            distance=case["distance"],
            p_spam=case["p_spam"],
            state_prep_op=qec.operation.prep0,
            num_shots=1,
            num_rounds=case["num_rounds"],
            num_logical=1,
            dem_filename=str(dem_file),
            save_dem=False,
            load_dem=True,
            decoder_window=case["decoder_window"],
            target_name="quantinuum",
            emulate=True,
            machine_name="",  # this should trigger the error
        )

    qec.finalize_decoders()
