# ============================================================================ #
# Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import pytest
import numpy as np
import cudaq
import cudaq_qec as qec

# This test checks that the correct device code is generated
# for different targets when using the QEC decoding APIs.


@cudaq.kernel
def kernel() -> int:
    # This call will generate different device code
    # depending on the target set in cudaq.
    qec.reset_decoder(0)
    return 0


def test_default_sim_target():
    # A default simulator target uses the simulation decoder
    cudaq.reset_target()
    kernel_code = str(kernel)
    print(kernel_code)
    assert "_ZN5cudaq3qec8decoding10simulation13reset_decoderEm" in kernel_code


@pytest.mark.skip(
    reason=
    "Disabling support for calling kernels defined in Python from kernels defined in C++."
)
def test_quantinuum_target():
    # A Quantinuum target uses the Quantinuum decoder
    cudaq.reset_target()
    cudaq.set_target("quantinuum", machine="Helios-1SC", emulate=True)
    kernel_code = str(kernel)
    print(kernel_code)
    assert "@reset_decoder_ui64" in kernel_code


@pytest.mark.skip(
    reason=
    "Disabling support for calling kernels defined in Python from kernels defined in C++."
)
def test_target_swap():
    # Swapping targets back and forth generates correct code each time
    cudaq.reset_target()
    cudaq.set_target("quantinuum", machine="Helios-1SC", emulate=True)
    kernel_code = str(kernel)
    print(kernel_code)
    assert "@reset_decoder_ui64" in kernel_code

    cudaq.set_target("stim")
    kernel_code = str(kernel)
    print(kernel_code)
    assert "_ZN5cudaq3qec8decoding10simulation13reset_decoderEm" in kernel_code

    cudaq.set_target("quantinuum", machine="Helios-1SC", emulate=True)
    kernel_code = str(kernel)
    print(kernel_code)
    assert "@reset_decoder_ui64" in kernel_code

    cudaq.reset_target()
    # Reset back to default simulator
    kernel_code = str(kernel)
    print(kernel_code)
    assert "_ZN5cudaq3qec8decoding10simulation13reset_decoderEm" in kernel_code
