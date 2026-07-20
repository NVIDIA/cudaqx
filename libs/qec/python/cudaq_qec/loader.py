# ============================================================================ #
# Copyright (c) 2024 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import cudaq
from ._pycudaqx_qec_the_suffix_matters_cudaq_qec import *


def qec_set_target_callback(target):
    if target.name == "quantinuum":
        qecrt.load_quantinuum_realtime_decoding()
    else:
        qecrt.load_simulation_realtime_decoding()
    qecrt.load_device_kernels()
