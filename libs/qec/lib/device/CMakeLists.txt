# ============================================================================ #
# Copyright (c) 2024 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

cudaqx_add_device_code(cudaq-qec
  SOURCES 
    memory_circuit.cpp
  COMPILER_FLAGS
    -DCUDAQ_QEC_DISABLE_REALTIME_DECODING
)
