# ============================================================================ #
# Copyright (c) 2024 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

cudaqx_add_device_code(cudaq-qec
  SOURCES
    steane_device.cpp
    surface_code_device.cpp
    repetition_device.cpp
)

target_sources(cudaq-qec PRIVATE steane.cpp repetition.cpp surface_code.cpp)
