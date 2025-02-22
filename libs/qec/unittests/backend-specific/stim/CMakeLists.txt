# ============================================================================ #
# Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

cudaqx_set_target(stim) 

add_executable(test_qec_stim test_qec_stim.cpp)
target_link_libraries(test_qec_stim
  PRIVATE
    GTest::gtest_main
    cudaq::cudaq_stim
    cudaq-qec
)
add_dependencies(CUDAQXQECUnitTests test_qec_stim)
gtest_discover_tests(test_qec_stim)
