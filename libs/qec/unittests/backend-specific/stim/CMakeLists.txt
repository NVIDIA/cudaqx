# ============================================================================ #
# Copyright (c) 2024 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.    #
# ============================================================================ #

# Compile test_qec_stim.cpp with nvq++ so that the inline __qpu__ kernels
# go through the MLIR pipeline, the same way memory-circuit kernels in
# libs/qec/lib/device are built.
add_executable(test_qec_stim)
set_target_properties(test_qec_stim PROPERTIES LINKER_LANGUAGE CXX)

cudaqx_add_device_code(test_qec_stim
  SOURCES
    test_qec_stim.cpp
)

target_link_directories(test_qec_stim
  PRIVATE ${CUDAQ_INSTALL_DIR}/lib)

target_link_libraries(test_qec_stim
  PRIVATE
    GTest::gtest_main
    cudaq-qec
    cudaq::cudaq cudaq::cudaq-common cudaq-mlir-runtime
    nvqir nvqir-stim)

add_dependencies(CUDAQXQECUnitTests test_qec_stim)
gtest_discover_tests(test_qec_stim)
