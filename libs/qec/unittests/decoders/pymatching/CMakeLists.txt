# ============================================================================ #
# Copyright (c) 2025 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# External Dependencies 
# ==============================================================================

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# Bug in GCC 12 leads to spurious warnings (-Wrestrict)
# https://gcc.gnu.org/bugzilla/show_bug.cgi?id=105329
if (CMAKE_COMPILER_IS_GNUCXX
  AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 12.0.0
  AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 13.0.0)
  target_compile_options(gtest PUBLIC
    $<$<COMPILE_LANGUAGE:CXX>:--param=evrp-mode=legacy>)
endif()
include(GoogleTest)

# ==============================================================================
add_compile_options(-Wno-attributes) 

add_executable(test_pymatching test_pymatching.cpp)
target_link_libraries(test_pymatching PRIVATE GTest::gtest_main cudaq-qec cudaq::cudaq)
add_dependencies(CUDAQXQECUnitTests test_pymatching)
gtest_discover_tests(test_pymatching)

if(CUDAQ_REALTIME_ROOT AND CUDAQ_REALTIME_INCLUDE_DIR)
  add_executable(test_pymatching_realtime test_pymatching_realtime.cpp)
  target_include_directories(test_pymatching_realtime PRIVATE
    ${CUDAQ_REALTIME_INCLUDE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
    ${CUDAQX_QEC_SOURCE_DIR}/lib/realtime
  )
  target_compile_definitions(test_pymatching_realtime PRIVATE
    CUDAQ_REALTIME_ROOT)
  target_link_libraries(test_pymatching_realtime PRIVATE
    GTest::gtest_main
    cudaq-qec
    cudaq-qec-realtime-decoding
    cudaq::cudaq
  )
  set_target_properties(test_pymatching_realtime PROPERTIES
    BUILD_RPATH "${CMAKE_BINARY_DIR}/lib;${CMAKE_BINARY_DIR}/lib/decoder-plugins"
  )
  add_dependencies(test_pymatching_realtime cudaq-qec-pymatching)
  add_dependencies(CUDAQXQECUnitTests test_pymatching_realtime)
  gtest_discover_tests(test_pymatching_realtime)
endif()
