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
  target_compile_options(gtest PRIVATE
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

if(CUDAQ_REALTIME_INCLUDE_DIR AND TARGET cudaq-qec-realtime-device-call)
  add_executable(test_pymatching_device_call_realtime)
  set_target_properties(test_pymatching_device_call_realtime PROPERTIES
    LINKER_LANGUAGE CXX)

  target_include_directories(test_pymatching_device_call_realtime PRIVATE
    ${CUDAQ_REALTIME_INCLUDE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
  )

  cudaqx_add_device_code(test_pymatching_device_call_realtime
    SOURCES
      test_pymatching_device_call_realtime.cpp
    COMPILER_FLAGS
      --target qpp-cpu
      -frealtime-lowering
  )

  target_link_directories(test_pymatching_device_call_realtime
    PRIVATE ${CUDAQ_INSTALL_DIR}/lib)

  target_link_libraries(test_pymatching_device_call_realtime PRIVATE
    cudaq-qec
    cudaq-qec-realtime-decoding
    cudaq-qec-realtime-decoding-simulation
    cudaq-qec-realtime-device-call
    cudaq::cudaq cudaq::cudaq-common cudaq-mlir-runtime
    nvqir nvqir-qpp
  )

  set_target_properties(test_pymatching_device_call_realtime PROPERTIES
    BUILD_RPATH
      "${CMAKE_BINARY_DIR}/lib;${CMAKE_BINARY_DIR}/lib/decoder-plugins;${CUDAQ_INSTALL_DIR}/lib"
  )

  add_dependencies(test_pymatching_device_call_realtime cudaq-qec-pymatching)
  add_dependencies(CUDAQXQECUnitTests test_pymatching_device_call_realtime)
  add_test(
    NAME decoders.pymatching.device_call_host_dispatch
    COMMAND test_pymatching_device_call_realtime
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )
  set_tests_properties(decoders.pymatching.device_call_host_dispatch
    PROPERTIES ENVIRONMENT "CUDAQ_DEVICE_CALL_CHANNEL=host_dispatch")
endif()
