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

if(CUDAQ_REALTIME_INCLUDE_DIR
   AND TARGET cudaq-qec-realtime-decoding
   AND TARGET cudaq-qec-pymatching)
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
else()
  message(STATUS "Skipping test_pymatching_realtime "
                 "(requires realtime headers, "
                 "cudaq-qec-realtime-decoding, and cudaq-qec-pymatching).")
endif()

set(_pymatching_cudaq_realtime_prefixes "")
if(CUDAQ_REALTIME_ROOT)
  list(APPEND _pymatching_cudaq_realtime_prefixes "${CUDAQ_REALTIME_ROOT}")
  list(APPEND _pymatching_cudaq_realtime_prefixes "${CUDAQ_REALTIME_ROOT}/build")
endif()
if(CUDAQ_INSTALL_PREFIX)
  list(APPEND _pymatching_cudaq_realtime_prefixes "${CUDAQ_INSTALL_PREFIX}")
endif()
if(CUDAQ_INSTALL_DIR)
  list(APPEND _pymatching_cudaq_realtime_prefixes "${CUDAQ_INSTALL_DIR}")
endif()

if(NOT CUDAQ_DEVICE_CALL_INCLUDE_DIR)
  find_path(CUDAQ_DEVICE_CALL_INCLUDE_DIR
    NAMES cudaq_internal/device_call/DeviceCallService.h
    HINTS ${_pymatching_cudaq_realtime_prefixes}
    PATH_SUFFIXES include runtime/internal/device_call/include
  )
endif()

set(_pymatching_device_call_runtime_link "")
if(TARGET cudaq::cudaq-device-call-runtime)
  set(_pymatching_device_call_runtime_link cudaq::cudaq-device-call-runtime)
elseif(TARGET cudaq-device-call-runtime)
  set(_pymatching_device_call_runtime_link cudaq-device-call-runtime)
else()
  if(NOT CUDAQ_DEVICE_CALL_RUNTIME_LIBRARY)
    find_library(CUDAQ_DEVICE_CALL_RUNTIME_LIBRARY
      NAMES cudaq-device-call-runtime
      PATHS ${_pymatching_cudaq_realtime_prefixes}
      PATH_SUFFIXES lib
    )
  endif()
  if(CUDAQ_DEVICE_CALL_RUNTIME_LIBRARY)
    set(_pymatching_device_call_runtime_link ${CUDAQ_DEVICE_CALL_RUNTIME_LIBRARY})
  endif()
endif()

if(CUDAQ_REALTIME_INCLUDE_DIR
   AND _pymatching_device_call_runtime_link
   AND TARGET cudaq-qec-realtime-decoding-host-dispatch
   AND TARGET cudaq-qec-pymatching)
  add_executable(test_pymatching_qpu_device_call)
  set_target_properties(test_pymatching_qpu_device_call PROPERTIES
    LINKER_LANGUAGE CXX
  )

  cudaqx_add_device_code(test_pymatching_qpu_device_call
    SOURCES
      test_pymatching_qpu_device_call.cpp
    COMPILER_FLAGS
      --target qpp-cpu
      -frealtime-lowering
  )

  target_link_directories(test_pymatching_qpu_device_call
    PRIVATE ${CUDAQ_INSTALL_DIR}/lib)

  if(UNIX AND NOT APPLE)
    target_link_options(test_pymatching_qpu_device_call PRIVATE
      "LINKER:--no-as-needed")
  endif()

  target_link_libraries(test_pymatching_qpu_device_call PRIVATE
    GTest::gtest_main
    cudaq-qec
    cudaq-qec-pymatching
    cudaq-qec-realtime-decoding
    cudaq-qec-realtime-decoding-host-dispatch
    ${_pymatching_device_call_runtime_link}
    cudaq::cudaq cudaq::cudaq-common cudaq-mlir-runtime
    nvqir nvqir-qpp)

  set(_pymatching_qpu_device_call_rpath
    "${CMAKE_BINARY_DIR}/lib"
    "${CMAKE_BINARY_DIR}/lib/decoder-plugins"
  )
  if(CUDAQ_REALTIME_LIBRARY)
    get_filename_component(_pymatching_qpu_realtime_lib_dir
      "${CUDAQ_REALTIME_LIBRARY}" DIRECTORY)
    list(APPEND _pymatching_qpu_device_call_rpath
      "${_pymatching_qpu_realtime_lib_dir}")
  endif()
  if(CUDAQ_INSTALL_DIR)
    list(APPEND _pymatching_qpu_device_call_rpath
      "${CUDAQ_INSTALL_DIR}/lib")
  endif()

  set_target_properties(test_pymatching_qpu_device_call PROPERTIES
    BUILD_RPATH "${_pymatching_qpu_device_call_rpath}"
  )

  add_dependencies(test_pymatching_qpu_device_call
    cudaq-qec-pymatching
    cudaq-qec-realtime-decoding-host-dispatch)
  add_dependencies(CUDAQXQECUnitTests test_pymatching_qpu_device_call)
  gtest_discover_tests(test_pymatching_qpu_device_call)
else()
  message(STATUS "Skipping test_pymatching_qpu_device_call "
                 "(requires realtime headers, cudaq-device-call-runtime, "
                 "cudaq-qec-realtime-decoding-host-dispatch, and "
                 "cudaq-qec-pymatching).")
endif()
