# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         #
# All rights reserved.                                                        #
#                                                                             #
# This source code and the accompanying materials are made available under    #
# the terms of the Apache License 2.0 which accompanies this distribution.   #
# ============================================================================ #

# Hololink bridge and playback tools
# ==============================================================================
# These targets are gated by CUDAQX_QEC_ENABLE_HOLOLINK_TOOLS and require an
# external holoscan-sensor-bridge build with DOCA support.
# They are NOT CI tests -- they need FPGA hardware or an FPGA emulator.

if (NOT HOLOSCAN_SENSOR_BRIDGE_SOURCE_DIR)
  message(FATAL_ERROR
    "HOLOSCAN_SENSOR_BRIDGE_SOURCE_DIR must be set when building hololink tools.")
endif()
if (NOT HOLOSCAN_SENSOR_BRIDGE_BUILD_DIR)
  message(FATAL_ERROR
    "HOLOSCAN_SENSOR_BRIDGE_BUILD_DIR must be set when building hololink tools.")
endif()

find_package(Threads REQUIRED)
find_package(CUDAToolkit REQUIRED)

# --------------------------------------------------------------------------- #
# Find Hololink core library
# --------------------------------------------------------------------------- #

find_library(HOLOLINK_CORE_LIB
  NAMES hololink_core
  PATHS
    "${HOLOSCAN_SENSOR_BRIDGE_BUILD_DIR}"
    "${HOLOSCAN_SENSOR_BRIDGE_BUILD_DIR}/src/hololink/core"
    "${HOLOSCAN_SENSOR_BRIDGE_BUILD_DIR}/lib"
  NO_DEFAULT_PATH)

if (NOT HOLOLINK_CORE_LIB)
  message(FATAL_ERROR
    "Could not find hololink_core library under ${HOLOSCAN_SENSOR_BRIDGE_BUILD_DIR}.")
endif()

# --------------------------------------------------------------------------- #
# Find GPU RoCE Transceiver library (static, from holoscan-sensor-bridge)
# --------------------------------------------------------------------------- #

find_library(GPU_ROCE_TRANSCEIVER_LIB
  NAMES gpu_roce_transceiver
  PATHS
    "${HOLOSCAN_SENSOR_BRIDGE_BUILD_DIR}"
    "${HOLOSCAN_SENSOR_BRIDGE_BUILD_DIR}/src/hololink/operators/gpu_roce_transceiver"
    "${HOLOSCAN_SENSOR_BRIDGE_BUILD_DIR}/lib"
  NO_DEFAULT_PATH)

if (NOT GPU_ROCE_TRANSCEIVER_LIB)
  message(WARNING
    "Could not find gpu_roce_transceiver library. "
    "hololink_mock_decoder_bridge will not be built.")
endif()

# --------------------------------------------------------------------------- #
# Find DOCA libraries (required by gpu_roce_transceiver)
# --------------------------------------------------------------------------- #

set(DOCA_PATH "/opt/mellanox/doca")

# Determine DOCA library directory based on architecture
if (CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64)|(AMD64|amd64)")
  set(DOCA_LIB_DIR "${DOCA_PATH}/lib/x86_64-linux-gnu")
elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "(aarch64)|(arm64)")
  set(DOCA_LIB_DIR "${DOCA_PATH}/lib/aarch64-linux-gnu")
else()
  set(DOCA_LIB_DIR "${DOCA_PATH}/lib")
endif()

find_path(DOCA_INCLUDE_DIR doca_verbs.h
  PATHS ${DOCA_PATH}/include
  NO_DEFAULT_PATH)

find_library(DOCA_VERBS_LIB doca_verbs
  PATHS ${DOCA_LIB_DIR}
  NO_DEFAULT_PATH)

find_library(DOCA_GPUNETIO_LIB doca_gpunetio
  PATHS ${DOCA_LIB_DIR}
  NO_DEFAULT_PATH)

find_library(DOCA_COMMON_LIB doca_common
  PATHS ${DOCA_LIB_DIR}
  NO_DEFAULT_PATH)

# --------------------------------------------------------------------------- #
# Find Holoscan (required by gpu_roce_transceiver -> holoscan::core)
# --------------------------------------------------------------------------- #

find_package(holoscan QUIET)

# --------------------------------------------------------------------------- #
# Find cuda-quantum realtime libraries
# --------------------------------------------------------------------------- #

set(_cudaq_realtime_prefixes "")
if(CUDAQ_REALTIME_ROOT)
  list(APPEND _cudaq_realtime_prefixes "${CUDAQ_REALTIME_ROOT}")
  list(APPEND _cudaq_realtime_prefixes "${CUDAQ_REALTIME_ROOT}/build")
endif()
if(CUDAQ_INSTALL_PREFIX)
  list(APPEND _cudaq_realtime_prefixes "${CUDAQ_INSTALL_PREFIX}")
endif()

find_path(CUDAQ_REALTIME_INCLUDE_DIR
  NAMES cudaq/nvqlink/daemon/dispatcher/cudaq_realtime.h
  PATHS ${_cudaq_realtime_prefixes}
  PATH_SUFFIXES include ../include
)

find_library(CUDAQ_REALTIME_LIBRARY
  NAMES cudaq-realtime
  PATHS ${_cudaq_realtime_prefixes}
  PATH_SUFFIXES lib
)

find_library(CUDAQ_REALTIME_DISPATCH_LIBRARY
  NAMES cudaq-realtime-dispatch
  PATHS ${_cudaq_realtime_prefixes}
  PATH_SUFFIXES lib
)

# =========================================================================== #
# hololink_mock_decoder_bridge
# =========================================================================== #
# Bridge tool: Hololink GPU-RoCE Transceiver <-> cudaq dispatch kernel (mock decoder)

if (GPU_ROCE_TRANSCEIVER_LIB AND CUDAQ_REALTIME_INCLUDE_DIR AND
    CUDAQ_REALTIME_LIBRARY AND CUDAQ_REALTIME_DISPATCH_LIBRARY AND
    DOCA_INCLUDE_DIR AND DOCA_VERBS_LIB AND DOCA_COMMON_LIB AND
    DOCA_GPUNETIO_LIB)

  message(STATUS "Building hololink_mock_decoder_bridge")
  message(STATUS "  GPU RoCE Transceiver: ${GPU_ROCE_TRANSCEIVER_LIB}")
  message(STATUS "  cuda-quantum realtime: ${CUDAQ_REALTIME_LIBRARY}")
  message(STATUS "  cuda-quantum dispatch: ${CUDAQ_REALTIME_DISPATCH_LIBRARY}")

  # The hololink_wrapper.cpp is compiled by the C++ compiler (not nvcc)
  # to isolate Hololink/Holoscan fmt dependencies from CUDA code.
  add_library(hololink_wrapper_bridge STATIC
    hololink_wrapper.cpp
  )

  target_include_directories(hololink_wrapper_bridge
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      "${HOLOSCAN_SENSOR_BRIDGE_SOURCE_DIR}/src"
      ${DOCA_INCLUDE_DIR}
  )

  target_link_libraries(hololink_wrapper_bridge
    PRIVATE
      ${GPU_ROCE_TRANSCEIVER_LIB}
  )

  # Suppress deprecated declarations warning for DOCA APIs
  target_compile_options(hololink_wrapper_bridge PRIVATE -Wno-deprecated-declarations)

  # The bridge tool itself is a CUDA executable
  add_executable(hololink_mock_decoder_bridge
    hololink_mock_decoder_bridge.cu
  )

  set_target_properties(hololink_mock_decoder_bridge PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_STANDARD 17
  )

  target_include_directories(hololink_mock_decoder_bridge
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CUDAToolkit_INCLUDE_DIRS}
      ${CMAKE_CURRENT_SOURCE_DIR}/../../include
      ${CMAKE_SOURCE_DIR}/libs/core/include
      ${CUDAQ_REALTIME_INCLUDE_DIR}
  )

  target_compile_definitions(hololink_mock_decoder_bridge PRIVATE
    BRIDGE_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../decoders/realtime/data"
  )

  target_link_libraries(hololink_mock_decoder_bridge
    PRIVATE
      CUDA::cudart
      Threads::Threads
      cudaq-qec-realtime-cudevice
      ${CUDAQ_REALTIME_LIBRARY}
      ${CUDAQ_REALTIME_DISPATCH_LIBRARY}
      hololink_wrapper_bridge
      ${GPU_ROCE_TRANSCEIVER_LIB}
      ${DOCA_VERBS_LIB}
      ${DOCA_GPUNETIO_LIB}
      ${DOCA_COMMON_LIB}
  )

  if (holoscan_FOUND)
    target_link_libraries(hololink_mock_decoder_bridge PRIVATE holoscan::core)
    target_link_libraries(hololink_wrapper_bridge PRIVATE holoscan::core)
  endif()

  # Ensure runtime can locate shared libraries
  get_filename_component(CUDAQ_REALTIME_LIB_DIR "${CUDAQ_REALTIME_LIBRARY}" DIRECTORY)
  set_target_properties(hololink_mock_decoder_bridge PROPERTIES
    BUILD_RPATH "${CUDAQ_REALTIME_LIB_DIR};${DOCA_LIB_DIR}"
    INSTALL_RPATH "${CUDAQ_REALTIME_LIB_DIR};${DOCA_LIB_DIR}"
  )

else()
  if (NOT GPU_ROCE_TRANSCEIVER_LIB)
    message(WARNING "gpu_roce_transceiver library not found. "
                    "hololink_mock_decoder_bridge will not be built.")
  endif()
  if (NOT CUDAQ_REALTIME_LIBRARY OR NOT CUDAQ_REALTIME_DISPATCH_LIBRARY)
    message(WARNING "cuda-quantum realtime libraries not found. "
                    "Set CUDAQ_REALTIME_ROOT to enable hololink_mock_decoder_bridge.")
  endif()
  if (NOT DOCA_INCLUDE_DIR OR NOT DOCA_VERBS_LIB)
    message(WARNING "DOCA libraries not found. "
                    "hololink_mock_decoder_bridge requires DOCA.")
  endif()
endif()
