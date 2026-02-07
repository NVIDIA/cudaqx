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
# Find transitive Hololink libraries (required by gpu_roce_transceiver)
# --------------------------------------------------------------------------- #

find_library(HOLOLINK_COMMON_LIB
  NAMES hololink
  PATHS
    "${HOLOSCAN_SENSOR_BRIDGE_BUILD_DIR}"
    "${HOLOSCAN_SENSOR_BRIDGE_BUILD_DIR}/src/hololink/common"
    "${HOLOSCAN_SENSOR_BRIDGE_BUILD_DIR}/lib"
  NO_DEFAULT_PATH)

find_library(ROCE_RECEIVER_LIB
  NAMES roce_receiver
  PATHS
    "${HOLOSCAN_SENSOR_BRIDGE_BUILD_DIR}"
    "${HOLOSCAN_SENSOR_BRIDGE_BUILD_DIR}/src/hololink/operators/roce_receiver"
    "${HOLOSCAN_SENSOR_BRIDGE_BUILD_DIR}/lib"
  NO_DEFAULT_PATH)

find_library(BASE_RECEIVER_OP_LIB
  NAMES base_receiver_op
  PATHS
    "${HOLOSCAN_SENSOR_BRIDGE_BUILD_DIR}"
    "${HOLOSCAN_SENSOR_BRIDGE_BUILD_DIR}/src/hololink/operators"
    "${HOLOSCAN_SENSOR_BRIDGE_BUILD_DIR}/lib"
  NO_DEFAULT_PATH)

find_library(IBVERBS_LIB NAMES ibverbs)

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
# hololink_fpga_syndrome_playback
# =========================================================================== #
# FPGA stimulus tool: loads syndromes into FPGA BRAM and triggers playback.

add_executable(hololink_fpga_syndrome_playback
  hololink_fpga_syndrome_playback.cpp)

target_include_directories(hololink_fpga_syndrome_playback
  PRIVATE "${HOLOSCAN_SENSOR_BRIDGE_SOURCE_DIR}/src")

if (DEFINED CUDAQ_INCLUDE_DIR AND CUDAQ_INCLUDE_DIR)
  target_include_directories(hololink_fpga_syndrome_playback
    PRIVATE "${CUDAQ_INCLUDE_DIR}")
endif()

if (DEFINED CUDAQ_INSTALL_DIR AND CUDAQ_INSTALL_DIR)
  target_include_directories(hololink_fpga_syndrome_playback
    PRIVATE "${CUDAQ_INSTALL_DIR}/include")
endif()

if (CUDAQ_REALTIME_INCLUDE_DIR)
  target_include_directories(hololink_fpga_syndrome_playback
    PRIVATE "${CUDAQ_REALTIME_INCLUDE_DIR}")
endif()

target_include_directories(hololink_fpga_syndrome_playback
  PRIVATE "${CUDAToolkit_INCLUDE_DIRS}")

target_link_libraries(hololink_fpga_syndrome_playback
  PRIVATE CUDA::cudart)

target_link_libraries(hololink_fpga_syndrome_playback
  PRIVATE ${HOLOLINK_CORE_LIB} Threads::Threads)

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

  # fmt is needed transitively by hololink/core/logging_internal.hpp
  find_path(FMT_INCLUDE_DIR
    NAMES fmt/format.h
    PATHS /opt/nvidia/holoscan /usr/local/cudaq /usr /usr/local
    PATH_SUFFIXES include
    NO_DEFAULT_PATH
  )

  target_include_directories(hololink_wrapper_bridge
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      "${HOLOSCAN_SENSOR_BRIDGE_SOURCE_DIR}/src"
      ${DOCA_INCLUDE_DIR}
      ${CUDAToolkit_INCLUDE_DIRS}
      ${FMT_INCLUDE_DIR}
  )

  target_link_libraries(hololink_wrapper_bridge
    PRIVATE
      ${GPU_ROCE_TRANSCEIVER_LIB}
  )

  # Suppress deprecated declarations warning for DOCA APIs
  target_compile_options(hololink_wrapper_bridge PRIVATE -Wno-deprecated-declarations)

  # The bridge tool is a C++ executable -- all CUDA device code
  # lives in the cudaq-qec-realtime-cudevice library.
  # LINKER_LANGUAGE CUDA forces nvcc for the final link step so that
  # device symbols from the linked static libraries are resolved,
  # while the .cpp source itself is compiled by the host C++ compiler.
  add_executable(hololink_mock_decoder_bridge
    hololink_mock_decoder_bridge.cpp
  )

  set_target_properties(hololink_mock_decoder_bridge PROPERTIES
    LINKER_LANGUAGE CUDA
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
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

  # Link order matters: static archives first, then shared libs that resolve
  # their symbols.  gpu_roce_transceiver -> roce_receiver -> base_receiver_op
  # -> hololink_core -> hololink (common) form the Hololink dependency chain.
  target_link_libraries(hololink_mock_decoder_bridge
    PRIVATE
      cudaq-qec-realtime-cudevice
      ${CUDAQ_REALTIME_DISPATCH_LIBRARY}
      hololink_wrapper_bridge
      ${GPU_ROCE_TRANSCEIVER_LIB}
      ${ROCE_RECEIVER_LIB}
      ${BASE_RECEIVER_OP_LIB}
      ${HOLOLINK_CORE_LIB}
      ${HOLOLINK_COMMON_LIB}
      ${CUDAQ_REALTIME_LIBRARY}
      CUDA::cudart
      CUDA::cuda_driver
      ${DOCA_VERBS_LIB}
      ${DOCA_GPUNETIO_LIB}
      ${DOCA_COMMON_LIB}
      ${IBVERBS_LIB}
      Threads::Threads
      ${CMAKE_DL_LIBS}
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

# =========================================================================== #
# hololink_fpga_emulator
# =========================================================================== #
# Software FPGA emulator: replaces the physical FPGA for testing.
# Uses libibverbs for RDMA (no DOCA / GPU dependency).

if (IBVERBS_LIB)
  message(STATUS "Building hololink_fpga_emulator")

  add_executable(hololink_fpga_emulator
    hololink_fpga_emulator.cpp)

  target_link_libraries(hololink_fpga_emulator
    PRIVATE
      ${IBVERBS_LIB}
      Threads::Threads
  )
else()
  message(WARNING "libibverbs not found. hololink_fpga_emulator will not be built.")
endif()
