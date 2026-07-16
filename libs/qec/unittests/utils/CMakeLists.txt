# ============================================================================ #
# Copyright (c) 2025-2026 NVIDIA Corporation & Affiliates.                    #
# All rights reserved.                                                        #
#                                                                             #
# This source code and the accompanying materials are made available under    #
# the terms of the Apache License 2.0 which accompanies this distribution.    #
# ============================================================================ #

# Hololink bridge and playback tools for QLDPC graph-launched decoder
# ==============================================================================
# Gated by CUDAQX_QEC_ENABLE_HOLOLINK_TOOLS.  Requires:
#   - holoscan-sensor-bridge build (hololink_core for playback)
#   - cuda-quantum/realtime build (libcudaq-realtime*, hololink bridge libs)
#   - DOCA 3.3+ (for gpu_roce_transceiver)
# These are NOT CI tests -- they need FPGA hardware or an FPGA emulator.

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
enable_language(CUDA)

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
    "hololink_qldpc_graph_decoder_bridge will not be built.")
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
  NAMES cudaq/realtime/daemon/dispatcher/cudaq_realtime.h
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

find_library(CUDAQ_REALTIME_HOST_DISPATCH_LIBRARY
  NAMES cudaq-realtime-host-dispatch
  PATHS ${_cudaq_realtime_prefixes}
  PATH_SUFFIXES lib
)

find_library(CUDAQ_REALTIME_BRIDGE_HOLOLINK_LIBRARY
  NAMES cudaq-realtime-bridge-hololink
  PATHS ${_cudaq_realtime_prefixes}
  PATH_SUFFIXES lib
)

# =========================================================================== #
# hololink_fpga_syndrome_playback
# =========================================================================== #

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
  PRIVATE
    ${HOLOLINK_CORE_LIB}
    CUDA::cudart
    CUDA::cuda_driver
    Threads::Threads)

# =========================================================================== #
# hololink_qldpc_graph_decoder_bridge
# =========================================================================== #
# Bridge tool: Hololink GPU-RoCE Transceiver <-> HOST_LOOP graph dispatch
# (CPU-launched CUDA graph with Relay BP decoder)

if (GPU_ROCE_TRANSCEIVER_LIB AND CUDAQ_REALTIME_INCLUDE_DIR AND
    CUDAQ_REALTIME_LIBRARY AND CUDAQ_REALTIME_HOST_DISPATCH_LIBRARY AND
    CUDAQ_REALTIME_BRIDGE_HOLOLINK_LIBRARY AND
    DOCA_INCLUDE_DIR AND DOCA_VERBS_LIB AND DOCA_COMMON_LIB AND
    DOCA_GPUNETIO_LIB)

  message(STATUS "Building hololink_qldpc_graph_decoder_bridge")
  message(STATUS "  GPU RoCE Transceiver: ${GPU_ROCE_TRANSCEIVER_LIB}")
  message(STATUS "  cuda-quantum realtime: ${CUDAQ_REALTIME_LIBRARY}")
  message(STATUS "  cuda-quantum host dispatch: ${CUDAQ_REALTIME_HOST_DISPATCH_LIBRARY}")
  message(STATUS "  cuda-quantum bridge hololink: ${CUDAQ_REALTIME_BRIDGE_HOLOLINK_LIBRARY}")

  # cudaq-qec-realtime-decoding transitively pulls in cudaq-qec, which
  # references CUDA-Q quantum runtime symbols.
  find_library(_CUDAQ_LIBRARY NAMES cudaq
    PATHS ${CUDAQ_INSTALL_PREFIX} $ENV{CUDAQ_INSTALL_PREFIX}
          ${CUDAQ_DIR}/.. $ENV{CUDAQ_DIR}/..
    PATH_SUFFIXES lib)
  find_library(_NVQIR_LIBRARY NAMES nvqir
    PATHS ${CUDAQ_INSTALL_PREFIX} $ENV{CUDAQ_INSTALL_PREFIX}
          ${CUDAQ_DIR}/.. $ENV{CUDAQ_DIR}/..
    PATH_SUFFIXES lib)

  add_executable(hololink_qldpc_graph_decoder_bridge
    hololink_qldpc_graph_decoder_bridge.cpp
    cuda_link_stub.cu
  )

  set_target_properties(hololink_qldpc_graph_decoder_bridge PROPERTIES
    LINKER_LANGUAGE CUDA
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
  )

  target_include_directories(hololink_qldpc_graph_decoder_bridge
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CUDAToolkit_INCLUDE_DIRS}
      ${CMAKE_CURRENT_SOURCE_DIR}/../../include
      ${CMAKE_SOURCE_DIR}/libs/core/include
      ${CUDAQ_REALTIME_INCLUDE_DIR}
  )

  target_link_libraries(hololink_qldpc_graph_decoder_bridge
    PRIVATE
      ${CUDAQ_REALTIME_DISPATCH_LIBRARY}
      ${CUDAQ_REALTIME_HOST_DISPATCH_LIBRARY}
      ${CUDAQ_REALTIME_BRIDGE_HOLOLINK_LIBRARY}
      ${GPU_ROCE_TRANSCEIVER_LIB}
      ${BASE_RECEIVER_OP_LIB}
      ${HOLOLINK_CORE_LIB}
      ${HOLOLINK_COMMON_LIB}
      ${CUDAQ_REALTIME_LIBRARY}
      cudaq-qec-realtime-decoding
      $<$<BOOL:${_CUDAQ_LIBRARY}>:${_CUDAQ_LIBRARY}>
      $<$<BOOL:${_NVQIR_LIBRARY}>:${_NVQIR_LIBRARY}>
      CUDA::cudart
      CUDA::cuda_driver
      ${DOCA_VERBS_LIB}
      ${DOCA_GPUNETIO_LIB}
      ${DOCA_COMMON_LIB}
      ${IBVERBS_LIB}
      Threads::Threads
      ${CMAKE_DL_LIBS}
  )

  target_link_options(hololink_qldpc_graph_decoder_bridge PRIVATE
    "LINKER:--allow-shlib-undefined"
  )

  if (holoscan_FOUND)
    target_link_libraries(hololink_qldpc_graph_decoder_bridge PRIVATE holoscan::core)
  endif()

  get_filename_component(CUDAQ_REALTIME_LIB_DIR "${CUDAQ_REALTIME_LIBRARY}" DIRECTORY)
  set_target_properties(hololink_qldpc_graph_decoder_bridge PROPERTIES
    BUILD_RPATH "${CUDAQ_REALTIME_LIB_DIR};${DOCA_LIB_DIR};${CMAKE_BINARY_DIR}/lib/decoder-plugins"
    INSTALL_RPATH "${CUDAQ_REALTIME_LIB_DIR};${DOCA_LIB_DIR};${CMAKE_BINARY_DIR}/lib/decoder-plugins"
  )

else()
  if (NOT GPU_ROCE_TRANSCEIVER_LIB)
    message(WARNING "gpu_roce_transceiver library not found. "
                    "hololink_qldpc_graph_decoder_bridge will not be built.")
  endif()
  if (NOT CUDAQ_REALTIME_LIBRARY OR NOT CUDAQ_REALTIME_HOST_DISPATCH_LIBRARY)
    message(WARNING "cuda-quantum realtime libraries not found. "
                    "Set CUDAQ_REALTIME_ROOT to enable hololink_qldpc_graph_decoder_bridge.")
  endif()
  if (NOT CUDAQ_REALTIME_BRIDGE_HOLOLINK_LIBRARY)
    message(WARNING "cuda-quantum bridge hololink library not found. "
                    "Set CUDAQ_REALTIME_ROOT to enable hololink_qldpc_graph_decoder_bridge.")
  endif()
  if (NOT DOCA_INCLUDE_DIR OR NOT DOCA_VERBS_LIB)
    message(WARNING "DOCA libraries not found. "
                    "hololink_qldpc_graph_decoder_bridge requires DOCA.")
  endif()
endif()

# hololink_predecoder_bridge is now in unittests/realtime/
