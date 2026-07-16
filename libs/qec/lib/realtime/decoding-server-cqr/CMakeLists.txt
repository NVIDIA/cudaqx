# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Handle both standalone (libs/qec) and full-repo builds.
if(CUDAQX_QEC_STANDALONE_BUILD)
  set(_core_include_dir "${CMAKE_SOURCE_DIR}/../core/include")
else()
  set(_core_include_dir "${CMAKE_SOURCE_DIR}/libs/core/include")
endif()

# ---------------------------------------------------------------------------
# Optional: GPU RoCE transport via Hololink Sensor Bridge + DOCA
#
# Library names and DOCA path mirror libs/qec/unittests/utils/CMakeLists.txt
# (the existing hololink_qldpc_graph_decoder_bridge target) so both targets
# find the same libraries from the same build tree.
# ---------------------------------------------------------------------------
set(CUDAQ_GPU_ROCE_AVAILABLE FALSE)

if(HOLOSCAN_SENSOR_BRIDGE_BUILD_DIR AND CUDAQ_REALTIME_INCLUDE_DIR)
  find_library(HOLOLINK_CORE_LIB
    NAMES hololink_core
    PATHS "${HOLOSCAN_SENSOR_BRIDGE_BUILD_DIR}"
          "${HOLOSCAN_SENSOR_BRIDGE_BUILD_DIR}/src/hololink/core"
          "${HOLOSCAN_SENSOR_BRIDGE_BUILD_DIR}/lib"
    NO_DEFAULT_PATH)

  find_library(GPU_ROCE_TRANSCEIVER_LIB
    NAMES gpu_roce_transceiver
    PATHS "${HOLOSCAN_SENSOR_BRIDGE_BUILD_DIR}"
          "${HOLOSCAN_SENSOR_BRIDGE_BUILD_DIR}/src/hololink/operators/gpu_roce_transceiver"
          "${HOLOSCAN_SENSOR_BRIDGE_BUILD_DIR}/lib"
    NO_DEFAULT_PATH)

  find_library(HOLOLINK_COMMON_LIB
    NAMES hololink
    PATHS "${HOLOSCAN_SENSOR_BRIDGE_BUILD_DIR}"
          "${HOLOSCAN_SENSOR_BRIDGE_BUILD_DIR}/src/hololink/common"
          "${HOLOSCAN_SENSOR_BRIDGE_BUILD_DIR}/lib"
    NO_DEFAULT_PATH)

  find_library(BASE_RECEIVER_OP_LIB
    NAMES base_receiver_op
    PATHS "${HOLOSCAN_SENSOR_BRIDGE_BUILD_DIR}"
          "${HOLOSCAN_SENSOR_BRIDGE_BUILD_DIR}/src/hololink/operators"
          "${HOLOSCAN_SENSOR_BRIDGE_BUILD_DIR}/lib"
    NO_DEFAULT_PATH)

  find_library(IBVERBS_LIB NAMES ibverbs)

  # DOCA at the standard Mellanox install path.
  # Arch-specific lib dir mirrors libs/qec/unittests/utils/CMakeLists.txt.
  set(_doca_root "/opt/mellanox/doca")
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64)|(AMD64|amd64)")
    set(_doca_lib_dir "${_doca_root}/lib/x86_64-linux-gnu")
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "(aarch64)|(arm64)")
    set(_doca_lib_dir "${_doca_root}/lib/aarch64-linux-gnu")
  else()
    set(_doca_lib_dir "${_doca_root}/lib")
  endif()
  find_library(DOCA_VERBS_LIB   doca_verbs   PATHS "${_doca_lib_dir}" NO_DEFAULT_PATH)
  find_library(DOCA_GPUNETIO_LIB doca_gpunetio PATHS "${_doca_lib_dir}" NO_DEFAULT_PATH)
  find_library(DOCA_COMMON_LIB  doca_common  PATHS "${_doca_lib_dir}" NO_DEFAULT_PATH)

  # hololink_wrapper.h lives in CUDAQ_REALTIME_INCLUDE_DIR.
  # Derive the lib search dir from the already-found CUDAQ_REALTIME_LIBRARY
  # (set by the parent CMakeLists or the CQR plugin detection above).
  if(CUDAQ_REALTIME_LIBRARY)
    get_filename_component(_cudaq_rt_lib_dir "${CUDAQ_REALTIME_LIBRARY}" DIRECTORY)
  else()
    get_filename_component(_cudaq_rt_lib_dir "${CUDAQ_REALTIME_INCLUDE_DIR}/../lib" ABSOLUTE)
  endif()
  find_library(CUDAQ_REALTIME_BRIDGE_HOLOLINK_LIBRARY
    NAMES cudaq-realtime-bridge-hololink
    HINTS "${_cudaq_rt_lib_dir}"
    NO_DEFAULT_PATH)

  # cudaq-realtime-dispatch: declares cudaq_create_dispatch_graph_regular,
  # cudaq_launch_dispatch_graph, cudaq_destroy_dispatch_graph, cudaq_function_entry_t.
  find_library(CUDAQ_REALTIME_DISPATCH_LIBRARY
    NAMES cudaq-realtime-dispatch
    HINTS "${_cudaq_rt_lib_dir}"
    NO_DEFAULT_PATH)

  # cudaq-realtime-host-dispatch: required transitive dep of the bridge path.
  find_library(CUDAQ_REALTIME_HOST_DISPATCH_LIBRARY
    NAMES cudaq-realtime-host-dispatch
    HINTS "${_cudaq_rt_lib_dir}"
    NO_DEFAULT_PATH)

  if(HOLOLINK_CORE_LIB AND GPU_ROCE_TRANSCEIVER_LIB AND HOLOLINK_COMMON_LIB
      AND BASE_RECEIVER_OP_LIB AND DOCA_VERBS_LIB AND DOCA_GPUNETIO_LIB
      AND DOCA_COMMON_LIB AND CUDAQ_REALTIME_BRIDGE_HOLOLINK_LIBRARY
      AND CUDAQ_REALTIME_DISPATCH_LIBRARY AND CUDAQ_REALTIME_HOST_DISPATCH_LIBRARY)
    set(CUDAQ_GPU_ROCE_AVAILABLE TRUE CACHE INTERNAL
      "GPU RoCE transport compiled into cudaq-qec-decoding-server")
    message(STATUS
      "cudaq-qec-decoding-server: GPU RoCE ENABLED (core=${HOLOLINK_CORE_LIB})")
  else()
    message(STATUS
      "cudaq-qec-decoding-server: GPU RoCE DISABLED "
      "(missing one or more of: hololink_core gpu_roce_transceiver hololink "
      "base_receiver_op ibverbs doca_* cudaq-realtime-bridge-hololink "
      "cudaq-realtime-dispatch cudaq-realtime-host-dispatch)")
  endif()
else()
  message(STATUS
    "cudaq-qec-decoding-server: GPU RoCE DISABLED "
    "(HOLOSCAN_SENSOR_BRIDGE_BUILD_DIR or CUDAQ_REALTIME_INCLUDE_DIR not set)")
endif()

message(STATUS
  "cudaq-qec-decoding-server: CPU RoCE DISABLED (skeleton only; ibverbs implementation pending)")

# ---------------------------------------------------------------------------
# Core decoding server (always built): transport abstraction, session registry,
# per-decoder worker threads, RPC dispatcher.  No dependency on CUDAQ_REALTIME.
# ---------------------------------------------------------------------------
add_library(cudaq-qec-decoding-server STATIC
  RoundAccumulator.cpp
  DecodingSession.cpp
  SessionRegistry.cpp
  RpcDispatcher.cpp
  DecodingServer.cpp
  CpuRoceTransceiver.cpp
  GpuRoceTransceiver.cpp
)

target_compile_features(cudaq-qec-decoding-server PUBLIC cxx_std_20)

target_include_directories(cudaq-qec-decoding-server
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CUDAQX_QEC_INCLUDE_DIR}>
    $<BUILD_INTERFACE:${_core_include_dir}>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(cudaq-qec-decoding-server
  PUBLIC
    # The decode-server core only needs the decoder interface + logger (in
    # cudaq-qec-decoders) and the realtime decoding config -- NOT the heavy
    # cudaq-qec (which pulls in libcustabilizer etc.). Keeping cudaq-qec here
    # violates the realtime-server dependency-closure contract.
    cudaq-qec-decoders
    cudaq-qec-realtime-decoding
  PRIVATE
    # DecodingSession worker threads pin themselves to their decoder's
    # cuda_device_id (cudaSetDevice).
    CUDA::cudart
)

if(CUDAQ_GPU_ROCE_AVAILABLE)
  target_compile_definitions(cudaq-qec-decoding-server PRIVATE CUDAQ_GPU_ROCE_AVAILABLE)
  target_include_directories(cudaq-qec-decoding-server PRIVATE
    "${CUDAQ_REALTIME_INCLUDE_DIR}"
    "${_doca_root}/include"
  )
  target_link_libraries(cudaq-qec-decoding-server PRIVATE
    "${CUDAQ_REALTIME_BRIDGE_HOLOLINK_LIBRARY}"
    "${GPU_ROCE_TRANSCEIVER_LIB}"
    "${BASE_RECEIVER_OP_LIB}"
    "${HOLOLINK_CORE_LIB}"
    "${HOLOLINK_COMMON_LIB}"
    "${DOCA_VERBS_LIB}"
    "${DOCA_GPUNETIO_LIB}"
    "${DOCA_COMMON_LIB}"
    $<$<BOOL:${IBVERBS_LIB}>:${IBVERBS_LIB}>
    CUDA::cudart
  )
endif()

set_target_properties(cudaq-qec-decoding-server PROPERTIES
  ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
  POSITION_INDEPENDENT_CODE ON
)

install(TARGETS cudaq-qec-decoding-server
  COMPONENT qec-lib
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

# ---------------------------------------------------------------------------
# CQR host-dispatch plugin (only when CUDAQ device_call headers are available):
# CqrTransceiver bridges DeviceCallService handler callbacks to ITransceiver,
# then hands off to DecodingServer for session management and decoding.
# ---------------------------------------------------------------------------
if(CUDAQ_QEC_DEVICE_CALL_SERVICE_AVAILABLE)
  add_library(cudaq-qec-realtime-decoding-server-cqr SHARED
    decoding_server_cqr.cpp
  )

  target_compile_options(cudaq-qec-realtime-decoding-server-cqr
    PRIVATE -fvisibility=hidden -fvisibility-inlines-hidden)

  target_include_directories(cudaq-qec-realtime-decoding-server-cqr
    PUBLIC
      $<BUILD_INTERFACE:${CUDAQX_QEC_INCLUDE_DIR}>
      $<INSTALL_INTERFACE:${CUDAQ_INCLUDE_DIR}>
      $<INSTALL_INTERFACE:include>
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CUDAQ_REALTIME_INCLUDE_DIR}
      ${CUDAQ_DEVICE_CALL_INCLUDE_DIR}
      ${CUDAToolkit_INCLUDE_DIRS}
  )

  target_link_options(cudaq-qec-realtime-decoding-server-cqr PRIVATE
    $<$<CXX_COMPILER_ID:GNU>:-Wl,--exclude-libs,libcudaq-qec-decoding-server.a>
  )

  target_link_libraries(cudaq-qec-realtime-decoding-server-cqr
    PUBLIC
      cudaq-qec-realtime-decoding
    PRIVATE
      cudaq-qec-decoding-server
      cudaq::cudaq-common
      ${CUDAQ_DEVICE_CALL_RUNTIME_LIBRARY}
  )

  set(_server_cqr_build_rpath "${CMAKE_BINARY_DIR}/lib")
  set(_server_cqr_install_rpath
    "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}")
  if(_CUDAQ_REALTIME_LIB_DIR)
    list(APPEND _server_cqr_build_rpath "${_CUDAQ_REALTIME_LIB_DIR}")
    list(APPEND _server_cqr_install_rpath "${_CUDAQ_REALTIME_LIB_DIR}")
  endif()
  if(_CUDAQ_DEVICE_CALL_RUNTIME_LIB_DIR)
    list(APPEND _server_cqr_build_rpath "${_CUDAQ_DEVICE_CALL_RUNTIME_LIB_DIR}")
    list(APPEND _server_cqr_install_rpath
      "${_CUDAQ_DEVICE_CALL_RUNTIME_LIB_DIR}")
  endif()

  set_target_properties(cudaq-qec-realtime-decoding-server-cqr PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    BUILD_RPATH "${_server_cqr_build_rpath}"
    INSTALL_RPATH "${_server_cqr_install_rpath}"
  )

  install(TARGETS cudaq-qec-realtime-decoding-server-cqr
    COMPONENT qec-lib
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  )
endif()
