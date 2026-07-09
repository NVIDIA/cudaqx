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
# Core decoder server (always built): transport abstraction, session registry,
# per-decoder worker threads, RPC dispatcher.  No dependency on CUDAQ_REALTIME.
# ---------------------------------------------------------------------------
add_library(cudaq-qec-decoder-server STATIC
  RoundAccumulator.cpp
  DecoderSession.cpp
  SessionRegistry.cpp
  RpcDispatcher.cpp
  DecoderServer.cpp
)

target_compile_features(cudaq-qec-decoder-server PUBLIC cxx_std_20)

target_include_directories(cudaq-qec-decoder-server
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CUDAQX_QEC_INCLUDE_DIR}>
    $<BUILD_INTERFACE:${_core_include_dir}>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(cudaq-qec-decoder-server
  PUBLIC
    cudaq-qec-decoders
    cudaq-qec-realtime-decoding
)

set_target_properties(cudaq-qec-decoder-server PROPERTIES
  ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
  POSITION_INDEPENDENT_CODE ON
)

install(TARGETS cudaq-qec-decoder-server
  COMPONENT qec-lib
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

# ---------------------------------------------------------------------------
# CQR host-dispatch plugin (only when CUDAQ device_call headers are available):
# CqrTransceiver bridges DeviceCallService handler callbacks to ITransceiver,
# then hands off to DecoderServer for session management and decoding.
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
    $<$<CXX_COMPILER_ID:GNU>:-Wl,--exclude-libs,libcudaq-qec-decoder-server.a>
  )

  target_link_libraries(cudaq-qec-realtime-decoding-server-cqr
    PUBLIC
      cudaq-qec-realtime-decoding
    PRIVATE
      cudaq-qec-decoder-server
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
