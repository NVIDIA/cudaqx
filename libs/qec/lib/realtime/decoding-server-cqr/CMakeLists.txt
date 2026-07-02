# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Decoding server (skeleton) for the cudaq-realtime (cqr) host-dispatch path.
#
# This is the server end of the wire: a cudaq_internal::device_call
# DeviceCallService that registers CUDAQ_DISPATCH_HOST_CALL handlers for the
# default-route RPCs (enqueue_syndromes / get_corrections / reset_decoder, per
# decoder_server_runtime.md) and drives the configured decoders via the
# cudaq-qec-realtime-decoding host API. A simulated QPU (../simulation-cqr) or,
# eventually, real hardware issues those RPCs over cudaq-realtime.
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
    ${CUDAQ_REALTIME_INCLUDE_DIR}
    ${CUDAQ_DEVICE_CALL_INCLUDE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_options(cudaq-qec-realtime-decoding-server-cqr PRIVATE
  $<$<CXX_COMPILER_ID:GNU>:-Wl,--exclude-libs,ALL>
)

target_link_libraries(cudaq-qec-realtime-decoding-server-cqr
  PUBLIC
    cudaq-qec-realtime-decoding
  PRIVATE
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
  list(APPEND _server_cqr_install_rpath "${_CUDAQ_DEVICE_CALL_RUNTIME_LIB_DIR}")
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
