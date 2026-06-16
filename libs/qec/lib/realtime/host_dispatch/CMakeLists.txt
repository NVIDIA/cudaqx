# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

if(NOT CUDAQ_REALTIME_INCLUDE_DIR OR NOT CUDAQ_DEVICE_CALL_INCLUDE_DIR
   OR NOT CUDAQ_DEVICE_CALL_RUNTIME_LIBRARY)
  message(WARNING "cuda-quantum realtime/device_call headers or runtime not found. "
                  "Skipping cudaq-qec-realtime-decoding-host-dispatch.")
  return()
endif()

add_library(cudaq-qec-realtime-decoding-host-dispatch SHARED
  host_dispatch_service.cpp
)

target_compile_options(cudaq-qec-realtime-decoding-host-dispatch
  PRIVATE -fvisibility=hidden -fvisibility-inlines-hidden)

cudaqx_add_device_code(cudaq-qec-realtime-decoding-host-dispatch
  SOURCES
    host_dispatch_device.cpp
  COMPILER_FLAGS
    -frealtime-lowering)

target_include_directories(cudaq-qec-realtime-decoding-host-dispatch
  PUBLIC
    $<BUILD_INTERFACE:${CUDAQX_QEC_INCLUDE_DIR}>
    $<INSTALL_INTERFACE:${CUDAQ_INCLUDE_DIR}>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    ${CUDAQ_REALTIME_INCLUDE_DIR}
    ${CUDAQ_DEVICE_CALL_INCLUDE_DIR}
)

target_link_options(cudaq-qec-realtime-decoding-host-dispatch PUBLIC
  $<$<CXX_COMPILER_ID:GNU>:-Wl,--exclude-libs,ALL>
)

target_link_libraries(cudaq-qec-realtime-decoding-host-dispatch
  PUBLIC
    cudaqx-core
    cudaq-qec
    cudaq-qec-realtime-decoding
  PRIVATE
    cudaq::cudaq-common
    ${CUDAQ_DEVICE_CALL_RUNTIME_LIBRARY}
    $<$<BOOL:${CUDAQ_REALTIME_LIBRARY}>:${CUDAQ_REALTIME_LIBRARY}>
    $<$<BOOL:${CUDAQ_REALTIME_HOST_DISPATCH_LIBRARY}>:${CUDAQ_REALTIME_HOST_DISPATCH_LIBRARY}>
)

set_target_properties(cudaq-qec-realtime-decoding-host-dispatch PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

if(_CUDAQ_REALTIME_LIB_DIR)
  set_target_properties(cudaq-qec-realtime-decoding-host-dispatch PROPERTIES
    BUILD_RPATH "${_CUDAQ_REALTIME_LIB_DIR};${CMAKE_BINARY_DIR}/lib"
    INSTALL_RPATH "${_CUDAQ_REALTIME_LIB_DIR};${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}"
  )
endif()

install(TARGETS cudaq-qec-realtime-decoding-host-dispatch
  COMPONENT qec-lib
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
