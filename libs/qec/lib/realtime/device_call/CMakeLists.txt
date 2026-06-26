# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

add_library(cudaq-qec-realtime-device-call SHARED
  qec_device_call_service.cpp
)

target_compile_options(cudaq-qec-realtime-device-call
  PRIVATE -fvisibility=hidden -fvisibility-inlines-hidden)

target_include_directories(cudaq-qec-realtime-device-call
  PRIVATE
    ${CUDAQ_REALTIME_INCLUDE_DIR}
    ${CUDAQ_DEVICE_CALL_INCLUDE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_options(cudaq-qec-realtime-device-call PRIVATE
  $<$<CXX_COMPILER_ID:GNU>:-Wl,--exclude-libs,ALL>
)

target_link_libraries(cudaq-qec-realtime-device-call
  PUBLIC
    cudaq-qec-realtime-decoding
  PRIVATE
    cudaq::cudaq-common
    ${CUDAQ_DEVICE_CALL_RUNTIME_LIBRARY}
)

set(_qec_device_call_build_rpath "${CMAKE_BINARY_DIR}/lib")
set(_qec_device_call_install_rpath "$ORIGIN")
if(_CUDAQ_DEVICE_CALL_RUNTIME_LIB_DIR)
  list(APPEND _qec_device_call_build_rpath
    "${_CUDAQ_DEVICE_CALL_RUNTIME_LIB_DIR}")
  list(APPEND _qec_device_call_install_rpath
    "${_CUDAQ_DEVICE_CALL_RUNTIME_LIB_DIR}")
endif()

set_target_properties(cudaq-qec-realtime-device-call PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
  BUILD_RPATH "${_qec_device_call_build_rpath}"
  INSTALL_RPATH "${_qec_device_call_install_rpath}"
)

install(TARGETS cudaq-qec-realtime-device-call
  COMPONENT qec-lib
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
