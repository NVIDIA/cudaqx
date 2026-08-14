# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# The playback emulator core: parser, timing loop, dispatch (stream_until),
# the three required session backends, syndrome sources, and the downstream
# CSV analyzer.
#
# This whole tool -- headers, lib, CLI, python bindings, tests -- lives
# self-contained under tools/playback-emulator/; this file is the one
# exception that reaches outside it, for the pieces it borrows from the
# existing decoding-server (RpcSlot.h et al.) and core (heterogeneous_map
# et al.), which are not this tool's to own a copy of.

if(CUDAQX_QEC_STANDALONE_BUILD)
  set(_playback_core_include_dir "${CMAKE_SOURCE_DIR}/../core/include")
  set(_playback_decoding_server_dir "${CMAKE_SOURCE_DIR}/lib/realtime/decoding-server-cqr")
else()
  set(_playback_core_include_dir "${CMAKE_SOURCE_DIR}/libs/core/include")
  set(_playback_decoding_server_dir "${CMAKE_SOURCE_DIR}/libs/qec/lib/realtime/decoding-server-cqr")
endif()

add_library(cudaq-qec-playback-emulator STATIC
  emulator.cpp
  parser.cpp
  session_inproc.cpp
  session_null.cpp
  session_udp.cpp
  syndrome_source.cpp
)

target_compile_features(cudaq-qec-playback-emulator PUBLIC cxx_std_20)

target_include_directories(cudaq-qec-playback-emulator
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
    $<BUILD_INTERFACE:${CUDAQX_QEC_INCLUDE_DIR}>
    $<BUILD_INTERFACE:${_playback_core_include_dir}>
    $<BUILD_INTERFACE:${CUDAQ_REALTIME_INCLUDE_DIR}>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    # RpcSlot.h / SessionRegistry.h / DecodingSession.h: the in-process and
    # UDP backends dispatch straight to these payload-level cores rather
    # than duplicating wire parsing/decoder-set construction.
    ${_playback_decoding_server_dir}
)

find_package(Threads REQUIRED)

target_link_libraries(cudaq-qec-playback-emulator
  PUBLIC
    cudaq-qec-decoding-server
    cudaq-qec-realtime-decoding
  PRIVATE
    cudaq-qec-decoders
    libstim
    Threads::Threads
)

set_target_properties(cudaq-qec-playback-emulator PROPERTIES
  ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
  POSITION_INDEPENDENT_CODE ON
)

# STATIC archive consumed only at link time (by the CLI tool, unit tests, and
# the Python bindings) -- not part of the installed SDK, so not installed.
