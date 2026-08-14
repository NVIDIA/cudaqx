# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# The playback emulator. Everything specific to
# this tool -- headers, library implementation, CLI, python bindings, and
# tests -- lives under this directory.

# Only built where the in-process/UDP backends' dependency is: a
# realtime-enabled CUDA-Q install (see lib/realtime/decoding-server-cqr's own
# guard, which cudaq-qec-decoding-server is gated behind).
if(NOT TARGET cudaq-qec-decoding-server)
  message(STATUS "Skipping playback-emulator tool: "
                 "cudaq-qec-decoding-server not available")
  return()
endif()

add_subdirectory(lib)

# The CLI entry point 
add_executable(playback-emulator
  playback_emulator_main.cpp
)

target_link_libraries(playback-emulator PRIVATE cudaq-qec-playback-emulator)

set_target_properties(playback-emulator PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  BUILD_RPATH "$ORIGIN/../lib"
  INSTALL_RPATH "$ORIGIN/../lib"
  INSTALL_RPATH_USE_LINK_PATH TRUE
)

install(TARGETS playback-emulator
  COMPONENT qec-tools
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

add_subdirectory(sifl)
