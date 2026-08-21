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

# cudaq-qec (pulled in by cudaq_memory_source, which shares a translation
# unit with the other syndrome sources) leaves undefined symbols for a
# concrete NVQIR backend that the final binary has to supply -- there is no
# C++-level equivalent of Python's cudaq.set_target() to pick one at runtime.
# Same set libs/qec/unittests/backend-specific/stim/test_qec_stim.cpp links.
target_link_directories(playback-emulator PRIVATE ${CUDAQ_INSTALL_DIR}/lib)
target_link_libraries(playback-emulator PRIVATE
  cudaq-qec-playback-emulator
  cudaq::cudaq
  cudaq::cudaq-common
  cudaq-mlir-runtime
  nvqir
  nvqir-stim
)

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
