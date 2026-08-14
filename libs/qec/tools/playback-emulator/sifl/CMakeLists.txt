# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# dummy_sifl_decoder: async-sleep decoder plugin used by SIFL to drive its
# closed-loop timing (see dummy_sifl_decoder.cpp for the state machine).
# Dlopen'd by name at runtime, so nothing here links it directly.
add_library(cudaq-qec-dummy-sifl-decoder SHARED
  dummy_sifl_decoder.cpp
)

target_include_directories(cudaq-qec-dummy-sifl-decoder
  PUBLIC
    ${CMAKE_SOURCE_DIR}/libs/qec/include
    ${CMAKE_SOURCE_DIR}/libs/core/include
)

target_link_libraries(cudaq-qec-dummy-sifl-decoder
  # Sibling decoder plugins (example, pymatching) link only
  # cudaq-qec-decoders: the decoder base class and the config-schema
  # registry both live there. Linking the full cudaq-qec pulls in NVQIR
  # quantum-kernel intrinsics this decoder never calls, which this plugin
  # cannot resolve at dlopen time in environments without the matching
  # NVQIR runtime -- it then fails to load silently (plugin_loader.cpp logs
  # and continues), so dummy_sifl_decoder's YAML schema never registers.
  PRIVATE cudaq-qec-decoders
)

set_target_properties(cudaq-qec-dummy-sifl-decoder PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/decoder-plugins
)

set_target_properties(cudaq-qec-dummy-sifl-decoder PROPERTIES
  BUILD_RPATH "$ORIGIN"
  INSTALL_RPATH "$ORIGIN:$ORIGIN/.."
  INSTALL_RPATH_USE_LINK_PATH TRUE
)

install(TARGETS cudaq-qec-dummy-sifl-decoder
  COMPONENT qec-decoder-plugins
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/decoder-plugins
)
