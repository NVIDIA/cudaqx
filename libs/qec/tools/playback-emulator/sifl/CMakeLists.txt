# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# multi_round_decoder: delegates a shot's variable-length stabilizer history
# to one of several monolithic sub-decoders, each built from a DEM analysed
# from scratch for its own round count (see multi_round_decoder.cpp).
# Dlopen'd by name at runtime, so nothing here links it directly.
add_library(cudaq-qec-multi-round-decoder SHARED
  multi_round_decoder.cpp
)

target_include_directories(cudaq-qec-multi-round-decoder
  PUBLIC
    ${CMAKE_SOURCE_DIR}/libs/qec/include
    ${CMAKE_SOURCE_DIR}/libs/core/include
)

target_link_libraries(cudaq-qec-multi-round-decoder
  PRIVATE cudaq-qec-decoders
)

set_target_properties(cudaq-qec-multi-round-decoder PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/decoder-plugins
)

set_target_properties(cudaq-qec-multi-round-decoder PROPERTIES
  BUILD_RPATH "$ORIGIN"
  INSTALL_RPATH "$ORIGIN:$ORIGIN/.."
  INSTALL_RPATH_USE_LINK_PATH TRUE
)

install(TARGETS cudaq-qec-multi-round-decoder
  COMPONENT qec-decoder-plugins
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/decoder-plugins
)
