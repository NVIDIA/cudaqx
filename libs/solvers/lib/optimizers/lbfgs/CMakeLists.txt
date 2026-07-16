# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

FetchContent_Declare(liblbfgs
  GIT_REPOSITORY https://github.com/chokkan/liblbfgs.git
  GIT_TAG master
)

if(NOT liblbfgs_POPULATED)
  FetchContent_Populate(liblbfgs)
  set(LBFGS_USE_SSE ON)
  set(LBFGS_lib_TARGET_NAME liblbfgs)
  set(LBFGS_BUILD_SHARED_LIBS OFF)
  add_subdirectory(${liblbfgs_SOURCE_DIR} ${liblbfgs_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

# ==============================================================================

target_sources(cudaq-solvers PRIVATE lbfgs.cpp LBFGSObjective.cpp)
target_link_libraries(cudaq-solvers PRIVATE LBFGS::lib)
