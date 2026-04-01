# ============================================================================ #
# Copyright (c) 2024 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.    #
# ============================================================================ #

#[=======================================================================[.rst:
FindcuStabilizer
----------------

Find the cuStabilizer library (shipped inside cuQuantum).

Uses the same ``CUQUANTUM_INSTALL_PREFIX`` convention as CUDA-Q.

Imported Targets
^^^^^^^^^^^^^^^^

``cuStabilizer::cuStabilizer``
  The cuStabilizer library.

Result Variables
^^^^^^^^^^^^^^^^

``cuStabilizer_FOUND``
``cuStabilizer_INCLUDE_DIR``
``cuStabilizer_LIBRARY``

Hints
^^^^^

``CUSTABILIZER_ROOT``
  Preferred search prefix.
``CUQUANTUM_INSTALL_PREFIX``
  cuQuantum installation prefix (same as CUDA-Q convention).

#]=======================================================================]

if(NOT CUSTABILIZER_ROOT AND NOT CUQUANTUM_INSTALL_PREFIX)
  set(CUQUANTUM_INSTALL_PREFIX "$ENV{CUQUANTUM_INSTALL_PREFIX}" CACHE PATH
    "Path to cuQuantum installation")
endif()

find_path(cuStabilizer_INCLUDE_DIR
  NAMES custabilizer.h
  HINTS
    ${CUSTABILIZER_ROOT}/include
    ${CUQUANTUM_INSTALL_PREFIX}/include
)

find_library(cuStabilizer_LIBRARY
  NAMES custabilizer libcustabilizer.so.0
  HINTS
    ${CUSTABILIZER_ROOT}/lib64
    ${CUSTABILIZER_ROOT}/lib
    ${CUQUANTUM_INSTALL_PREFIX}/lib64
    ${CUQUANTUM_INSTALL_PREFIX}/lib
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(cuStabilizer
  REQUIRED_VARS cuStabilizer_INCLUDE_DIR cuStabilizer_LIBRARY
)

if(cuStabilizer_FOUND AND NOT TARGET cuStabilizer::cuStabilizer)
  add_library(cuStabilizer::cuStabilizer INTERFACE IMPORTED GLOBAL)
  set_target_properties(cuStabilizer::cuStabilizer PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${cuStabilizer_INCLUDE_DIR}"
  )
  if(cuStabilizer_LIBRARY)
    set_target_properties(cuStabilizer::cuStabilizer PROPERTIES
      INTERFACE_LINK_LIBRARIES "${cuStabilizer_LIBRARY}"
    )
  endif()
endif()
