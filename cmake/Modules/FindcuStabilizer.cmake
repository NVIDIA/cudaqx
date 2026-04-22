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

Find the cuStabilizer library (shipped as a pip wheel: custabilizer-cu12 / cu13).

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
  Preferred search prefix.  Accepted as a CMake variable (``-DCUSTABILIZER_ROOT=...``)
  or an environment variable.

#]=======================================================================]

cmake_policy(SET CMP0144 NEW)

if(NOT CUSTABILIZER_ROOT AND DEFINED ENV{CUSTABILIZER_ROOT})
  set(CUSTABILIZER_ROOT "$ENV{CUSTABILIZER_ROOT}")
endif()

find_path(cuStabilizer_INCLUDE_DIR
  NAMES custabilizer.h
  HINTS
    ${CUSTABILIZER_ROOT}/include
)

find_library(cuStabilizer_LIBRARY
  NAMES custabilizer libcustabilizer.so.0
  HINTS
    ${CUSTABILIZER_ROOT}/lib64
    ${CUSTABILIZER_ROOT}/lib
)

set(cuStabilizer_VERSION "")
if(cuStabilizer_INCLUDE_DIR AND EXISTS "${cuStabilizer_INCLUDE_DIR}/custabilizer.h")
  file(STRINGS "${cuStabilizer_INCLUDE_DIR}/custabilizer.h"
       _custab_major_line REGEX "^#define[ \t]+CUSTABILIZER_MAJOR[ \t]+[0-9]+")
  file(STRINGS "${cuStabilizer_INCLUDE_DIR}/custabilizer.h"
       _custab_minor_line REGEX "^#define[ \t]+CUSTABILIZER_MINOR[ \t]+[0-9]+")
  file(STRINGS "${cuStabilizer_INCLUDE_DIR}/custabilizer.h"
       _custab_patch_line REGEX "^#define[ \t]+CUSTABILIZER_PATCH[ \t]+[0-9]+")
  if(_custab_major_line AND _custab_minor_line AND _custab_patch_line)
    string(REGEX REPLACE "^#define[ \t]+CUSTABILIZER_MAJOR[ \t]+([0-9]+).*$" "\\1"
           _custab_major "${_custab_major_line}")
    string(REGEX REPLACE "^#define[ \t]+CUSTABILIZER_MINOR[ \t]+([0-9]+).*$" "\\1"
           _custab_minor "${_custab_minor_line}")
    string(REGEX REPLACE "^#define[ \t]+CUSTABILIZER_PATCH[ \t]+([0-9]+).*$" "\\1"
           _custab_patch "${_custab_patch_line}")
    set(cuStabilizer_VERSION "${_custab_major}.${_custab_minor}.${_custab_patch}")
  endif()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(cuStabilizer
  REQUIRED_VARS cuStabilizer_INCLUDE_DIR cuStabilizer_LIBRARY
  VERSION_VAR cuStabilizer_VERSION
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
