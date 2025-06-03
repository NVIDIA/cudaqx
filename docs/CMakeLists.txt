# ============================================================================ #
# Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

find_package(Doxygen REQUIRED)
find_package(Sphinx REQUIRED)

# Find all the public headers
set(CUDAQX_PUBLIC_HEADERS)
file(GLOB_RECURSE LIB_PUBLIC_HEADERS "../libs/core/include/*.h")
list(APPEND CUDAQX_PUBLIC_HEADERS ${LIB_PUBLIC_HEADERS})
foreach(lib ${CUDAQX_ALL_LIBS})
  file(GLOB_RECURSE LIB_PUBLIC_HEADERS "../libs/${lib}/include/*.h")
  list(APPEND CUDAQX_PUBLIC_HEADERS ${LIB_PUBLIC_HEADERS})
endforeach()

# Find all the public Python files
set(CUDAQX_PUBLIC_PYTHON)
foreach(lib ${CUDAQX_ALL_LIBS})
  file(GLOB_RECURSE LIB_PUBLIC_PYTHON "../libs/${lib}/python/*.py")
  list(APPEND CUDAQX_PUBLIC_PYTHON ${LIB_PUBLIC_PYTHON})
endforeach()

# ==============================================================================
# Doxygen
# ==============================================================================

# We need to give doxygen a space separated list of files
list(JOIN CUDAQX_PUBLIC_HEADERS " " HEADERS_LIST)

set(DOXYGEN_INPUT ${HEADERS_LIST})
set(DOXYGEN_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/doxygen)
set(DOXYGEN_INDEX_FILE ${DOXYGEN_OUTPUT_DIR}/xml/index.xml)
set(DOXYFILE_IN ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in)
set(DOXYFILE_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

configure_file(${DOXYFILE_IN} ${DOXYFILE_OUT} @ONLY)

# Only regenerate Doxygen when the Doxyfile or public headers change
add_custom_command(
  OUTPUT ${DOXYGEN_INDEX_FILE}
  DEPENDS ${CUDAQX_PUBLIC_HEADERS}
  COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYFILE_OUT}
  MAIN_DEPENDENCY ${DOXYFILE_OUT} ${DOXYFILE_IN}
  COMMENT "Generating docs"
)

add_custom_target(doxygen_docs DEPENDS ${DOXYGEN_INDEX_FILE})

# ==============================================================================

add_custom_target(docs DEPENDS doxygen_docs)
