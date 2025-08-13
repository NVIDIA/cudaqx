# ============================================================================ #
# Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

#[=======================================================================[.rst:
cudaqx_add_device_code
----------------------------

Add NVQ++ custom-compiled quantum device source files to a target library using nvq++.

This function compiles specified source files using the nvq++ compiler
from the CUDAQ installation and adds the resulting object files to the
given library target.

.. command:: add_custom_compiled_sources

  .. code-block:: cmake

    cudaqx_add_device_code(
      <library_name>
      SOURCES <source1> [<source2> ...]
      [COMPILER_FLAGS <flag1> [<flag2> ...]]
    )

  ``<library_name>``
    The name of the existing library target to which the compiled
    sources will be added.

  ``SOURCES <source1> [<source2> ...]``
    A list of source files to be compiled.

  ``COMPILER_FLAGS <flag1> [<flag2> ...]``
    Optional. A list of compiler flags to be passed to nvq++.

This function creates custom commands to compile each source file using
nvq++, generates custom targets for each compilation, and adds the
resulting object files to the specified library target.

Note: This function assumes that the CUDAQ_INSTALL_DIR variable is set
to the CUDAQ installation directory.

Note: You can use DEPENDS_ON if you want to delay compilation until some other
target has been built.

Example usage:
  cudaqx_add_device_code(
    my_library
    SOURCES
      ${CMAKE_CURRENT_SOURCE_DIR}/file1.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/file2.cpp
    COMPILER_FLAGS
      --enable-mlir
      -v
    DEPENDS_ON
      SomeOtherTarget
  )

#]=======================================================================]
function(cudaqx_add_device_code LIBRARY_NAME)
  set(options)
  set(oneValueArgs)
  set(multiValueArgs SOURCES COMPILER_FLAGS DEPENDS_ON)
  cmake_parse_arguments(ARGS "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  if(NOT DEFINED CUDAQ_INSTALL_DIR)
    message(FATAL_ERROR "CUDAQ_INSTALL_DIR must be defined")
  endif()

  if(NOT ARGS_SOURCES)
    message(FATAL_ERROR "At least one SOURCE file is required")
  endif()

  set(COMPILER ${CUDAQ_INSTALL_DIR}/bin/nvq++)

  # It might be that our CXX toolchain is installed in non-standard path and
  # `cudaq-quake`, being a clang-based compiler, won't be able to find it. In
  # such cases, setting CMAKE_CXX_COMPILER_EXTERNAL_TOOLCHAIN will allows us
  # to tell `cudaq-quake` where to look for the toolchain. (This happens when
  # building wheels inside the manylinux container, for example.)
  if (CMAKE_CXX_COMPILER_EXTERNAL_TOOLCHAIN)
    set(ARGS_COMPILER_FLAGS "${ARGS_COMPILER_FLAGS} --gcc-install-dir=${CMAKE_CXX_COMPILER_EXTERNAL_TOOLCHAIN}")
  endif()

  set(prop "$<TARGET_PROPERTY:${LIBRARY_NAME},INCLUDE_DIRECTORIES>")
  foreach(source ${ARGS_SOURCES})
    get_filename_component(filename ${source} NAME_WE)
    set(output_file "${CMAKE_CURRENT_BINARY_DIR}/${LIBRARY_NAME}_${filename}.o")
    cmake_path(GET output_file FILENAME baseName)

    add_custom_command(
      OUTPUT ${output_file}
      COMMAND ${COMPILER}
        ${ARGS_COMPILER_FLAGS} -c -fPIC --enable-mlir
        ${CMAKE_CURRENT_SOURCE_DIR}/${source} -o ${baseName}
        "$<$<BOOL:${prop}>:-I $<JOIN:${prop}, -I >>"
      DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${source} ${ARGS_DEPENDS_ON}
      COMMENT "Compiling ${source} with nvq++"
      VERBATIM
    )

    list(APPEND object_files ${output_file})
    list(APPEND custom_targets ${LIBRARY_NAME}_${filename}_target)

    add_custom_target(${LIBRARY_NAME}_${filename}_target DEPENDS ${output_file})
  endforeach()

  add_dependencies(${LIBRARY_NAME} ${custom_targets})
  target_sources(${LIBRARY_NAME} PRIVATE ${object_files})
endfunction()

#[=======================================================================[.rst:
cudaqx_set_target
-------------------------
   Set up a CUDA-QX target with the specified name.

   This function creates an interface library for the given CUDA-QX
   target, links it to the main CUDAQ library, and adds target-specific libraries.

   :param TARGETNAME: The name of the CUDA-QX target to set up.

   .. note::
      This function will create an interface library
      named ``cudaq_${TARGETNAME}`` and an alias target ``cudaq::cudaq_${TARGETNAME}``.

   **Example:**

   .. code-block:: cmake

      cudaqx_set_target(my_target)

   This will:

   1. Create an interface library ``cudaq_my_target``
   2. Link it to ``cudaq::cudaq``
   3. Link it to ``cudaq::cudaq-my_target-target``
   4. Create an alias target ``cudaq::cudaq_my_target``

   This function simplifies the process of setting up CUDA-QX targets by
   automating the creation of interface libraries and establishing the necessary linkages.
#]=======================================================================]
function(cudaqx_set_target TARGETNAME)
  message(STATUS "Setting CUDA-QX Target = ${TARGETNAME}")

  # Create a new interface target
  add_library(cudaq_${TARGETNAME} INTERFACE)

  # Link to the original cudaq target
  target_link_libraries(cudaq_${TARGETNAME} INTERFACE cudaq::cudaq)

  # Add the additional target-specific library
  target_link_libraries(cudaq_${TARGETNAME} INTERFACE cudaq::cudaq-${TARGETNAME}-target)

  # Create an alias target to make it easier to use
  add_library(cudaq::cudaq_${TARGETNAME} ALIAS cudaq_${TARGETNAME})
endfunction()

#[=======================================================================[.rst:
cudaqx_add_pymodule
-------------------------
    This is a helper function to add CUDAQ-QX libraries' python modules. It's
    main purpose is to create a custom target, cudaqx-pymodules, which depends
    on all libraries' python modules.

#]=======================================================================]
function(cudaqx_add_pymodule module)
  pybind11_add_module(${module} ${ARGN})
  add_dependencies(cudaqx-pymodules ${module})
endfunction()
