# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Realtime-decoding example.
#
# This builds ONE source (surface_code_realtime_decoding.cpp) two ways, against
# the *installed* CUDA-Q / CUDA-QX SDK -- nothing here reaches into a build tree:
#
#   surface_code_realtime_decoding       generator: --target stim; writes the
#                                        decoder config + syndrome files.
#   surface_code_realtime_decoding-cqr   lowered kernel: adds -frealtime-lowering
#                                        -DQEC_APP_CQR; the live QPU kernel that
#                                        streams syndromes to the delivered
#                                        decoding_server over UDP.
#
# The decoding_server and the FPGA playback tool are DELIVERABLES (installed,
# not built here); run_realtime_decoding.sh resolves them from --install-prefix.

cmake_minimum_required(VERSION 3.23)

# The lowered-kernel binary links the realtime dispatch archive (relocatable
# CUDA device code), so the project enables CUDA and needs an architecture for
# the device-link step. Default to 80 (A100) -- an architecture the shipped
# SDK's dispatch archive includes; override to match your GPU:
# -DCMAKE_CUDA_ARCHITECTURES=90 for Hopper, 100 for Blackwell (e.g. GB200).
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 80)
endif()

project(realtime_decoding_demo LANGUAGES CXX CUDA)

# ---------------------------------------------------------------------------- #
# Locate the installed SDK.
# ---------------------------------------------------------------------------- #
# CUDA-Q: provides nvq++, cudaq/*.h, cudaq/realtime.h and the core runtime libs.
if(NOT CUDAQ_INSTALL_DIR)
  if(DEFINED ENV{CUDA_QUANTUM_PATH})
    set(CUDAQ_INSTALL_DIR "$ENV{CUDA_QUANTUM_PATH}")
  else()
    set(CUDAQ_INSTALL_DIR "/usr/local/cudaq")
  endif()
endif()
if(NOT EXISTS "${CUDAQ_INSTALL_DIR}/bin/nvq++")
  message(FATAL_ERROR
    "nvq++ not found under CUDAQ_INSTALL_DIR=${CUDAQ_INSTALL_DIR}. "
    "Pass -DCUDAQ_INSTALL_DIR=<cuda-quantum install prefix>.")
endif()

# CUDA-QX: provides cudaq/qec/*.h and the QEC + realtime-decoding libs/plugins.
if(NOT CUDAQX_INSTALL_DIR)
  if(DEFINED ENV{CUDAQX_INSTALL_DIR})
    set(CUDAQX_INSTALL_DIR "$ENV{CUDAQX_INSTALL_DIR}")
  else()
    message(FATAL_ERROR
      "Set -DCUDAQX_INSTALL_DIR=<cuda-qx install prefix> (the directory that "
      "holds include/cudaq/qec and lib/libcudaq-qec.so).")
  endif()
endif()
if(NOT EXISTS "${CUDAQX_INSTALL_DIR}/include/cudaq/qec/code.h")
  message(FATAL_ERROR
    "CUDAQX_INSTALL_DIR=${CUDAQX_INSTALL_DIR} does not look like a CUDA-QX "
    "install (missing include/cudaq/qec/code.h).")
endif()

set(NVQPP "${CUDAQ_INSTALL_DIR}/bin/nvq++")
find_package(CUDAToolkit REQUIRED)

# The realtime dispatch archive carries relocatable (RDC) device code, so the
# -cqr executable must be linked with a CUDA device-link step (see below).
find_library(CUDAQ_REALTIME_DISPATCH_LIB cudaq-realtime-dispatch
  PATHS "${CUDAQ_INSTALL_DIR}" "${CUDAQ_REALTIME_DIR}" PATH_SUFFIXES lib REQUIRED)

link_directories(
  "${CUDAQX_INSTALL_DIR}/lib"
  "${CUDAQX_INSTALL_DIR}/lib/decoder-plugins"
  "${CUDAQ_INSTALL_DIR}/lib"
  "${CUDAQ_INSTALL_DIR}/lib/plugins")

set(_src "${CMAKE_CURRENT_SOURCE_DIR}/surface_code_realtime_decoding.cpp")

# Only the CUDA-QX headers need an explicit -I; nvq++ already knows its own
# CUDA-Q include dir (cudaq/*.h, cudaq/realtime.h, common/*.h).
set(_nvqpp_includes "-I${CUDAQX_INSTALL_DIR}/include")

# CUDA-Q core libraries, listed AFTER the qec + simulation libs on purpose.
# The simulation backend's static initializer registers into a global map owned
# by libcudaq-common; ELF runs initializers in reverse DT_NEEDED (link) order,
# so the qec/simulation libs must be listed FIRST (they initialize last) and the
# cudaq core libs LAST (they initialize first) or the registration dereferences
# a not-yet-constructed map and segfaults at startup.
set(_cudaq_core_libs
  cudaq-mlir-runtime nvqir nvqir-stim
  cudaq-qec-realtime-decoding cudaq-qec-decoders
  cudaq cudaq-platform-default cudaq-em-default
  cudaq-common cudaq-operator cudaq-logger)

# ---------------------------------------------------------------------------- #
# Helper: compile the source with nvq++ into an object, in an isolated working
# directory (nvq++ drops intermediates named after the source, so two compiles
# of the same file must not share a cwd).
# ---------------------------------------------------------------------------- #
function(_add_nvqpp_object out_var obj_name)
  set(_obj "${CMAKE_CURRENT_BINARY_DIR}/${obj_name}.o")
  set(_wd  "${CMAKE_CURRENT_BINARY_DIR}/${obj_name}.nvqpp")
  file(MAKE_DIRECTORY "${_wd}")
  add_custom_command(
    OUTPUT "${_obj}"
    COMMAND ${NVQPP} ${ARGN} -c -fPIC "${_src}" -o "${_obj}" ${_nvqpp_includes}
    DEPENDS "${_src}"
    WORKING_DIRECTORY "${_wd}"
    COMMENT "Compiling surface_code_realtime_decoding.cpp with nvq++ (${obj_name})"
    VERBATIM)
  set_source_files_properties("${_obj}" PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE)
  set(${out_var} "${_obj}" PARENT_SCOPE)
endfunction()

# ---------------------------------------------------------------------------- #
# Generator (plain): writes the decoder config + syndrome files.
# ---------------------------------------------------------------------------- #
_add_nvqpp_object(_gen_obj generator --target stim)
add_executable(surface_code_realtime_decoding "${_gen_obj}")
set_target_properties(surface_code_realtime_decoding PROPERTIES LINKER_LANGUAGE CXX)
# --no-as-needed: nvqir-stim (the --target stim simulator) and
# cudaq-platform-default are pulled in at runtime by registration, not by a
# direct symbol reference, so the default --as-needed would drop them from
# DT_NEEDED and the process would crash at startup with no platform.
target_link_libraries(surface_code_realtime_decoding PRIVATE
  -Wl,--no-as-needed
  cudaq-qec cudaq-qec-realtime-decoding
  cudaq-qec-realtime-decoding-simulation ${_cudaq_core_libs}
  -Wl,--as-needed
  "${CUDAQ_REALTIME_DISPATCH_LIB}")
target_link_options(surface_code_realtime_decoding PRIVATE
  LINKER:--allow-shlib-undefined LINKER:--export-dynamic)

# ---------------------------------------------------------------------------- #
# Lowered kernel (-cqr): the live syndrome source over UDP.
# LINKER_LANGUAGE CUDA + CUDA_RESOLVE_DEVICE_SYMBOLS runs the device-link
# (nvcc -dlink) that resolves the dispatch archive's RDC registration symbols.
# ---------------------------------------------------------------------------- #
_add_nvqpp_object(_cqr_obj generator-cqr
  --target stim -frealtime-lowering -DQEC_APP_CQR)
add_executable(surface_code_realtime_decoding-cqr "${_cqr_obj}")
set_target_properties(surface_code_realtime_decoding-cqr PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
  CUDA_STANDARD 17
  LINKER_LANGUAGE CUDA)
target_link_libraries(surface_code_realtime_decoding-cqr PRIVATE
  -Wl,--no-as-needed
  cudaq-qec cudaq-qec-realtime-decoding
  cudaq-qec-realtime-decoding-simulation-cqr
  # The in-process decoding service; also exports
  # cudaqx_qec_device_call_dispatch_count, which the app prints as evidence
  # that an external server (not the in-process service) did the decoding.
  cudaq-qec-realtime-decoding-server-cqr
  cudaq-device-call-runtime
  ${_cudaq_core_libs}
  -Wl,--as-needed
  "${CUDAQ_REALTIME_DISPATCH_LIB}" CUDA::cudart)
target_link_options(surface_code_realtime_decoding-cqr PRIVATE
  LINKER:--allow-shlib-undefined LINKER:--export-dynamic)
