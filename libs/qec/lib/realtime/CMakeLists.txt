# ============================================================================ #
# Copyright (c) 2024 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

find_package(LLVM 16.0.6 EXACT REQUIRED CONFIG)

# Enable CUDA for device code
if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  
  # Find cuda-quantum realtime headers for device library
  set(CUDAQ_REALTIME_ROOT "" CACHE PATH "Prefix to cuda-quantum install")
  set(_cudaq_realtime_prefixes "")
  if(CUDAQ_REALTIME_ROOT)
    list(APPEND _cudaq_realtime_prefixes "${CUDAQ_REALTIME_ROOT}")
    list(APPEND _cudaq_realtime_prefixes "${CUDAQ_REALTIME_ROOT}/build")
  endif()
  if(CUDAQ_INSTALL_PREFIX)
    list(APPEND _cudaq_realtime_prefixes "${CUDAQ_INSTALL_PREFIX}")
  endif()
  
  find_path(CUDAQ_REALTIME_INCLUDE_DIR
    NAMES cudaq/nvqlink/daemon/dispatcher/cudaq_realtime.h
    PATHS ${_cudaq_realtime_prefixes}
    PATH_SUFFIXES include ../include
  )
  
  if(CUDAQ_REALTIME_INCLUDE_DIR)
    message(STATUS "Found cuda-quantum realtime headers at ${CUDAQ_REALTIME_INCLUDE_DIR}")
    
    # Create static device library for device code
    add_library(cudaq-qec-realtime-cudevice STATIC
      gpu_kernels.cu
    )
    
    set_target_properties(cudaq-qec-realtime-cudevice PROPERTIES
      CUDA_SEPARABLE_COMPILATION ON
      CUDA_STANDARD 17
      POSITION_INDEPENDENT_CODE ON
    )
    
    # Handle both standalone (libs/qec) and full repo builds
    if(CUDAQX_QEC_STANDALONE_BUILD)
      set(_core_include_dir "${CMAKE_SOURCE_DIR}/../core/include")
    else()
      set(_core_include_dir "${CMAKE_SOURCE_DIR}/libs/core/include")
    endif()
    
    target_include_directories(cudaq-qec-realtime-cudevice PUBLIC
      $<BUILD_INTERFACE:${CUDAQX_QEC_INCLUDE_DIR}>
      $<BUILD_INTERFACE:${_core_include_dir}>
      $<BUILD_INTERFACE:${CUDAQ_REALTIME_INCLUDE_DIR}>
      $<INSTALL_INTERFACE:include>
    )
    
    target_link_libraries(cudaq-qec-realtime-cudevice PUBLIC CUDA::cudart)
    
    set_target_properties(cudaq-qec-realtime-cudevice PROPERTIES
      ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    )
    
    install(TARGETS cudaq-qec-realtime-cudevice
      COMPONENT qec-lib
      ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    )
  else()
    message(WARNING "cuda-quantum realtime headers not found. "
                    "Device library cudaq-qec-realtime-device will not be built. "
                    "Set CUDAQ_REALTIME_ROOT to enable it.")
  endif()
endif()

# Shared library for host-side API
add_library(cudaq-qec-realtime-decoding SHARED
  realtime_decoding.cpp
  config.cpp
)

target_compile_options(cudaq-qec-realtime-decoding 
  PRIVATE -fvisibility=hidden -fvisibility-inlines-hidden)

target_include_directories(cudaq-qec-realtime-decoding
  PUBLIC 
    $<BUILD_INTERFACE:${CUDAQX_QEC_INCLUDE_DIR}>
    $<BUILD_INTERFACE:${LLVM_INCLUDE_DIRS}>
    $<INSTALL_INTERFACE:${CUDAQ_INCLUDE_DIR}>
    $<INSTALL_INTERFACE:include>
)

target_link_options(cudaq-qec-realtime-decoding PUBLIC
  $<$<CXX_COMPILER_ID:GNU>:-Wl,--exclude-libs,ALL>
)

target_link_libraries(cudaq-qec-realtime-decoding
  PUBLIC 
    cudaqx-core
    cudaq-qec
  PRIVATE
    cudaq::cudaq-common
    LLVMSupport
)

set_target_properties(cudaq-qec-realtime-decoding PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)


install(TARGETS cudaq-qec-realtime-decoding
  COMPONENT qec-lib
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

add_subdirectory(quantinuum)
add_subdirectory(simulation)
