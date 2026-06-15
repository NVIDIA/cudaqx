# ============================================================================ #
# Copyright (c) 2024 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

find_package(LLVM 22.1.4 EXACT REQUIRED CONFIG)

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
    NAMES cudaq/realtime/daemon/dispatcher/cudaq_realtime.h
    HINTS ${_cudaq_realtime_prefixes}
    PATH_SUFFIXES include
    NO_DEFAULT_PATH
  )
  if(NOT CUDAQ_REALTIME_INCLUDE_DIR)
    find_path(CUDAQ_REALTIME_INCLUDE_DIR
      NAMES cudaq/realtime/daemon/dispatcher/cudaq_realtime.h
      PATHS ${_cudaq_realtime_prefixes}
      PATH_SUFFIXES include
    )
  endif()
  if(NOT CUDAQ_REALTIME_INCLUDE_DIR)
    find_path(CUDAQ_REALTIME_INCLUDE_DIR
      NAMES cudaq/nvqlink/daemon/dispatcher/cudaq_realtime.h
      HINTS ${_cudaq_realtime_prefixes}
      PATH_SUFFIXES include ../include
    )
  endif()
  
  if(CUDAQ_REALTIME_INCLUDE_DIR)
    message(STATUS "Found cuda-quantum realtime headers at ${CUDAQ_REALTIME_INCLUDE_DIR}")
    find_library(CUDAQ_REALTIME_LIBRARY
      NAMES cudaq-realtime
      PATHS ${_cudaq_realtime_prefixes}
      PATH_SUFFIXES lib
    )
    if(CUDAQ_REALTIME_LIBRARY)
      message(STATUS "Found cuda-quantum realtime library at ${CUDAQ_REALTIME_LIBRARY}")
      get_filename_component(_CUDAQ_REALTIME_LIB_DIR "${CUDAQ_REALTIME_LIBRARY}" DIRECTORY)
    endif()

    # The host-side dispatch loop (cudaq_host_dispatcher_loop), used by
    # qec_realtime_session, lives in libcudaq-realtime-host-dispatch -- separate
    # from libcudaq-realtime.  Locate it so cudaq-qec-realtime-decoding can link
    # it (mirrors the cudaq-realtime-pipeline target below).
    find_library(CUDAQ_REALTIME_HOST_DISPATCH_LIBRARY
      NAMES cudaq-realtime-host-dispatch
      PATHS ${_cudaq_realtime_prefixes}
      PATH_SUFFIXES lib
    )
    if(CUDAQ_REALTIME_HOST_DISPATCH_LIBRARY)
      message(STATUS "Found cuda-quantum realtime host dispatch library at ${CUDAQ_REALTIME_HOST_DISPATCH_LIBRARY}")
    endif()
    
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
  qec_realtime_session.cpp
  rpc_producer.cpp
)

target_compile_options(cudaq-qec-realtime-decoding 
  PRIVATE -fvisibility=hidden -fvisibility-inlines-hidden)

target_include_directories(cudaq-qec-realtime-decoding
  PUBLIC 
    $<BUILD_INTERFACE:${CUDAQX_QEC_INCLUDE_DIR}>
    $<BUILD_INTERFACE:${LLVM_INCLUDE_DIRS}>
    $<INSTALL_INTERFACE:${CUDAQ_INCLUDE_DIR}>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    $<$<BOOL:${CUDAQ_REALTIME_INCLUDE_DIR}>:${CUDAQ_REALTIME_INCLUDE_DIR}>
    ${CUDAToolkit_INCLUDE_DIRS}
)

if(CUDAQ_REALTIME_INCLUDE_DIR)
  target_compile_definitions(cudaq-qec-realtime-decoding PRIVATE
    CUDAQ_REALTIME_ROOT)
endif()

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
    $<$<BOOL:${CUDAQ_REALTIME_LIBRARY}>:${CUDAQ_REALTIME_LIBRARY}>
    $<$<BOOL:${CUDAQ_REALTIME_HOST_DISPATCH_LIBRARY}>:${CUDAQ_REALTIME_HOST_DISPATCH_LIBRARY}>
)

set_target_properties(cudaq-qec-realtime-decoding PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

if(_CUDAQ_REALTIME_LIB_DIR)
  set_target_properties(cudaq-qec-realtime-decoding PROPERTIES
    BUILD_RPATH "${_CUDAQ_REALTIME_LIB_DIR};${CMAKE_BINARY_DIR}/lib"
    INSTALL_RPATH "${_CUDAQ_REALTIME_LIB_DIR};${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}"
  )
endif()


install(TARGETS cudaq-qec-realtime-decoding
  COMPONENT qec-lib
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

# ---------------------------------------------------------------------------
# RealtimePipeline shared library
# Requires pre-installed cudaq-realtime (set CUDAQ_REALTIME_ROOT)
# ---------------------------------------------------------------------------
if(CMAKE_CUDA_COMPILER AND CUDAQ_REALTIME_INCLUDE_DIR)
    find_library(_CUDAQ_RT_LIB cudaq-realtime
      PATHS ${_cudaq_realtime_prefixes} PATH_SUFFIXES lib)
    find_library(_CUDAQ_RT_HD_LIB cudaq-realtime-host-dispatch
      PATHS ${_cudaq_realtime_prefixes} PATH_SUFFIXES lib)

    if(_CUDAQ_RT_LIB AND _CUDAQ_RT_HD_LIB)
      message(STATUS "RealtimePipeline: building with CUDAQ_REALTIME_INCLUDE_DIR=${CUDAQ_REALTIME_INCLUDE_DIR}")

      add_library(cudaq-realtime-pipeline SHARED
        realtime_pipeline.cu
      )

      get_filename_component(_cuda_bin_pl "${CMAKE_CUDA_COMPILER}" DIRECTORY)
      get_filename_component(_cuda_root_pl "${_cuda_bin_pl}" DIRECTORY)
      set(_cuda_cccl_include_pl "${_cuda_root_pl}/include/cccl")

      target_include_directories(cudaq-realtime-pipeline
        PUBLIC
          $<BUILD_INTERFACE:${CUDAQX_QEC_INCLUDE_DIR}>
          $<BUILD_INTERFACE:${CUDAQ_REALTIME_INCLUDE_DIR}>
          $<BUILD_INTERFACE:${_cuda_cccl_include_pl}>
          $<INSTALL_INTERFACE:include>
      )

      target_link_libraries(cudaq-realtime-pipeline
        PUBLIC CUDA::cudart_static
        PRIVATE ${_CUDAQ_RT_LIB} ${_CUDAQ_RT_HD_LIB}
      )

      option(ENABLE_NVTX "Enable NVTX profiling ranges" OFF)
      if(ENABLE_NVTX)
        target_compile_definitions(cudaq-realtime-pipeline PRIVATE ENABLE_NVTX)
        message(STATUS "NVTX profiling enabled for cudaq-realtime-pipeline")
      endif()


      get_filename_component(_CUDAQ_RT_LIB_DIR "${_CUDAQ_RT_LIB}" DIRECTORY)
      set_target_properties(cudaq-realtime-pipeline PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
        BUILD_RPATH "${_CUDAQ_RT_LIB_DIR};${CMAKE_BINARY_DIR}/lib"
      )

      install(TARGETS cudaq-realtime-pipeline
        COMPONENT qec-lib
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
      )
    else()
      message(STATUS "RealtimePipeline: skipping (cudaq-realtime or cudaq-realtime-host-dispatch not found)")
    endif()
else()
  if(CMAKE_CUDA_COMPILER)
    message(STATUS "RealtimePipeline: skipping (CUDAQ_REALTIME_INCLUDE_DIR not set)")
  endif()
endif()

add_subdirectory(quantinuum)
add_subdirectory(simulation)
