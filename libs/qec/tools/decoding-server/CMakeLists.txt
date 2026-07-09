# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# qec_decoding_daemon is a decoder-agnostic decoding-server process: the
# service end of a CUDA-Q device_call transport, wiring the libcudaq-realtime
# HOST_CALL dispatcher onto a transceiver's rings and serving the
# decoding-server-cqr function table (enqueue_syndromes / get_corrections /
# reset_decoder). Its decoders are selected entirely by a YAML config file
# (see decoding_server_config.yaml) and its transport by --transport=udp
# (loopback-capable, runs anywhere) or --transport=cpu_roce (real RDMA wire,
# compiled in when libcudaq-realtime-cpu-roce-transport + ibverbs are found).
#
# The two-process test (unittests/realtime/test_decoding_server.cpp) spawns
# this binary as a subprocess.

if(CUDAQ_REALTIME_INCLUDE_DIR AND TARGET cudaq-qec-realtime-decoding-server-cqr)
  # The daemon needs the DeviceCallService plugin headers, the libcudaq-realtime
  # shared lib (cudaq_host_dispatcher_loop lives there, device-linked), and the
  # cpu_transport UDP ring transceiver static lib.
  find_path(QEC_UDP_DEVICE_CALL_INCLUDE_DIR
    NAMES cudaq/realtime/device_call_service.h
          cudaq_internal/device_call/DeviceCallService.h
    HINTS ${CUDAQ_REALTIME_ROOT} ${CUDAQ_INSTALL_DIR}
    PATH_SUFFIXES include)
  find_library(QEC_UDP_REALTIME_LIBRARY cudaq-realtime
    HINTS ${CUDAQ_REALTIME_ROOT} ${CUDAQ_INSTALL_DIR}
    PATH_SUFFIXES lib)
  find_library(QEC_UDP_TRANSPORT_LIBRARY cudaq-realtime-udp-transport
    HINTS ${CUDAQ_REALTIME_ROOT} ${CUDAQ_INSTALL_DIR}
    PATH_SUFFIXES lib)
  # The host-side ring dispatch loop (cudaq_host_ring_dispatch_loop) lives in
  # its own static archive since the realtime host-dispatch split; it is no
  # longer part of libcudaq-realtime.
  find_library(QEC_HOST_DISPATCH_LIBRARY cudaq-realtime-host-dispatch
    HINTS ${CUDAQ_REALTIME_ROOT} ${CUDAQ_INSTALL_DIR}
    PATH_SUFFIXES lib)
  # Optional: the daemon's --transport=cpu_roce path (real RDMA wire) needs
  # the CPU RoCE transceiver and libibverbs; without them the daemon still
  # builds with the udp transport only.
  find_library(QEC_CPU_ROCE_TRANSPORT_LIBRARY cudaq-realtime-cpu-roce-transport
    HINTS ${CUDAQ_REALTIME_ROOT} ${CUDAQ_INSTALL_DIR}
    PATH_SUFFIXES lib)
  find_library(QEC_IBVERBS_LIBRARY ibverbs)

  if(QEC_UDP_DEVICE_CALL_INCLUDE_DIR AND QEC_UDP_REALTIME_LIBRARY
     AND QEC_UDP_TRANSPORT_LIBRARY AND QEC_HOST_DISPATCH_LIBRARY)
    add_executable(qec_decoding_daemon qec_decoding_daemon.cpp)
    target_include_directories(qec_decoding_daemon PRIVATE
      ${CUDAQ_REALTIME_INCLUDE_DIR}
      ${QEC_UDP_DEVICE_CALL_INCLUDE_DIR}
      ${CUDAToolkit_INCLUDE_DIRS}
    )
    target_link_directories(qec_decoding_daemon
      PRIVATE ${CUDAQ_INSTALL_DIR}/lib)
    target_link_libraries(qec_decoding_daemon PRIVATE
      cudaq-qec
      cudaq-qec-realtime-decoding
      cudaq-qec-realtime-decoding-server-cqr
      ${QEC_UDP_REALTIME_LIBRARY}
      ${QEC_UDP_TRANSPORT_LIBRARY}
      ${QEC_HOST_DISPATCH_LIBRARY}
      cudaq::cudaq cudaq::cudaq-common cudaq-mlir-runtime
      nvqir nvqir-stim
      CUDA::cudart
    )
    if(QEC_CPU_ROCE_TRANSPORT_LIBRARY AND QEC_IBVERBS_LIBRARY)
      target_compile_definitions(qec_decoding_daemon PRIVATE
        QEC_HAVE_CPU_ROCE_TRANSPORT)
      target_link_libraries(qec_decoding_daemon PRIVATE
        ${QEC_CPU_ROCE_TRANSPORT_LIBRARY}
        ${QEC_IBVERBS_LIBRARY})
      message(STATUS "qec_decoding_daemon: cpu_roce transport enabled")
    endif()
    if(CUDAQ_GPU_ROCE_AVAILABLE)
      # Re-find in case the CQR subdir cache entry didn't propagate here.
      if(NOT CUDAQ_REALTIME_DISPATCH_LIBRARY)
        find_library(CUDAQ_REALTIME_DISPATCH_LIBRARY cudaq-realtime-dispatch
          HINTS ${CUDAQ_REALTIME_ROOT} ${CUDAQ_INSTALL_DIR}
          PATH_SUFFIXES lib)
      endif()
      if(NOT CUDAQ_REALTIME_DISPATCH_LIBRARY)
        message(FATAL_ERROR
          "CUDAQ_GPU_ROCE_AVAILABLE is set but cudaq-realtime-dispatch was "
          "not found (searched CUDAQ_REALTIME_ROOT=${CUDAQ_REALTIME_ROOT} and "
          "CUDAQ_INSTALL_DIR=${CUDAQ_INSTALL_DIR}).  "
          "GpuRoceTransceiver::launch_scheduler resolves "
          "cudaq_create/launch/destroy_dispatch_graph_regular via dlsym; "
          "without this library in the daemon image those lookups will fail "
          "at runtime.  Set CUDAQ_REALTIME_ROOT to the CUDA-Q realtime "
          "install prefix and re-run CMake.")
      endif()
      target_compile_definitions(qec_decoding_daemon PRIVATE
        QEC_HAVE_GPU_ROCE_TRANSPORT)
      target_include_directories(qec_decoding_daemon PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../../lib/realtime/decoding-server-cqr
      )
      if(NOT TARGET cudaq-qec-realtime-cudevice-proprietary)
        message(FATAL_ERROR
          "CUDAQ_GPU_ROCE_AVAILABLE is set but the CMake target "
          "cudaq-qec-realtime-cudevice-proprietary is not defined.  "
          "GpuRoceTransceiver::launch_scheduler resolves the three "
          "populate-shim symbols "
          "(cudaqx_qec_realtime_dispatch_populate_*_device_entry) via "
          "dlsym; without this archive absorbed as WHOLE_ARCHIVE those "
          "lookups will fail at runtime.  Ensure the proprietary static "
          "library target is available before configuring the daemon.")
      endif()
      target_link_libraries(qec_decoding_daemon PRIVATE
        cudaq-qec-decoder-server
        ${CUDAQ_REALTIME_DISPATCH_LIBRARY}
        $<LINK_LIBRARY:WHOLE_ARCHIVE,cudaq-qec-realtime-cudevice-proprietary>
      )
      target_link_options(qec_decoding_daemon PRIVATE
        "LINKER:--export-dynamic"
      )
      message(STATUS "qec_decoding_daemon: gpu_roce transport enabled")
    endif()
    set_target_properties(qec_decoding_daemon PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
      BUILD_RPATH
        "${CMAKE_BINARY_DIR}/lib;${CMAKE_BINARY_DIR}/lib/decoder-plugins;${CUDAQ_INSTALL_DIR}/lib"
    )
    # Decoder plugins are loaded at runtime by decoder type name; the example
    # configs currently select pymatching / multi_error_lut.
    add_dependencies(qec_decoding_daemon cudaq-qec-pymatching)

    # Place the example decoder configs next to the daemon binary.
    configure_file(
      ${CMAKE_CURRENT_SOURCE_DIR}/decoding_server_config.yaml
      ${CMAKE_BINARY_DIR}/bin/decoding_server_config.yaml
      COPYONLY)
    configure_file(
      ${CMAKE_CURRENT_SOURCE_DIR}/decoding_server_config_multi_error_lut.yaml
      ${CMAKE_BINARY_DIR}/bin/decoding_server_config_multi_error_lut.yaml
      COPYONLY)
  else()
    message(STATUS "qec_decoding_daemon: cudaq-realtime udp transport not "
                   "found; decoding-server tool will not be built.")
  endif()
endif()
