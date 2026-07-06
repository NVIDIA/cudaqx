# ============================================================================ #
# Copyright (c) 2024 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

#------------------------------------------------------------------------------#
# Maintenance note: to generate QIR files and corresponding config files for
# Quantinuum, you may manually run tests like the following:
#    CUDAQ_DUMP_JIT_IR=1 KEEP_LOG_FILES=1 ctest -R quantinuum
# The resulting files will be in:
#    build/libs/qec/unittests/realtime/app_examples/config-*.yml
#    build/libs/qec/unittests/realtime/app_examples/qir-*.ll
#------------------------------------------------------------------------------#

# Resolve cudaq-realtime dispatch for app examples that opt into
# CUDAQ_QEC_REALTIME_MODE=inproc_rpc.
set(_app_cudaq_realtime_prefixes "")
if(CUDAQ_REALTIME_ROOT)
  list(APPEND _app_cudaq_realtime_prefixes "${CUDAQ_REALTIME_ROOT}")
  list(APPEND _app_cudaq_realtime_prefixes "${CUDAQ_REALTIME_ROOT}/build")
endif()
if(CUDAQ_INSTALL_PREFIX)
  list(APPEND _app_cudaq_realtime_prefixes "${CUDAQ_INSTALL_PREFIX}")
endif()
find_library(_APP_CUDAQ_REALTIME_DISPATCH_LIB cudaq-realtime-dispatch
  PATHS ${_app_cudaq_realtime_prefixes} PATH_SUFFIXES lib)

function(qec_realtime_app_link_options _tgt)
  target_link_options(${_tgt} PRIVATE
    "LINKER:--allow-shlib-undefined"
    "LINKER:--export-dynamic"
  )
  if(_APP_CUDAQ_REALTIME_DISPATCH_LIB)
    target_link_libraries(${_tgt} PRIVATE
      ${_APP_CUDAQ_REALTIME_DISPATCH_LIB})
  endif()
  if(TARGET cudaq-qec-realtime-cudevice-proprietary)
    target_link_libraries(${_tgt} PRIVATE
      CUDA::cudart
      $<LINK_LIBRARY:WHOLE_ARCHIVE,cudaq-qec-realtime-cudevice-proprietary>)
    set_property(TARGET ${_tgt} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
    set_property(TARGET ${_tgt} PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    set_property(TARGET ${_tgt} PROPERTY CUDA_STANDARD 17)
    set_property(TARGET ${_tgt} PROPERTY LINKER_LANGUAGE CUDA)
  endif()
endfunction()

#------------------------------------------------------------------------------#
# Surface code 1, fully local, regular Stim simulator
add_executable(surface_code-1-local)
set_target_properties(surface_code-1-local PROPERTIES LINKER_LANGUAGE CXX)

cudaqx_add_device_code(surface_code-1-local
  SOURCES
    surface_code-1.cpp
  COMPILER_FLAGS
    --target stim
)

target_link_directories(surface_code-1-local
  PRIVATE ${CUDAQ_INSTALL_DIR}/lib)

target_link_libraries(surface_code-1-local
  PRIVATE
    cudaq-qec
    cudaq-qec-realtime-decoding
    cudaq-qec-realtime-decoding-simulation
    cudaq::cudaq cudaq::cudaq-common cudaq-mlir-runtime
    nvqir nvqir-stim)
qec_realtime_app_link_options(surface_code-1-local)

add_test(
  NAME app_examples.surface_code-1-local-test-distance-3
  COMMAND
    bash "${CMAKE_CURRENT_SOURCE_DIR}/surface_code-1-test.sh"
      ${CMAKE_CURRENT_BINARY_DIR}/surface_code-1-local
      ${CMAKE_CURRENT_BINARY_DIR}/surface_code-1-local
      3 40 40 NULL 12 6
      ${CMAKE_BINARY_DIR}/lib
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

if(_APP_CUDAQ_REALTIME_DISPATCH_LIB
   AND TARGET cudaq-qec-realtime-cudevice-proprietary)
  add_test(
    NAME app_examples.surface_code-1-local-test-distance-3-inproc-rpc
    COMMAND
      bash "${CMAKE_CURRENT_SOURCE_DIR}/surface_code-1-test.sh"
        ${CMAKE_CURRENT_BINARY_DIR}/surface_code-1-local
        ${CMAKE_CURRENT_BINARY_DIR}/surface_code-1-local
        3 60 40 NULL 12 6
        ${CMAKE_BINARY_DIR}/lib
        nv-qldpc-decoder
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )
  set_tests_properties(
    app_examples.surface_code-1-local-test-distance-3-inproc-rpc
    PROPERTIES ENVIRONMENT
      "CUDAQ_QEC_REALTIME_MODE=inproc_rpc;EXTRA_CLI_ARGS=--use-relay-bp")
endif()

# Test with sliding_window decoder
add_test(
  NAME app_examples.surface_code-1-local-test-distance-3-sliding-window
  COMMAND
    bash "${CMAKE_CURRENT_SOURCE_DIR}/surface_code-1-test.sh"
      ${CMAKE_CURRENT_BINARY_DIR}/surface_code-1-local
      ${CMAKE_CURRENT_BINARY_DIR}/surface_code-1-local
      3 40 40 NULL 12 6
      ${CMAKE_BINARY_DIR}/lib
      sliding_window 6 1
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

# This must be disabled for now because the multi_error_lut decoder is not
# powerful enough to pass this test. The nv-qldpc-decoder can pass this test,
# but that is not available on the GitHub repo.
# add_test(
#   NAME app_examples.surface_code-1-local-test-distance-5
#   COMMAND
#     bash "${CMAKE_CURRENT_SOURCE_DIR}/surface_code-1-test.sh"
#       ${CMAKE_CURRENT_BINARY_DIR}/surface_code-1-local
#       ${CMAKE_CURRENT_BINARY_DIR}/surface_code-1-local
#       5 40 40 NULL 20 10
#       ${CMAKE_BINARY_DIR}/lib
#   WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
# )

# Surface code 1 Python test
add_test(
  NAME app_examples.py-surface_code-1-test
  COMMAND
    pytest ${CMAKE_CURRENT_SOURCE_DIR}/surface_code-1-test.py
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

# End of surface code 1, fully local, regular Stim simulator
#------------------------------------------------------------------------------#


#------------------------------------------------------------------------------#
# Surface code 1, --target quantinuum --emulate
add_executable(surface_code-1-quantinuum-emulate)
set_target_properties(surface_code-1-quantinuum-emulate PROPERTIES LINKER_LANGUAGE CXX)

add_dependencies(surface_code-1-quantinuum-emulate surface_code-1-local)

cudaqx_add_device_code(surface_code-1-quantinuum-emulate
  SOURCES
    surface_code-1.cpp
  COMPILER_FLAGS
    --target quantinuum --emulate --quantinuum-machine Helios-Fake -DMANUALLY_INJECT_ERRORS=1 
  DEPENDS_ON
    # Must wait for the local executable to be built because it has the same
    # surface_code-1.cpp file, and the nvq++ compiler will use the same filename
    # prefix (surface_code-1) for intermediate files. They will stomp on each
    # other if you don't add this.
    surface_code-1-local
)

target_link_directories(surface_code-1-quantinuum-emulate
  PRIVATE ${CUDAQ_INSTALL_DIR}/lib)

# Add -Wl,--export-dynamic to the link line.
target_link_options(surface_code-1-quantinuum-emulate PRIVATE -Wl,--export-dynamic)
target_link_libraries(surface_code-1-quantinuum-emulate
  PRIVATE
    cudaq-qec
    cudaq-qec-realtime-decoding
    cudaq-qec-realtime-decoding-quantinuum
    cudaq::cudaq cudaq::cudaq-common cudaq-mlir-runtime cudaq-rest-qpu
    nvqir nvqir-stim)
qec_realtime_app_link_options(surface_code-1-quantinuum-emulate)

add_test(
  NAME app_examples.surface_code-1-quantinuum-emulate-test-distance-3-in-process
  COMMAND
    bash "${CMAKE_CURRENT_SOURCE_DIR}/surface_code-1-test.sh"
      ${CMAKE_CURRENT_BINARY_DIR}/surface_code-1-local
      ${CMAKE_CURRENT_BINARY_DIR}/surface_code-1-quantinuum-emulate
      3 0 1000 NULL 12 6
      ${CMAKE_BINARY_DIR}/lib
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

add_test(
  NAME app_examples.surface_code-1-quantinuum-emulate-test-distance-5-in-process
  COMMAND
    bash "${CMAKE_CURRENT_SOURCE_DIR}/surface_code-1-test.sh"
      ${CMAKE_CURRENT_BINARY_DIR}/surface_code-1-local
      ${CMAKE_CURRENT_BINARY_DIR}/surface_code-1-quantinuum-emulate
      5 0 0 NULL 20 10
      ${CMAKE_BINARY_DIR}/lib
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)


# End of surface code 1, --target quantinuum --emulate
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
# Surface code 2, fully local, regular Stim simulator
add_executable(surface_code-2-local)
set_target_properties(surface_code-2-local PROPERTIES LINKER_LANGUAGE CXX)

cudaqx_add_device_code(surface_code-2-local
  SOURCES
    surface_code-2.cpp
  COMPILER_FLAGS
    --target stim
)

target_link_directories(surface_code-2-local
  PRIVATE ${CUDAQ_INSTALL_DIR}/lib)

target_link_libraries(surface_code-2-local
  PRIVATE
    cudaq-qec
    cudaq-qec-realtime-decoding
    cudaq-qec-realtime-decoding-simulation
    cudaq::cudaq cudaq::cudaq-common cudaq-mlir-runtime
    nvqir nvqir-stim)
qec_realtime_app_link_options(surface_code-2-local)

add_test(
  NAME app_examples.surface_code-2-local-test-distance-3
  COMMAND
    bash "${CMAKE_CURRENT_SOURCE_DIR}/surface_code-2-test.sh"
      ${CMAKE_CURRENT_BINARY_DIR}/surface_code-2-local
      ${CMAKE_CURRENT_BINARY_DIR}/surface_code-2-local
      3 10 10 NULL
      ${CMAKE_BINARY_DIR}/lib
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
# This must be disabled for now because the multi_error_lut decoder is not
# powerful enough to pass this test. The nv-qldpc-decoder can pass this test,
# but that is not available on the GitHub repo.
# add_test(
#   NAME app_examples.surface_code-2-local-test-distance-5
#   COMMAND
#     bash "${CMAKE_CURRENT_SOURCE_DIR}/surface_code-2-test.sh"
#       ${CMAKE_CURRENT_BINARY_DIR}/surface_code-2-local
#       ${CMAKE_CURRENT_BINARY_DIR}/surface_code-2-local
#       5 10 10 NULL
#       ${CMAKE_BINARY_DIR}/lib
#   WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
# )

# End of surface code 2, fully local, regular Stim simulator
#------------------------------------------------------------------------------#


#------------------------------------------------------------------------------#
# Surface code 2, --target quantinuum --emulate
add_executable(surface_code-2-quantinuum-emulate)
set_target_properties(surface_code-2-quantinuum-emulate PROPERTIES LINKER_LANGUAGE CXX)

add_dependencies(surface_code-2-quantinuum-emulate surface_code-2-local)

cudaqx_add_device_code(surface_code-2-quantinuum-emulate
  SOURCES
    surface_code-2.cpp
  COMPILER_FLAGS
    --target quantinuum --emulate --quantinuum-machine Helios-Fake -DMANUALLY_INJECT_ERRORS=1
  DEPENDS_ON
    # Must wait for the local executable to be built because it has the same
    # surface_code-2.cpp file, and the nvq++ compiler will use the same filename
    # prefix (surface_code-2) for intermediate files. They will stomp on each
    # other if you don't add this.
    surface_code-2-local
)

target_link_directories(surface_code-2-quantinuum-emulate
  PRIVATE ${CUDAQ_INSTALL_DIR}/lib)

# Add -Wl,--export-dynamic to the link line.
target_link_options(surface_code-2-quantinuum-emulate PRIVATE -Wl,--export-dynamic)
target_link_libraries(surface_code-2-quantinuum-emulate
  PRIVATE
    cudaq-qec
    cudaq-qec-realtime-decoding
    cudaq-qec-realtime-decoding-quantinuum
    cudaq::cudaq cudaq::cudaq-common cudaq-mlir-runtime cudaq-rest-qpu
    nvqir nvqir-stim)
qec_realtime_app_link_options(surface_code-2-quantinuum-emulate)

add_test(
  NAME app_examples.surface_code-2-quantinuum-emulate-test-distance-3-in-process
  COMMAND
    bash "${CMAKE_CURRENT_SOURCE_DIR}/surface_code-2-test.sh"
      ${CMAKE_CURRENT_BINARY_DIR}/surface_code-2-local
      ${CMAKE_CURRENT_BINARY_DIR}/surface_code-2-quantinuum-emulate
      3 0 1000 NULL
      ${CMAKE_BINARY_DIR}/lib
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

add_test(
  NAME app_examples.surface_code-2-quantinuum-emulate-test-distance-5-in-process
  COMMAND
    bash "${CMAKE_CURRENT_SOURCE_DIR}/surface_code-2-test.sh"
      ${CMAKE_CURRENT_BINARY_DIR}/surface_code-2-local
      ${CMAKE_CURRENT_BINARY_DIR}/surface_code-2-quantinuum-emulate
      5 0 0 NULL
      ${CMAKE_BINARY_DIR}/lib
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)


# End of surface code 2, --target quantinuum --emulate
#------------------------------------------------------------------------------#


macro(add_surface_code_test base_name exe_local exe_remote dist server)
  # For now, disable 0.015 error rate because the multi_error_lut decoder is not
  # powerful enough to pass this test. The nv-qldpc-decoder can pass this test,
  # but that is not available on the GitHub repo.
  foreach(err_rate IN ITEMS 0.010)
    foreach(prep IN ITEMS prep0 prepp)
      add_test(
        NAME ${base_name}-dist-${dist}-${prep}-err-${err_rate}
        COMMAND
          bash "${CMAKE_CURRENT_SOURCE_DIR}/surface_code-3-test.sh"
            ${exe_local}
            ${exe_remote}
            ${dist} ${server} ${err_rate} "${prep}"
            ${CMAKE_BINARY_DIR}/quantinuum_decoding_server/lib
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
      )
    endforeach()
  endforeach()
endmacro()

#------------------------------------------------------------------------------#
# Surface code 3, fully local, regular Stim simulator
add_executable(surface_code-3-local)
set_target_properties(surface_code-3-local PROPERTIES LINKER_LANGUAGE CXX)

cudaqx_add_device_code(surface_code-3-local
  SOURCES
    surface_code-3.cpp
  COMPILER_FLAGS
    --target stim
)

target_link_directories(surface_code-3-local
  PRIVATE ${CUDAQ_INSTALL_DIR}/lib)

target_link_libraries(surface_code-3-local
  PRIVATE
    cudaq-qec
    cudaq-qec-realtime-decoding
    cudaq-qec-realtime-decoding-simulation
    cudaq::cudaq cudaq::cudaq-common cudaq-mlir-runtime
    nvqir nvqir-stim)
qec_realtime_app_link_options(surface_code-3-local)

# --- Local tests ---
add_surface_code_test(
  app_examples.surface_code-3-local-test
  ${CMAKE_CURRENT_BINARY_DIR}/surface_code-3-local
  ${CMAKE_CURRENT_BINARY_DIR}/surface_code-3-local
  3 NULL
)
add_surface_code_test(
  app_examples.surface_code-3-local-test
  ${CMAKE_CURRENT_BINARY_DIR}/surface_code-3-local
  ${CMAKE_CURRENT_BINARY_DIR}/surface_code-3-local
  5 NULL
)

# End of surface code 3, fully local, regular Stim simulator
#------------------------------------------------------------------------------#


#------------------------------------------------------------------------------#
# Surface code 3, --target quantinuum --emulate
add_executable(surface_code-3-quantinuum-emulate)
set_target_properties(surface_code-3-quantinuum-emulate PROPERTIES LINKER_LANGUAGE CXX)

add_dependencies(surface_code-3-quantinuum-emulate surface_code-3-local)

cudaqx_add_device_code(surface_code-3-quantinuum-emulate
  SOURCES
    surface_code-3.cpp
  COMPILER_FLAGS
    --target quantinuum --emulate --quantinuum-machine Helios-Fake -DMANUALLY_INJECT_ERRORS=1
  DEPENDS_ON
    # Must wait for the local executable to be built because it has the same
    # surface_code-3.cpp file, and the nvq++ compiler will use the same filename
    # prefix (surface_code-3) for intermediate files. They will stomp on each
    # other if you don't add this.
    surface_code-3-local
)

target_link_directories(surface_code-3-quantinuum-emulate
  PRIVATE ${CUDAQ_INSTALL_DIR}/lib)

# Add -Wl,--export-dynamic to the link line.
target_link_options(surface_code-3-quantinuum-emulate PRIVATE -Wl,--export-dynamic)
target_link_libraries(surface_code-3-quantinuum-emulate
  PRIVATE
    cudaq-qec
    cudaq-qec-realtime-decoding
    cudaq-qec-realtime-decoding-quantinuum
    cudaq::cudaq cudaq::cudaq-common cudaq-mlir-runtime cudaq-rest-qpu
    nvqir nvqir-stim)
qec_realtime_app_link_options(surface_code-3-quantinuum-emulate)


    # --- Quantinuum emulate (in-process) ---
add_surface_code_test(
  app_examples.surface_code-3-quantinuum-emulate-test-in-process
  ${CMAKE_CURRENT_BINARY_DIR}/surface_code-3-local
  ${CMAKE_CURRENT_BINARY_DIR}/surface_code-3-quantinuum-emulate
  3 NULL
)
add_surface_code_test(
  app_examples.surface_code-3-quantinuum-emulate-test-in-process
  ${CMAKE_CURRENT_BINARY_DIR}/surface_code-3-local
  ${CMAKE_CURRENT_BINARY_DIR}/surface_code-3-quantinuum-emulate
  5 NULL
)

# End of surface code 3, --target quantinuum --emulate
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
# Surface code 4: YAML-switchable realtime decoders (pymatching / trt+pymatching
# / nv-qldpc relay-bp). The decoder is selected at realtime from the loaded YAML
# config (--yaml).
add_executable(surface_code-4-yaml)
set_target_properties(surface_code-4-yaml PROPERTIES LINKER_LANGUAGE CXX)

cudaqx_add_device_code(surface_code-4-yaml
  SOURCES
    surface_code-4-yaml.cpp
  COMPILER_FLAGS
    --target stim
)

target_link_directories(surface_code-4-yaml
  PRIVATE ${CUDAQ_INSTALL_DIR}/lib)

target_link_libraries(surface_code-4-yaml
  PRIVATE
    cudaq-qec
    cudaq-qec-realtime-decoding
    cudaq-qec-realtime-decoding-simulation
    cudaq::cudaq cudaq::cudaq-common cudaq-mlir-runtime
    nvqir nvqir-stim)
qec_realtime_app_link_options(surface_code-4-yaml)

if(TARGET cudaq-qec-pymatching)
  add_dependencies(surface_code-4-yaml cudaq-qec-pymatching)
endif()
if(TARGET cudaq-qec-trt-decoder)
  add_dependencies(surface_code-4-yaml cudaq-qec-trt-decoder)
endif()

# Surface-code-4 YAML tests: exercise both d=3 and d=5 for each decoder entry.
# All entries run through the realtime inproc RPC mode; relay-BP is registered
# only when the private cudevice/archive and external decoder plugin are
# configured.
find_package(Python COMPONENTS Interpreter QUIET)
set(_SC4_TRY_TRT_TEST FALSE)
if(Python_Interpreter_FOUND)
  execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import onnx"
    RESULT_VARIABLE _sc4_onnx_python_result
    OUTPUT_QUIET
    ERROR_QUIET)
  if(_sc4_onnx_python_result EQUAL 0)
    set(_SC4_TRY_TRT_TEST TRUE)
  endif()
endif()

function(add_surface_code_4_yaml_test test_suffix distance num_rounds
         decoder_type num_shots onnx_path)
  set(_sc4_env "CUDAQ_QEC_REALTIME_MODE=inproc_rpc")
  if(Python_Interpreter_FOUND)
    list(APPEND _sc4_env "PYTHON=${Python_EXECUTABLE}")
  endif()

  add_test(
    NAME app_examples.surface_code-4-yaml-${test_suffix}
    COMMAND
      bash "${CMAKE_CURRENT_SOURCE_DIR}/surface_code-4-yaml-test.sh"
        ${CMAKE_CURRENT_BINARY_DIR}/surface_code-4-yaml
        ${distance} ${num_rounds}
        ${decoder_type} ${num_shots} "${onnx_path}" ${ARGN}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )
  set_tests_properties(
    app_examples.surface_code-4-yaml-${test_suffix}
    PROPERTIES ENVIRONMENT "${_sc4_env}")
endfunction()

# This example decodes ONE volume of num_rounds rounds (no sliding windows), so
# each case runs num_rounds == distance rounds.
function(add_surface_code_4_yaml_decoder_cases label decoder_type onnx_path)
  add_surface_code_4_yaml_test(${label}-d3 3 3
    ${decoder_type} 200 "${onnx_path}" ${ARGN})
  add_surface_code_4_yaml_test(${label}-d5 5 5
    ${decoder_type} 200 "${onnx_path}" ${ARGN})
endfunction()

add_surface_code_4_yaml_decoder_cases(pymatching pymatching "")

# TensorRT predecoder + PyMatching global decoder. The shell driver generates a
# tiny ONNX model sized for each test case: output [pre_L=0, residual], so TRT
# preserves the syndrome and PyMatching performs the correction.
if(TARGET cudaq-qec-trt-decoder AND _SC4_TRY_TRT_TEST)
  add_surface_code_4_yaml_decoder_cases(trt-pymatching trt_decoder AUTO)
endif()

# nv-qldpc relay-BP requires both the external nv-qldpc decoder plugin and the
# proprietary cudevice archive. Match the other realtime graph/inproc tests:
# register only when those dependencies are configured; do not runtime-skip.
if(_APP_CUDAQ_REALTIME_DISPATCH_LIB
   AND TARGET cudaq-qec-realtime-cudevice-proprietary
   AND (QEC_EXTERNAL_DECODERS OR DEFINED ENV{QEC_EXTERNAL_DECODERS}))
  add_surface_code_4_yaml_decoder_cases(relay-bp nv-qldpc-decoder ""
    --use-relay-bp)
endif()

# Relaxed round rule (T=6 not a multiple of d=5) and multi-logical patches, each
# through the full generate + reload + decode driver -- so no hand-written
# generate/reload duplication, and both get the driver's residual-LER check.
add_surface_code_4_yaml_test(d5t6 5 6 pymatching 20 "")
add_surface_code_4_yaml_test(multilogical 3 3 pymatching 20 "" --num_logical 2)

# --------------------------------------------------------------------------- #
# Negative / boundary tests: pin the validation behavior (distance/round guards,
# YAML-to-geometry binding, X-basis-bundle rejection, replay guards including
# structurally-incomplete captures, and multi-logical reload) so it cannot
# silently regress. Native ctests: PASS_REGULAR_EXPRESSION asserts the specific
# diagnostic (not merely a nonzero exit), and FIXTURES_SETUP makes a failed
# fixture fail its dependents. Self-contained -- no external Ising artifacts or
# ONNX models required.
# --------------------------------------------------------------------------- #
set(_sc4_exe ${CMAKE_CURRENT_BINARY_DIR}/surface_code-4-yaml)
set(_sc4_neg_dir ${CMAKE_CURRENT_BINARY_DIR}/negative)
file(MAKE_DIRECTORY ${_sc4_neg_dir})
set(_sc4_neg_env "CUDAQ_QEC_REALTIME_MODE=inproc_rpc"
                 "CUDAQ_DEFAULT_SIMULATOR=stim")

# A rejection case must satisfy BOTH conditions: a nonzero exit AND the expected
# diagnostic. CTest's PASS_REGULAR_EXPRESSION ignores the exit code, so a wrapper
# enforces both -- a future regression that prints the message but returns 0
# then fails the test. The bash script is a literal bracket argument; the
# diagnostic, executable, and app args are passed as positionals so the
# diagnostic (which may contain quotes or leading dashes) needs no escaping.
function(_sc4_reject name diagnostic)
  add_test(NAME app_examples.surface_code-4-yaml-negative-${name}
    COMMAND bash -c [==[
diag=$1; shift
if out=$("$@" 2>&1); then
  printf '%s\n' "$out"
  echo "NEGATIVE TEST FAIL: command returned 0; expected a nonzero exit"
  exit 1
fi
printf '%s\n' "$out"
printf '%s' "$out" | grep -qF -- "$diag" || {
  echo "NEGATIVE TEST FAIL: missing expected diagnostic: $diag"
  exit 1
}
]==] _ "${diagnostic}" ${_sc4_exe} ${ARGN}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
  set_tests_properties(app_examples.surface_code-4-yaml-negative-${name}
    PROPERTIES ENVIRONMENT "${_sc4_neg_env}")
endfunction()

# Positive-path assertion: the command must exit 0 AND print the expected
# marker. Uses a wrapper for the same reason as _sc4_reject -- CTest's
# PASS_REGULAR_EXPRESSION ignores the exit code, and here we want both.
function(_sc4_accept name marker)
  add_test(NAME app_examples.surface_code-4-yaml-${name}
    COMMAND bash -c [==[
mark=$1; shift
if ! out=$("$@" 2>&1); then
  printf '%s\n' "$out"
  echo "POSITIVE TEST FAIL: command returned nonzero; expected success"
  exit 1
fi
printf '%s\n' "$out"
printf '%s' "$out" | grep -qF -- "$mark" || {
  echo "POSITIVE TEST FAIL: missing expected output: $mark"
  exit 1
}
]==] _ "${marker}" ${_sc4_exe} ${ARGN}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
  set_tests_properties(app_examples.surface_code-4-yaml-${name}
    PROPERTIES ENVIRONMENT "${_sc4_neg_env}")
endfunction()

# distance guard: an even or too-small distance is rejected (one guard, one
# diagnostic) before code construction; d=4 covers it.
_sc4_reject(distance-even "distance must be an odd integer"
  --distance 4 --num_rounds 4 --decoder_type pymatching
  --save_dem ${_sc4_neg_dir}/even.yml)

# rounds guard: a single round has no cross-round detectors, so require >= 2.
_sc4_reject(rounds-too-few "must be >= 2"
  --distance 3 --num_rounds 1 --decoder_type pymatching
  --save_dem ${_sc4_neg_dir}/t1.yml)

# replay requires --yaml.
_sc4_reject(replay-needs-yaml "load_syndrome requires --yaml"
  --load_syndrome ${_sc4_neg_dir}/none.txt --distance 3 --num_rounds 3)

# X-basis Ising bundle rejected (basis must be Z). Fabricate just the
# metadata.txt the loader inspects; the dummy ONNX is never opened because the
# basis check throws first.
file(WRITE ${_sc4_neg_dir}/xbundle/metadata.txt
  "distance=3\nn_rounds=3\nbasis=X\ncode_rotation=XV\n")
_sc4_reject(xbasis-bundle "basis='X'"
  --distance 3 --num_rounds 3 --decoder_type trt_decoder
  --onnx_path ${_sc4_neg_dir}/dummy.onnx --ising_bundle ${_sc4_neg_dir}/xbundle
  --save_dem ${_sc4_neg_dir}/xb.yml)

# Geometry binding: a d5/T5 config rejected when loaded as d3/T15 (same 120
# detectors, different measurement-buffer span).
add_test(NAME app_examples.surface_code-4-yaml-negative-d5t5-generate
  COMMAND ${_sc4_exe} --distance 5 --num_rounds 5 --num_shots 20 --p_spam 0.01
    --decoder_type pymatching --save_dem ${_sc4_neg_dir}/d5t5.yml
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
set_tests_properties(app_examples.surface_code-4-yaml-negative-d5t5-generate
  PROPERTIES FIXTURES_SETUP sc4_d5t5 ENVIRONMENT "${_sc4_neg_env}")
_sc4_reject(geometry-mismatch "measurement-buffer span"
  --yaml ${_sc4_neg_dir}/d5t5.yml --distance 3 --num_rounds 15
  --num_shots 20 --p_spam 0.01)
set_tests_properties(app_examples.surface_code-4-yaml-negative-geometry-mismatch
  PROPERTIES FIXTURES_REQUIRED sc4_d5t5)

# Replay structural integrity: build a valid d3 capture, then derive a truncated
# (footer dropped) and a corrections-flipped copy. Replaying a missing,
# truncated, or correction-mismatched capture must fail with a diagnostic.
add_test(NAME app_examples.surface_code-4-yaml-negative-replay-fixtures
  COMMAND bash -c "set -e
'${_sc4_exe}' --distance 3 --num_rounds 3 --num_shots 5 --p_spam 0.01 --decoder_type pymatching --save_dem '${_sc4_neg_dir}/d3.yml'
'${_sc4_exe}' --yaml '${_sc4_neg_dir}/d3.yml' --distance 3 --num_rounds 3 --num_shots 5 --p_spam 0.01 --save_syndrome '${_sc4_neg_dir}/cap.txt'
sed '/^CORRECTIONS_START/,$d' '${_sc4_neg_dir}/cap.txt' > '${_sc4_neg_dir}/cap_trunc.txt'
sed '/CORRECTIONS_START/,/CORRECTIONS_END/{s/^0$/X/;s/^1$/0/;s/^X$/1/}' '${_sc4_neg_dir}/cap.txt' > '${_sc4_neg_dir}/cap_mismatch.txt'"
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
set_tests_properties(app_examples.surface_code-4-yaml-negative-replay-fixtures
  PROPERTIES FIXTURES_SETUP sc4_replay ENVIRONMENT "${_sc4_neg_env}")

_sc4_reject(replay-missing-file "Could not open syndrome file"
  --yaml ${_sc4_neg_dir}/d3.yml --load_syndrome ${_sc4_neg_dir}/missing.txt
  --distance 3 --num_rounds 3)
set_tests_properties(app_examples.surface_code-4-yaml-negative-replay-missing-file
  PROPERTIES FIXTURES_REQUIRED sc4_replay)
_sc4_reject(replay-truncated "missing the CORRECTIONS_START/CORRECTIONS_END"
  --yaml ${_sc4_neg_dir}/d3.yml --load_syndrome ${_sc4_neg_dir}/cap_trunc.txt
  --distance 3 --num_rounds 3 --num_shots 5)
set_tests_properties(app_examples.surface_code-4-yaml-negative-replay-truncated
  PROPERTIES FIXTURES_REQUIRED sc4_replay)
_sc4_reject(replay-correction-mismatch "replay correction mismatch"
  --yaml ${_sc4_neg_dir}/d3.yml --load_syndrome ${_sc4_neg_dir}/cap_mismatch.txt
  --distance 3 --num_rounds 3 --num_shots 5)
set_tests_properties(app_examples.surface_code-4-yaml-negative-replay-correction-mismatch
  PROPERTIES FIXTURES_REQUIRED sc4_replay)

# Positive replay: the untampered capture must round-trip -- every replayed
# correction matches the captured run. Proves the load/replay path works on good
# input, not just that it rejects tampered captures.
_sc4_accept(replay-roundtrip "SUCCESS: All corrections match"
  --yaml ${_sc4_neg_dir}/d3.yml --load_syndrome ${_sc4_neg_dir}/cap.txt
  --distance 3 --num_rounds 3 --num_shots 5)
set_tests_properties(app_examples.surface_code-4-yaml-replay-roundtrip
  PROPERTIES FIXTURES_REQUIRED sc4_replay)
