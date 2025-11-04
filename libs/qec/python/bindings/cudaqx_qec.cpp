/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "py_code.h"
#include "py_decoder.h"
#include "py_decoding.h"
#include "py_decoding_config.h"
#include "py_surface_code.h"

#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

PYBIND11_MODULE(_pycudaqx_qec_the_suffix_matters_cudaq_qec, mod) {
  mod.doc() = "Python bindings for the CUDA-Q QEC Libraries.";
  cudaq::qec::bindCode(mod);
  cudaq::qec::bindDecoder(mod);
  cudaq::qec::decoding::config::bindDecodingConfig(mod);
  cudaq::qec::decoding::bindDecoding(mod);
  cudaq::qec::surface_code::bindSurfaceCode(mod);
}
