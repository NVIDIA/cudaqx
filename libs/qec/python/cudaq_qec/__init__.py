# ============================================================================ #
# Copyright (c) 2024 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

from .patch import patch
from ._pycudaqx_qec_the_suffix_matters_cudaq_qec import *

code = qecrt.code
Code = qecrt.Code
decoder = qecrt.decoder
Decoder = qecrt.Decoder
TwoQubitDepolarization = qecrt.TwoQubitDepolarization
TwoQubitBitFlip = qecrt.TwoQubitBitFlip
operation = qecrt.operation
get_code = qecrt.get_code
get_available_codes = qecrt.get_available_codes
get_decoder = qecrt.get_decoder
DecoderResult = qecrt.DecoderResult
generate_random_bit_flips = qecrt.generate_random_bit_flips
sample_memory_circuit = qecrt.sample_memory_circuit
sample_code_capacity = qecrt.sample_code_capacity

from .plugins import decoders, codes
import pkgutil, importlib, traceback


def iter_namespace(ns_pkg):
    return pkgutil.iter_modules(ns_pkg.__path__, ns_pkg.__name__ + ".")


for finder, name, ispkg in iter_namespace(plugins.decoders):
    try:
        importlib.import_module(name)
    except ModuleNotFoundError:
        pass

for finder, name, ispkg in iter_namespace(plugins.codes):
    try:
        importlib.import_module(name)
    except ModuleNotFoundError as e:
        pass
