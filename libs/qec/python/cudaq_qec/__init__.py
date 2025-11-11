# ============================================================================ #
# Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

from .patch import patch
from ._pycudaqx_qec_the_suffix_matters_cudaq_qec import *

__version__ = qecrt.__version__
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
DetectorErrorModel = qecrt.DetectorErrorModel
generate_random_bit_flips = qecrt.generate_random_bit_flips
sample_memory_circuit = qecrt.sample_memory_circuit
sample_code_capacity = qecrt.sample_code_capacity
dem_from_memory_circuit = qecrt.dem_from_memory_circuit
x_dem_from_memory_circuit = qecrt.x_dem_from_memory_circuit
z_dem_from_memory_circuit = qecrt.z_dem_from_memory_circuit

dump_pcm = qecrt.dump_pcm
generate_random_pcm = qecrt.generate_random_pcm
get_pcm_for_rounds = qecrt.get_pcm_for_rounds
get_sorted_pcm_column_indices = qecrt.get_sorted_pcm_column_indices
pcm_is_sorted = qecrt.pcm_is_sorted
reorder_pcm_columns = qecrt.reorder_pcm_columns
shuffle_pcm_columns = qecrt.shuffle_pcm_columns
simplify_pcm = qecrt.simplify_pcm
sort_pcm_columns = qecrt.sort_pcm_columns
pcm_extend_to_n_rounds = qecrt.pcm_extend_to_n_rounds
compute_msm = qecrt.compute_msm
construct_mz_table = qecrt.construct_mz_table
generate_timelike_sparse_detector_matrix = qecrt.generate_timelike_sparse_detector_matrix
pcm_to_sparse_vec = qecrt.pcm_to_sparse_vec

multi_decoder_config = qecrt.config.multi_decoder_config
decoder_config = qecrt.config.decoder_config
nv_qldpc_decoder_config = qecrt.config.nv_qldpc_decoder_config
multi_error_lut_config = qecrt.config.multi_error_lut_config
configure_decoders_from_file = qecrt.config.configure_decoders_from_file
configure_decoders_from_str = qecrt.config.configure_decoders_from_str
finalize_decoders = qecrt.config.finalize_decoders
configure_decoders = qecrt.config.configure_decoders

stabilizer_grid = qecrt.stabilizer_grid
role_to_str = qecrt.role_to_str

from .plugins import decoders, codes
import pkgutil, importlib, traceback


def iter_namespace(ns_pkg):
    return pkgutil.iter_modules(ns_pkg.__path__, ns_pkg.__name__ + ".")


for finder, name, ispkg in iter_namespace(plugins.decoders):
    try:
        importlib.import_module(name)
    except (ModuleNotFoundError, ImportError) as e:
        pass

for finder, name, ispkg in iter_namespace(plugins.codes):
    try:
        importlib.import_module(name)
    except (ModuleNotFoundError, ImportError) as e:
        pass

import cudaq
from .loader import qec_set_target_callback

cudaq.register_set_target_callback(qec_set_target_callback, "cudaq_qec")
