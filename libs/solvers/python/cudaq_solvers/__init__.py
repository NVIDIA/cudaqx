# ============================================================================ #
# Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

try:
    from ._pycudaqx_solvers_the_suffix_matters_cudaq_solvers import *
    from ._pycudaqx_solvers_the_suffix_matters_cudaq_solvers import __version__
except ImportError as _e:
    from ._system_dep_check import raise_if_missing_system_dep
    raise_if_missing_system_dep(_e, "cudaq-solvers")
    raise
try:
    from .gqe_algorithm.gqe import gqe
except ImportError:

    def gqe(*args, **kwargs):
        raise ImportError(
            "Failed to load GQE solver due to missing dependencies. "
            "Recommend installing the required dependencies with: "
            "'pip install cudaq-solvers[gqe]'")
