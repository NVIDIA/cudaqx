# ============================================================================ #
# Copyright (c) 2024 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
import cudaq_qec as qec
import numpy as np


@qec.decoder("example_byod")
class ExampleDecoder:

    def __init__(self, H, **kwargs):
        qec.Decoder.__init__(self, H)
        self.H = H
        if 'weights' in kwargs:
            print(kwargs['weights'])

    def decode(self, syndrome):
        res = qec.DecoderResult()
        res.converged = True
        res.result = np.random.random(len(syndrome)).tolist()
        res.opt_results = None
        return res
