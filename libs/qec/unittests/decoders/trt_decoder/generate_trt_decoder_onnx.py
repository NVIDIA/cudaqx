#!/usr/bin/env python3
# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import argparse
import os

import onnx
from onnx import TensorProto, helper


def make_identity_model(output_path, name, elem_type, shape):
    input_info = helper.make_tensor_value_info("input", elem_type, shape)
    output_info = helper.make_tensor_value_info("output", elem_type, shape)
    identity = helper.make_node("Identity", ["input"], ["output"])
    graph = helper.make_graph([identity], name, [input_info], [output_info])
    model = helper.make_model(graph,
                              opset_imports=[helper.make_opsetid("", 19)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    onnx.save(model, output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Generate small ONNX fixtures for trt_decoder tests.")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    make_identity_model(
        os.path.join(args.out_dir, "trt_dynamic_identity.onnx"),
        "trt_dynamic_identity",
        TensorProto.FLOAT,
        ["batch", 3],
    )
    make_identity_model(
        os.path.join(args.out_dir, "trt_uint8_identity.onnx"),
        "trt_uint8_identity",
        TensorProto.UINT8,
        [1, 3],
    )


if __name__ == "__main__":
    main()
