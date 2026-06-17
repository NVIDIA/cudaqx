#!/usr/bin/env python3
# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import argparse
from pathlib import Path

import onnx
from onnx import TensorProto, helper


def make_float_tensor(name, shape, values):
    return helper.make_tensor(name, TensorProto.FLOAT, shape, values)


def make_int64_tensor(name, shape, values):
    return helper.make_tensor(name, TensorProto.INT64, shape, values)


def build_predecoder_onnx(output_path, num_detectors, num_observables,
                          pre_l_detector_index):
    output_size = num_observables + num_detectors
    x = helper.make_tensor_value_info("detectors", TensorProto.FLOAT,
                                      [1, num_detectors])
    y = helper.make_tensor_value_info("preL_residual", TensorProto.FLOAT,
                                      [1, output_size])

    initializers = [
        make_int64_tensor("pre_l_detector_index", [1], [pre_l_detector_index]),
        make_float_tensor("pre_l_mask", [1, num_observables],
                          [0.0] * num_observables),
        make_float_tensor("residual_mask", [1, num_detectors],
                          [1.0] * num_detectors),
    ]

    nodes = [
        helper.make_node("Gather", ["detectors", "pre_l_detector_index"],
                         ["selected_detector"],
                         axis=1),
        helper.make_node("Mul", ["selected_detector", "pre_l_mask"], ["pre_L"]),
        helper.make_node("Mul", ["detectors", "residual_mask"], ["residual"]),
        helper.make_node("Concat", ["pre_L", "residual"], ["preL_residual"],
                         axis=1),
    ]

    graph = helper.make_graph(nodes,
                              "trt_split_predecoder", [x], [y],
                              initializer=initializers)
    model = helper.make_model(graph,
                              opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Generate a tiny TRT predecoder ONNX with "
        "[pre_L | residual] output.")
    parser.add_argument("--num-detectors", type=int, required=True)
    parser.add_argument("--num-observables", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--pre-l-detector-index", type=int, default=0)
    args = parser.parse_args()

    if args.num_detectors <= 0:
        raise RuntimeError("--num-detectors must be positive.")
    if args.num_observables <= 0:
        raise RuntimeError("--num-observables must be positive.")
    if not 0 <= args.pre_l_detector_index < args.num_detectors:
        raise RuntimeError("--pre-l-detector-index is out of range.")

    build_predecoder_onnx(Path(args.out), args.num_detectors,
                          args.num_observables, args.pre_l_detector_index)
    print(f"Wrote split predecoder: in[{args.num_detectors}] -> "
          f"out[{args.num_observables + args.num_detectors}] at {args.out}")


if __name__ == "__main__":
    main()
