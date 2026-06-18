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


def sparse_rows(values):
    rows = []
    row = []
    for value in values:
        if value == -1:
            rows.append(row)
            row = []
        else:
            row.append(value)
    if row:
        rows.append(row)
    return rows


def select_decoder(config, decoder_id):
    decoders = config.get("decoders", [])
    if not decoders:
        raise RuntimeError("Config does not contain any decoders.")

    if decoder_id is None:
        if len(decoders) != 1:
            raise RuntimeError("Config contains multiple decoders; pass "
                               "--decoder-id to choose one.")
        return decoders[0]

    for decoder in decoders:
        if decoder.get("id") == decoder_id:
            return decoder
    raise RuntimeError(f"Decoder id {decoder_id} not found in config.")


def find_boundary_absorption(decoder_config):
    block_size = decoder_config["block_size"]
    h_rows = sparse_rows(decoder_config["H_sparse"])
    o_rows = sparse_rows(decoder_config.get("O_sparse", []))

    col_to_detector_rows = [[] for _ in range(block_size)]
    for row_idx, row in enumerate(h_rows):
        for col_idx in row:
            if 0 <= col_idx < block_size:
                col_to_detector_rows[col_idx].append(row_idx)

    for col_idx, detector_rows in enumerate(col_to_detector_rows):
        if len(detector_rows) != 1:
            continue
        obs_bits = [1.0 if col_idx in row else 0.0 for row in o_rows]
        if any(obs_bits):
            return detector_rows[0], obs_bits, col_idx

    return None


def make_float_tensor(name, shape, values):
    return helper.make_tensor(name, TensorProto.FLOAT, shape, values)


def make_int64_tensor(name, shape, values):
    return helper.make_tensor(name, TensorProto.INT64, shape, values)


def build_predecoder_onnx(output_path,
                          num_detectors,
                          num_observables,
                          pre_l_detector_index,
                          pre_l_mask=None,
                          residual_mask=None):
    output_size = num_observables + num_detectors
    if pre_l_mask is None:
        pre_l_mask = [0.0] * num_observables
    if residual_mask is None:
        residual_mask = [1.0] * num_detectors

    x = helper.make_tensor_value_info("detectors", TensorProto.FLOAT,
                                      [1, num_detectors])
    y = helper.make_tensor_value_info("preL_residual", TensorProto.FLOAT,
                                      [1, output_size])

    initializers = [
        make_int64_tensor("pre_l_detector_index", [1], [pre_l_detector_index]),
        make_float_tensor("pre_l_mask", [1, num_observables], pre_l_mask),
        make_float_tensor("residual_mask", [1, num_detectors], residual_mask),
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
    parser.add_argument("--config",
                        help="Optional surface_code-1 YAML config generated "
                        "by --save_dem. If provided, model dimensions are read "
                        "from the selected decoder config.")
    parser.add_argument("--decoder-id", type=int, default=None)
    parser.add_argument("--identity",
                        action="store_true",
                        help="With --config, force pure identity: pre_L=0, "
                        "residual=input.")
    parser.add_argument("--num-detectors", type=int)
    parser.add_argument("--num-observables", type=int)
    parser.add_argument("--out", required=True)
    parser.add_argument("--pre-l-detector-index", type=int, default=0)
    args = parser.parse_args()

    pre_l_detector_index = args.pre_l_detector_index
    pre_l_mask = None
    residual_mask = None
    detail = "pre_L=0, residual=input"
    mode = "identity"

    if args.config:
        import yaml
        with open(args.config, "r", encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file)
        decoder_config = select_decoder(config, args.decoder_id)
        args.num_detectors = decoder_config["syndrome_size"]
        args.num_observables = len(
            sparse_rows(decoder_config.get("O_sparse", [])))

        absorption = None if args.identity else find_boundary_absorption(
            decoder_config)
        if absorption is not None:
            detector_idx, obs_bits, col_idx = absorption
            pre_l_detector_index = detector_idx
            pre_l_mask = obs_bits
            residual_mask = [1.0] * args.num_detectors
            residual_mask[detector_idx] = 0.0
            mode = "near-identity"
            detail = (f"absorbed boundary column {col_idx}: detector "
                      f"{detector_idx}, observable mask "
                      f"{[int(bit) for bit in obs_bits]}")

    if args.num_detectors is None or args.num_observables is None:
        raise RuntimeError("--config or both --num-detectors and "
                           "--num-observables are required.")
    if args.num_detectors <= 0:
        raise RuntimeError("--num-detectors must be positive.")
    if args.num_observables <= 0:
        raise RuntimeError("--num-observables must be positive.")
    if not 0 <= pre_l_detector_index < args.num_detectors:
        raise RuntimeError("--pre-l-detector-index is out of range.")

    build_predecoder_onnx(Path(args.out), args.num_detectors,
                          args.num_observables, pre_l_detector_index,
                          pre_l_mask, residual_mask)
    print(f"Wrote {mode} split predecoder: in[{args.num_detectors}] -> "
          f"out[{args.num_observables + args.num_detectors}] ({detail}) "
          f"at {args.out}")


if __name__ == "__main__":
    main()
