# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Rebuild the independent reference color-code detector error models."""

import argparse
from pathlib import Path

import cudaq

from cudaq_qec.plugins.codes.color_code import ColorCodeGeometry

DISTANCE = 5
ROUNDS = 5
NOISE_PROBABILITY = 0.01


class _ReferencePlan:
    """Host-side feedback sets used by the reference detector circuit."""

    def __init__(self, distance: int, rounds: int, basis: str):
        if basis not in ("X", "Z"):
            raise ValueError("basis must be 'X' or 'Z'")
        if rounds < 2:
            raise ValueError("rounds must be >= 2")

        self.grid = ColorCodeGeometry(distance)
        self.rounds = rounds
        self.basis = basis
        grid = self.grid
        num_plaquettes = grid.num_plaquettes
        z_side_data = grid.z_side_data()

        def feedback_set(support):
            support = set(support)
            return [
                a for a, side in enumerate(z_side_data)
                if len(support.intersection(side)) % 2 == 1
            ]

        self.prev_sets = [
            sorted({p} ^ set(feedback_set(grid.plaquettes[p]['data_qubits'])))
            for p in range(num_plaquettes)
        ]
        self.obs_plaquettes = sorted(feedback_set(grid.logical_qubits))


def _kernel_args(plan: _ReferencePlan, p_cx: float):
    """Flatten the reference circuit's variable-sized host data."""
    if p_cx < 0:
        raise ValueError(f"p_cx must be non-negative, got {p_cx}")

    grid = plan.grid
    num_data = grid.num_data
    num_plaquettes = grid.num_plaquettes
    layers = grid._superdense_cnot_layers()
    schedule = [qubit for layer in layers for edge in layer for qubit in edge]
    schedule_offsets = [0]
    for layer in layers:
        schedule_offsets.append(schedule_offsets[-1] + 2 * len(layer))

    prev_stride = max(1, max(len(s) for s in plan.prev_sets))
    prev_flat = []
    prev_counts = []
    for support in plan.prev_sets:
        prev_counts.append(len(support))
        prev_flat.extend(support + [0] * (prev_stride - len(support)))

    support_stride = max(
        len(plaquette['data_qubits']) for plaquette in grid.plaquettes)
    support_flat = []
    support_counts = []
    for plaquette in grid.plaquettes:
        support = list(plaquette['data_qubits'])
        support_counts.append(len(support))
        support_flat.extend(support + [0] * (support_stride - len(support)))

    return (num_data, num_plaquettes, plan.rounds,
            1 if plan.basis == "Z" else 0, float(p_cx), schedule,
            schedule_offsets, prev_flat, prev_counts, prev_stride, support_flat,
            support_counts, support_stride, plan.obs_plaquettes,
            list(grid.logical_qubits))


@cudaq.kernel
def _reference_superdense_memory(
        num_data: int, num_plaquettes: int, num_rounds: int, z_basis: int,
        p_cx: float, schedule: list[int], schedule_offsets: list[int],
        prev_flat: list[int], prev_counts: list[int], prev_stride: int,
        support_flat: list[int], support_counts: list[int], support_stride: int,
        obs_plaquettes: list[int], logical: list[int]):
    """Independent reference circuit with noise on every stabilizer round."""
    qubits = cudaq.qvector(num_data + 2 * num_plaquettes)

    if z_basis == 0:
        for i in range(num_data):
            h(qubits[i])

    for i in range(num_plaquettes):
        h(qubits[num_data + i])
    for layer in range(8):
        for k in range(schedule_offsets[layer], schedule_offsets[layer + 1], 2):
            x.ctrl(qubits[schedule[k]], qubits[schedule[k + 1]])
            if p_cx > 0.0:
                cudaq.apply_noise(cudaq.Depolarization2, p_cx,
                                  qubits[schedule[k]], qubits[schedule[k + 1]])

    z_previous = mz(qubits[num_data + num_plaquettes:num_data +
                           2 * num_plaquettes])
    if z_basis == 1:
        for j in range(len(obs_plaquettes)):
            cudaq.logical_observable(z_previous[obs_plaquettes[j]])
    for i in range(num_plaquettes):
        h(qubits[num_data + i])
    x_previous = mz(qubits[num_data:num_data + num_plaquettes])

    if z_basis == 1:
        for p in range(num_plaquettes):
            cudaq.detector(z_previous[p])
    else:
        for p in range(num_plaquettes):
            cudaq.detector(x_previous[p])

    for round_index in range(1, num_rounds):
        for i in range(num_plaquettes):
            reset(qubits[num_data + i])
            h(qubits[num_data + i])
            reset(qubits[num_data + num_plaquettes + i])
        for layer in range(8):
            for k in range(schedule_offsets[layer], schedule_offsets[layer + 1],
                           2):
                x.ctrl(qubits[schedule[k]], qubits[schedule[k + 1]])
                if p_cx > 0.0:
                    cudaq.apply_noise(cudaq.Depolarization2, p_cx,
                                      qubits[schedule[k]],
                                      qubits[schedule[k + 1]])

        z_current = mz(qubits[num_data + num_plaquettes:num_data +
                              2 * num_plaquettes])
        if z_basis == 1:
            for j in range(len(obs_plaquettes)):
                cudaq.logical_observable(z_current[obs_plaquettes[j]])
        for i in range(num_plaquettes):
            h(qubits[num_data + i])
        x_current = mz(qubits[num_data:num_data + num_plaquettes])

        for p in range(num_plaquettes):
            cudaq.detector(x_current[p], x_previous[p])
        for p in range(num_plaquettes):
            if prev_counts[p] == 0:
                cudaq.detector(z_current[p])
            else:
                previous = [
                    z_previous[prev_flat[p * prev_stride + j]]
                    for j in range(prev_counts[p])
                ]
                cudaq.detector(z_current[p], previous)
        z_previous = z_current
        x_previous = x_current

    if z_basis == 0:
        for i in range(num_data):
            h(qubits[i])
    data_measurements = mz(qubits[0:num_data])

    for p in range(num_plaquettes):
        data_support = [
            data_measurements[support_flat[p * support_stride + j]]
            for j in range(support_counts[p])
        ]
        if z_basis == 1:
            if prev_counts[p] == 0:
                cudaq.detector(data_support)
            else:
                previous = [
                    z_previous[prev_flat[p * prev_stride + j]]
                    for j in range(prev_counts[p])
                ]
                cudaq.detector(data_support, previous)
        else:
            cudaq.detector(x_previous[p], data_support)

    if z_basis == 1:
        for j in range(len(logical)):
            cudaq.logical_observable(data_measurements[logical[j]])
    else:
        cudaq.logical_observable(data_measurements)


def _header() -> str:
    return ("# Reference superdense color-code DEM, d=5, 5 rounds, "
            "DEPOLARIZE2(0.01)\n"
            "# per CX, ALL ROUNDS NOISY (physically stricter convention).\n"
            "# Reproduce from the repository root:\n"
            "# PYTHONPATH=build/python:/usr/local/cudaq python3 "
            "libs/qec/python/tests/generate_color_code_goldens.py "
            "--output-dir libs/qec/python/tests/data\n"
            "# Generator: independent reference superdense memory kernel "
            "(cudaq.dem_from_kernel).\n")


def generate_goldens(output_dir: Path) -> list[Path]:
    """Generate the X- and Z-basis golden DEM files in ``output_dir``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for basis in ("X", "Z"):
        plan = _ReferencePlan(DISTANCE, ROUNDS, basis)
        text = cudaq.dem_from_kernel(_reference_superdense_memory,
                                     *_kernel_args(plan, NOISE_PROBABILITY),
                                     noise_model=cudaq.NoiseModel())
        path = output_dir / (
            f"superdense_golden_d{DISTANCE}_r{ROUNDS}_allnoisy_{basis}.dem")
        path.write_text(_header() + text)
        paths.append(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    cudaq.set_target("stim")
    for path in generate_goldens(args.output_dir):
        print(path)


if __name__ == "__main__":
    main()
