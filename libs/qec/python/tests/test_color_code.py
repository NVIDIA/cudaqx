# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
import pytest
import numpy as np
import cudaq
import cudaq_qec as qec

import os

from cudaq_qec.plugins.codes.color_code import (
    ColorCodeGeometry, SuperdensePlan, superdense_cnot_layers, z_side_data,
    superdense_dem, superdense_sample, superdense_memory, _kernel_args)


@pytest.fixture(scope="module", autouse=True)
def set_target():
    cudaq.set_target("stim")
    yield
    cudaq.reset_target()


# Truth data transcribed from the source module docstring ("Reference
# plaquettes for verification"). Keyed by frozenset of data-qubit ids.
REFERENCE_PLAQUETTES = {
    3: {
        frozenset([0, 1, 2, 3]): ('green', 'boundary'),
        frozenset([2, 3, 4, 5]): ('blue', 'boundary'),
        frozenset([1, 3, 5, 6]): ('red', 'boundary'),
    },
    5: {
        frozenset([0, 1, 2, 3]): ('green', 'boundary'),
        frozenset([2, 3, 4, 5, 7, 8]): ('blue', 'bulk'),
        frozenset([1, 3, 5, 6]): ('red', 'boundary'),
        frozenset([5, 6, 8, 9, 12, 13]): ('green', 'bulk'),
        frozenset([7, 8, 11, 12, 15, 16]): ('red', 'bulk'),
        frozenset([4, 7, 10, 11]): ('green', 'boundary'),
        frozenset([10, 11, 14, 15]): ('blue', 'boundary'),
        frozenset([12, 13, 16, 17]): ('blue', 'boundary'),
        frozenset([9, 13, 17, 18]): ('red', 'boundary'),
    },
    7: {
        frozenset([0, 1, 2, 3]): ('green', 'boundary'),
        frozenset([2, 3, 4, 5, 7, 8]): ('blue', 'bulk'),
        frozenset([1, 3, 5, 6]): ('red', 'boundary'),
        frozenset([5, 6, 8, 9, 12, 13]): ('green', 'bulk'),
        frozenset([7, 8, 11, 12, 15, 16]): ('red', 'bulk'),
        frozenset([4, 7, 10, 11]): ('green', 'boundary'),
        frozenset([10, 11, 14, 15, 19, 20]): ('blue', 'bulk'),
        frozenset([12, 13, 16, 17, 21, 22]): ('blue', 'bulk'),
        frozenset([9, 13, 17, 18]): ('red', 'boundary'),
        frozenset([14, 19, 24, 25]): ('green', 'boundary'),
        frozenset([15, 16, 20, 21, 26, 27]): ('green', 'bulk'),
        frozenset([17, 18, 22, 23, 28, 29]): ('green', 'bulk'),
        frozenset([19, 20, 25, 26, 31, 32]): ('red', 'bulk'),
        frozenset([21, 22, 27, 28, 33, 34]): ('red', 'bulk'),
        frozenset([23, 29, 35, 36]): ('red', 'boundary'),
        frozenset([24, 25, 30, 31]): ('blue', 'boundary'),
        frozenset([26, 27, 32, 33]): ('blue', 'boundary'),
        frozenset([28, 29, 34, 35]): ('blue', 'boundary'),
    },
}


def support_matrix(grid):
    """[num_plaquettes, num_data] 0/1 plaquette support matrix."""
    S = np.zeros((grid.num_plaquettes, grid.num_data), dtype=np.uint8)
    for i, plaq in enumerate(grid.plaquettes):
        S[i, plaq['data_qubits']] = 1
    return S


@pytest.mark.parametrize("d", [3, 5, 7, 9, 11, 13])
def test_geometry_counts(d):
    grid = ColorCodeGeometry(d)
    assert grid.num_data == (3 * d * d + 1) // 4
    assert grid.num_plaquettes == (3 * (d * d - 1)) // 8
    assert grid.n_rows == d + (d - 1) // 2
    assert grid.n_cols == d
    assert len(grid.plaquettes) == grid.num_plaquettes
    assert len(grid.qubit_to_coord) == grid.num_data
    assert len(grid.data_qubits) == grid.num_data
    assert len(grid.xcheck_qubits) == grid.num_plaquettes
    assert len(grid.zcheck_qubits) == grid.num_plaquettes
    assert len(grid.all_qubits) == grid.num_data + 2 * grid.num_plaquettes


@pytest.mark.parametrize("d", [1, 2, 4, 6])
def test_invalid_distance_rejected(d):
    with pytest.raises(ValueError, match="odd"):
        ColorCodeGeometry(d)


@pytest.mark.parametrize("d", [3, 5, 7])
def test_reference_plaquettes(d):
    grid = ColorCodeGeometry(d)
    actual = {
        frozenset(p['data_qubits']): (p['color'], p['type'])
        for p in grid.plaquettes
    }
    assert actual == REFERENCE_PLAQUETTES[d]


@pytest.mark.parametrize("d", [3, 5, 7])
def test_plaquettes_sorted_by_grid_position(d):
    grid = ColorCodeGeometry(d)
    positions = [p['grid_pos'] for p in grid.plaquettes]
    assert positions == sorted(positions)
    # Each syndrome maps to a distinct data-qubit grid cell (needed for the
    # CNN grid embedding to be collision-free).
    assert len(set(positions)) == len(positions)


@pytest.mark.parametrize("d", [3, 5, 7])
def test_syndrome_mapping_rule(d):
    # Right boundary (red weight-4) maps to top-left; everything else to
    # top-right (from the source module docstring).
    grid = ColorCodeGeometry(d)
    for i, plaq in enumerate(grid.plaquettes):
        coords = [(q, grid.qubit_to_coord[q]) for q in plaq['data_qubits']]
        top_row = max(c[0] for _, c in coords)
        top = [(q, c) for q, c in coords if c[0] == top_row]
        if plaq['color'] == 'red' and plaq['type'] == 'boundary':
            expected = min(top, key=lambda x: x[1][1])[0]
        else:
            expected = max(top, key=lambda x: x[1][1])[0]
        assert plaq['mapped_qubit'] == expected
        assert grid.stab_to_data_idx[i] == expected
        assert plaq['grid_pos'] == grid.qubit_to_grid[expected]


@pytest.mark.parametrize("d", [3, 5, 7, 9])
def test_css_commutation(d):
    # Self-dual CSS: X-stab i and Z-stab j commute iff their supports overlap
    # on an even number of qubits. Since X and Z supports are identical
    # plaquettes, check (S @ S.T) mod 2 == 0 (diagonal included: weights are
    # 4 or 6, i.e. even).
    grid = ColorCodeGeometry(d)
    S = support_matrix(grid)
    assert not ((S @ S.T) % 2).any()


@pytest.mark.parametrize("d", [3, 5, 7, 9])
def test_logical_operator(d):
    grid = ColorCodeGeometry(d)
    # Bottom edge, weight exactly d.
    assert len(grid.logical_qubits) == d
    bottom_row = min(coord[0] for coord in grid.qubit_to_coord.values())
    assert all(
        grid.qubit_to_coord[q][0] == bottom_row for q in grid.logical_qubits)
    # Logical operator commutes with every stabilizer (even overlap with
    # every plaquette).
    S = support_matrix(grid)
    L = np.zeros(grid.num_data, dtype=np.uint8)
    L[grid.logical_qubits] = 1
    assert not ((S @ L) % 2).any()


def test_top_qubit_at_origin():
    # "Top qubit (qubit 0) is always at (row=0, col=0)" (module docstring).
    for d in (3, 5, 7):
        grid = ColorCodeGeometry(d)
        assert grid.qubit_to_coord[0] == (0, 0)


# ---------------------------------------------------------------------------
# Layout helper methods
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("d", [3, 5, 7])
def test_get_grid_array(d):
    grid = ColorCodeGeometry(d)
    arr = grid.get_grid_array()
    assert arr.shape == (grid.n_rows, grid.n_cols)
    non_pad = arr[arr >= 0]
    assert sorted(non_pad.tolist()) == list(range(grid.num_data))
    for qid, (gr, gc) in grid.qubit_to_grid.items():
        assert arr[gr, gc] == qid


@pytest.mark.parametrize("d", [3, 5, 7])
def test_get_syndrome_grid_indices(d):
    grid = ColorCodeGeometry(d)
    idx = grid.get_syndrome_grid_indices()
    assert idx.shape == (grid.num_plaquettes,)
    for i, plaq in enumerate(grid.plaquettes):
        gr, gc = plaq['grid_pos']
        assert idx[i] == gr * grid.n_cols + gc
    # Distinct plaquettes map to distinct flat grid cells.
    assert len(set(idx.tolist())) == len(idx)


@pytest.mark.parametrize("d", [3, 5, 7])
@pytest.mark.parametrize("id_order", ["rtl", "ltr"])
@pytest.mark.parametrize("flip_rows", [False, True])
def test_circuit_physical_layout(d, id_order, flip_rows):
    grid = ColorCodeGeometry(d)
    layout = grid.get_circuit_physical_layout(id_order=id_order,
                                              flip_rows=flip_rows)
    n_total = grid.num_data + 2 * grid.num_plaquettes
    # Every qubit id placed exactly once, on distinct sites, inside the
    # (n_rows) x (2d-1) rectangle.
    assert set(layout.keys()) == set(range(n_total))
    coords = list(layout.values())
    assert len(set(coords)) == len(coords)
    for r, c in coords:
        assert 0 <= r < grid.n_rows
        assert 0 <= c < 2 * d - 1


def test_circuit_physical_layout_flip_rows_is_reflection():
    grid = ColorCodeGeometry(5)
    base = grid.get_circuit_physical_layout()
    flipped = grid.get_circuit_physical_layout(flip_rows=True)
    H = grid.n_rows
    for q, (r, c) in base.items():
        assert flipped[q] == (H - 1 - r, c)


def test_circuit_physical_layout_rejects_bad_id_order():
    grid = ColorCodeGeometry(3)
    with pytest.raises(ValueError, match="id_order"):
        grid.get_circuit_physical_layout(id_order="bogus")


@pytest.mark.parametrize("d", [3, 5])
def test_superdense_plaquette_labels(d):
    grid = ColorCodeGeometry(d)
    for i, plaq in enumerate(grid.plaquettes):
        labels = grid.superdense_plaquette(i)
        assert labels['a1'] == plaq['x_ancilla']
        assert labels['a2'] == plaq['z_ancilla']
        if plaq['weight'] == 6:
            qs = [labels[f'q{k}'] for k in range(1, 7)]
            assert sorted(qs) == sorted(plaq['data_qubits'])
        else:  # weight 4: q3/q4 are the unused w6 labels
            assert labels['q3'] == -1
            assert labels['q4'] == -1
            qs = [labels['q1'], labels['q2'], labels['q5'], labels['q6']]
            assert sorted(qs) == sorted(plaq['data_qubits'])


def test_superdense_plaquette_bad_index():
    grid = ColorCodeGeometry(3)
    with pytest.raises(IndexError):
        grid.superdense_plaquette(-1)
    with pytest.raises(IndexError):
        grid.superdense_plaquette(grid.num_plaquettes)


def test_print_structure_smoke(capsys):
    ColorCodeGeometry(3).print_structure()
    out = capsys.readouterr().out
    assert "Triangular Color Code - Distance 3" in out
    assert "Plaquettes (sorted by grid position" in out


# ---------------------------------------------------------------------------
# Pauli-word builders (plugin glue)
# ---------------------------------------------------------------------------
from cudaq_qec.plugins.codes.color_code import (_logical_pauli_words,
                                                _plaquette_pauli_words)


@pytest.mark.parametrize("d", [3, 5, 7])
def test_plaquette_pauli_words(d):
    grid = ColorCodeGeometry(d)
    x_words, z_words = _plaquette_pauli_words(grid)
    assert len(x_words) == len(z_words) == grid.num_plaquettes
    for x_word, z_word, plaq in zip(x_words, z_words, grid.plaquettes):
        assert len(x_word) == len(z_word) == grid.num_data
        support = set(plaq['data_qubits'])
        assert {i for i, ch in enumerate(x_word) if ch == 'X'} == support
        assert set(x_word) <= {'I', 'X'}
        assert {i for i, ch in enumerate(z_word) if ch == 'Z'} == support
        assert set(z_word) <= {'I', 'Z'}


def test_plaquette_pauli_words_d3_explicit():
    # d=3 plaquettes in grid-sorted order: green [0,1,2,3] (maps to D0),
    # red [1,3,5,6] (maps to D1), blue [2,3,4,5] (maps to D3).
    grid = ColorCodeGeometry(3)
    x_words, z_words = _plaquette_pauli_words(grid)
    assert z_words == ["ZZZZIII", "IZIZIZZ", "IIZZZZI"]
    assert x_words == ["XXXXIII", "IXIXIXX", "IIXXXXI"]


@pytest.mark.parametrize("d", [3, 5, 7])
def test_logical_pauli_words(d):
    grid = ColorCodeGeometry(d)
    x_word, z_word = _logical_pauli_words(grid)
    support = set(grid.logical_qubits)
    assert {i for i, ch in enumerate(x_word) if ch == 'X'} == support
    assert {i for i, ch in enumerate(z_word) if ch == 'Z'} == support
    assert len(x_word) == len(z_word) == grid.num_data


def test_logical_pauli_words_d3_explicit():
    # d=3 bottom edge is qubits 4,5,6 — same as the Steane-style tail.
    grid = ColorCodeGeometry(3)
    assert _logical_pauli_words(grid) == ("IIIIXXX", "IIIIZZZ")


# ---------------------------------------------------------------------------
# Plugin registration and framework interface
# ---------------------------------------------------------------------------


def test_color_code_registered():
    assert 'color_code' in qec.get_available_codes()


def test_get_code_requires_distance():
    with pytest.raises(RuntimeError, match="distance"):
        qec.get_code('color_code')


def test_get_code_rejects_even_distance():
    with pytest.raises((ValueError, RuntimeError), match="odd"):
        qec.get_code('color_code', distance=4)


@pytest.mark.parametrize("d", [3, 5])
def test_plugin_interface(d):
    code = qec.get_code('color_code', distance=d)
    assert isinstance(code, qec.Code)
    n = (3 * d * d + 1) // 4
    P = (3 * (d * d - 1)) // 8
    assert code.get_num_data_qubits() == n
    assert code.get_num_ancilla_x_qubits() == P
    assert code.get_num_ancilla_z_qubits() == P
    assert code.get_num_ancilla_qubits() == 2 * P
    assert code.get_num_x_stabilizers() == P
    assert code.get_num_z_stabilizers() == P


@pytest.mark.parametrize("d", [3, 5])
def test_plugin_parity_matrices(d):
    code = qec.get_code('color_code', distance=d)
    n = (3 * d * d + 1) // 4
    P = (3 * (d * d - 1)) // 8
    parity = code.get_parity()
    assert parity.shape == (2 * P, 2 * n)
    parity_x = code.get_parity_x()
    parity_z = code.get_parity_z()
    assert parity_x.shape == (P, n)
    assert parity_z.shape == (P, n)
    # Self-dual: X and Z parity blocks contain the same rows (possibly in a
    # different canonical sort order).
    x_rows = {tuple(r) for r in parity_x}
    z_rows = {tuple(r) for r in parity_z}
    assert x_rows == z_rows
    # Rows are exactly the plaquette supports.
    grid = ColorCodeGeometry(d)
    expected = {tuple(r) for r in support_matrix(grid)}
    assert x_rows == expected


@pytest.mark.parametrize("d", [3, 5])
def test_plugin_observables(d):
    code = qec.get_code('color_code', distance=d)
    n = (3 * d * d + 1) // 4
    grid = ColorCodeGeometry(d)
    Lz = code.get_observables_z()
    Lx = code.get_observables_x()
    assert Lz.shape == (1, n)
    assert Lx.shape == (1, n)
    assert set(np.flatnonzero(Lz[0])) == set(grid.logical_qubits)
    assert set(np.flatnonzero(Lx[0])) == set(grid.logical_qubits)


def test_plugin_stabilizers_list():
    code = qec.get_code('color_code', distance=3)
    stabs = code.get_stabilizers()
    words = {s.get_pauli_word() for s in stabs}
    assert words == {
        "ZZZZIII",
        "IZIZIZZ",
        "IIZZZZI",
        "XXXXIII",
        "IXIXIXX",
        "IIXXXXI",
    }


# ---------------------------------------------------------------------------
# Superdense memory circuit
# ---------------------------------------------------------------------------


def _compass_z_side(grid, p_idx):
    # Independent oracle for the Z-ancilla coupling set, from the superdense
    # compass labels: bulk -> {q4,q5,q6}; boundary green -> {q1,q2,q6};
    # boundary red -> {q5}; boundary blue -> {q5,q6}.
    labels = grid.superdense_plaquette(p_idx)
    plaq = grid.plaquettes[p_idx]
    if plaq['weight'] == 6:
        keys = ['q4', 'q5', 'q6']
    elif plaq['color'] == 'green':
        keys = ['q1', 'q2', 'q6']
    elif plaq['color'] == 'red':
        keys = ['q5']
    else:
        keys = ['q5', 'q6']
    return sorted(labels[k] for k in keys)


@pytest.mark.parametrize("d", [3, 5, 7])
def test_schedule_shape_and_disjointness(d):
    grid = ColorCodeGeometry(d)
    N, P = grid.num_data, grid.num_plaquettes
    layers = superdense_cnot_layers(grid)
    assert len(layers) == 8
    # Layers 1 and 8: pure X-ancilla -> Z-ancilla of the same plaquette.
    for li in (0, 7):
        assert sorted(layers[li]) == [(N + i, N + P + i) for i in range(P)]
    # Every layer's endpoints are disjoint (parallel-executable).
    for layer in layers:
        seen = [q for e in layer for q in e]
        assert len(seen) == len(set(seen))
    # Layers 2-4: data is control; layers 5-7: data is target.
    for li in (1, 2, 3):
        assert all(c < N and t >= N for c, t in layers[li])
    for li in (4, 5, 6):
        assert all(c >= N and t < N for c, t in layers[li])
    # Mirror property: layers 5-7 contain exactly the reversed edges of 2-4.
    fwd = {frozenset(e) for li in (1, 2, 3) for e in layers[li]}
    bwd = {frozenset(e) for li in (4, 5, 6) for e in layers[li]}
    assert fwd == bwd
    # Every data-ancilla edge stays inside its plaquette's support.
    for li in (1, 2, 3, 4, 5, 6):
        for c, t in layers[li]:
            dq, anc = (c, t) if c < N else (t, c)
            p_idx = (anc - N) % P
            assert dq in grid.plaquettes[p_idx]['data_qubits']


@pytest.mark.parametrize("d", [3, 5, 7])
def test_z_side_data_matches_compass_rule(d):
    grid = ColorCodeGeometry(d)
    zs = z_side_data(grid)
    assert len(zs) == grid.num_plaquettes
    for p_idx in range(grid.num_plaquettes):
        assert zs[p_idx] == _compass_z_side(grid, p_idx)


@pytest.mark.parametrize("d", [3, 5, 7])
def test_every_plaquette_qubit_coupled_exactly_once_per_side(d):
    # Each data qubit of a plaquette couples to exactly one of the pair
    # (a1 or a2), once on the forward side and once on the mirror side.
    grid = ColorCodeGeometry(d)
    N, P = grid.num_data, grid.num_plaquettes
    layers = superdense_cnot_layers(grid)
    for p_idx, plaq in enumerate(grid.plaquettes):
        anc_pair = {N + p_idx, N + P + p_idx}
        fwd = [e for li in (1, 2, 3) for e in layers[li] if (e[1] in anc_pair)]
        touched = sorted(e[0] for e in fwd)
        assert touched == sorted(plaq['data_qubits'])


# Ground truth: detector sets of the reference superdense construction after
# stim's with_inlined_feedback(), d=3, 3 rounds.
EXPECTED_D3_R3_Z = {
    "detectors": [((0.0, 1.0, 0.0, 4.0), [0]), ((1.0, 1.0, 0.0, 3.0), [1]),
                  ((2.0, 1.0, 0.0, 5.0), [2]), ((0.0, 1.0, 0.0, 1.0), [3, 9]),
                  ((1.0, 1.0, 0.0, 0.0), [4, 10]),
                  ((2.0, 1.0, 0.0, 2.0), [5, 11]),
                  ((0.0, 1.0, 0.0, 4.0), [2, 6]), ((1.0, 1.0, 0.0, 3.0), [7]),
                  ((2.0, 1.0, 0.0, 5.0), [0, 2, 8]),
                  ((0.0, 1.0, 0.0, 1.0), [9, 15]),
                  ((1.0, 1.0, 0.0, 0.0), [10, 16]),
                  ((2.0, 1.0, 0.0, 2.0), [11, 17]),
                  ((0.0, 1.0, 0.0, 4.0), [8, 12]), ((1.0, 1.0, 0.0, 3.0), [13]),
                  ((2.0, 1.0, 0.0, 5.0), [6, 8, 14]),
                  ((0.0, 1.0, 1.0, 4.0), [14, 18, 19, 20, 21]),
                  ((1.0, 1.0, 1.0, 3.0), [19, 21, 23, 24]),
                  ((2.0, 1.0, 1.0, 5.0), [12, 14, 20, 21, 22, 23])],
    "observable": [1, 2, 7, 8, 13, 14, 22, 23, 24],
}

EXPECTED_D3_R3_X = {
    "detectors": [((0.0, 1.0, 0.0, 1.0), [3]), ((1.0, 1.0, 0.0, 0.0), [4]),
                  ((2.0, 1.0, 0.0, 2.0), [5]), ((0.0, 1.0, 0.0, 1.0), [3, 9]),
                  ((1.0, 1.0, 0.0, 0.0), [4, 10]),
                  ((2.0, 1.0, 0.0, 2.0), [5, 11]),
                  ((0.0, 1.0, 0.0, 4.0), [2, 6]), ((1.0, 1.0, 0.0, 3.0), [7]),
                  ((2.0, 1.0, 0.0, 5.0), [0, 2, 8]),
                  ((0.0, 1.0, 0.0, 1.0), [9, 15]),
                  ((1.0, 1.0, 0.0, 0.0), [10, 16]),
                  ((2.0, 1.0, 0.0, 2.0), [11, 17]),
                  ((0.0, 1.0, 0.0, 4.0), [8, 12]), ((1.0, 1.0, 0.0, 3.0), [13]),
                  ((2.0, 1.0, 0.0, 5.0), [6, 8, 14]),
                  ((0.0, 1.0, 1.0, 1.0), [15, 18, 19, 20, 21]),
                  ((1.0, 1.0, 1.0, 0.0), [16, 19, 21, 23, 24]),
                  ((2.0, 1.0, 1.0, 2.0), [17, 20, 21, 22, 23])],
    "observable": [18, 19, 20, 21, 22, 23, 24],
}


@pytest.mark.parametrize("basis,expected", [("Z", EXPECTED_D3_R3_Z),
                                            ("X", EXPECTED_D3_R3_X)])
def test_plan_matches_reference_d3_r3(basis, expected):
    plan = SuperdensePlan(3, 3, basis)
    got = [(tuple(c), sorted(s))
           for c, s in zip(plan.detector_coords, plan.detectors)]
    assert got == expected["detectors"]
    assert sorted(plan.observable) == expected["observable"]


@pytest.mark.parametrize("d,rounds,basis",
                         [(3, 3, "Z"), (3, 4, "X"), (5, 3, "Z"), (5, 4, "X"),
                          (7, 5, "Z"), (7, 5, "X")])
def test_plan_structure(d, rounds, basis):
    plan = SuperdensePlan(d, rounds, basis)
    P = plan.grid.num_plaquettes
    N = plan.grid.num_data
    assert len(plan.detectors) == 2 * P * rounds
    assert len(plan.detector_coords) == len(plan.detectors)
    assert plan.num_records == 2 * P * rounds + N
    for s in plan.detectors:
        assert all(0 <= i < plan.num_records for i in s)
    # Round-0 detectors are single-record, memory basis.
    for p in range(P):
        (rec,) = plan.detectors[p]
        assert rec == (p if basis == "Z" else P + p)


def test_detector_bits_xor():
    plan = SuperdensePlan(3, 3, "Z")
    m = np.zeros((2, plan.num_records), dtype=np.uint8)
    # Flip exactly ONE record of detector 5 in shot 1: precisely the
    # detectors containing that record must fire.
    rec = sorted(plan.detectors[5])[0]
    m[1, rec] = 1
    bits = plan.detector_bits(m)
    assert bits.shape == (2, len(plan.detectors))
    assert not bits[0].any()
    expected = {j for j, s in enumerate(plan.detectors) if rec in s}
    assert set(np.flatnonzero(bits[1]).tolist()) == expected
    assert 5 in expected


@pytest.mark.parametrize("d,rounds,basis", [(3, 3, "Z"), (3, 3, "X"),
                                            (5, 3, "Z"), (5, 3, "X")])
def test_dem_construction_succeeds(d, rounds, basis):
    # stim validates every declared detector is deterministic; passing this
    # call at all is the determinism proof for the inlined parities.
    dem = superdense_dem(d, rounds, basis, p_cx=0.01)
    plan = SuperdensePlan(d, rounds, basis)
    assert dem.num_detectors() == len(plan.detectors)
    assert dem.num_observables() == 1
    assert dem.num_error_mechanisms() > 0
    assert dem.detector_error_matrix.shape[0] == len(plan.detectors)


@pytest.mark.parametrize("basis,expected", [("Z", 0), ("X", 0)])
def test_noiseless_sampling_deterministic(basis, expected):
    syndromes, data, logical = superdense_sample(3,
                                                 3,
                                                 basis,
                                                 p_cx=0.0,
                                                 shots=50)
    assert syndromes.shape == (50, 18)
    assert data.shape == (50, 7)
    assert not syndromes.any()
    assert np.all(logical == expected)


def test_noisy_sampling_fires_detectors():
    syndromes, _, _ = superdense_sample(3, 4, "Z", p_cx=0.05, shots=200)
    assert syndromes.any()
    # No detector column should fire in (nearly) all shots — that would
    # indicate a broken parity rather than noise.
    assert syndromes.mean(axis=0).max() < 0.75


def test_decoding_reduces_logical_errors():
    cudaq.set_random_seed(13)
    d, rounds, p = 3, 4, 0.01
    dem = superdense_dem(d, rounds, "Z", p_cx=p)
    syndromes, _, logical = superdense_sample(d,
                                              rounds,
                                              "Z",
                                              p_cx=p,
                                              shots=1000)
    decoder = qec.get_decoder('single_error_lut', dem.detector_error_matrix)
    dr = decoder.decode_batch(syndromes)
    err = np.asarray(dr.result, dtype=np.uint8)
    pred = ((dem.observables_flips_matrix @ err.T) % 2).flatten()
    n_without = int(logical.sum())
    n_with = int((pred ^ logical).sum())
    print(f"logical errors without decoding: {n_without}, with: {n_with}")
    assert n_with < n_without


_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def _dem_signature(dem_text, stim_mod):
    # Canonical multiset {(detectors, observables) -> merged probability}.
    def merge(p, q):
        return p + q - 2.0 * p * q

    sig = {}
    for inst in stim_mod.DetectorErrorModel(dem_text).flattened():
        if inst.type != "error":
            continue
        prob = inst.args_copy()[0]
        dets, obss = [], []
        for t in inst.targets_copy():
            if t.is_relative_detector_id():
                dets.append(t.val)
            elif t.is_logical_observable_id():
                obss.append(t.val)
        key = (tuple(sorted(dets)), tuple(sorted(obss)))
        sig[key] = merge(sig[key], prob) if key in sig else prob
    return sig


@pytest.mark.parametrize("d,rounds,basis", [(5, 5, "Z"), (5, 5, "X")])
def test_dem_matches_reference_golden(d, rounds, basis):
    # d=5 is the smallest distance exercising every structural element: bulk
    # weight-6 plaquettes of all three colors plus every boundary type.
    stim_mod = pytest.importorskip("stim")
    with open(
            os.path.join(_DATA_DIR,
                         f"superdense_golden_d{d}_r{rounds}_{basis}.dem")) as f:
        golden = f.read()
    golden = "\n".join(l for l in golden.splitlines() if not l.startswith("#"))
    plan = SuperdensePlan(d, rounds, basis)
    # An (empty) noise model must be passed or in-kernel apply_noise is a
    # silent no-op and the DEM comes back with zero error mechanisms.
    ours_text = cudaq.dem_from_kernel(superdense_memory,
                                      *_kernel_args(plan, 0.01),
                                      noise_model=cudaq.NoiseModel())
    ref = _dem_signature(golden, stim_mod)
    got = _dem_signature(ours_text, stim_mod)
    assert set(ref) == set(got), (
        f"error-mechanism signatures differ: ref-only "
        f"{sorted(set(ref) - set(got))[:5]}, ours-only "
        f"{sorted(set(got) - set(ref))[:5]}")
    for k in ref:
        assert np.isclose(ref[k], got[k], atol=1e-6, rtol=1e-4), (
            f"probability mismatch at {k}: ref={ref[k]}, ours={got[k]}")


_COORD_COLOR = {'red': 0., 'green': 1., 'blue': 2.}


@pytest.mark.parametrize("d", [3, 5, 7])
@pytest.mark.parametrize("basis", ["Z", "X"])
def test_detector_coords_contract(d, basis):
    # Emission-order blocks: round-0 (P, memory basis), then per round r >= 1
    # an X block (P) followed by a Z block (P), then the final block (P,
    # memory basis). Chromobius annotation: c = color + 3 for Z-checks.
    rounds = 4
    plan = SuperdensePlan(d, rounds, basis)
    P = plan.grid.num_plaquettes
    assert len(plan.detector_coords) == len(plan.detectors) == 2 * P * rounds

    def block_is_z(det_idx):
        if det_idx < P:
            return basis == "Z"
        if det_idx >= 2 * P * (rounds - 1) + P:
            return basis == "Z"
        return ((det_idx - P) // P) % 2 == 1

    for i, (gr, gc, t, c) in enumerate(plan.detector_coords):
        p = i % P
        plaq = plan.grid.plaquettes[p]
        assert (gr, gc) == tuple(map(float, plaq['grid_pos']))
        is_final = i >= 2 * P * (rounds - 1) + P
        assert t == (1.0 if is_final else 0.0)
        expected_c = _COORD_COLOR[plaq['color']] + (3.0
                                                    if block_is_z(i) else 0.0)
        assert c == expected_c


@pytest.mark.parametrize("basis,expected", [("Z", 0), ("X", 0)])
def test_minimal_two_round_experiment(basis, expected):
    # rounds=2 is the smallest experiment: round 0 plus the final round, with
    # no bulk rounds in between.
    dem = superdense_dem(3, 2, basis, p_cx=0.01)
    plan = SuperdensePlan(3, 2, basis)
    assert dem.num_detectors() == len(plan.detectors) == 12
    assert dem.num_error_mechanisms() > 0
    syndromes, data, logical = superdense_sample(3,
                                                 2,
                                                 basis,
                                                 p_cx=0.0,
                                                 shots=30)
    assert syndromes.shape == (30, 12)
    assert not syndromes.any()
    assert np.all(logical == expected)


if __name__ == "__main__":
    pytest.main()
