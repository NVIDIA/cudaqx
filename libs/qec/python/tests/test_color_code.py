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

from cudaq_qec.plugins.codes.color_code import ColorCodeGeometry


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


@pytest.mark.parametrize("d", [3, 5, 7, 9])
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


def test_plaquette_pauli_words_d3_explicit():
    # d=3 plaquettes in grid-sorted order: green [0,1,2,3] (maps to D0),
    # red [1,3,5,6] (maps to D1), blue [2,3,4,5] (maps to D3).
    grid = ColorCodeGeometry(3)
    x_words, z_words = _plaquette_pauli_words(grid)
    assert z_words == ["ZZZZIII", "IZIZIZZ", "IIZZZZI"]
    assert x_words == ["XXXXIII", "IXIXIXX", "IIXXXXI"]


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


if __name__ == "__main__":
    pytest.main()
