# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
import os

import pytest
import numpy as np
import cudaq
import cudaq_qec as qec

from cudaq_qec.plugins.codes import color_code as _color_code_mod
from cudaq_qec.plugins.codes.color_code import (ColorCodeGeometry,
                                                _logical_pauli_words,
                                                _plaquette_pauli_words)


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


@pytest.mark.parametrize("distance",
                         [None, True, False, "3", 3.0, -3, 0, 1, 2, 4, 6])
def test_invalid_distance_rejected(distance):
    with pytest.raises(ValueError, match="odd integer"):
        ColorCodeGeometry(distance)


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


def test_plaquette_pauli_words_d3_explicit():
    # d=3 plaquettes in grid-sorted order: green [0,1,2,3] (maps to D0),
    # red [1,3,5,6] (maps to D1), blue [2,3,4,5] (maps to D3).
    grid = ColorCodeGeometry(3)
    x_words, z_words = _plaquette_pauli_words(grid)
    assert z_words == ["ZZZZIII", "IZIZIZZ", "IIZZZZI"]
    assert x_words == ["XXXXIII", "IXIXIXX", "IIXXXXI"]


def test_logical_pauli_words_d3_explicit():
    # Logical X is the all-data representative (X on all 7 data qubits);
    # logical Z is the bottom edge (qubits 4,5,6 — the Steane-style tail).
    grid = ColorCodeGeometry(3)
    assert _logical_pauli_words(grid) == ("XXXXXXX", "IIIIZZZ")


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


@pytest.mark.parametrize("distance", [None, True, False, "3", 3.0, -3, 0])
def test_get_code_rejects_invalid_distance(distance):
    with pytest.raises(ValueError, match="odd integer"):
        qec.get_code('color_code', distance=distance)


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
    # Z observable: bottom-edge weight-d representative.
    assert set(np.flatnonzero(Lz[0])) == set(grid.logical_qubits)
    # X observable: all-data representative (all ones), matching the reference
    # construction's all-data logical_observable.
    assert np.all(Lx[0] == 1)
    assert set(np.flatnonzero(Lx[0])) == set(range(n))


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
# Superdense CX-layer schedule (host-side builders)
# ---------------------------------------------------------------------------


def test_z_side_data_d3_truth_table():
    # Grid order: p0 green [0,1,2,3], p1 red [1,3,5,6], p2 blue [2,3,4,5].
    grid = ColorCodeGeometry(3)
    assert grid.z_side_data() == [[0, 1, 3], [6], [3, 5]]


@pytest.mark.parametrize("d", [3, 5, 7])
def test_rowmajor_schedule_has_mirrored_data_layers(d):
    layers = ColorCodeGeometry(d)._rowmajor_cnot_layers()
    assert len(layers) == 8
    assert layers[0] == layers[7]
    for forward, mirrored in zip(layers[1:4], layers[4:7]):
        assert mirrored == sorted(
            (target, control) for control, target in forward)


@pytest.mark.parametrize("d", [3, 5])
def test_superdense_schedule_has_eight_terminated_layers(d):
    schedule = ColorCodeGeometry(d).superdense_schedule()
    assert schedule.count(-1) == 8
    assert schedule[-1] == -1


# ---------------------------------------------------------------------------
# Captured superdense stabilizer round
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module", autouse=True)
def plugin_color_code():
    # Defensive re-registration: guarantee qec.get_code('color_code') resolves
    # to this plugin's ColorCode for every test in this module, independent of
    # collection order or of any other module registering the same name.
    # Returns the plugin module for factory spying.
    qec.code('color_code')(_color_code_mod.ColorCode)
    return _color_code_mod


def test_make_stabilizer_round_returns_kernel():
    # The factory turns a captured flat schedule into a CUDA-Q kernel usable as
    # the stabilizer_round operation encoding.
    grid = ColorCodeGeometry(3)
    k = _color_code_mod._make_stabilizer_round(grid.superdense_schedule())
    assert k is not None
    assert hasattr(k, "name")  # PyKernelDecorator


@pytest.mark.parametrize("d", [3, 5])
def test_stabilizer_round_consumes_superdense_schedule(d, monkeypatch,
                                                       plugin_color_code):
    # Gate-sequence check: the registered round must be the factory product
    # built from *exactly* geometry.superdense_schedule() (the reference
    # per-round gate sequence). A spy on the factory captures what
    # ColorCode.__init__ hands it.
    captured = {}
    real_factory = _color_code_mod._make_stabilizer_round

    def spy(schedule_flat):
        captured["schedule"] = list(schedule_flat)
        return real_factory(schedule_flat)

    monkeypatch.setattr(_color_code_mod, "_make_stabilizer_round", spy)
    # get_code re-runs ColorCode.__init__, which must build the round from the
    # captured schedule (get_code returns the C++ Code wrapper, so we assert on
    # the spied factory argument rather than on operation_encodings).
    qec.get_code("color_code", distance=d)
    grid = ColorCodeGeometry(d)
    assert "schedule" in captured, \
        "ColorCode.__init__ did not build stabilizer_round via the factory"
    assert captured["schedule"] == grid.superdense_schedule()


# ---------------------------------------------------------------------------
# Declared inlined-feedback matrices + plaquette-order assertion
#
# Record order [Z][X]: record k in [0,P) is plaquette k's Z-ancilla (a2);
# record P+k is plaquette k's X-ancilla (a1); numCols = 2P. The matrices are
# unit-tested on a directly instantiated ColorCode (the pure-Python getters),
# then exercised through the framework in the E2E block below.
# ---------------------------------------------------------------------------


def _fb_set(z_side_data, support):
    """Reference F(S) = {a : |z_side_data[a] ∩ S| odd}, mirroring the module
    helper but re-derived independently in the test."""
    s = set(support)
    return [
        a for a, zd in enumerate(z_side_data)
        if len(s.intersection(zd)) % 2 == 1
    ]


def test_get_inlined_feedback_d3_reference():
    code = _color_code_mod.ColorCode(distance=3)
    fb = code.get_inlined_feedback()
    assert fb.dtype == np.uint8
    assert fb.shape == (6, 6)
    # Rows Z0,Z1,Z2,X0,X1,X2; columns [Z][X]. Diagonal entries are intentional
    # (duplicate-record XOR cancellation in cudaq::detector).
    expected = np.array([
        [1, 0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ],
                        dtype=np.uint8)
    np.testing.assert_array_equal(fb, expected)


def test_get_observable_inlined_feedback_z_d3_reference():
    code = _color_code_mod.ColorCode(distance=3)
    obz = code.get_observable_inlined_feedback_z()
    assert obz.dtype == np.uint8
    assert obz.shape == (1, 6)
    np.testing.assert_array_equal(
        obz, np.array([[0, 1, 1, 0, 0, 0]], dtype=np.uint8))


def test_observable_inlined_feedback_x_absent():
    # The X-basis logical (logical X read from data measured in X) commutes with
    # the schedule's X byproduct, so no observable feedback is needed. The getter
    # is intentionally absent; the bridge maps the missing method to the empty
    # default (verified as no-feedback by the deterministic X-basis E2E tests).
    code = _color_code_mod.ColorCode(distance=3)
    assert not hasattr(code, "get_observable_inlined_feedback_x")


@pytest.mark.parametrize("d", [3, 5, 7])
def test_feedback_matrices_shape_dtype_and_x_block_zero(d):
    code = _color_code_mod.ColorCode(distance=d)
    P = code.grid.num_plaquettes
    fb = code.get_inlined_feedback()
    obz = code.get_observable_inlined_feedback_z()
    assert fb.shape == (2 * P, 2 * P) and fb.dtype == np.uint8
    assert obz.shape == (1, 2 * P) and obz.dtype == np.uint8
    # X records (>= P) neither herald nor receive the byproduct.
    assert not fb[P:, :].any()
    assert not fb[:, P:].any()
    assert not obz[:, P:].any()


def test_feedback_F_sets_d5_spot_check():
    # Brief's triple-verified d=5 cross-check of F(supp) per plaquette and
    # F(logical), plus that the assembled matrices agree with F.
    grid = ColorCodeGeometry(5)
    zsd = grid.z_side_data()
    expected_F = {
        0: [0, 2],
        1: [1, 4],
        2: [0, 2, 5],
        3: [3, 7],
        4: [1, 4, 8],
        5: [2, 5],
        6: [6],
        7: [3],
        8: [4],
    }
    for j, plaq in enumerate(grid.plaquettes):
        assert _fb_set(zsd, plaq['data_qubits']) == expected_F[j]
    assert _fb_set(zsd, grid.logical_qubits) == [5, 6, 7, 8]

    code = _color_code_mod.ColorCode(distance=5)
    fb = code.get_inlined_feedback()
    obz = code.get_observable_inlined_feedback_z()
    for j in range(grid.num_plaquettes):
        assert sorted(np.flatnonzero(fb[j]).tolist()) == expected_F[j]
    assert sorted(np.flatnonzero(obz[0]).tolist()) == [5, 6, 7, 8]


@pytest.mark.parametrize("d", [3, 5, 7, 9])
def test_plaquette_order_assertion_passes(d):
    grid = ColorCodeGeometry(d)
    # Does not raise for any supported distance.
    _color_code_mod._assert_plaquette_order(grid.plaquettes)
    mins = [min(p['data_qubits']) for p in grid.plaquettes]
    assert all(a < b for a, b in zip(mins, mins[1:]))


def test_plaquette_order_assertion_rejects_bad_order():
    # Two plaquettes sharing a min data-qubit index (0) — not strictly
    # increasing, so record k could not map to plaquette k.
    bad = [{'data_qubits': [0, 1, 2, 3]}, {'data_qubits': [0, 2, 4, 5]}]
    with pytest.raises(ValueError, match="strictly-increasing"):
        _color_code_mod._assert_plaquette_order(bad)


# ---------------------------------------------------------------------------
# Framework-path E2E payoff (stim target) — the declared feedback makes the
# X-basis memory deterministic and the DEM/decode pipeline work in both bases.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("prep,basis,obs_getter,expected", [
    (qec.operation.prep0, 'z', 'get_observables_z', 0),
    (qec.operation.prep1, 'z', 'get_observables_z', 1),
    (qec.operation.prepp, 'x', 'get_observables_x', 0),
    (qec.operation.prepm, 'x', 'get_observables_x', 1),
])
@pytest.mark.parametrize("nRounds", [1, 2, 4])
def test_noiseless_memory_deterministic_both_bases(plugin_color_code, prep,
                                                   basis, obs_getter, expected,
                                                   nRounds):
    code = qec.get_code('color_code', distance=3)
    sample_fn = (qec.z_sample_memory_circuit
                 if basis == 'z' else qec.x_sample_memory_circuit)
    syndromes, data = sample_fn(code, prep, 20, nRounds, cudaq.NoiseModel())
    # Noise-free: every detector is deterministic, so all syndromes vanish.
    assert not np.any(syndromes)
    L = getattr(code, obs_getter)()
    logical = ((L @ data.T) % 2).flatten()
    assert np.all(logical == expected)


def test_z_dem_from_memory_circuit_shapes(plugin_color_code):
    code = qec.get_code('color_code', distance=3)
    P = code.get_num_z_stabilizers()
    noise = cudaq.NoiseModel()
    noise.add_all_qubit_channel("x", cudaq.Depolarization2(0.01), 1)
    nRounds = 4
    dem = qec.z_dem_from_memory_circuit(code, qec.operation.prep0, nRounds,
                                        noise)
    assert dem.detector_error_matrix.shape[0] == (nRounds + 1) * P
    assert len(dem.error_rates) == dem.detector_error_matrix.shape[1]
    assert dem.observables_flips_matrix.shape == (
        1, dem.detector_error_matrix.shape[1])
    assert all(rate > 0 for rate in dem.error_rates)
    # Detector rows line up column-for-column with sampled Z syndromes.
    syndromes, _ = qec.z_sample_memory_circuit(code, qec.operation.prep0, 10,
                                               nRounds, noise)
    assert syndromes.shape[1] == dem.detector_error_matrix.shape[0]


def test_x_dem_from_memory_circuit_shapes(plugin_color_code):
    code = qec.get_code('color_code', distance=3)
    P = code.get_num_x_stabilizers()
    noise = cudaq.NoiseModel()
    noise.add_all_qubit_channel("x", cudaq.Depolarization2(0.01), 1)
    nRounds = 3
    dem = qec.x_dem_from_memory_circuit(code, qec.operation.prepp, nRounds,
                                        noise)
    # X-basis keeps only the X-stabilizer detectors: (nRounds + 1) * P rows.
    assert dem.detector_error_matrix.shape[0] == (nRounds + 1) * P
    assert dem.detector_error_matrix.shape[1] > 0
    assert len(dem.error_rates) == dem.detector_error_matrix.shape[1]
    syndromes, _ = qec.x_sample_memory_circuit(code, qec.operation.prepp, 10,
                                               nRounds, noise)
    assert syndromes.shape[1] == dem.detector_error_matrix.shape[0]


def test_noisy_sampling_shapes_and_activity(plugin_color_code):
    code = qec.get_code('color_code', distance=3)
    P = code.get_num_z_stabilizers()
    noise = cudaq.NoiseModel()
    noise.add_all_qubit_channel("x", cudaq.Depolarization2(0.05), 1)
    nShots, nRounds = 100, 4
    syndromes, data = qec.z_sample_memory_circuit(code, qec.operation.prep0,
                                                  nShots, nRounds, noise)
    assert syndromes.shape == (nShots, (nRounds + 1) * P)
    assert data.shape == (nShots, code.get_num_data_qubits())
    assert np.any(syndromes)


def test_decoding_reduces_logical_errors_x_basis(plugin_color_code):
    # Decode smoke in the X basis. The X basis has no observable feedback
    # (absent get_observable_inlined_feedback_x), so Lx @ data IS the DEM
    # observable and a valid ground truth. The Z-basis observable is the
    # byproduct-corrected DEM observable whose feedback (herald) records
    # sample_memory_circuit does not return, so a Z-basis truth from data alone
    # would be wrong — the realtime-only "DEM crux".
    cudaq.set_random_seed(13)
    code = qec.get_code('color_code', distance=3)
    Lx = code.get_observables_x()
    noise = cudaq.NoiseModel()
    noise.add_all_qubit_channel("x", cudaq.Depolarization2(0.01), 1)
    nRounds, nShots = 6, 400

    dem = qec.x_dem_from_memory_circuit(code, qec.operation.prepp, nRounds,
                                        noise)
    syndromes, data = qec.x_sample_memory_circuit(code, qec.operation.prepp,
                                                  nShots, nRounds, noise)
    logical = ((Lx @ data.T) % 2).flatten().astype(np.uint8)

    decoder = qec.get_decoder('single_error_lut', dem.detector_error_matrix)
    dr = decoder.decode_batch(np.asarray(syndromes, dtype=np.uint8))
    err_pred = np.asarray(dr.result, dtype=np.uint8)
    predictions = ((dem.observables_flips_matrix @ err_pred.T) % 2).flatten()

    n_without = int(np.sum(logical))
    n_with = int(np.sum(predictions ^ logical))
    print(f"logical errors without decoding: {n_without}, with: {n_with}")
    assert n_with < n_without


def test_declared_feedback_makes_x_path_deterministic(plugin_color_code):
    # With the byproduct-correcting CX dropped from the round and no feedback
    # declared, the X-basis (prepp) DEM was rejected with "non-deterministic
    # detectors". With the feedback matrices declared it now builds — the flip
    # from negative to positive control.
    code = qec.get_code("color_code", distance=3)
    noise = cudaq.NoiseModel()
    noise.add_all_qubit_channel("x", cudaq.Depolarization2(0.01), 1)
    dem = qec.x_dem_from_memory_circuit(code, qec.operation.prepp, 3, noise)
    assert dem.detector_error_matrix.shape[0] > 0
    assert dem.detector_error_matrix.shape[1] > 0


def test_z_path_never_raises(plugin_color_code):
    # The Z-basis heralds are Z-ancilla records that are deterministically 0
    # in noiseless prep0, so the Z path was deterministic even with the bare
    # (no-feedback) round and remains so with the feedback declared.
    code = qec.get_code("color_code", distance=3)
    noise = cudaq.NoiseModel()
    noise.add_all_qubit_channel("x", cudaq.Depolarization2(0.01), 1)
    dem = qec.z_dem_from_memory_circuit(code, qec.operation.prep0, 3, noise)
    assert dem.detector_error_matrix.shape[0] > 0


def test_undeclared_feedback_breaks_determinism(plugin_color_code):
    # Retained bare-round negative control: a code that reuses the plugin's
    # bare superdense round (via delegation) but declares NO feedback getters
    # hits the non-deterministic-detector rejection on the X path, proving the
    # declared matrices are load-bearing. (A plain ColorCode subclass cannot be used:
    # qec.code() copies only the decorated class's own __dict__ and reparents
    # to Code, dropping inherited methods — hence the delegation.)
    @qec.code('color_code_bare_control')
    class ColorCodeBareControl:

        def __init__(self, **kwargs):
            qec.Code.__init__(self)
            self._c = _color_code_mod.ColorCode(**kwargs)
            self.stabilizers = self._c.stabilizers
            self.pauli_observables = self._c.pauli_observables
            self.operation_encodings = self._c.operation_encodings

        def get_num_data_qubits(self):
            return self._c.get_num_data_qubits()

        def get_num_ancilla_x_qubits(self):
            return self._c.get_num_ancilla_x_qubits()

        def get_num_ancilla_z_qubits(self):
            return self._c.get_num_ancilla_z_qubits()

        def get_num_ancilla_qubits(self):
            return self._c.get_num_ancilla_qubits()

        def get_num_x_stabilizers(self):
            return self._c.get_num_x_stabilizers()

        def get_num_z_stabilizers(self):
            return self._c.get_num_z_stabilizers()

    code = qec.get_code('color_code_bare_control', distance=3)
    noise = cudaq.NoiseModel()
    noise.add_all_qubit_channel("x", cudaq.Depolarization2(0.01), 1)
    with pytest.raises(ValueError, match="non-deterministic detectors"):
        qec.x_dem_from_memory_circuit(code, qec.operation.prepp, 3, noise)


# ---------------------------------------------------------------------------
# Golden oracle: framework DEM vs the reference superdense DEMs,
# d=5, rounds=5, DEPOLARIZE2(0.01) per schedule CX.
#
# The full (both-basis-detector) framework DEM comes from the non-directional
# qec.dem_from_memory_circuit: it keeps all 2*P*rounds detectors, matching the
# golden's layout. The directional z/x_dem_from_memory_circuit slice out the
# off-basis cross-round detectors ((rounds+1)*P rows) and so cannot be compared
# row-for-row against the full 90-detector golden.
#
# The reference files are the ALL-ROUNDS-NOISY convention (every stabilizer
# round noisy — physically stricter). They were generated by the reference
# superdense construction (superdense_memory kernel, cudaq.dem_from_kernel);
# the same generator byte-reproduces the reference construction's original
# perfect-final-round DEMs when its two noise guards (R > 1 and r < R - 1) are
# restored. Those perfect-final-round originals are retained outside the test
# suite.
#
# Under the all-rounds-noisy convention the framework DEM equals the reference
# mechanism-for-mechanism, including the observable-flip column, at bit-exact
# probabilities (0 diff, both bases, verified when the goldens were generated).
# This pins the plugin's schedule, [Z][X] record order, feedback matrices, the
# logical-X observable representative, and the detector-alignment permutation
# below as exact. A comparison against the perfect-final-round originals awaits
# a framework noiseless-final-round capability (memory_circuit noises every
# stabilizer round; there is no perfect-final-round knob).
# ---------------------------------------------------------------------------

_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def test_golden_generator_configuration():
    from generate_color_code_goldens import (DISTANCE, NOISE_PROBABILITY,
                                             ROUNDS)
    assert (DISTANCE, ROUNDS, NOISE_PROBABILITY) == (5, 5, 0.01)


def test_golden_headers_document_reproduction_command():
    command = ("python3 libs/qec/python/tests/generate_color_code_goldens.py "
               "--output-dir libs/qec/python/tests/data")
    for basis in ("X", "Z"):
        path = os.path.join(_DATA_DIR,
                            f"superdense_golden_d5_r5_allnoisy_{basis}.dem")
        with open(path) as stream:
            header = "".join(stream.readline() for _ in range(10))
        assert command in header


def _merge_error_prob(p, q):
    # Two independent error mechanisms with identical detector/observable
    # support combine as p*(1-q) + q*(1-p).
    return p + q - 2.0 * p * q


def _golden_signature(dem_text, stim_mod):
    """Canonical {(detector-support, observable-support): merged prob} of a
    stim DEM text. flattened() expands repeat/shift blocks; detector targets
    across '^' decomposition separators are unioned (separators ignored), so
    the signature is invariant to graphlike error decomposition."""
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
        sig[key] = _merge_error_prob(sig[key], prob) if key in sig else prob
    return sig


def _framework_detector_permutation(P, num_rounds):
    """Framework detector index -> golden emission index.

    Both circuits emit P round-0 fixed detectors (prep basis), then num_rounds-1
    interior rounds each carrying a full [Z-block(P), X-block(P)] ancilla pair,
    then P boundary detectors. memory_circuit.cpp emits each interior round
    Z-block-first (the round returns mz([*ancz, *ancx]), so cross-round records
    j < P are the Z ancillas); the reference superdense_memory kernel emits each
    interior round X-block-first. Round-0 and boundary blocks are identical. So
    the alignment swaps the two P-sized halves of every interior round and is the
    identity elsewhere -- derived from the two emission orders, not fit to the
    DEM contents."""
    total = 2 * P * num_rounds
    perm = {}
    for i in range(total):
        if i < P or i >= total - P:
            perm[i] = i
            continue
        local = i - P
        rr = local // (2 * P)
        within = local % (2 * P)
        base = P + rr * (2 * P)
        perm[i] = base + (within + P if within < P else within - P)
    assert sorted(perm.values()) == list(range(total))
    return perm


def _framework_signature(dem, perm):
    """Canonical {(detector-support, observable-support): merged prob} of a
    framework detector_error_model, with detector indices mapped through perm."""
    dm = np.asarray(dem.detector_error_matrix, dtype=np.uint8)
    obs = np.asarray(dem.observables_flips_matrix, dtype=np.uint8)
    rates = np.asarray(dem.error_rates, dtype=float)
    sig = {}
    for c in range(dm.shape[1]):
        dets = tuple(sorted(perm[int(r)] for r in np.flatnonzero(dm[:, c])))
        ob = tuple(np.flatnonzero(obs[:, c]).tolist())
        key = (dets, ob)
        sig[key] = _merge_error_prob(sig[key],
                                     rates[c]) if key in sig else rates[c]
    return sig


@pytest.mark.parametrize("basis,prep", [("Z", qec.operation.prep0),
                                        ("X", qec.operation.prepp)])
def test_dem_matches_reference_golden(plugin_color_code, basis, prep):
    """Framework DEM equals the reference superdense DEM, d=5, rounds=5.

    The goldens (superdense_golden_d5_r5_allnoisy_{Z,X}.dem) are the
    ALL-ROUNDS-NOISY convention: every stabilizer round carries
    DEPOLARIZE2(0.01) on each schedule CX (physically stricter than a perfect
    final round). They were generated by the reference superdense construction;
    that same generator byte-reproduces the reference perfect-final-round DEMs
    (retained outside the test suite) when its two noise guards are restored,
    so the goldens carry the reference construction exactly, only under the
    stricter noise convention. d=5 is the smallest distance exercising every
    structural element: bulk weight-6 plaquettes of all three colors plus every
    boundary type. The comparison is exact -- same canonical merged signatures,
    the observable-flip column included, bit-exact probability equality under
    the structural detector permutation. A comparison against the
    perfect-final-round originals awaits a framework noiseless-final-round
    capability (memory_circuit noises every round)."""
    stim_mod = pytest.importorskip("stim")
    d, rounds = 5, 5
    code = qec.get_code('color_code', distance=d)
    P = code.get_num_z_stabilizers()
    noise = cudaq.NoiseModel()
    noise.add_all_qubit_channel("x", cudaq.Depolarization2(0.01), 1)
    dem = qec.dem_from_memory_circuit(code, prep, rounds, noise)
    perm = _framework_detector_permutation(P, rounds)
    got = _framework_signature(dem, perm)
    with open(
            os.path.join(
                _DATA_DIR,
                f"superdense_golden_d{d}_r{rounds}_allnoisy_{basis}.dem")) as f:
        golden_text = f.read()
    # Belt-and-suspenders: the golden files carry a fixed number of error
    # mechanisms (2957 Z / 2958 X); a mismatch means a corrupted or truncated
    # golden, not a plugin regression.
    n_mechanisms = sum(
        1 for inst in stim_mod.DetectorErrorModel(golden_text).flattened()
        if inst.type == "error")
    assert n_mechanisms == {
        "Z": 2957,
        "X": 2958
    }[basis], (
        f"golden file parsed {n_mechanisms} error mechanisms; expected "
        f"{'2957' if basis == 'Z' else '2958'} — golden corrupted/truncated?")
    ref = _golden_signature(golden_text, stim_mod)
    assert set(ref) == set(got), (
        f"error-mechanism signatures differ: ref-only "
        f"{sorted(set(ref) - set(got))[:5]}, ours-only "
        f"{sorted(set(got) - set(ref))[:5]}")
    # Bit-exact: the framework and generator DEMs agree to the last float bit
    # (isolation max diff 0.0), so assert equality rather than a tolerance.
    for k in ref:
        assert ref[k] == got[k], (
            f"probability mismatch at {k}: ref={ref[k]}, ours={got[k]}")


if __name__ == "__main__":
    pytest.main()
