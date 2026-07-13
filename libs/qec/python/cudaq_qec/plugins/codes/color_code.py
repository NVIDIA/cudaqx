# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""
Triangular Color Code with rectangular grid embedding for CNN pre-decoder.

The data qubits are embedded in a (n_rows x n_cols) rectangular grid where:
- n_rows = d + (d-1)//2
- n_cols = d

Coordinate system:
- Top qubit (qubit 0) is always at (row=0, col=0)
- Row index decreases (more negative) going down the triangle
- Column is centered around 0, expanding symmetrically

Syndrome-to-grid mapping:
- For all stabilizers EXCEPT right boundary: map to top-right data qubit
- For right boundary (red weight-4): map to top-left data qubit

Reference plaquettes for verification:

d=3 (7 data qubits, 3 plaquettes, 4x3 grid):
  [0,1,2,3]: green (boundary)
  [2,3,4,5]: blue (boundary)
  [1,3,5,6]: red (boundary)

d=5 (19 data qubits, 9 plaquettes, 7x5 grid):
  [0,1,2,3]: green (boundary)
  [2,3,4,5,7,8]: blue (bulk)
  [1,3,5,6]: red (boundary)
  [5,6,8,9,12,13]: green (bulk)
  [7,8,11,12,15,16]: red (bulk)
  [4,7,10,11]: green (boundary)
  [10,11,14,15]: blue (boundary)
  [12,13,16,17]: blue (boundary)
  [9,13,17,18]: red (boundary)

d=7 (37 data qubits, 18 plaquettes, 10x7 grid):
  [0,1,2,3]: green (boundary)
  [2,3,4,5,7,8]: blue (bulk)
  [1,3,5,6]: red (boundary)
  [5,6,8,9,12,13]: green (bulk)
  [7,8,11,12,15,16]: red (bulk)
  [4,7,10,11]: green (boundary)
  [10,11,14,15,19,20]: blue (bulk)
  [12,13,16,17,21,22]: blue (bulk)
  [9,13,17,18]: red (boundary)
  [14,19,24,25]: green (boundary)
  [15,16,20,21,26,27]: green (bulk)
  [17,18,22,23,28,29]: green (bulk)
  [19,20,25,26,31,32]: red (bulk)
  [21,22,27,28,33,34]: red (bulk)
  [23,29,35,36]: red (boundary)
  [24,25,30,31]: blue (boundary)
  [26,27,32,33]: blue (boundary)
  [28,29,34,35]: blue (boundary)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional

import cudaq
import cudaq_qec as qec
from cudaq_qec import patch


class ColorCodeGeometry:
    """
    Triangular color code with rectangular grid embedding.

    Qubit numbering (for distance d):
    - Data qubits: [0, num_data)
    - X-check ancillas: [num_data, num_data + num_plaquettes)
    - Z-check ancillas: [num_data + num_plaquettes, num_data + 2*num_plaquettes)

    Data qubits are numbered top-to-bottom, left-to-right.
    Ancillas are numbered by their mapped grid position (top-to-bottom, left-to-right).

    Args:
        distance: Code distance (odd integer >= 3)

    Attributes:
        data_qubits: Array of data qubit indices [0, num_data)
        xcheck_qubits: Array of X-check ancilla indices
        zcheck_qubits: Array of Z-check ancilla indices
        stab_to_data_idx: Maps stabilizer index to data qubit grid position
    """

    def __init__(self, distance: int):
        if distance < 3 or distance % 2 == 0:
            raise ValueError("Distance must be odd and >= 3")

        self.distance = distance
        self.n_rows = distance + (distance - 1) // 2
        self.n_cols = distance
        self.num_data = (3 * distance * distance + 1) // 4
        self.num_plaquettes = (3 * (distance * distance - 1)) // 8

        # Generate data qubit grid layout
        self._generate_data_qubit_grid()

        # Generate plaquettes (stabilizers) algorithmically
        self._generate_plaquettes()

        # Compute syndrome-to-data mapping and sort plaquettes by grid position
        self._compute_syndrome_mapping_and_sort()

        # Create qubit index arrays (after sorting)
        self.data_qubits = np.arange(self.num_data)
        self.xcheck_qubits = np.arange(self.num_data,
                                       self.num_data + self.num_plaquettes)
        self.zcheck_qubits = np.arange(self.num_data + self.num_plaquettes,
                                       self.num_data + 2 * self.num_plaquettes)
        self.all_qubits = np.arange(self.num_data + 2 * self.num_plaquettes)

        # Find the logical operator support
        self._find_logical_qubits()

    def _get_row_width(self, row_idx: int) -> int:
        """Get number of qubits in a given row (0-indexed from top)."""
        group = row_idx // 3
        pos = row_idx % 3
        return 2 * group + 1 if pos < 2 else 2 * group + 2

    def _get_row_start_col(self, width: int) -> int:
        """Get starting column for a row of given width (centered around 0)."""
        return -(width // 2)

    def _generate_data_qubit_grid(self):
        """Generate data qubit positions on rectangular grid."""
        self.qubit_to_coord = {}  # qubit_id -> (row, col)
        self.coord_to_qubit = {}  # (row, col) -> qubit_id
        self.grid_to_qubit = {
        }  # (grid_row, grid_col) -> qubit_id (0-indexed grid)
        self.qubit_to_grid = {}  # qubit_id -> (grid_row, grid_col)

        qubit_id = 0
        for row_idx in range(self.n_rows):
            width = self._get_row_width(row_idx)
            start_col = self._get_row_start_col(width)
            row = -row_idx  # User's coordinate: row 0 at top, negative going down

            for i in range(width):
                col = start_col + i

                self.qubit_to_coord[qubit_id] = (row, col)
                self.coord_to_qubit[(row, col)] = qubit_id

                # Also store 0-indexed grid position for CNN
                grid_row = row_idx
                grid_col = col + (self.n_cols // 2)
                self.grid_to_qubit[(grid_row, grid_col)] = qubit_id
                self.qubit_to_grid[qubit_id] = (grid_row, grid_col)

                qubit_id += 1

        assert qubit_id == self.num_data, f"Expected {self.num_data} qubits, got {qubit_id}"

    def _try_pattern(self, row: int, col: int,
                     pattern: List[Tuple[int, int]]) -> Optional[List[int]]:
        """Try to form a plaquette with given pattern from anchor position."""
        qubits = []
        for dr, dc in pattern:
            pos = (row + dr, col + dc)
            if pos in self.coord_to_qubit:
                qubits.append(self.coord_to_qubit[pos])
            else:
                return None
        return sorted(qubits)

    def _generate_plaquettes(self):
        """Generate plaquette connectivity algorithmically."""
        colors = ['green', 'blue', 'red']
        plaquettes = []
        added_plaqs = set()
        used_bulk = {c: set() for c in colors}

        # Plaquette patterns (relative to anchor position)
        pattern_w6 = [(0, 0), (0, 1), (-1, 0), (-1, 1), (-2, 0),
                      (-2, 1)]  # 3x2 bulk
        pattern_top = [(0, 0), (-1, 0), (-2, -1), (-2, 0)]  # top cap
        pattern_left = [(-2, 0), (-2, 1), (-1, 1), (0, 1)]  # left boundary
        pattern_right = [(-2, 0), (-2, 1), (-1, 0), (0, 0)]  # right boundary
        pattern_bottom = [(-1, 0), (-1, 1), (0, 0), (0, 1)]  # bottom 2x2

        def add_plaq(qubits, color, ptype, check_bulk_overlap=True):
            key = tuple(sorted(qubits))
            if key in added_plaqs:
                return False
            if check_bulk_overlap and ptype == 'bulk':
                if any(q in used_bulk[color] for q in qubits):
                    return False
                used_bulk[color].update(qubits)
            plaquettes.append((qubits, color, ptype))
            added_plaqs.add(key)
            return True

        # 1. Top plaquette (green) - always qubits 0,1,2,3
        add_plaq([0, 1, 2, 3], 'green', 'boundary', False)

        # 2. Weight-6 bulk plaquettes
        color_occurrence = {'green': 0, 'blue': 0, 'red': 0}

        for row_idx in range(2, self.n_rows - 2):
            row = -row_idx
            color = colors[row % 3]
            occ = color_occurrence[color]
            color_occurrence[color] += 1

            # Compute valid column positions for this color
            if color == 'green':
                # Green: alternates between center+even offsets and odd offsets
                if occ == 0:
                    valid_cols = [0]
                elif occ % 2 == 1:  # Odd occurrence: use odd offsets
                    valid_cols = sorted(
                        [c for c in range(-occ, occ + 1) if c % 2 != 0])
                else:  # Even occurrence: use even offsets (including 0)
                    valid_cols = sorted(
                        [c for c in range(-occ, occ + 1) if c % 2 == 0])
            else:
                # Blue/Red: start from -(occ+1), step by 2
                start = -(occ + 1)
                num_plaqs = occ + 1
                valid_cols = [start + 2 * i for i in range(num_plaqs)]

            for col in valid_cols:
                qubits = self._try_pattern(row, col, pattern_w6)
                if qubits:
                    add_plaq(qubits, color, 'bulk', True)

        # 3. Left boundary (green) - every 3 rows starting from row_idx=3
        for row_idx in range(3, self.n_rows - 2, 3):
            row = -row_idx
            width = self._get_row_width(row_idx)
            anchor_col = self._get_row_start_col(width) - 1
            qubits = self._try_pattern(row, anchor_col, pattern_left)
            if qubits:
                add_plaq(qubits, 'green', 'boundary', False)

        # 4. Right boundary (red) - every 3 rows starting from row_idx=1
        for row_idx in range(1, self.n_rows - 2, 3):
            row = -row_idx
            width = self._get_row_width(row_idx)
            anchor_col = self._get_row_start_col(width) + width - 1
            qubits = self._try_pattern(row, anchor_col, pattern_right)
            if qubits:
                add_plaq(qubits, 'red', 'boundary', False)

        # 5. Bottom boundary (blue) - 2x2 blocks on second-to-last row, step by 2
        second_last_row_idx = self.n_rows - 2
        row = -second_last_row_idx
        width = self._get_row_width(second_last_row_idx)
        start_col = self._get_row_start_col(width)
        for col in range(start_col, start_col + width - 1, 2):
            qubits = self._try_pattern(row, col, pattern_bottom)
            if qubits:
                add_plaq(qubits, 'blue', 'boundary', False)

        # Store raw plaquettes temporarily (will be sorted later)
        self._raw_plaquettes = plaquettes

        assert len(plaquettes) == self.num_plaquettes, \
            f"Expected {self.num_plaquettes} plaquettes, got {len(plaquettes)}"

    def _get_mapped_data_qubit(self, data_qubits: List[int], color: str,
                               ptype: str) -> int:
        """
        Get the data qubit that a plaquette's syndrome maps to.

        Rules:
        - Right boundary (red weight-4): top-left data qubit
        - All others: top-right data qubit
        """
        # Get coordinates for all data qubits in plaquette
        coords = [(q, self.qubit_to_coord[q]) for q in data_qubits]

        # Find top row (highest, i.e., least negative)
        top_row = max(c[0] for _, c in coords)
        top_qubits = [(q, c) for q, c in coords if c[0] == top_row]

        # Determine if this is right boundary (red weight-4)
        is_right_boundary = (color == 'red' and ptype == 'boundary')

        if is_right_boundary:
            # Top-left: minimum column
            return min(top_qubits, key=lambda x: x[1][1])[0]
        else:
            # Top-right: maximum column
            return max(top_qubits, key=lambda x: x[1][1])[0]

    def _compute_syndrome_mapping_and_sort(self):
        """Compute syndrome-to-data mapping and sort plaquettes by grid position."""
        # Compute mapped data qubit for each plaquette
        plaq_with_mapping = []
        for qubits, color, ptype in self._raw_plaquettes:
            mapped_qubit = self._get_mapped_data_qubit(qubits, color, ptype)
            grid_pos = self.qubit_to_grid[mapped_qubit]
            plaq_with_mapping.append({
                'data_qubits': qubits,
                'color': color,
                'type': ptype,
                'weight': len(qubits),
                'mapped_qubit': mapped_qubit,
                'grid_pos': grid_pos,
            })

        # Sort by grid position: top-to-bottom (row), left-to-right (col)
        plaq_with_mapping.sort(
            key=lambda p: (p['grid_pos'][0], p['grid_pos'][1]))

        # Assign ancilla IDs based on sorted order
        # X-ancilla at index i: num_data + i
        # Z-ancilla at index i: num_data + num_plaquettes + i
        self.plaquettes = []
        self.stab_to_data_idx = np.zeros(self.num_plaquettes, dtype=np.int32)

        for plaq_idx, plaq in enumerate(plaq_with_mapping):
            x_ancilla_id = self.num_data + plaq_idx
            z_ancilla_id = self.num_data + self.num_plaquettes + plaq_idx

            self.plaquettes.append({
                'x_ancilla': x_ancilla_id,
                'z_ancilla': z_ancilla_id,
                'data_qubits': plaq['data_qubits'],
                'weight': plaq['weight'],
                'type': plaq['type'],
                'color': plaq['color'],
                'mapped_qubit': plaq['mapped_qubit'],
                'grid_pos': plaq['grid_pos'],
            })

            self.stab_to_data_idx[plaq_idx] = plaq['mapped_qubit']

        # Clean up temporary storage
        del self._raw_plaquettes

    def _find_logical_qubits(self):
        """Find the data qubits supporting the logical X/Z operators.

        For the triangular color code, the minimal logical operator is the
        bottom edge (blue boundary), which has exactly d qubits.
        This is preferred over using all data qubits because:
        - Measurement errors scale as O(d) instead of O(d²)
        - No need for boundary detectors in memory experiments

        Parity-check and observable matrices are not stored here: the code
        object returned by qec.get_code('color_code', ...) provides them in
        the framework's canonical form via get_parity_x()/get_parity_z()
        and get_observables_x()/get_observables_z().
        """
        # Find bottom edge qubits (the row with minimum row coordinate)
        bottom_row = min(
            self.qubit_to_coord[q][0] for q in range(self.num_data))
        self.logical_qubits = sorted([
            q for q in range(self.num_data)
            if self.qubit_to_coord[q][0] == bottom_row
        ])

    def get_grid_array(self) -> np.ndarray:
        """Return 2D array of qubit IDs on the grid (-1 for padding)."""
        grid = np.full((self.n_rows, self.n_cols), -1, dtype=np.int32)
        for qid, (grid_row, grid_col) in self.qubit_to_grid.items():
            grid[grid_row, grid_col] = qid
        return grid

    def get_syndrome_grid_indices(self) -> np.ndarray:
        """
        Return array mapping stabilizer index to flat grid index.

        For use with reshape_stabilizers_to_grid functions.
        Returns array of shape (num_plaquettes,) where each entry is the
        flat index (row * n_cols + col) into the n_rows x n_cols grid.
        """
        indices = np.zeros(self.num_plaquettes, dtype=np.int32)
        for i, plaq in enumerate(self.plaquettes):
            grid_row, grid_col = plaq['grid_pos']
            indices[i] = grid_row * self.n_cols + grid_col
        return indices

    def print_structure(self):
        """Print code structure summary."""
        print(f"Triangular Color Code - Distance {self.distance}")
        print(f"  Grid size: {self.n_rows} x {self.n_cols}")
        print(f"  Data qubits: {self.num_data} (IDs: 0-{self.num_data-1})")
        print(
            f"  X-check ancillas: {self.num_plaquettes} (IDs: {self.xcheck_qubits[0]}-{self.xcheck_qubits[-1]})"
        )
        print(
            f"  Z-check ancillas: {self.num_plaquettes} (IDs: {self.zcheck_qubits[0]}-{self.zcheck_qubits[-1]})"
        )
        print(f"  Total qubits: {len(self.all_qubits)}")
        print(f"  Plaquettes: {len(self.plaquettes)}")
        print()

        # Print grid layout
        print("Data qubit grid (user coordinates):")
        for row_idx in range(self.n_rows):
            row = -row_idx
            width = self._get_row_width(row_idx)
            start_col = self._get_row_start_col(width)

            qubits_str = []
            for i in range(width):
                col = start_col + i
                if (row, col) in self.coord_to_qubit:
                    qid = self.coord_to_qubit[(row, col)]
                    qubits_str.append(f"D{qid:2d}")

            indent = "  " * (self.n_cols // 2 - (-start_col))
            print(f"  row {row:3d}: {indent}{' '.join(qubits_str)}")
        print()

        # Print CNN grid
        print("CNN grid layout (0-indexed, -1 = padding):")
        grid = self.get_grid_array()
        print("         " + "  ".join(f"c{c}" for c in range(self.n_cols)))
        for r in range(self.n_rows):
            row_str = " ".join(f"{grid[r,c]:3d}" if grid[r, c] >= 0 else "  ."
                               for c in range(self.n_cols))
            print(f"  row {r}: {row_str}")
        print()

        # Print plaquettes with syndrome mapping
        print(
            "Plaquettes (sorted by grid position, top-to-bottom, left-to-right):"
        )
        boundary_count = bulk_count = 0
        for i, plaq in enumerate(self.plaquettes):
            if plaq['type'] == 'boundary':
                boundary_count += 1
            else:
                bulk_count += 1
            grid_pos = plaq['grid_pos']
            print(
                f"  Plaq {i:2d} (X:{plaq['x_ancilla']:2d}, Z:{plaq['z_ancilla']:2d}, {plaq['color']:5s}, "
                f"{plaq['type']:8s}, w{plaq['weight']}): "
                f"{plaq['data_qubits']} -> D{plaq['mapped_qubit']} @ grid({grid_pos[0]},{grid_pos[1]})"
            )
        print(
            f"\nSummary: {boundary_count} boundary + {bulk_count} bulk = {len(self.plaquettes)} plaquettes"
        )
        print()

    def superdense_plaquette(self, plaq_idx: int) -> Dict[str, int]:
        """
        Return a canonical labeling for a plaquette for the superdense circuit.

        Labels follow the convention discussed in chat:
        - a1: X-ancilla (prepared in |+>, measured in X)
        - a2: Z-ancilla (prepared in |0>, measured in Z)
        - q1..q6: data qubits ordered by compass position around the plaquette (for weight-6):
            q1 = NW, q2 = W, q3 = SW, q4 = NE, q5 = E, q6 = SE

        Weight-6 plaquettes occupy a 3x2 block in (row, col) coordinates.

        Weight-4 plaquettes are embedded into the same frame by populating only:
          - q1, q2 (feed into a1 via the q*->a1 half)
          - q5, q6 (feed into a2 via the q*->a2 half)
        and setting q3 and q4 to -1 (missing). This matches using the same global 8-step schedule,
        while weight-4 plaquettes naturally skip the third pair steps.
        """
        if plaq_idx < 0 or plaq_idx >= len(self.plaquettes):
            raise IndexError(f"plaq_idx out of range: {plaq_idx}")

        plaq = self.plaquettes[plaq_idx]
        w = int(plaq["weight"])
        if w not in (4, 6):
            raise ValueError(
                f"Unsupported plaquette weight={w} for plaq_idx={plaq_idx}")

        data = list(plaq["data_qubits"])
        coords = {q: self.qubit_to_coord[q] for q in data}  # (row, col)

        rows = sorted({r for r, _ in coords.values()},
                      reverse=True)  # top (largest) -> bottom (smallest)
        cols = sorted({c for _, c in coords.values()})  # left -> right

        out: Dict[str, int] = {
            "a1": int(plaq["x_ancilla"]),
            "a2": int(plaq["z_ancilla"]),
        }

        coord_to_qid = {v: k for k, v in coords.items()}

        if w == 6:
            if len(rows) != 3 or len(cols) != 2:
                raise ValueError(
                    "Expected weight-6 plaquette data qubits to occupy exactly 3 distinct rows and 2 distinct cols "
                    f"but got rows={rows}, cols={cols} for plaq_idx={plaq_idx}, data_qubits={data}, coords={coords}"
                )

            row_top, row_mid, row_bot = rows
            col_left, col_right = cols

            # West column (left): q1 (NW), q2 (W), q3 (SW)
            # East column (right): q4 (NE), q5 (E), q6 (SE)
            expected_positions = {
                "q1": (row_top, col_left),
                "q2": (row_mid, col_left),
                "q3": (row_bot, col_left),
                "q4": (row_top, col_right),
                "q5": (row_mid, col_right),
                "q6": (row_bot, col_right),
            }

            missing = []
            for label, pos in expected_positions.items():
                qid = coord_to_qid.get(pos)
                if qid is None:
                    missing.append((label, pos))
                else:
                    out[label] = int(qid)

            if missing:
                raise ValueError(
                    f"Could not assign all q1..q6 labels for plaq_idx={plaq_idx}; missing={missing}; "
                    f"data_qubits={data}; coords={coords}")
            return out

        # --- weight-4 embedding into q1/q2/q5/q6; q3/q4 are missing ---
        out["q3"] = -1
        out["q4"] = -1

        # Case A: South boundary 2x2 block (two rows, two cols)
        if len(rows) == 2 and len(cols) == 2:
            row_top, row_bot = rows  # already sorted top->bottom
            col_left, col_right = cols

            # User-confirmed for south boundary:
            # q1 = NW (top-left), q2 = W (bottom-left), q3 = NE (top-right), q4 = E (bottom-right).
            # We embed into the w6 schedule by mapping:
            #   q1 -> q1, q2 -> q2, q3 -> q5, q4 -> q6  (and q3/q4 are the unused w6 labels).
            pos_q1 = (row_top, col_left)
            pos_q2 = (row_bot, col_left)
            pos_q5 = (row_top, col_right)  # NE
            pos_q6 = (row_bot, col_right)  # E / SE

            for key, pos in [("q1", pos_q1), ("q2", pos_q2), ("q5", pos_q5),
                             ("q6", pos_q6)]:
                qid = coord_to_qid.get(pos)
                if qid is None:
                    raise ValueError(
                        f"Missing expected {key} position {pos} for w4 2x2 plaq_idx={plaq_idx}, coords={coords}"
                    )
                out[key] = int(qid)
            return out

        # Case B: L-shape boundary (three rows, two cols, four points).
        # Empirically in this construction, one column has 3 qubits (dense) and the other has 1 (sparse),
        # with the sparse qubit on the bottom row.
        if len(rows) == 3 and len(cols) == 2:
            col_a, col_b = cols
            pts_a = [q for q, (r, c) in coords.items() if c == col_a]
            pts_b = [q for q, (r, c) in coords.items() if c == col_b]

            if len(pts_a) == 3 and len(pts_b) == 1:
                dense_col, sparse_col = col_a, col_b
            elif len(pts_b) == 3 and len(pts_a) == 1:
                dense_col, sparse_col = col_b, col_a
            else:
                raise ValueError(
                    f"Unexpected w4 L-shape column counts for plaq_idx={plaq_idx}: cols={cols}, counts={[len(pts_a), len(pts_b)]}, coords={coords}"
                )

            row_top, row_mid, row_bot = rows

            # Use dense column for q1/q2 (feeding a1), and use (sparse bottom, dense bottom)
            # for q5/q6 (feeding a2). This uses all 4 data qubits and is consistent across boundary types.
            pos_q1 = (row_top, dense_col)
            pos_q2 = (row_mid, dense_col)
            pos_q5 = (row_bot, sparse_col)
            pos_q6 = (row_bot, dense_col)

            for key, pos in [("q1", pos_q1), ("q2", pos_q2), ("q5", pos_q5),
                             ("q6", pos_q6)]:
                qid = coord_to_qid.get(pos)
                if qid is None:
                    raise ValueError(
                        f"Missing expected {key} position {pos} for w4 L-shape plaq_idx={plaq_idx}, coords={coords}"
                    )
                out[key] = int(qid)
            return out

        raise ValueError(
            f"Unsupported w4 geometry for plaq_idx={plaq_idx}: distinct_rows={len(rows)}, distinct_cols={len(cols)}, coords={coords}"
        )

        return out

    # ------------------------------------------------------------------ #
    # Superdense paired-ancilla CX-layer schedule (host-side builders)
    # ------------------------------------------------------------------ #

    def _rowmajor_cnot_layers(self) -> List[List[Tuple[int, int]]]:
        """The 8 CNOT layers of the superdense schedule in the row-major
        inverted-triangle numbering.

        The triangle is oriented with its widest row (d data qubits) on top,
        narrowing to a single qubit at the bottom.

        Returns a length-8 list; each element is a list of (control, target)
        pairs where data qubits are 0..num_data-1 (row-major, top-to-bottom)
        and ancillas are num_data..num_data+2*num_plaquettes-1, grouped as
        (a1, a2) = (num_data + 2p, num_data + 2p + 1) with a1 the X-ancilla
        and a2 the Z-ancilla of plaquette p.
        """
        d = int(self.distance)
        num_data = (3 * d * d + 1) // 4
        num_plaquettes = (3 * (d * d - 1)) // 8
        num_rows_data_qubits = (3 * d - 1) // 2

        layers: List[List[Tuple[int, int]]] = []

        # tt=1..8 correspond to 8 layers
        for tt in range(1, 9):
            edges: List[Tuple[int, int]] = []

            if tt == 1 or tt == 8:
                # CNOT between the two ancilla qubits: control |+> (a1),
                # target |0> (a2)
                index = num_data
                for _ in range(num_plaquettes):
                    a1 = index
                    a2 = index + 1
                    edges.append((a1, a2))
                    index += 2

            elif tt == 2:
                # First sequence (data = control)
                data_qubit_index = d
                index_ancilla = num_data
                cols = d - 1
                for rr in range(num_rows_data_qubits - 1):
                    if rr % 3 == 0:
                        for _ in range(cols):
                            edges.append((data_qubit_index, index_ancilla))
                            data_qubit_index += 1
                            index_ancilla += 1
                        cols -= 1
                    elif rr % 3 == 1:
                        for cc in range(cols):
                            edges.append((data_qubit_index, index_ancilla))
                            data_qubit_index += 1
                            if cc == cols - 1:
                                index_ancilla += 3
                            else:
                                index_ancilla += 1
                    else:  # rr % 3 == 2
                        for _ in range(cols):
                            edges.append((data_qubit_index, index_ancilla))
                            data_qubit_index += 1
                            index_ancilla += 1
                        cols -= 1

            elif tt == 3:
                # Second sequence (data = control)
                data_qubit_index = 0
                index_ancilla = num_data
                cols = d - 1
                for rr in range(num_rows_data_qubits - 1):
                    if rr % 3 == 0:
                        for cc in range(cols):
                            edges.append((data_qubit_index, index_ancilla))
                            index_ancilla += 1
                            if cc == cols - 1:
                                data_qubit_index += 3
                            else:
                                data_qubit_index += 1
                        cols -= 1
                    elif rr % 3 == 1:
                        for cc in range(cols):
                            edges.append((data_qubit_index, index_ancilla))
                            data_qubit_index += 1
                            if cc == cols - 1:
                                index_ancilla += 3
                            else:
                                index_ancilla += 1
                    else:  # rr % 3 == 2
                        for _ in range(cols):
                            edges.append((data_qubit_index, index_ancilla))
                            data_qubit_index += 1
                            index_ancilla += 1
                        cols -= 1

            elif tt == 4:
                # Third sequence (data = control)
                data_qubit_index = 1
                index_ancilla = num_data + d - 1
                cols = d - 1
                for rr in range(num_rows_data_qubits - 2):
                    if rr % 3 == 0:
                        for _ in range(cols):
                            edges.append((data_qubit_index, index_ancilla))
                            data_qubit_index += 1
                            index_ancilla += 1
                    elif rr % 3 == 1:
                        for _ in range(cols):
                            edges.append((data_qubit_index, index_ancilla))
                            data_qubit_index += 1
                            index_ancilla += 1
                        cols -= 2
                    else:  # rr % 3 == 2
                        for cc in range(cols):
                            edges.append((data_qubit_index, index_ancilla))
                            index_ancilla += 1
                            if cc == cols - 1:
                                data_qubit_index += 3
                            else:
                                data_qubit_index += 1

            elif tt == 5:
                # First sequence (data = target) => ancilla is control
                data_qubit_index = d
                index_ancilla = num_data
                cols = d - 1
                for rr in range(num_rows_data_qubits - 1):
                    if rr % 3 == 0:
                        for _ in range(cols):
                            edges.append((index_ancilla, data_qubit_index))
                            data_qubit_index += 1
                            index_ancilla += 1
                        cols -= 1
                    elif rr % 3 == 1:
                        for cc in range(cols):
                            edges.append((index_ancilla, data_qubit_index))
                            data_qubit_index += 1
                            if cc == cols - 1:
                                index_ancilla += 3
                            else:
                                index_ancilla += 1
                    else:  # rr % 3 == 2
                        for _ in range(cols):
                            edges.append((index_ancilla, data_qubit_index))
                            data_qubit_index += 1
                            index_ancilla += 1
                        cols -= 1

            elif tt == 6:
                # Second sequence (data = target) => ancilla is control
                data_qubit_index = 0
                index_ancilla = num_data
                cols = d - 1
                for rr in range(num_rows_data_qubits - 1):
                    if rr % 3 == 0:
                        for cc in range(cols):
                            edges.append((index_ancilla, data_qubit_index))
                            index_ancilla += 1
                            if cc == cols - 1:
                                data_qubit_index += 3
                            else:
                                data_qubit_index += 1
                        cols -= 1
                    elif rr % 3 == 1:
                        for cc in range(cols):
                            edges.append((index_ancilla, data_qubit_index))
                            data_qubit_index += 1
                            if cc == cols - 1:
                                index_ancilla += 3
                            else:
                                index_ancilla += 1
                    else:  # rr % 3 == 2
                        for _ in range(cols):
                            edges.append((index_ancilla, data_qubit_index))
                            data_qubit_index += 1
                            index_ancilla += 1
                        cols -= 1

            else:  # tt == 7
                # Third sequence (data = target) => ancilla is control
                data_qubit_index = 1
                index_ancilla = num_data + d - 1
                cols = d - 1
                for rr in range(num_rows_data_qubits - 2):
                    if rr % 3 == 0:
                        for _ in range(cols):
                            edges.append((index_ancilla, data_qubit_index))
                            data_qubit_index += 1
                            index_ancilla += 1
                    elif rr % 3 == 1:
                        for _ in range(cols):
                            edges.append((index_ancilla, data_qubit_index))
                            data_qubit_index += 1
                            index_ancilla += 1
                        cols -= 2
                    else:  # rr % 3 == 2
                        for cc in range(cols):
                            edges.append((index_ancilla, data_qubit_index))
                            index_ancilla += 1
                            if cc == cols - 1:
                                data_qubit_index += 3
                            else:
                                data_qubit_index += 1

            layers.append(sorted(edges))

        return layers

    def _rowmajor_to_unified(self) -> Dict[int, int]:
        """Bijection from the row-major inverted-triangle qubit numbering of
        :meth:`_rowmajor_cnot_layers` to the unified label space (data
        0..N-1, X-ancilla N+i, Z-ancilla N+P+i for plaquette i).

        Data qubits: the grid's coordinates put row 0 at the top with rows
        decreasing (more negative) downward, so sorting data ids by (row, col)
        ascending enumerates them bottom-to-top, left-to-right -- exactly the
        row-major order of the inverted triangle (widest row first).

        Ancillas: plaquettes are matched by their data-qubit support sets (the
        data endpoints coupled to an ancilla pair across all layers), then
        (a1, a2) = (num_data + 2p, num_data + 2p + 1) maps to
        (N + i, N + P + i) for the matching plaquette index i.
        """
        N = int(self.num_data)
        P = int(self.num_plaquettes)

        ordered_data = sorted(
            range(N),
            key=lambda q:
            (self.qubit_to_coord[q][0], self.qubit_to_coord[q][1]))
        data_to_rowmajor = {q: i for i, q in enumerate(ordered_data)}

        layers = self._rowmajor_cnot_layers()

        # Support set (in row-major data ids) of each row-major plaquette.
        support_to_plaq = {}
        for p in range(P):
            a1 = N + 2 * p
            a2 = N + 2 * p + 1
            ds = set()
            for layer in layers:
                for c, t in layer:
                    if c < N and t in (a1, a2):
                        ds.add(c)
                    if c in (a1, a2) and t < N:
                        ds.add(t)
            support_to_plaq[tuple(sorted(ds))] = p

        mapping = {i: q for q, i in data_to_rowmajor.items()}
        for i, plaq in enumerate(self.plaquettes):
            key = tuple(sorted(
                data_to_rowmajor[q] for q in plaq['data_qubits']))
            p = support_to_plaq.get(key)
            if p is None:
                raise ValueError(
                    f"Plaquette {i} (data qubits {plaq['data_qubits']}) has no "
                    f"support-set match in the schedule")
            mapping[N + 2 * p] = N + i
            mapping[N + 2 * p + 1] = N + P + i
        return mapping

    def _superdense_cnot_layers(self) -> List[List[Tuple[int, int]]]:
        """The eight CNOT layers of the paired-ancilla superdense schedule in
        the unified label space: data 0..N-1, X-ancilla N+i and Z-ancilla
        N+P+i of plaquette i. Layers 0 and 7 entangle each plaquette's
        ancilla pair; layers 1-3 couple data (control) into the pair; layers
        4-6 apply the mirrored ancilla-controlled CNOTs.
        """
        mapping = self._rowmajor_to_unified()
        return [
            sorted((mapping[c], mapping[t])
                   for c, t in layer)
            for layer in self._rowmajor_cnot_layers()
        ]

    def z_side_data(self) -> List[List[int]]:
        """Per plaquette (grid order), the data qubits coupled to its
        Z-ancilla by the schedule; a conditional X on exactly these qubits is
        the byproduct the Z-ancilla readout heralds."""
        layers = self._superdense_cnot_layers()
        N, P = self.num_data, self.num_plaquettes
        out = [set() for _ in range(P)]
        for li in (1, 2, 3):
            for c, t in layers[li]:
                if t >= N + P:
                    out[t - N - P].add(c)
        return [sorted(s) for s in out]

    def superdense_schedule(self) -> List[int]:
        """Flattened one-round CX-layer schedule over the unified index space
        (data 0..N-1, X-ancilla N..N+P-1, Z-ancilla N+P..N+2P-1).

        Encoding: the eight layers are concatenated as flat integers; within a
        layer each (control, target) CNOT contributes the two ints control,
        target, and every layer is closed by a -1 terminator. All qubit ids
        are non-negative, so -1 unambiguously marks a layer boundary. A
        consumer reads ints two at a time as (control, target) pairs and
        advances to the next layer on -1.
        """
        schedule: List[int] = []
        for layer in self._superdense_cnot_layers():
            for c, t in layer:
                schedule.append(int(c))
                schedule.append(int(t))
            schedule.append(-1)
        return schedule


# ---------------------------------------------------------------------------
# CUDA-QX plugin glue: Pauli words over the data qubits
# ---------------------------------------------------------------------------


def _plaquette_pauli_words(
        grid: "ColorCodeGeometry") -> Tuple[List[str], List[str]]:
    """Return (x_words, z_words): one pure-X and one pure-Z Pauli word per
    plaquette, over the data qubits, in plaquette (grid-sorted) order.

    The code is a self-dual CSS code: the X and Z stabilizers of a plaquette
    share the same support, so both words are derived from the same
    ``data_qubits`` list. Word index 0 is data qubit 0 (CUDA-QX convention:
    leftmost Pauli character acts on qubit 0).
    """
    x_words: List[str] = []
    z_words: List[str] = []
    for plaq in grid.plaquettes:
        chars = ['I'] * grid.num_data
        for q in plaq['data_qubits']:
            chars[q] = 'X'
        x_words.append(''.join(chars))
        chars = ['I'] * grid.num_data
        for q in plaq['data_qubits']:
            chars[q] = 'Z'
        z_words.append(''.join(chars))
    return x_words, z_words


def _logical_pauli_words(grid: "ColorCodeGeometry") -> Tuple[str, str]:
    """Return (x_word, z_word) for the logical X/Z operators.

    The logical X is the all-data-qubit representative (X on every one of the
    N data qubits), matching the reference construction's all-data
    ``logical_observable``. The logical Z is the bottom-edge representative
    (``logical_qubits``, weight exactly d) — see
    ColorCodeGeometry._find_logical_qubits for why the bottom edge is
    preferred for Z (O(d) measurement error, no boundary detectors).

    The representatives are deliberately asymmetric. Both are valid: X on all
    N data qubits commutes with every plaquette (all even weight, 4 or 6) and
    anti-commutes with the bottom-edge logical Z (overlap = the d bottom-edge
    qubits, d odd), so the pair is a logical X/Z.
    """
    x_word = 'X' * grid.num_data
    z_chars = ['I'] * grid.num_data
    for q in grid.logical_qubits:
        z_chars[q] = 'Z'
    return x_word, ''.join(z_chars)


# ---------------------------------------------------------------------------
# Inlined-feedback declarations
#
# The registered stabilizer_round replays the reference superdense schedule
# with the byproduct left BARE: the mirrored ancilla-controlled CNOTs (layers
# 5-7) leave an uncorrected X byproduct on each plaquette's Z-ancilla-coupled
# data qubits (ColorCodeGeometry.z_side_data()), heralded by that Z-ancilla's
# readout record. The framework absorbs the byproduct downstream by XOR-ing the
# heralding records into the affected detectors / observables via these matrices
# (see libs/qec/lib/device/inlined_feedback.h), instead of a coherent in-round
# correction CX.
# ---------------------------------------------------------------------------


def _feedback_set(z_side_data: List[List[int]], support) -> List[int]:
    """``F(S) = {a in [0,P) : |z_side_data[a] ∩ S| odd}``.

    ``z_side_data[a]`` is the data-qubit set coupled to plaquette ``a``'s
    Z-ancilla by the schedule; its readout heralds a conditional X on exactly
    those qubits. A record support ``S`` (a detector's or observable's
    data-qubit support) picks up plaquette ``a``'s byproduct iff ``S`` overlaps
    ``z_side_data[a]`` on an odd number of qubits.
    """
    s = set(support)
    return [
        a for a, zd in enumerate(z_side_data)
        if len(s.intersection(zd)) % 2 == 1
    ]


def _assert_plaquette_order(plaquettes: List[dict]) -> None:
    """Require min data-qubit index strictly increasing across grid-ordered
    plaquettes; raise ValueError otherwise.

    ``to_parity_matrix`` (stabilizer_utils.cpp) sorts the Z-stabilizers by the
    index of their first ``Z`` character, i.e. by min data-qubit index. The
    memory circuit uses parity row k for the boundary detector of measurement
    record k, and record k is plaquette k's Z-ancilla in this module's grid
    order. Record k therefore maps to plaquette k only when the grid order is
    already the min-data-qubit sort — i.e. min data-qubit is strictly
    increasing across grid-ordered plaquettes (also guaranteeing the sort keys
    are unique, so the non-stable std::sort cannot permute them). The declared
    feedback matrices are indexed by that record order, so a violation would
    silently misalign them.
    """
    mins = [min(p['data_qubits']) for p in plaquettes]
    for k in range(1, len(mins)):
        if mins[k] <= mins[k - 1]:
            raise ValueError(
                "[color_code] plaquette order violates the min-data-qubit "
                "strictly-increasing invariant required for measurement "
                "record k to map to plaquette k (the framework sorts "
                "stabilizers by min data-qubit index); min-data-qubit "
                f"sequence={mins}.")


# ---------------------------------------------------------------------------
# CUDA-Q kernels
#
# These operate on the framework `patch` type (data / ancx / ancz qubit
# views). The memory-circuit driver (libs/qec/lib/device/memory_circuit.cpp)
# wraps stabilizer_round with cudaq::detector / cudaq::detectors /
# cudaq::logical_observable annotations, so DEM extraction and
# detector-ordered sampling need no extra work here.
# ---------------------------------------------------------------------------


@cudaq.kernel
def prep0(logicalQubit: patch):
    # Transversal |0...0> prep; the first stabilizer round projects into the
    # codespace with deterministic Z stabilizers.
    reset(logicalQubit.data)


@cudaq.kernel
def prep1(logicalQubit: patch):
    # X on every data qubit is a logical-X representative: every plaquette
    # has even weight (4 or 6) so it commutes with all Z stabilizers, and it
    # flips the weight-d (odd) bottom-edge logical Z.
    prep0(logicalQubit)
    x(logicalQubit.data)


@cudaq.kernel
def prepp(logicalQubit: patch):
    # The code is self-dual CSS (identical X/Z supports), so transversal H
    # preserves the stabilizer group and maps logical |0> to |+>.
    prep0(logicalQubit)
    h(logicalQubit.data)


@cudaq.kernel
def prepm(logicalQubit: patch):
    prep0(logicalQubit)
    x(logicalQubit.data)
    h(logicalQubit.data)


def _make_stabilizer_round(schedule_flat: list[int]):
    """Build the per-round ``stabilizer_round`` kernel that replays the
    reference superdense CX schedule exactly, as captured from
    :meth:`ColorCodeGeometry.superdense_schedule`.

    The factory closes over the flat, ``-1``-terminated schedule so the kernel
    walks a concrete gate list (a ``@cudaq.kernel`` defined inside a factory
    captures a host ``list[int]``, and the capture survives the C++
    ``memory_circuit`` -> registered-Python-sub-kernel dispatch; each registered
    instance keeps its own capture).

    Schedule encoding (unified index space, data ``0..N-1``, X-ancilla
    ``N..N+P-1``, Z-ancilla ``N+P..N+2P-1``): ints are read two at a time as
    ``(control, target)`` CNOT pairs; a ``-1`` closes the current layer. All
    qubit ids are non-negative, so ``-1`` unambiguously marks a layer boundary.
    Applying the pairs in flat order (skipping the ``-1`` markers) reproduces
    the eight-layer gate sequence exactly: layers are emitted in order and each
    layer's gates act on disjoint qubits, so the within-layer order is
    immaterial.

    No byproduct-correction gates are emitted. The eight schedule layers include
    the mirrored ancilla-controlled CNOTs (layers 5-7) that leave an uncorrected
    X byproduct on the Z-ancilla-coupled data qubits; the correction is absorbed
    downstream by the code's inlined-feedback declarations rather than by a
    coherent in-round CX.
    """

    @cudaq.kernel
    def _round(logicalQubit: patch, x_stabilizers: list[int],
               z_stabilizers: list[int]) -> list[cudaq.measure_handle]:
        # x_stabilizers / z_stabilizers arrive from the framework but are
        # unused: the captured schedule alone drives the gate sequence.
        num_data = len(logicalQubit.data)
        P = len(logicalQubit.ancx)

        # X-ancilla (a1) prep in |+>.
        h(logicalQubit.ancx)

        # Walk the captured schedule: (control, target) CNOT pairs, layers
        # separated by -1. Index -> patch view: i < N -> data[i];
        # N <= i < N+P -> ancx[i-N]; N+P <= i < N+2P -> ancz[i-N-P].
        i = 0
        n = len(schedule_flat)
        while i < n:
            if schedule_flat[i] == -1:
                i = i + 1
            else:
                ci = schedule_flat[i]
                ti = schedule_flat[i + 1]
                if ci < num_data:
                    if ti < num_data:
                        x.ctrl(logicalQubit.data[ci], logicalQubit.data[ti])
                    elif ti < num_data + P:
                        x.ctrl(logicalQubit.data[ci],
                               logicalQubit.ancx[ti - num_data])
                    else:
                        x.ctrl(logicalQubit.data[ci],
                               logicalQubit.ancz[ti - num_data - P])
                elif ci < num_data + P:
                    if ti < num_data:
                        x.ctrl(logicalQubit.ancx[ci - num_data],
                               logicalQubit.data[ti])
                    elif ti < num_data + P:
                        x.ctrl(logicalQubit.ancx[ci - num_data],
                               logicalQubit.ancx[ti - num_data])
                    else:
                        x.ctrl(logicalQubit.ancx[ci - num_data],
                               logicalQubit.ancz[ti - num_data - P])
                else:
                    if ti < num_data:
                        x.ctrl(logicalQubit.ancz[ci - num_data - P],
                               logicalQubit.data[ti])
                    elif ti < num_data + P:
                        x.ctrl(logicalQubit.ancz[ci - num_data - P],
                               logicalQubit.ancx[ti - num_data])
                    else:
                        x.ctrl(logicalQubit.ancz[ci - num_data - P],
                               logicalQubit.ancz[ti - num_data - P])
                i = i + 2

        # Rotate a1 back to the computational basis for the X readout.
        h(logicalQubit.ancx)

        # [Z][X] record order (ancz first) — the order the framework detectors
        # and the inlined-feedback matrices assume.
        results = mz([*logicalQubit.ancz, *logicalQubit.ancx])

        reset(logicalQubit.ancx)
        reset(logicalQubit.ancz)
        return results

    return _round


# ---------------------------------------------------------------------------
# Plugin registration
# ---------------------------------------------------------------------------


@qec.code('color_code')
class ColorCode:
    """Triangular color code plugin for CUDA-QX.

    One ancilla per stabilizer per basis: with P = 3(d^2-1)/8 plaquettes the
    patch holds (3d^2+1)/4 data qubits plus P X-ancillas and P Z-ancillas.
    ``distance`` (odd, >= 3) must be provided via
    ``qec.get_code('color_code', distance=d)``.

    Memory-circuit sampling for this Python feedback code currently requires
    ``cudaq.set_target('stim')``; the ``qpp-cpu`` target does not preserve the
    required cross-round measurement correlations.

    The generic :class:`qec.Code` object returned by :func:`qec.get_code`
    consumes ``distance`` but does not expose the Python implementation's
    ``distance`` or ``grid`` attributes. Callers that need the plaquette
    layout, syndrome-to-grid mapping, or CNN embedding should construct
    :class:`ColorCodeGeometry` with the same distance.
    """

    def __init__(self, **kwargs):
        # The Code binding accepts no constructor arguments; kwargs from
        # qec.get_code(...) are consumed here on the Python side.
        qec.Code.__init__(self)
        if 'distance' not in kwargs:
            raise RuntimeError(
                "[color_code] distance not provided. distance must be "
                "provided via qec.get_code('color_code', distance=d).")
        self.distance = kwargs['distance']
        self.grid = ColorCodeGeometry(self.distance)

        # The declared feedback matrices are indexed by the measurement-record
        # order (record k = plaquette k's Z-ancilla in grid order). That only
        # matches the framework's parity-row order when the grid order is the
        # min-data-qubit sort; fail loudly otherwise.
        _assert_plaquette_order(self.grid.plaquettes)

        x_words, z_words = _plaquette_pauli_words(self.grid)
        self.stabilizers = [
            cudaq.SpinOperator.from_word(w) for w in z_words + x_words
        ]
        obs_x, obs_z = _logical_pauli_words(self.grid)
        self.pauli_observables = [
            cudaq.SpinOperator.from_word(w) for w in (obs_x, obs_z)
        ]

        self.operation_encodings = {
            qec.operation.prep0:
                prep0,
            qec.operation.prep1:
                prep1,
            qec.operation.prepp:
                prepp,
            qec.operation.prepm:
                prepm,
            # Per-instance round: replays this code's captured reference
            # superdense schedule with the byproduct left uncorrected; the
            # inlined-feedback getters declared below (get_inlined_feedback,
            # get_observable_inlined_feedback_z) absorb it downstream.
            qec.operation.stabilizer_round:
                _make_stabilizer_round(self.grid.superdense_schedule()),
        }

    def get_num_data_qubits(self):
        return self.grid.num_data

    def get_num_ancilla_x_qubits(self):
        return self.grid.num_plaquettes

    def get_num_ancilla_z_qubits(self):
        return self.grid.num_plaquettes

    def get_num_ancilla_qubits(self):
        return 2 * self.grid.num_plaquettes

    def get_num_x_stabilizers(self):
        return self.grid.num_plaquettes

    def get_num_z_stabilizers(self):
        return self.grid.num_plaquettes

    # ------------------------------------------------------------------ #
    # Inlined-feedback declarations. Records are ordered [Z][X]:
    # record k in [0,P) is plaquette k's Z-ancilla (a2); record P+k is
    # plaquette k's X-ancilla (a1); numCols = 2P. Matrices are row-major
    # uint8 (the bridge coerces dtype but requires a 2-D array).
    # ------------------------------------------------------------------ #

    def get_inlined_feedback(self):
        """[2P x 2P] detector feedback: fb[j][k] = 1 iff detector record j and
        herald record k are both Z records (< P) and plaquette k is in
        F(support of plaquette j).

        The bare schedule leaves an X byproduct on plaquette k's z_side_data,
        heralded by Z-record k; it flips plaquette j's Z detector iff that
        detector's support overlaps z_side_data[k] oddly (k in F(supp j)). X
        records (>= P) neither carry nor receive the byproduct (an X byproduct
        commutes with the X stabilizers), so those rows and columns are zero.
        Diagonal entries (k in F(supp k)) are intentional: cudaq::detector
        XOR-cancels the duplicated record.
        """
        P = self.grid.num_plaquettes
        z_side_data = self.grid.z_side_data()
        fb = np.zeros((2 * P, 2 * P), dtype=np.uint8)
        for j, plaq in enumerate(self.grid.plaquettes):
            for k in _feedback_set(z_side_data, plaq['data_qubits']):
                fb[j, k] = 1
        return fb

    def get_observable_inlined_feedback_z(self):
        """[1 x 2P] logical-Z observable feedback: column k = 1 iff k is a Z
        record (< P) and plaquette k is in F(logical_qubits).

        The logical Z (bottom edge, ``logical_qubits``) picks up plaquette k's
        Z-ancilla-heralded X byproduct iff it overlaps z_side_data[k] oddly.
        """
        P = self.grid.num_plaquettes
        z_side_data = self.grid.z_side_data()
        fb = np.zeros((1, 2 * P), dtype=np.uint8)
        for k in _feedback_set(z_side_data, self.grid.logical_qubits):
            fb[0, k] = 1
        return fb

    # get_observable_inlined_feedback_x is intentionally NOT defined: the
    # X-basis logical (logical X read from data measured in X) commutes with
    # the schedule's X byproduct, so it needs no feedback. The bridge maps the
    # absent method to the base-class empty tensor, which flattens to no
    # observable feedback on the X path (py_code.cpp get_observable_inlined_
    # feedback_x -> code::... empty; flatten_feedback_tensor empty -> {}).


if __name__ == "__main__":
    for d in [3, 5, 7]:
        print("=" * 60)
        c = ColorCodeGeometry(d)
        c.print_structure()
