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

    # ----------------------------------------------------------------------------------
    # Circuit-only physical layout (rectangular grid with blanks, including ancillas)
    # ----------------------------------------------------------------------------------
    def get_circuit_physical_layout(
            self,
            *,
            id_order: str = "rtl",
            flip_rows: bool = False) -> Dict[int, Tuple[int, int]]:
        """
        Return a *circuit-only* physical layout mapping qubit_id -> (r, c) on a rectangular grid.

        This layout is meant ONLY for reasoning about 2D nearest-neighbor connectivity constraints during
        circuit construction. It does not affect the existing coordinate systems used elsewhere:
          - `qubit_to_coord` / `coord_to_qubit` (triangular user coords)
          - `qubit_to_grid` / `grid_to_qubit` (CNN rectangular embedding)

        Layout rules (as provided by user, generalized for odd distance d):
          - Number of rows equals `self.n_rows = d + (d-1)//2`.
          - Row lengths follow: 1, 3, 4, 5, 7, 8, 9, 11, 12, 13, ... (increments 2,1,1 repeating).
          - Within each row, tokens alternate between data sites (D) and ancilla pairs (X,Z), and the X/Z
            ancillas are adjacent horizontally with Z immediately to the left of X (left-to-right: Z then X).
          - Row start offsets create a triangular envelope; the resulting rectangular width is (2*d - 1).

        Mapping convention:
          - Data sites are assigned ids 0..num_data-1 using a deterministic per-row order controlled by `id_order`.
          - Ancillas are assigned ids in two global blocks (all X first, then all Z), and within each block we also
            use `id_order` per row.
          - NOTE: Even though physically we place Z before X in each row, we still assign X ids first globally.

        Args:
          id_order:
            - "rtl": assign ids within each row from right-to-left. This matches the legacy triangle's upside-down
              convention and makes the legacy schedule nearest-neighbor on this grid for odd distances tested.
            - "ltr": assign ids within each row from left-to-right (natural scan order).
          flip_rows:
            - If True, return coordinates after reflecting the row index: (r, c) -> (n_rows - 1 - r, c).
              This is useful when you want the same schedule embedded onto an up-facing vs down-facing triangle.
        """
        d = int(self.distance)
        width = 2 * d - 1
        n_rows = int(self.n_rows)
        if id_order not in ("rtl", "ltr"):
            raise ValueError("id_order must be 'rtl' or 'ltr'")

        # Row lengths: 1, 3, 4, 5, 7, 8, 9, 11, ...
        def row_len(row_idx: int) -> int:
            if row_idx == 0:
                return 1
            g = row_idx // 3
            pos = row_idx % 3
            # pos 0: +1 from previous end of triple => 2g+5 for g>=1 (works out from pattern)
            # easiest: build from recurrence: L0=1; delta pattern [2,1,1] repeating.
            # but closed form below matches the observed sequence:
            if pos == 0:
                return 2 * g + 5
            if pos == 1:
                return 2 * g + 3
            # pos == 2
            return 2 * g + 4

        # More robust: generate lengths by recurrence to avoid mistakes.
        lengths = [1]
        deltas = [2, 1, 1]
        for i in range(1, n_rows):
            lengths.append(lengths[-1] + deltas[(i - 1) % 3])

        # Row start offsets (column indices in [-d+1, d-1])
        starts = [0]
        for i in range(1, n_rows):
            # first delta=2 and first delta=1 both shift start left by 1; second delta=1 keeps start
            starts.append(starts[-1] + (-1 if (i - 1) % 3 in (0, 1) else 0))

        # Token generators per row type (Z then X)
        def tokens_for_row(row_idx: int, L: int) -> List[str]:
            if row_idx == 0:
                return ["D"]
            pos = row_idx % 3
            out: List[str] = []
            if pos == 0:
                # start with "DD"
                out.extend(["D", "D"])
            elif pos == 1:
                # start with "D"
                out.append("D")
            else:  # pos == 2
                # start with "ZX"
                out.extend(["Z", "X"])

            # Alternate between ZX and DD blocks.
            # Determine next block type based on what we ended with.
            next_block = "ZX" if (len(out) > 0 and out[-1] == "D") else "DD"
            while len(out) < L:
                if next_block == "ZX":
                    out.extend(["Z", "X"])
                    next_block = "DD"
                else:
                    out.extend(["D", "D"])
                    next_block = "ZX"
            return out[:L]

        # Collect token coordinates; assign ids afterward using `id_order`.
        layout: Dict[int, Tuple[int, int]] = {}
        d_positions: List[Tuple[int, int]] = []
        x_positions: List[Tuple[int, int]] = []
        z_positions: List[Tuple[int, int]] = []

        for r in range(n_rows):
            L = lengths[r]
            start = starts[r]
            toks = tokens_for_row(r, L)
            for j, tok in enumerate(toks):
                col = start + j
                # shift to [0..width-1]
                c = col + (d - 1)
                if not (0 <= c < width):
                    raise AssertionError(
                        f"Physical layout column out of range: row={r} col={col} width={width}"
                    )
                if tok == "D":
                    d_positions.append((r, c))
                elif tok == "X":
                    x_positions.append((r, c))
                elif tok == "Z":
                    z_positions.append((r, c))
                else:
                    raise AssertionError(f"Unknown token {tok}")

        if len(d_positions) != int(self.num_data):
            raise AssertionError(
                f"Expected exactly num_data={self.num_data} data sites in physical layout, got {len(d_positions)}"
            )

        def _row_sort_key(rc: Tuple[int, int]) -> Tuple[int, int]:
            rr, cc = rc
            return (rr, -cc) if id_order == "rtl" else (rr, cc)

        # Assign data ids
        data_id = 0
        for rc in sorted(d_positions, key=_row_sort_key):
            layout[data_id] = rc
            data_id += 1

        # Assign ancilla ids: X first globally, then Z (regardless of left-to-right placement).
        if len(x_positions) != int(
                self.num_plaquettes) or len(z_positions) != int(
                    self.num_plaquettes):
            raise AssertionError(
                f"Expected exactly num_plaquettes={self.num_plaquettes} X and Z ancillas in physical layout, "
                f"got X={len(x_positions)} Z={len(z_positions)}")

        x_id = int(self.num_data)
        for (r, c) in sorted(x_positions, key=_row_sort_key):
            layout[x_id] = (r, c)
            x_id += 1

        z_id = int(self.num_data + self.num_plaquettes)
        for (r, c) in sorted(z_positions, key=_row_sort_key):
            layout[z_id] = (r, c)
            z_id += 1

        if data_id != int(self.num_data) or x_id != int(
                self.num_data +
                self.num_plaquettes) or z_id != int(self.num_data +
                                                    2 * self.num_plaquettes):
            raise AssertionError(
                "Physical layout did not assign all qubits: "
                f"data={data_id}/{self.num_data} x={x_id - self.num_data}/{self.num_plaquettes} z={z_id - (self.num_data + self.num_plaquettes)}/{self.num_plaquettes}"
            )

        if flip_rows:
            H = n_rows
            return {q: (H - 1 - rc[0], rc[1]) for q, rc in layout.items()}
        return layout

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

    Both logicals use the bottom-edge representative (``logical_qubits``,
    weight exactly d) — see ColorCodeGeometry._find_logical_qubits for why
    the bottom edge is preferred over the full data-qubit support.
    """
    chars = ['I'] * grid.num_data
    for q in grid.logical_qubits:
        chars[q] = 'X'
    x_word = ''.join(chars)
    return x_word, x_word.replace('X', 'Z')


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


@cudaq.kernel
def stabilizer_round(logicalQubit: patch, x_stabilizers: list[int],
                     z_stabilizers: list[int]) -> list[cudaq.measure_handle]:
    # x_stabilizers / z_stabilizers are the flattened
    # [num_stabilizers x num_data] parity rows handed over by the framework
    # (row i, column q at index i * num_data + q).
    num_data = len(logicalQubit.data)

    # X-stabilizer half: ancilla in |+>, CX from ancilla onto plaquette data.
    h(logicalQubit.ancx)
    for xi in range(len(logicalQubit.ancx)):
        for di in range(num_data):
            if x_stabilizers[xi * num_data + di] == 1:
                x.ctrl(logicalQubit.ancx[xi], logicalQubit.data[di])
    h(logicalQubit.ancx)

    # Z-stabilizer half: CX from plaquette data onto ancilla.
    for zi in range(len(logicalQubit.ancz)):
        for di in range(num_data):
            if z_stabilizers[zi * num_data + di] == 1:
                x.ctrl(logicalQubit.data[di], logicalQubit.ancz[zi])

    # [Z][X] measurement order matches the framework's per-round detector
    # layout.
    results = mz([*logicalQubit.ancz, *logicalQubit.ancx])

    reset(logicalQubit.ancx)
    reset(logicalQubit.ancz)
    return results


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

    The full grid geometry (plaquette layout, syndrome-to-grid mapping, CNN
    embedding, physical circuit layout) is exposed on the ``grid`` attribute,
    an instance of :class:`ColorCodeGeometry`.
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

        x_words, z_words = _plaquette_pauli_words(self.grid)
        self.stabilizers = [
            cudaq.SpinOperator.from_word(w) for w in z_words + x_words
        ]
        obs_x, obs_z = _logical_pauli_words(self.grid)
        self.pauli_observables = [
            cudaq.SpinOperator.from_word(w) for w in (obs_x, obs_z)
        ]

        self.operation_encodings = {
            qec.operation.prep0: prep0,
            qec.operation.prep1: prep1,
            qec.operation.prepp: prepp,
            qec.operation.prepm: prepm,
            qec.operation.stabilizer_round: stabilizer_round,
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


# ---------------------------------------------------------------------------
# Superdense memory circuit
#
# Superdense memory circuit for the triangular color code.
#
# Each plaquette measures both its X and Z stabilizer in one round using a
# paired ancilla: the X-ancilla is prepared in |+>, the Z-ancilla in |0>, and
# an 8-CNOT-layer schedule entangles the pair (layers 1 and 8), couples each
# plaquette's data qubits into the pair with data as control (layers 2-4), and
# mirrors those couplings with the ancillas as control (layers 5-7). The
# mirrored half applies, per plaquette, an X on the data qubits coupled to the
# Z-ancilla conditioned on that ancilla's state; rather than measuring
# mid-circuit and feeding forward, the detectors carry the feedback-inlined
# parities, so the circuit stays straight-line Clifford.
#
# Label space used by this module: with N data qubits and P plaquettes,
# data qubits are 0..N-1, N + i is the X-ancilla of plaquette i, and
# N + P + i is the Z-ancilla of plaquette i (plaquette index order of
# :class:`ColorCodeGeometry`).
#
# Entry points: :func:`superdense_dem` (detector error model) and
# :func:`superdense_sample` (detector-ordered syndromes, final data readout,
# and logical parity). Both drive the :func:`superdense_memory` kernel, whose
# detector/observable annotations are emitted from a
# :class:`SuperdensePlan` — the host-side source of truth for the record
# layout, the per-detector measurement sets, and the detector coordinates.
#
# Decoder integration: `cudaq.detector` carries no coordinates, so the DEM
# returned by :func:`superdense_dem` is coordinate-free.
# ``SuperdensePlan(distance, rounds, basis).detector_coords`` supplies the
# 4-tuple ``(grid_row, grid_col, t, c)`` for each detector row, in emission
# order, with ``c = color + 3*basis`` (red/green/blue = 0/1/2; +3 for
# Z-checks) — the annotation convention Chromobius expects for color-code
# decoding.
# ---------------------------------------------------------------------------


def _rowmajor_cnot_layers(d: int) -> list[list[tuple[int, int]]]:
    """
    The 8 CNOT layers of the superdense schedule in row-major inverted-triangle
    numbering.

    The triangle is oriented with its widest row (d data qubits) on top,
    narrowing to a single qubit at the bottom.

    Returns:
        layers: length-8 list; each element is a list of (control, target)
        pairs where
          - data qubits: 0..num_data-1, row-major top-to-bottom
          - ancillas: num_data..num_data+2*num_plaquettes-1, grouped as
            (a1, a2) = (num_data + 2p, num_data + 2p + 1) with a1 the
            X-ancilla and a2 the Z-ancilla of plaquette p
    """
    d = int(d)
    num_data = (3 * d * d + 1) // 4
    num_plaquettes = (3 * (d * d - 1)) // 8
    num_rows_data_qubits = (3 * d - 1) // 2

    layers: list[list[tuple[int, int]]] = []

    # tt=1..8 correspond to 8 layers
    for tt in range(1, 9):
        edges: list[tuple[int, int]] = []

        if tt == 1 or tt == 8:
            # CNOT between the two ancilla qubits: control |+> (a1), target |0> (a2)
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


def _rowmajor_to_unified(grid: ColorCodeGeometry) -> dict[int, int]:
    """
    Bijection from the row-major inverted-triangle qubit numbering of
    :func:`_rowmajor_cnot_layers` to the unified label space.

    Data qubits: the grid's coordinates put row 0 at the top with rows
    decreasing (more negative) downward, so sorting data ids by (row, col)
    ascending enumerates them bottom-to-top, left-to-right — exactly the
    row-major order of the inverted triangle (widest row first).

    Ancillas: plaquettes are matched by their data-qubit support sets (the
    data endpoints coupled to an ancilla pair across all layers), then
    (a1, a2) = (num_data + 2p, num_data + 2p + 1) maps to
    (N + i, N + P + i) for the matching plaquette index i.
    """
    N = int(grid.num_data)
    P = int(grid.num_plaquettes)

    ordered_data = sorted(
        range(N),
        key=lambda q: (grid.qubit_to_coord[q][0], grid.qubit_to_coord[q][1]))
    data_to_rowmajor = {q: i for i, q in enumerate(ordered_data)}

    layers = _rowmajor_cnot_layers(grid.distance)

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
    for i, plaq in enumerate(grid.plaquettes):
        key = tuple(sorted(data_to_rowmajor[q] for q in plaq['data_qubits']))
        p = support_to_plaq.get(key)
        if p is None:
            raise ValueError(
                f"Plaquette {i} (data qubits {plaq['data_qubits']}) has no "
                f"support-set match in the schedule")
        mapping[N + 2 * p] = N + i
        mapping[N + 2 * p + 1] = N + P + i
    return mapping


def superdense_cnot_layers(
        grid: ColorCodeGeometry) -> list[list[tuple[int, int]]]:
    """Eight CNOT layers of the paired-ancilla superdense schedule.

    Unified label space: 0..N-1 data qubit ids, N+i the X-ancilla and
    N+P+i the Z-ancilla of plaquette i. Layers 1 and 8 entangle each
    plaquette's ancilla pair; layers 2-4 couple data (control) into the
    pair; layers 5-7 apply the mirrored ancilla-controlled CNOTs.
    """
    mapping = _rowmajor_to_unified(grid)
    return [
        sorted((mapping[c], mapping[t])
               for c, t in layer)
        for layer in _rowmajor_cnot_layers(grid.distance)
    ]


def z_side_data(grid: ColorCodeGeometry) -> list[list[int]]:
    """Per plaquette, the data qubits coupled to its Z-ancilla by the
    schedule; a conditional X on exactly these qubits is the byproduct the
    Z-ancilla readout heralds."""
    layers = superdense_cnot_layers(grid)
    N, P = grid.num_data, grid.num_plaquettes
    out = [set() for _ in range(P)]
    for li in (1, 2, 3):
        for c, t in layers[li]:
            if t >= N + P:
                out[t - N - P].add(c)
    return [sorted(s) for s in out]


_COLOR_IDX = {'red': 0, 'green': 1, 'blue': 2}


class SuperdensePlan:
    """Detector/observable layout of the superdense memory circuit.

    Record indexing: round r contributes records [2*P*r, 2*P*r + P) for the
    Z-ancillas (plaquette order) then [2*P*r + P, 2*P*(r+1)) for the
    X-ancillas; the final transversal data measurement occupies
    [2*P*rounds, 2*P*rounds + N) in data-qubit order.

    The Z-ancilla readouts herald conditional-X byproducts on their
    schedule-side data qubits; detectors and the Z-basis observable carry
    the absorbed correction records instead of physical feedback gates.
    """

    def __init__(self, distance, rounds, basis):
        if basis not in ('X', 'Z'):
            raise ValueError("basis must be 'X' or 'Z'")
        if rounds < 2:
            raise ValueError("rounds must be >= 2")
        self.grid = ColorCodeGeometry(distance)
        self.rounds = rounds
        self.basis = basis
        grid = self.grid
        P, N = grid.num_plaquettes, grid.num_data
        self.num_records = 2 * P * rounds + N

        zside = z_side_data(grid)

        def F(support):
            s = set(support)
            return [a for a in range(P) if len(set(zside[a]) & s) % 2 == 1]

        self.prev_sets = [
            sorted({p} ^ set(F(grid.plaquettes[p]['data_qubits'])))
            for p in range(P)
        ]
        self.obs_plaqs = sorted(F(grid.logical_qubits))

        Z = lambda p, r: 2 * P * r + p
        X = lambda p, r: 2 * P * r + P + p
        D = lambda q: 2 * P * rounds + q

        def coords(p, t, z_type):
            gp = grid.plaquettes[p]['grid_pos']
            c = _COLOR_IDX[grid.plaquettes[p]['color']] + (3 if z_type else 0)
            return (float(gp[0]), float(gp[1]), float(t), float(c))

        dets, cs = [], []
        for p in range(P):
            if basis == 'Z':
                dets.append(frozenset({Z(p, 0)}))
                cs.append(coords(p, 0, True))
            else:
                dets.append(frozenset({X(p, 0)}))
                cs.append(coords(p, 0, False))
        for r in range(1, rounds):
            for p in range(P):
                dets.append(frozenset({X(p, r), X(p, r - 1)}))
                cs.append(coords(p, 0, False))
            for p in range(P):
                dets.append(
                    frozenset({Z(p, r)}) |
                    {Z(a, r - 1) for a in self.prev_sets[p]})
                cs.append(coords(p, 0, True))
        for p in range(P):
            supp = grid.plaquettes[p]['data_qubits']
            if basis == 'Z':
                dets.append(
                    frozenset(D(q) for q in supp) |
                    {Z(a, rounds - 1) for a in self.prev_sets[p]})
                cs.append(coords(p, 1, True))
            else:
                dets.append(
                    frozenset({X(p, rounds - 1)}) | {D(q) for q in supp})
                cs.append(coords(p, 1, False))
        self.detectors = dets
        self.detector_coords = cs

        if basis == 'Z':
            obs = {D(q) for q in grid.logical_qubits}
            for r in range(rounds):
                obs ^= {Z(a, r) for a in self.obs_plaqs}
            self.observable = frozenset(obs)
        else:
            self.observable = frozenset(D(q) for q in range(N))

    def detector_bits(self, measurements):
        m = np.asarray(measurements, dtype=np.uint8)
        out = np.zeros((m.shape[0], len(self.detectors)), dtype=np.uint8)
        for j, s in enumerate(self.detectors):
            out[:, j] = np.bitwise_xor.reduce(m[:, sorted(s)], axis=1)
        return out

    def observable_bits(self, measurements):
        m = np.asarray(measurements, dtype=np.uint8)
        return np.bitwise_xor.reduce(m[:, sorted(self.observable)], axis=1)


def _kernel_args(plan, p_cx):
    if p_cx < 0:
        raise ValueError(f"p_cx must be non-negative, got {p_cx}")
    grid = plan.grid
    N, P, R = grid.num_data, grid.num_plaquettes, plan.rounds
    layers = superdense_cnot_layers(grid)
    sched = [q for layer in layers for e in layer for q in e]
    sched_off = [0]
    for layer in layers:
        sched_off.append(sched_off[-1] + 2 * len(layer))
    # prev_sets and final supports, padded to fixed strides
    ps_stride = max(len(s) for s in plan.prev_sets)
    ps_flat, ps_cnt = [], []
    for s in plan.prev_sets:
        ps_cnt.append(len(s))
        ps_flat.extend(s + [0] * (ps_stride - len(s)))
    supp_stride = 6
    supp_flat, supp_cnt = [], []
    for p in range(P):
        supp = grid.plaquettes[p]['data_qubits']
        supp_cnt.append(len(supp))
        supp_flat.extend(list(supp) + [0] * (supp_stride - len(supp)))
    obs = sorted(plan.obs_plaqs)
    logical = list(grid.logical_qubits)
    return (N, P, R, 1 if plan.basis == 'Z' else 0, float(p_cx), sched,
            sched_off, ps_flat, ps_cnt, ps_stride, supp_flat, supp_cnt,
            supp_stride, obs, logical)


@cudaq.kernel
def superdense_memory(N: int, P: int, R: int, z_basis: int, p_cx: float,
                      sched: list[int], sched_off: list[int],
                      ps_flat: list[int], ps_cnt: list[int], ps_stride: int,
                      supp_flat: list[int], supp_cnt: list[int],
                      supp_stride: int, obs: list[int], logical: list[int]):
    # One register indexed directly by the unified labels: q[0..N) data,
    # q[N..N+P) X-ancillas, q[N+P..N+2P) Z-ancillas (plaquette order), so
    # the sched entries need no decoding. Per round the Z-ancillas are
    # measured before the X-ancillas and the final data measurement comes
    # last: exactly the SuperdensePlan record layout.
    q = cudaq.qvector(N + 2 * P)

    if z_basis == 0:
        for i in range(N):
            h(q[i])

    # Round 0. Fresh qubits start in |0>, so the per-round ancilla resets
    # are elided; only the X-ancilla |+> prep is needed.
    for i in range(P):
        h(q[N + i])
    for li in range(8):
        for k in range(sched_off[li], sched_off[li + 1], 2):
            x.ctrl(q[sched[k]], q[sched[k + 1]])
            # Perfect-final-round convention: noise on every schedule CX of
            # every round except the last.
            if p_cx > 0.0 and R > 1:
                cudaq.apply_noise(cudaq.Depolarization2, p_cx, q[sched[k]],
                                  q[sched[k + 1]])
    zprev = mz(q[N + P:N + 2 * P])
    if z_basis == 1:
        # Z-basis observable: the obs_plaqs Z-ancilla records of every
        # round accumulate into L0 (inlined feed-forward byproduct).
        for j in range(len(obs)):
            cudaq.logical_observable(zprev[obs[j]])
    for i in range(P):
        h(q[N + i])
    xprev = mz(q[N:N + P])
    # Round-0 detectors: memory-basis single-record checks.
    if z_basis == 1:
        for p in range(P):
            cudaq.detector(zprev[p])
    else:
        for p in range(P):
            cudaq.detector(xprev[p])

    for r in range(1, R):
        for i in range(P):
            reset(q[N + i])
            h(q[N + i])
            reset(q[N + P + i])
        for li in range(8):
            for k in range(sched_off[li], sched_off[li + 1], 2):
                x.ctrl(q[sched[k]], q[sched[k + 1]])
                if p_cx > 0.0 and r < R - 1:
                    cudaq.apply_noise(cudaq.Depolarization2, p_cx, q[sched[k]],
                                      q[sched[k + 1]])
        zcurr = mz(q[N + P:N + 2 * P])
        if z_basis == 1:
            for j in range(len(obs)):
                cudaq.logical_observable(zcurr[obs[j]])
        for i in range(P):
            h(q[N + i])
        xcurr = mz(q[N:N + P])
        for p in range(P):
            cudaq.detector(xcurr[p], xprev[p])
        for p in range(P):
            # Z comparison detector: current Z record against the previous
            # round's prev_set records (plaquette indices into zprev).
            if ps_cnt[p] == 0:
                cudaq.detector(zcurr[p])
            else:
                psel = [
                    zprev[ps_flat[p * ps_stride + j]] for j in range(ps_cnt[p])
                ]
                cudaq.detector(zcurr[p], psel)
        zprev = zcurr
        xprev = xcurr

    if z_basis == 0:
        for i in range(N):
            h(q[i])
    dm = mz(q[0:N])
    for p in range(P):
        dsel = [dm[supp_flat[p * supp_stride + j]] for j in range(supp_cnt[p])]
        if z_basis == 1:
            if ps_cnt[p] == 0:
                cudaq.detector(dsel)
            else:
                psel = [
                    zprev[ps_flat[p * ps_stride + j]] for j in range(ps_cnt[p])
                ]
                cudaq.detector(dsel, psel)
        else:
            cudaq.detector(xprev[p], dsel)
    if z_basis == 1:
        for j in range(len(logical)):
            cudaq.logical_observable(dm[logical[j]])
    else:
        cudaq.logical_observable(dm)


def superdense_dem(distance, rounds, basis, p_cx):
    """Detector error model of the superdense memory circuit.

    The returned model's detector rows follow the plan's emission order;
    chromobius coordinates for each row are available from
    SuperdensePlan(distance, rounds, basis).detector_coords.
    """
    plan = SuperdensePlan(distance, rounds, basis)
    # An (empty) noise model must be passed or in-kernel apply_noise is a
    # silent no-op and the DEM comes back with zero error mechanisms.
    text = cudaq.dem_from_kernel(superdense_memory,
                                 *_kernel_args(plan, p_cx),
                                 noise_model=cudaq.NoiseModel())
    return qec.dem_from_stim_text(text)


def superdense_sample(distance, rounds, basis, p_cx, shots):
    """Sample the superdense memory circuit.

    Returns:
        (syndromes, data, logical): syndromes is a (shots, 2*P*rounds) uint8
        array of detector bits in plan emission order; data is a (shots, N)
        array of raw final data-qubit bits; logical is a (shots,) array of
        observable parity.
    """
    plan = SuperdensePlan(distance, rounds, basis)
    # Same activation requirement as in superdense_dem: without a noise
    # model argument the in-kernel apply_noise calls do nothing.
    res = cudaq.sample(superdense_memory,
                       *_kernel_args(plan, p_cx),
                       shots_count=shots,
                       explicit_measurements=True,
                       noise_model=cudaq.NoiseModel())
    # get_sequential_data() yields one '0'/'1' string per shot in record
    # order; convert per character (np.array on the raw strings would parse
    # each whole string as a single number).
    m = np.array([[c == '1' for c in s] for s in res.get_sequential_data()],
                 dtype=np.uint8)
    if m.shape[1] != plan.num_records:
        raise RuntimeError(
            f"expected {plan.num_records} measurement records per shot, "
            f"got {m.shape[1]}")
    N = plan.grid.num_data
    return (plan.detector_bits(m), m[:, -N:], plan.observable_bits(m))


if __name__ == "__main__":
    for d in [3, 5, 7]:
        print("=" * 60)
        c = ColorCodeGeometry(d)
        c.print_structure()
