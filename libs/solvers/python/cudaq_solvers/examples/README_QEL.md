# Quantum Exact Lanczos (QEL) Algorithm

## Overview

The Quantum Exact Lanczos (QEL) algorithm is a hybrid quantum-classical method for finding ground state energies of quantum many-body systems. QEL uses block encoding and quantum phase estimation techniques to build a Krylov subspace by collecting moments via quantum measurements, then solves a generalized eigenvalue problem classically to extract eigenvalues.

## Key Features

- **Block Encoding**: Efficient encoding of Hamiltonians into unitary operators
- **Pauli LCU**: Optimized Linear Combination of Unitaries for Pauli Hamiltonians
- **Krylov Subspace Methods**: Builds a compressed representation of the Hamiltonian
- **Hybrid Approach**: Quantum moment collection + classical diagonalization
- **Flexible**: Returns matrices for user to diagonalize with preferred library

## Algorithm Overview

### 1. Block Encoding

QEL uses block encoding to represent the Hamiltonian H as a unitary operator U acting on an extended Hilbert space:

```
(⟨0|_anc ⊗ I_sys) U (|0⟩_anc ⊗ I_sys) = H/α
```

where:
- `α` is the normalization constant (typically ||H||₁)
- Ancilla qubits encode the Hamiltonian structure
- System qubits represent the physical quantum state

The `PauliLCU` implementation optimizes this for Hamiltonians expressed as sums of Pauli strings, requiring only `⌈log₂(# terms)⌉` ancilla qubits.

### 2. Moment Collection

QEL collects moments μₖ = ⟨ψ|Hᵏ|ψ⟩ by:
1. Applying powers of the block encoding
2. Measuring expectation values
3. Using amplitude amplification via reflection operators

### 3. Krylov Matrix Construction

From collected moments, QEL constructs:
- **Hamiltonian matrix H**: Represents H in the Krylov basis
- **Overlap matrix S**: Accounts for non-orthogonality of the basis

### 4. Classical Eigenvalue Extraction

Solve the generalized eigenvalue problem:

```
H|v⟩ = E·S|v⟩
```

Then convert scaled eigenvalues to physical energies:

```
E_physical = E_scaled * α + constant_term
```

## Python API

### Classes

#### `BlockEncoding` (Abstract Base Class)

Abstract interface for block encoding implementations.

**Properties:**
- `num_ancilla: int` - Number of ancilla qubits required
- `num_system: int` - Number of system qubits
- `normalization: float` - Normalization constant α

**Methods:**
- `prepare(ancilla)` - Apply PREPARE operation
- `unprepare(ancilla)` - Apply PREPARE† (adjoint)
- `select(ancilla, system)` - Apply SELECT operation
- `apply(ancilla, system)` - Full block encoding (PREPARE → SELECT → PREPARE†)

#### `PauliLCU` (inherits from `BlockEncoding`)

Pauli Linear Combination of Unitaries block encoding.

```python
PauliLCU(hamiltonian: SpinOperator, num_qubits: int)
```

**Parameters:**
- `hamiltonian`: Target Hamiltonian as `cudaq.SpinOperator`
- `num_qubits`: Number of system qubits

**Additional Methods:**
- `get_angles()` - State preparation angles (for debugging)
- `get_term_controls()` - Binary control patterns
- `get_term_ops()` - Pauli operations
- `get_term_lengths()` - Operators per term
- `get_term_signs()` - Coefficient signs

**Example:**
```python
from cudaq import spin
import cudaq_solvers as solvers

h = 0.5 * spin.x(0) + 0.3 * spin.z(0)
encoding = solvers.PauliLCU(h, num_qubits=1)

print(f"Ancilla: {encoding.num_ancilla}")  # 1
print(f"Normalization: {encoding.normalization}")  # 0.8
```

#### `QELResult`

Result structure containing Krylov matrices and metadata.

**Attributes:**
- `hamiltonian_matrix: List[float]` - Flattened H matrix (row-major)
- `overlap_matrix: List[float]` - Flattened S matrix (row-major)
- `moments: List[float]` - Collected moments
- `krylov_dimension: int` - Krylov subspace dimension
- `constant_term: float` - Constant from Hamiltonian
- `normalization: float` - Block encoding normalization α
- `num_ancilla: int` - Ancilla qubits used
- `num_system: int` - System qubits

**Methods:**
- `get_hamiltonian_matrix()` - Returns H as 2D NumPy array
- `get_overlap_matrix()` - Returns S as 2D NumPy array
- `get_moments()` - Returns moments as NumPy array

### Functions

#### `quantum_exact_lanczos`

Main QEL algorithm function.

```python
quantum_exact_lanczos(
    hamiltonian: SpinOperator,
    num_qubits: int,
    n_electrons: int,
    **kwargs
) -> QELResult
```

**Parameters:**
- `hamiltonian`: Target Hamiltonian
- `num_qubits`: Number of system qubits
- `n_electrons`: Number of electrons (for Hartree-Fock initial state)

**Keyword Arguments:**
- `krylov_dim: int = 10` - Krylov subspace dimension
- `shots: int = -1` - Measurement shots (-1 for exact simulation)
- `verbose: bool = False` - Enable detailed logging

**Returns:**
- `QELResult` containing Krylov matrices and metadata

## Usage Examples

### Basic Usage

```python
import numpy as np
from scipy import linalg as la
from cudaq import spin
import cudaq_solvers as solvers

# Define H2 Hamiltonian
h2 = (-1.0523732 + 
      0.39793742 * spin.z(0) - 
      0.39793742 * spin.z(1) - 
      0.01128010 * spin.z(2) + 
      0.01128010 * spin.z(3) + 
      0.18093120 * spin.x(0) * spin.x(1) * spin.y(2) * spin.y(3))

# Run QEL
result = solvers.quantum_exact_lanczos(
    h2,
    num_qubits=4,
    n_electrons=2,
    krylov_dim=5
)

# Extract matrices
H = result.get_hamiltonian_matrix()
S = result.get_overlap_matrix()

# Solve generalized eigenvalue problem
eigenvalues = la.eigh(H, S, eigvals_only=True)

# Convert to physical energies
energies = eigenvalues * result.normalization + result.constant_term

# Filter to Chebyshev range
mask = np.abs(eigenvalues) <= 1.0
physical_energies = energies[mask]

ground_energy = physical_energies.min()
print(f"Ground state: {ground_energy:.6f} Ha")
```

### With Overlap Matrix Regularization

For numerical stability, add small regularization:

```python
# Run QEL
result = solvers.quantum_exact_lanczos(
    hamiltonian,
    num_qubits=4,
    n_electrons=2,
    krylov_dim=8
)

# Extract and regularize
H = result.get_hamiltonian_matrix()
S = result.get_overlap_matrix() + 1e-12 * np.eye(8)

# Solve
eigenvalues = la.eigh(H, S, eigvals_only=True)
energies = eigenvalues * result.normalization + result.constant_term

# Filter and extract ground state
mask = np.abs(eigenvalues) <= 1.0
ground_energy = energies[mask].min()
```

### Convergence Study

```python
krylov_dimensions = [3, 5, 7, 10, 15]

for kdim in krylov_dimensions:
    result = solvers.quantum_exact_lanczos(
        hamiltonian,
        num_qubits=4,
        n_electrons=2,
        krylov_dim=kdim
    )
    
    H = result.get_hamiltonian_matrix()
    S = result.get_overlap_matrix() + 1e-12 * np.eye(kdim)
    
    eigenvalues = la.eigh(H, S, eigvals_only=True)
    energies = eigenvalues * result.normalization + result.constant_term
    
    mask = np.abs(eigenvalues) <= 1.0
    ground = energies[mask].min()
    
    print(f"Krylov dim {kdim}: E = {ground:.6f} Ha")
```

### Using Block Encoding Directly

```python
from cudaq import spin
import cudaq
import cudaq_solvers as solvers

# Create Hamiltonian and encoding
h = 0.7 * spin.z(0) + 0.3 * spin.x(0)
encoding = solvers.PauliLCU(h, num_qubits=1)

# Use in quantum kernel
@cudaq.kernel
def circuit():
    anc = cudaq.qvector(encoding.num_ancilla)
    sys = cudaq.qvector(encoding.num_system)
    
    # Initialize system state
    x(sys[0])  # |1⟩ state
    
    # Apply block encoding
    encoding.apply(anc, sys)
    
    # Measure
    mz(sys)

result = cudaq.sample(circuit)
```

## Algorithm Parameters

### Krylov Dimension

The Krylov dimension controls the size of the subspace:
- **Smaller** (3-5): Faster, less accurate
- **Larger** (10-20): Slower, more accurate
- **Rule of thumb**: Start with 10, increase if needed

Trade-off:
- Larger dimension → more quantum measurements → longer runtime
- Larger dimension → better eigenvalue approximation

### Shots

Number of measurement shots per observable:
- `-1`: Exact simulation (statevector)
- `1000-10000`: Typical for noisy quantum hardware
- Higher shots → better statistical accuracy → longer runtime

### Initial State

QEL uses Hartree-Fock initialization:
- First `n_electrons` qubits in |1⟩ state
- Remaining qubits in |0⟩ state

For other initial states, modify the C++ implementation.

## Eigenvalue Filtering

QEL uses Chebyshev polynomial expansion, so valid eigenvalues satisfy |λ| ≤ 1:

```python
eigenvalues = la.eigh(H, S, eigvals_only=True)

# Filter to valid range
mask = np.abs(eigenvalues) <= 1.0
valid_eigenvalues = eigenvalues[mask]

# Convert to physical energies
energies = valid_eigenvalues * result.normalization + result.constant_term
```

Eigenvalues outside [-1, 1] are numerical artifacts and should be discarded.

## Numerical Considerations

### Overlap Matrix Conditioning

The overlap matrix S can be ill-conditioned. Regularization helps:

```python
epsilon = 1e-12
S_reg = S + epsilon * np.eye(result.krylov_dimension)
```

### Eigenvalue Solver Choice

Options for solving `H|v⟩ = E·S|v⟩`:
- `scipy.linalg.eigh`: Standard, reliable
- `numpy.linalg.eigh`: Faster, but may fail for ill-conditioned S
- `scipy.linalg.eig`: General solver, returns complex eigenvalues

Recommendation: Use `scipy.linalg.eigh` with regularized S.

## Performance Considerations

### Ancilla Qubits

Required ancilla: `⌈log₂(# terms)⌉`
- 2 terms → 1 ancilla
- 10 terms → 4 ancilla
- 100 terms → 7 ancilla
- 1024 terms → 10 ancilla (maximum supported)

Large Hamiltonians may require truncation.

### Quantum Circuit Depth

Circuit depth per moment: O(Krylov_dim × # ancilla qubits)

Deeper circuits → more noise on real hardware → consider:
- Reducing Krylov dimension
- Using error mitigation techniques
- Truncating Hamiltonian

## Comparison with Other Methods

| Method | Quantum Cost | Classical Cost | Accuracy |
|--------|--------------|----------------|----------|
| QEL | O(Krylov_dim²) | O(Krylov_dim³) | High |
| VQE | O(iterations × terms) | Low | Variable |
| QPE | O(2^precision) | Low | Very High |
| Exact Diagonalization | None | O(2^3n) | Exact |

QEL is particularly useful when:
- System is too large for exact diagonalization (> 20 qubits)
- VQE optimization is challenging (barren plateaus, local minima)
- QPE precision requirements are too high

## References

1. Dong et al., "Ground state preparation and energy estimation on early fault-tolerant quantum computers via quantum eigenvalue transformation of unitary matrices" (2023)
2. Lin & Tong, "Near-optimal ground state preparation" (2020)
3. Low & Chuang, "Hamiltonian simulation by qubitization" (2017)

## See Also

- [qel_h2_example.py](qel_h2_example.py) - Complete working examples
- [test_quantum_exact_lanczos.py](../../tests/test_quantum_exact_lanczos.py) - Unit tests
- CUDA-Q Solvers documentation - Full API reference

