import cudaq
import cudaq_solvers as solvers
from cudaq import spin
from typing import List


def get_identity(n_qubits: int) -> cudaq.SpinOperator:
    """
    Generate identity operator for n qubits.
    
    Args:
        n_qubits: Number of qubits.
    
    Returns:
        Identity SpinOperator (I ⊗ I ⊗ ... ⊗ I).
    """
    In = cudaq.spin.i(0)
    for q in range(1, n_qubits):
        In = In * cudaq.spin.i(q)
    return 1.0 * cudaq.SpinOperator(In)


def get_gqe_pauli_pool(num_qubits: int,
                       num_electrons: int,
                       params: List[float]) -> List[cudaq.SpinOperator]:
    """
    Generate a GQE operator pool based on individual UCCSD Pauli terms with parameter scaling.
    
    This function creates a pool by:
    1. Getting UCCSD operators
    2. Extracting Pauli string patterns from each term (ignoring original coefficients)
    3. Creating a separate SpinOperator for each Pauli string pattern
    4. Scaling each pattern by different parameter values
    
    Args:
        num_qubits: Total number of qubits in the system.
        num_electrons: Number of electrons in the system.
        params: List of parameter coefficients for scaling operators.
    
    Returns:
        List of cudaq.SpinOperator objects, each representing a single
        parameterized Pauli string.
    
    Example:
        >>> params = [0.01, -0.01, 0.05, -0.05, 0.1, -0.1]
        >>> pool = get_gqe_pauli_pool(num_qubits=4, num_electrons=2, params=params)
    """
    # Get base UCCSD operators
    uccsd_operators = solvers.get_operator_pool(
        "uccsd", num_qubits=num_qubits, num_electrons=num_electrons)
    
    # Start with identity operator
    pool = []
    pool.append(get_identity(num_qubits))
    
    # Extract individual Pauli string patterns (ignoring coefficients)
    individual_terms = []
    for op in uccsd_operators:
        for term in op:  # Iterate over terms in the SpinOperator
            # Get the Pauli word pattern and reconstruct as SpinOperator
            pauli_word = term.get_pauli_word(num_qubits)
            
            # Build spin operator from pauli_word string
            pauli_op = None
            for qubit_idx, pauli_char in enumerate(pauli_word):
                if pauli_char == 'I':
                    gate = spin.i(qubit_idx)
                elif pauli_char == 'X':
                    gate = spin.x(qubit_idx)
                elif pauli_char == 'Y':
                    gate = spin.y(qubit_idx)
                elif pauli_char == 'Z':
                    gate = spin.z(qubit_idx)
                else:
                    continue
                
                if pauli_op is None:
                    pauli_op = gate
                else:
                    pauli_op = pauli_op * gate
            
            if pauli_op is not None:
                individual_terms.append(cudaq.SpinOperator(pauli_op))
    
    # Add parameterized individual terms to pool
    for term_op in individual_terms:
        for param in params:
            pool.append(param * term_op)
    
    return pool
