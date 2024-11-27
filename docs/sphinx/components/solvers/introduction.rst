CUDA-Q Solvers Library
=======================

Overview
--------
The CUDA-Q Solvers library provides high-level quantum-classical hybrid 
algorithms and supporting infrastructure for quantum chemistry and 
optimization problems. It features implementations of VQE, ADAPT-VQE, 
and supporting utilities for Hamiltonian generation and operator pool management.

Core Components
-----------------

1. **Variational Algorithms**:

   * Variational Quantum Eigensolver (VQE)
   * Adaptive Derivative-Assembled Pseudo-Trotter VQE (ADAPT-VQE)

2. **Quantum Chemistry Tools**:

   * Molecular Hamiltonian Generation
   * One-Particle Operator Creation
   * Geometry Management

3. **Operator Infrastructure**:

   * Operator Pool Generation
   * Fermion-to-Qubit Mappings
   * Gradient Computation

Operator Infrastructure 
------------------------

Molecular Hamiltonian Options
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :code:`molecule_options` structure provides extensive configuration for molecular calculations in CUDA-QX.

+---------------------+---------------+------------------+------------------------------------------+
| Option              | Type          | Default          | Description                              |
+=====================+===============+==================+==========================================+
| driver              | string        | "RESTPySCFDriver"| Quantum chemistry driver backend         |
+---------------------+---------------+------------------+------------------------------------------+
| fermion_to_spin     | string        | "jordan_wigner"  | Fermionic to qubit operator mapping      |
+---------------------+---------------+------------------+------------------------------------------+
| type                | string        | "gas_phase"      | Type of molecular system                 |
+---------------------+---------------+------------------+------------------------------------------+
| symmetry            | bool          | false            | Use molecular symmetry                   |
+---------------------+---------------+------------------+------------------------------------------+
| memory              | double        | 4000.0           | Memory allocation (MB)                   |
+---------------------+---------------+------------------+------------------------------------------+
| cycles              | size_t        | 100              | Maximum SCF cycles                       |
+---------------------+---------------+------------------+------------------------------------------+
| initguess           | string        | "minao"          | Initial SCF guess method                 |
+---------------------+---------------+------------------+------------------------------------------+
| UR                  | bool          | false            | Enable unrestricted calculations         |
+---------------------+---------------+------------------+------------------------------------------+
| nele_cas            | optional      | nullopt          | Number of electrons in active space      |
|                     | <size_t>      |                  |                                          |
+---------------------+---------------+------------------+------------------------------------------+
| norb_cas            | optional      | nullopt          | Number of spatial orbitals in            |
|                     | <size_t>      |                  | in active space                          |
+---------------------+---------------+------------------+------------------------------------------+
| MP2                 | bool          | false            | Enable MP2 calculations                  |
+---------------------+---------------+------------------+------------------------------------------+
| natorb              | bool          | false            | Use natural orbitals                     |
+---------------------+---------------+------------------+------------------------------------------+
| casci               | bool          | false            | Perform CASCI calculations               |
+---------------------+---------------+------------------+------------------------------------------+
| ccsd                | bool          | false            | Perform CCSD calculations                |
+---------------------+---------------+------------------+------------------------------------------+
| casscf              | bool          | false            | Perform CASSCF calculations              |
+---------------------+---------------+------------------+------------------------------------------+
| integrals_natorb    | bool          | false            | Use natural orbitals for integrals       |
+---------------------+---------------+------------------+------------------------------------------+
| integrals_casscf    | bool          | false            | Use CASSCF orbitals for integrals        |
+---------------------+---------------+------------------+------------------------------------------+
| potfile             | optional      | nullopt          | Path to external potential file          |
|                     | <string>      |                  |                                          |
+---------------------+---------------+------------------+------------------------------------------+
| verbose             | bool          | false            | Enable detailed output logging           |
+---------------------+---------------+------------------+------------------------------------------+

Example Usage
^^^^^^^^^^^^^

.. tab:: Python

    .. code-block:: python

        import cudaq_solvers as solvers
        
        # Configure molecular options
        options = {
            'fermion_to_spin': 'jordan_wigner',
            'casci': True,
            'memory': 8000.0,
            'verbose': True
        }
        
        # Create molecular Hamiltonian
        molecule = solvers.create_molecule(
            geometry=[('H', (0., 0., 0.)), 
                    ('H', (0., 0., 0.7474))],
            basis='sto-3g',
            spin=0,
            charge=0,
            **options
        )

.. tab:: C++

    .. code-block:: cpp

        using namespace cudaq::solvers; 

        // Configure molecular options
        molecule_options options;
        options.fermion_to_spin = "jordan_wigner";
        options.casci = true;
        options.memory = 8000.0;
        options.verbose = true;
        
        // Create molecular geometry
        auto geometry = molecular_geometry({
            atom{"H", {0.0, 0.0, 0.0}},
            atom{"H", {0.0, 0.0, 0.7474}}
        });
        
        // Create molecular Hamiltonian
        auto molecule = create_molecule(
            geometry,
            "sto-3g",
            0,  // spin
            0,  // charge
            options
        );

Variational Quantum Eigensolver (VQE)
--------------------------------------

The VQE algorithm finds the minimum eigenvalue of a 
Hamiltonian using a hybrid quantum-classical approach.

VQE Examples
-------------

The VQE implementation supports multiple usage patterns with different levels of customization.

Basic Usage
^^^^^^^^^^^

.. tab:: Python

    .. code-block:: python

        import cudaq
        from cudaq import spin
        import cudaq_solvers as solvers

        # Define quantum kernel (ansatz)
        @cudaq.kernel
        def ansatz(theta: float):
            q = cudaq.qvector(2)
            x(q[0])
            ry(theta, q[1])
            x.ctrl(q[1], q[0])

        # Define Hamiltonian
        H = 5.907 - 2.1433 * spin.x(0) * spin.x(1) - \
            2.1433 * spin.y(0) * spin.y(1) + \
            0.21829 * spin.z(0) - 6.125 * spin.z(1)

        # Run VQE with defaults (cobyla optimizer)
        energy, parameters, data = solvers.vqe(
            lambda thetas: ansatz(thetas[0]),
            H,
            initial_parameters=[0.0],
            verbose=True
        )
        print(f"Ground state energy: {energy}")

.. tab:: C++

    .. code-block:: cpp
        
        #include "cudaq.h"
        
        #include "cudaq/solvers/operators.h"
        #include "cudaq/solvers/vqe.h"
        
        // Define quantum kernel
        struct ansatz {
          void operator()(std::vector<double> theta) __qpu__ {
              cudaq::qvector q(2);
              x(q[0]);
              ry(theta[0], q[1]);
              x<cudaq::ctrl>(q[1], q[0]);
          }
        };
        
        // Create Hamiltonian
        auto H = 5.907 - 2.1433 * x(0) * x(1) - 
                2.1433 * y(0) * y(1) +
                0.21829 * z(0) - 6.125 * z(1);
        
        // Run VQE with default optimizer
        auto result = cudaq::solvers::vqe(
            ansatz{},
            H,
            {0.0},  // Initial parameters
            {{"verbose", true}}
        );
        printf("Ground state energy: %lf\n", result.energy);

Custom Optimization
^^^^^^^^^^^^^^^^^^^

.. tab:: Python

    .. code-block:: python

        # Using L-BFGS-B optimizer with parameter-shift gradients
        energy, parameters, data = solvers.vqe(
            lambda thetas: ansatz(thetas[0]),
            H,
            initial_parameters=[0.0],
            optimizer='lbfgs',
            gradient='parameter_shift',
            verbose=True
        )

        # Using SciPy optimizer directly
        from scipy.optimize import minimize
        
        def callback(xk):
            exp_val = cudaq.observe(ansatz, H, xk[0]).expectation()
            print(f"Energy at iteration: {exp_val}")
        
        energy, parameters, data = solvers.vqe(
            lambda thetas: ansatz(thetas[0]),
            H,
            initial_parameters=[0.0],
            optimizer=minimize,
            callback=callback,
            method='L-BFGS-B',
            jac='3-point',
            tol=1e-4,
            options={'disp': True}
        )

.. tab:: C++

    .. code-block:: cpp

        // Using L-BFGS optimizer with central difference gradients
        auto optimizer = cudaq::optim::optimizer::get("lbfgs");
        auto gradient = cudaq::observe_gradient::get(
            "central_difference", 
            ansatz{}, 
            H
        );
        
        auto result = cudaq::solvers::vqe(
            ansatz{},
            H,
            *optimizer,
            *gradient,
            {0.0},  // Initial parameters
            {{"verbose", true}}
        );

Shot-based Simulation
^^^^^^^^^^^^^^^^^^^^^

.. tab:: Python

    .. code-block:: python

        # Run VQE with finite shots
        energy, parameters, data = solvers.vqe(
            lambda thetas: ansatz(thetas[0]),
            H,
            initial_parameters=[0.0],
            shots=10000,
            max_iterations=10,
            verbose=True
        )
        
        # Analyze measurement data
        for iteration in data:
            counts = iteration.result.counts()
            print("\nMeasurement counts:")
            print("XX basis:", counts.get_register_counts('XX'))
            print("YY basis:", counts.get_register_counts('YY'))
            print("ZI basis:", counts.get_register_counts('ZI'))
            print("IZ basis:", counts.get_register_counts('IZ'))

.. tab:: C++

    .. code-block:: cpp

        // Run VQE with finite shots
        auto optimizer = cudaq::optim::optimizer::get("lbfgs");
        auto gradient = cudaq::observe_gradient::get(
            "parameter_shift",
            ansatz{},
            H
        );
        
        auto result = cudaq::solvers::vqe(
            ansatz{},
            H,
            *optimizer,
            *gradient,
            {0.0},
            {
                {"shots", 10000},
                {"verbose", true}
            }
        );
        
        // Analyze measurement data
        for (auto& iteration : result.iteration_data) {
            std::cout << "Iteration type: " 
                    << (iteration.type == observe_execution_type::gradient 
                        ? "gradient" : "function") 
                    << "\n";
            iteration.result.dump();
        }

ADAPT-VQE
---------

The Adaptive Derivative-Assembled Pseudo-Trotter Variational Quantum Eigensolver (ADAPT-VQE) 
is an advanced quantum algorithm that dynamically builds a problem-tailored ansatz 
based on operator gradients.

Key Features
^^^^^^^^^^^^

* Dynamic ansatz construction
* Gradient-based operator selection
* Automatic termination criteria
* Support for various operator pools
* Compatible with multiple optimizers

Basic Usage
^^^^^^^^^^^^

.. tab:: Python

    .. code-block:: python

        import cudaq
        import cudaq_solvers as solvers
        
        # Define molecular geometry
        geometry = [
            ('H', (0., 0., 0.)), 
            ('H', (0., 0., 0.7474))
        ]
        
        # Create molecular Hamiltonian
        molecule = solvers.create_molecule(
            geometry,
            'sto-3g',
            spin=0,
            charge=0,
            casci=True
        )
        
        # Generate operator pool
        operators = solvers.get_operator_pool(
            "spin_complement_gsd",
            num_orbitals=molecule.n_orbitals
        )
        
        numElectrons = molecule.n_electrons

        # Define initial state preparation
        @cudaq.kernel
        def initial_state(q: cudaq.qview):
            for i in range(numElectrons):
                x(q[i])
        
        # Run ADAPT-VQE
        energy, parameters, operators = solvers.adapt_vqe(
            initial_state,
            molecule.hamiltonian,
            operators,
            verbose=True
        )
        print(f"Ground state energy: {energy}")

.. tab:: C++

    .. code-block:: cpp

        #include "cudaq/solvers/adapt.h"
        #include "cudaq/solvers/operators.h"

        // compile with 
        // nvq++ adaptEx.cpp --enable-mlir -lcudaq-solvers
        // ./a.out 

        int main() {
            // Define initial state preparation
            auto initial_state = [](cudaq::qvector<>& q) __qpu__ {
                for (std::size_t i = 0; i < 2; ++i)
                    x(q[i]);
            };
            
            // Create Hamiltonian (H2 molecule example)
            cudaq::solvers::molecular_geometry geometry{{"H", {0., 0., 0.}},
                                                {"H", {0., 0., .7474}}};
            auto molecule = cudaq::solvers::create_molecule(
                geometry, "sto-3g", 0, 0, {.casci = true, .verbose = true});
            
            auto h = molecule.hamiltonian;
            
            // Generate operator pool
            auto operators = cudaq::solvers::get_operator_pool(
                "spin_complement_gsd", {
                {"num-orbitals", h.num_qubits() / 2}
            });
            
            // Run ADAPT-VQE
            auto [energy, parameters, selected_ops] = 
                cudaq::solvers::adapt_vqe(
                    initial_state,
                    h,
                    operators,
                    {
                        {"grad_norm_tolerance", 1e-3},
                        {"verbose", true}
                    }
                );
        }

Advanced Usage
^^^^^^^^^^^^^^^

Custom Optimization Settings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. tab:: Python

    .. code-block:: python

        # Using L-BFGS-B optimizer with central difference gradients
        energy, parameters, operators = solvers.adapt_vqe(
            initial_state,
            molecule.hamiltonian,
            operators,
            optimizer='lbfgs',
            gradient='central_difference',
            verbose=True
        )
        
        # Using SciPy optimizer directly
        from scipy.optimize import minimize
        energy, parameters, operators = solvers.adapt_vqe(
            initial_state,
            molecule.hamiltonian,
            operators,
            optimizer=minimize,
            method='L-BFGS-B',
            jac='3-point',
            tol=1e-8,
            options={'disp': True}
        )

.. tab:: C++

    .. code-block:: cpp

        // Using L-BFGS optimizer with central difference gradients
        auto optimizer = cudaq::optim::optimizer::get("lbfgs");
        auto [energy, parameters, operators] = 
            cudaq::solvers::adapt_vqe(
                initial_state{},
                h,
                operators,
                *optimizer,
                "central_difference",
                {
                    {"grad_norm_tolerance", 1e-3},
                    {"verbose", true}
                }
            );

Available Operator Pools
^^^^^^^^^^^^^^^^^^^^^^^^^

CUDA-QX provides several pre-built operator pools for ADAPT-VQE:

* **spin_complement_gsd**: Spin-complemented generalized singles and doubles
* **uccsd**: UCCSD operators
* **qaoa**: QAOA mixer excitation operators

.. code-block:: python

    # Generate different operator pools
    gsd_ops = solvers.get_operator_pool(
        "spin_complement_gsd",
        num_orbitals=molecule.n_orbitals
    )
    
    uccsd_ops = solvers.get_operator_pool(
        "uccsd",
        num_orbitals=molecule.n_orbitals,
        num_electrons=molecule.n_electrons
    )

Algorithm Parameters
^^^^^^^^^^^^^^^^^^^^^^

ADAPT-VQE supports various configuration options:

* **grad_norm_tolerance**: Convergence threshold for operator gradients
* **max_iterations**: Maximum number of ADAPT iterations
* **verbose**: Enable detailed output
* **shots**: Number of measurements for shot-based simulation

.. code-block:: python

    energy, parameters, operators = solvers.adapt_vqe(
        initial_state,
        hamiltonian,
        operators,
        grad_norm_tolerance=1e-3,
        max_iterations=20,
        verbose=True,
        shots=10000
    )

Results Analysis
^^^^^^^^^^^^^^^^^

The algorithm returns three components:

1. **energy**: Final ground state energy
2. **parameters**: Optimized parameters for each selected operator
3. **operators**: List of selected operators in order of application

.. code-block:: python

    # Analyze results
    print(f"Final energy: {energy}")
    print("\nSelected operators and parameters:")
    for param, op in zip(parameters, operators):
        print(f"θ = {param:.6f} : {op}")