Variational Quantum Eigensolver (VQE)
-------------------------------------

The Variational Quantum Eigensolver (VQE) is a hybrid quantum-classical algorithm designed to find the ground state energy of a quantum system. It combines quantum computation with classical optimization to iteratively improve an approximation of the ground state.

Key features of VQE:

- Hybrid approach: Utilizes both quantum and classical resources efficiently.
- Variational method: Uses a parameterized quantum circuit (ansatz) to prepare trial states.
- Iterative optimization: Classical optimizer adjusts circuit parameters to minimize energy.
- Flexibility: Can be applied to various problems in quantum chemistry and materials science.

VQE Algorithm Overview:

1. Prepare an initial quantum state using a parameterized circuit (ansatz).
2. Measure the expectation value of the Hamiltonian.
3. Use a classical optimizer to adjust circuit parameters.
4. Repeat steps 1-3 until convergence or a stopping criterion is met.

CUDA-Q Solvers Implementation
+++++++++++++++++++++++++++++

CUDA-Q Solvers provides a high-level interface for running VQE simulations. Here's how to use it in both Python and C++:

.. tab:: Python

   .. literalinclude:: ../../examples/solvers/python/uccsd_vqe.py
      :language: python
      :start-after: [Begin Documentation]

.. tab:: C++

   .. literalinclude:: ../../examples/solvers/cpp/uccsd_vqe.cpp
      :language: cpp
      :start-after: [Begin Documentation]

   Compile and run with

   .. code-block:: bash

      nvq++ --enable-mlir -lcudaq-solvers uccsd_vqe.cpp -o uccsd_vqe
      ./uccsd_vqe

Code Explanation
++++++++++++++++

1. Molecule Creation:
   - Both examples start by defining the molecular geometry (H2 molecule).
   - The `create_molecule` function generates the molecular Hamiltonian.

2. Ansatz Definition:
   - The UCCSD (Unitary Coupled Cluster Singles and Doubles) ansatz is used.
   - In Python, it's defined as a `cudaq.kernel`.
   - In C++, it's defined as a lambda function within the VQE call.

3. VQE Execution:
   - The `solvers.vqe` function (Python) or `solvers::vqe` (C++) is called.
   - It takes the ansatz, Hamiltonian, initial parameters, and optimization settings.

4. Optimization:
   - Python uses SciPy's `minimize` function with L-BFGS-B method.
   - C++ uses CUDA-Q Solvers' built-in optimizer.
   - Either language can make use of CUDA-QX builtin optimizers.

5. Results:
   - Both versions print the final ground state energy.

The CUDA-Q Solvers implementation of VQE provides a high-level interface that handles the quantum-classical hybrid optimization loop, making it easy to apply VQE to molecular systems. Users can focus on defining the problem (molecule and ansatz) while CUDA-Q Solvers manages the complex interaction between quantum and classical resources.
