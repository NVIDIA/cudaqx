Quantum Approximate Optimization Algorithm (QAOA)
-------------------------------------------------

The Quantum Approximate Optimization Algorithm (QAOA) is a hybrid quantum-classical algorithm that solves combinatorial optimization problems.

Key features of QAOA:

- Hybrid approach: Utilizes both quantum and classical resources efficiently.
- Iterative optimization: Classical optimizer adjusts circuit parameters to minimize energy.
- NISQ compatibility: This algorithm is designed to run on the noisy quantum computers of today.
- Flexibility: Can be applied to various problems in quantum chemistry and optimization problems broadly.

.. tab:: Python

   .. literalinclude:: ../../examples/solvers/python/molecular_docking_qaoa.py
      :language: python
      :start-after: [Begin Documentation]

.. tab:: C++

   .. literalinclude:: ../../examples/solvers/cpp/molecular_docking_qaoa.cpp
      :language: cpp
      :start-after: [Begin Documentation]

   Compile and run with

   .. code-block:: bash

      nvq++ --enable-mlir -lcudaq-solvers molecular_docking_qaoa.cpp -o molecular_docking_qaoa
      ./molecular_docking_qaoa

CUDA-Q Solvers Implementation
+++++++++++++++++++++++++++++
Here's how to use CUDA-Q Solvers to solve the Maximum Clique Problem using QAOA:

Code Explanation
++++++++++++++++
1. Graph Creation:
    - A NetworkX graph is created to represent the problem.
    - Nodes and edges are added with specific weights.

2. Clique Hamiltonian Generation:
    - `solvers.get_clique_hamiltonian` is used to create the Hamiltonian for the Maximum Clique Problem.
    - The penalty term and number of QAOA layers are defined.

3. QAOA Parameter Setup:
    - The number of required parameters is calculated using `solvers.get_num_qaoa_parameters`.
    - Randomly generate initial parameters.

4. QAOA Execution with `solvers.qaoa`:
    - Call the solver with the Hamiltonian, number of QAOA layers, and whether you want full parametrization and counterdiabatic driving.
    - Full parameterization: Uses an optimization parameter for every term in the clique Hamiltonian and the mixer Hamiltonian.
    - Counterdiabatic driving: Adds extra Ry rotations at the end of each layer.

5. Results Analysis:
    - The optimal energy, sampled states, and most probable configuration are printed.

This implementation showcases the power of CUDA-Q Solvers in solving combinatorial optimization problems using hybrid quantum-classical algorithms.
By using CUDA-Q Solvers with the networkx library, we very quickly set up and ran a QAOA application to compute optimal configurations for a molecular docking problem.



