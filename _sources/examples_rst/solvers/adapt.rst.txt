ADAPT-VQE
---------

ADAPT-VQE is an advanced quantum algorithm designed to improve upon the
standard Variational Quantum Eigensolver (VQE) approach for solving quantum
chemistry problems. It addresses key challenges faced by traditional VQE
methods by dynamically constructing a problem-specific ansatz, offering
several advantages:

- Faster convergence: Adaptively selects the most impactful operators, potentially achieving convergence more quickly than fixed-ansatz VQE methods.
- Enhanced efficiency: Builds a compact ansatz tailored to the specific problem, potentially reducing overall circuit depth.
- Increased accuracy: Has demonstrated the ability to outperform standard VQE approaches in terms of accuracy for certain molecular systems.
- Adaptability: Automatically adjusts to different molecular systems without requiring significant user intervention or prior knowledge of the system's electronic structure.

The ADAPT-VQE algorithm works by iteratively growing the quantum circuit
ansatz, selecting operators from a predefined pool based on their gradient
magnitudes. This adaptive approach allows the algorithm to focus
computational resources on the most relevant aspects of the problem,
potentially leading to more efficient and accurate simulations of molecular
systems on quantum computers.

Here we demonstrate how to use the CUDA-Q Solvers library to execute the ADAPT-VQE algorithm.

.. tab:: Python

   .. literalinclude:: ../../examples/solvers/python/adapt_h2.py
      :language: python
      :start-after: [Begin Documentation]

.. tab:: C++

   .. literalinclude:: ../../examples/solvers/cpp/adapt_h2.cpp
      :language: cpp
      :start-after: [Begin Documentation]

   Compile and run with

   .. code:: bash

       nvq++ --enable-mlir -lcudaq-solvers adapt_h2.cpp -o adapt_h2
      ./adapt_h2
