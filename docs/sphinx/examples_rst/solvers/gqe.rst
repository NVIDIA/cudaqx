Generative Quantum Eigensolver (GQE)
-------------------------------------

The GQE algorithm has presented a novel approach to variational optimization that leverages generative AI. It is an active research topic. 
A core aspect of this research is with regards to the core cost / loss function the algorithm leverages. The current implementation provides 
a cost function suitable to small scale simulation. The GQE implementation in CUDA-Q Solvers is based on this paper: `The generative quantum eigensolver 
(GQE) and its application for ground state search <https://arxiv.org/abs/2401.09253>`_.

GQE Algorithm Overview:

1. Initialize or load a pre-trained generative model
2. Generate candidate quantum circuits
3. Evaluate circuit performance on target Hamiltonian
4. Update the generative model based on results
5. Repeat generation and optimization until convergence


CUDA-Q Solvers Implementation
+++++++++++++++++++++++++++++

CUDA-Q Solvers provides a high-level interface for running GQE simulations. Here's how to use it:

.. tab:: Python

   .. literalinclude:: ../../examples/solvers/python/gqe_h2.py
      :language: python
      :start-after: [Begin Documentation]

The CUDA-Q Solvers implementation of GQE provides a flexible framework for adaptive circuit construction and optimization. 
The algorithm can efficiently utilize multiple QPUs through MPI for parallel operator evaluation, making it suitable for larger quantum systems. 

.. note::

   The GQE implementation is a Python-only implementation.


   
