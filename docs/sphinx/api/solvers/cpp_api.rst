CUDA-Q Solvers C++ API
******************************

.. doxygenclass:: cudaq::solvers::operator_pool 
    :members:

.. doxygenclass:: cudaq::solvers::spin_complement_gsd 
.. doxygenclass:: cudaq::solvers::uccsd 
.. doxygenclass:: cudaq::solvers::qaoa_pool 

.. doxygenfunction:: cudaq::solvers::get_operator_pool 
    
.. doxygenstruct:: cudaq::solvers::atom 
    :members:

.. doxygenclass:: cudaq::solvers::molecular_geometry 
    :members: 

.. doxygenstruct:: cudaq::solvers::molecular_hamiltonian 
    :members: 

.. doxygenstruct:: cudaq::solvers::molecule_options 
    :members:

.. doxygenfunction:: cudaq::solvers::create_molecule 

.. doxygenfunction:: cudaq::solvers::get_maxcut_hamiltonian 

.. doxygenfunction:: cudaq::solvers::get_clique_hamiltonian 

.. doxygenfunction:: cudaq::solvers::one_particle_op 

.. doxygentypedef:: cudaq::ParameterizedKernel
.. doxygentypedef:: cudaq::optim::optimization_result 
.. doxygenclass:: cudaq::optim::optimizable_function 
.. doxygenclass:: cudaq::optim::optimizer 
    :members:
.. doxygenclass:: cudaq::optim::cobyla 
.. doxygenclass:: cudaq::optim::lbfgs 
.. doxygenclass:: cudaq::observe_gradient 
    :members:
.. doxygenstruct:: cudaq::observe_iteration
    :members:
.. doxygenclass:: cudaq::central_difference
.. doxygenclass:: cudaq::forward_difference
.. doxygenclass:: cudaq::parameter_shift

.. doxygenenum:: cudaq::observe_execution_type
 
.. doxygenstruct:: cudaq::solvers::vqe_result
.. doxygenfunction:: cudaq::solvers::vqe(QuantumKernel &&, const spin_op &, const std::string &, const std::string &, const std::vector<double> &, heterogeneous_map)
.. doxygenfunction:: cudaq::solvers::vqe(QuantumKernel &&, const spin_op &, const std::string &, const std::vector<double> &, heterogeneous_map)
.. doxygenfunction:: cudaq::solvers::vqe(QuantumKernel &&, const spin_op &, const std::string &, observe_gradient &, const std::vector<double> &, heterogeneous_map)
.. doxygenfunction:: cudaq::solvers::vqe(QuantumKernel &&, const spin_op &, optim::optimizer &, const std::string &, const std::vector<double> &, heterogeneous_map)
.. doxygenfunction:: cudaq::solvers::vqe(QuantumKernel &&, const spin_op &, optim::optimizer &, const std::vector<double> &, heterogeneous_map)
.. doxygenfunction:: cudaq::solvers::vqe(QuantumKernel &&, const spin_op &, optim::optimizer &, observe_gradient &, const std::vector<double> &, heterogeneous_map)

.. doxygentypedef:: cudaq::solvers::adapt::result 
.. doxygenfunction:: cudaq::solvers::adapt_vqe(const cudaq::qkernel<void(cudaq::qvector<>&)> &, const spin_op &, const std::vector<spin_op> &, const heterogeneous_map)
.. doxygenfunction:: cudaq::solvers::adapt_vqe(const cudaq::qkernel<void(cudaq::qvector<>&)> &, const spin_op &, const std::vector<spin_op> &, const optim::optimizer&, const heterogeneous_map)
.. doxygenfunction:: cudaq::solvers::adapt_vqe(const cudaq::qkernel<void(cudaq::qvector<>&)> &, const spin_op &, const std::vector<spin_op> &, const optim::optimizer&, const std::string&, const heterogeneous_map)

.. doxygentypedef:: cudaq::solvers::stateprep::excitation_list 
.. doxygenfunction:: cudaq::solvers::stateprep::get_uccsd_excitations
.. doxygenfunction:: cudaq::solvers::stateprep::get_num_uccsd_parameters
.. doxygenfunction:: cudaq::solvers::stateprep::single_excitation
.. doxygenfunction:: cudaq::solvers::stateprep::double_excitation
.. doxygenfunction:: cudaq::solvers::stateprep::uccsd(cudaq::qview<>, const std::vector<double>&, std::size_t, std::size_t)
.. doxygenfunction:: cudaq::solvers::stateprep::uccsd(cudaq::qview<>, const std::vector<double>&, std::size_t)


.. doxygenstruct:: cudaq::solvers::qaoa_result
    :members:
.. doxygenfunction:: cudaq::solvers::qaoa(const cudaq::spin_op &, const cudaq::spin_op &, const optim::optimizer &, std::size_t, const std::vector<double> &, const heterogeneous_map)
.. doxygenfunction:: cudaq::solvers::qaoa(const cudaq::spin_op &, const optim::optimizer &, std::size_t, const std::vector<double> &, const heterogeneous_map)
.. doxygenfunction:: cudaq::solvers::qaoa(const cudaq::spin_op &, std::size_t, const std::vector<double> &, const heterogeneous_map)
.. doxygenfunction:: cudaq::solvers::qaoa(const cudaq::spin_op &, const cudaq::spin_op &, std::size_t, const std::vector<double> &, const heterogeneous_map)
.. doxygenfunction:: cudaq::solvers::get_num_qaoa_parameters(const cudaq::spin_op &, const cudaq::spin_op &, std::size_t, const heterogeneous_map)
.. doxygenfunction:: cudaq::solvers::get_num_qaoa_parameters(const cudaq::spin_op &, std::size_t, const heterogeneous_map)
