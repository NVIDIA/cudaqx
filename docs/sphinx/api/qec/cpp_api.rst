CUDA-Q QEC C++ API
******************************

Code
=============

.. doxygenclass:: cudaq::qec::code
    :members:

.. doxygenstruct:: cudaq::qec::patch
    :members:

.. doxygenclass:: cudaq::qec::repetition::repetition
    :members:

.. doxygenclass:: cudaq::qec::steane::steane
    :members:

.. doxygenclass:: cudaq::qec::surface_code::stabilizer_grid
    :members:

.. doxygenclass:: cudaq::qec::surface_code::surface_code
    :members:


Decoder
=============

.. doxygenclass:: cudaq::qec::decoder
    :members:

.. doxygenstruct:: cudaq::qec::decoder_result
    :members:


Common
=============

.. doxygentypedef:: cudaq::qec::float_t

.. doxygenenum:: cudaq::qec::operation

.. doxygenfunction:: cudaq::qec::sample_code_capacity(const cudaqx::tensor<uint8_t> &, std::size_t, double)
.. doxygenfunction:: cudaq::qec::sample_code_capacity(const cudaqx::tensor<uint8_t> &, std::size_t, double, unsigned)
.. doxygenfunction:: cudaq::qec::sample_code_capacity(const code &, std::size_t, double)
.. doxygenfunction:: cudaq::qec::sample_code_capacity(const code &, std::size_t, double, unsigned)

.. doxygenfunction:: cudaq::qec::sample_memory_circuit(const code &, std::size_t, std::size_t)
.. doxygenfunction:: cudaq::qec::sample_memory_circuit(const code &, std::size_t, std::size_t, cudaq::noise_model &)
.. doxygenfunction:: cudaq::qec::sample_memory_circuit(const code &, operation, std::size_t, std::size_t)
.. doxygenfunction:: cudaq::qec::sample_memory_circuit(const code &, operation, std::size_t, std::size_t, cudaq::noise_model &)
