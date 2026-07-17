CUDA-QX Namespaces and Core Library C++ API
********************************************

Namespaces
==========
.. doxygennamespace:: cudaqx
    :desc-only:
.. doxygennamespace:: cudaq
    :desc-only:
.. doxygennamespace:: cudaq::qec
    :desc-only:
.. doxygennamespace:: cudaq::qec::decoding
    :desc-only:
.. doxygennamespace:: cudaq::qec::decoding::config
    :desc-only:
.. doxygennamespace:: cudaq::qec::detail
    :desc-only:
.. doxygennamespace:: cudaq::qec::realtime
    :desc-only:
.. doxygennamespace:: cudaq::qec::realtime::experimental
    :desc-only:
.. doxygennamespace:: cudaq::qec::steane
    :desc-only:
.. doxygennamespace:: cudaq::qec::surface_code
    :desc-only:
.. doxygennamespace:: cudaq::qec::repetition
    :desc-only:
.. doxygennamespace:: cudaq::solvers
    :desc-only:
.. doxygennamespace:: cudaq::solvers::stateprep
    :desc-only:
.. doxygennamespace:: cudaq::solvers::adapt
    :desc-only:
.. doxygennamespace:: cudaq::optim
    :desc-only:

Core
=============

.. doxygenclass:: cudaqx::extension_point
    :members:

.. doxygendefine:: CUDAQ_EXTENSION_CREATOR_FUNCTION
.. doxygendefine:: CUDAQ_EXTENSION_CUSTOM_CREATOR_FUNCTION
.. doxygendefine:: CUDAQ_EXTENSION_CUSTOM_CREATOR_FUNCTION_WITH_NAME
.. doxygendefine:: CUDAQ_EXT_PT_REGISTER_TYPE
.. doxygendefine:: INSTANTIATE_REGISTRY_NO_ARGS
.. doxygendefine:: INSTANTIATE_REGISTRY

.. doxygenclass:: cudaqx::heterogeneous_map
    :members:

.. doxygenclass:: cudaqx::tear_down
    :members:

.. doxygenclass:: cudaqx::details::tensor_impl
    :members:

.. doxygenclass:: cudaqx::tensor
    :members:

.. doxygenclass:: cudaqx::graph
    :members:
