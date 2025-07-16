TensorNetworkDecoder Example
===========================

This example demonstrates how to use the ``TensorNetworkDecoder`` class from the ``cudaq_qec`` library to decode syndromes for a simple quantum error-correcting code using tensor networks.

Overview
--------

The ``TensorNetworkDecoder`` constructs a tensor network representation of a quantum code given its parity check matrix, logical observable(s), and noise model. It can decode individual syndromes or batches of syndromes, returning the probability that a logical observable has flipped.

Key Steps
---------

1. **Define the parity check matrix**: This matrix encodes the structure of the quantum code. In the example, a simple [3,1] repetition code is used.

2. **Specify the logical observable**: This is typically a row vector indicating which qubits participate in the logical operator.

3. **Set the noise model**: The example uses a factorized noise model with independent bit-flip probability for each qubit.

4. **Instantiate the decoder**: Create a ``TensorNetworkDecoder`` object with the code parameters.

5. **Decode syndromes**: Use the ``decode`` method for single syndromes or ``decode_batch`` for multiple syndromes.

Usage
-----

.. code-block:: python

    import cudaq_qec as qec
    import numpy as np

    # Define code parameters
    H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
    logical_obs = np.array([[1, 1, 1]], dtype=np.uint8)
    noise_model = [0.1, 0.1, 0.1]

    decoder = qec.get_decoder("tensor_network_decoder", H, logical_obs=logical_obs, noise_model=noise_model, dtype="float32", device="cpu")

    # Decode a single syndrome
    syndrome = [0.0, 1.0]
    result = decoder.decode(syndrome)
    print(result.result)

    # Decode a batch of syndromes
    syndrome_batch = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    batch_results = decoder.decode_batch(syndrome_batch)
    for res in batch_results:
        print(res.result)

Output
------

The decoder returns the probability that the logical observable has flipped for each syndrome. This can be used to assess the performance of the code and the decoder under different error scenarios.

See Also
--------

- ``cudaq_qec.plugins.decoders.tensor_network_decoder``
- ``quimb.tensor.TensorNetwork``