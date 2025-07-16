.. class:: cudaq_qec.plugin.decoders.tensor_network_decoder.TensorNetworkDecoder

    A general class for tensor network decoders for quantum error correction codes.

    This decoder constructs a tensor network representation of a quantum code using its parity check matrix, logical observables, and noise model. The tensor network is based on the Tanner graph of the code and can be contracted to compute the probability that a logical observable has flipped, given a syndrome.

    The decoder supports both single-syndrome and batch decoding, and can run on CPU or GPU (using cuTensorNet if available).

    The Tensor Network Decoder is a Python-only implementation. C++ APIs are not available for this decoder.

    For related work, see:
    `Tensor Network Decoding Beyond 2D <https://arxiv.org/abs/2310.10722>`_


    .. note::
      It is recommended to create decoders using the `cudaq_qec` plugin API:

      .. code-block:: python

        import cudaq_qec as qec
        import numpy as np

        # Example: [3,1] repetition code
        H = np.array([[1, 1, 0],
                [0, 1, 1]], dtype=np.uint8)
        logical_obs = np.array([[1, 1, 1]], dtype=np.uint8)
        noise_model = [0.1, 0.1, 0.1]

        decoder = qec.get_decoder("tensor_network_decoder", H, logical_obs=logical_obs, noise_model=noise_model)

        syndrome = [0.0, 1.0]
        result = decoder.decode(syndrome)
        
    .. rubric:: Tensor Network Structure

    The tensor network constructed by this decoder is based on the Tanner graph of the code, extended with noise and logical observable tensors. The structure is illustrated below:

    .. code-block:: none

              open/output index < logical observable
                  --------
                     |
        s1      s2   |     s3   < syndromes               : product of 2D vectors [1 , 1-2pi] (pi is the probability detector i flipped)
        |       |    |     |                        ----|
        c1      c2  l1     c3   < checks / logical      | : delta tensors
        |     / |   | \    |                            |
        H   H   H   H  H   H    < Hadamard matrices     | TANNER (bipartite) GRAPH
          \ |   |  /   |  /                             |
            e1  e2     e3       < errors                | : delta tensors
            |   |     /                            -----|
             \ /     /
            P(e1, e2, e3)       < noise / error model     : classical probability density

        ci, ej, lk are delta tensors represented sparsely as indices.

    :param H: Parity check matrix (numpy.ndarray), shape (num_checks, num_qubits)
    :param logical_obs: Logical observable matrix (numpy.ndarray), shape (1, num_qubits)
    :param noise_model: Noise model, either a list of probabilities (length = num_qubits) or a quimb.tensor.TensorNetwork
    :param check_inds: (optional) List of check index names
    :param error_inds: (optional) List of error index names
    :param logical_inds: (optional) List of logical index names
    :param logical_tags: (optional) List of logical tags
    :param contract_noise_model: (bool, optional) Whether to contract the noise model at initialization (default: True)
    :param dtype: (str, optional) Data type for tensors (default: "float32")
    :param device: (str, optional) Device for tensor operations ("cpu", "cuda", or "cuda:X", default: "cuda")

    **Methods**

    .. method:: decode(syndrome)

        Decode a single syndrome by contracting the tensor network.

        :param syndrome: List of float values (soft-decision probabilities) for each check.
        :returns: DecoderResult with the probability that the logical observable flipped.

    .. method:: decode_batch(syndrome_batch)

        Decode a batch of syndromes.

        :param syndrome_batch: numpy.ndarray of shape (batch_size, num_checks)
        :returns: List of DecoderResult objects with the probability that the logical observable has flipped for each syndrome.

    .. method:: replace_logical_observable(logical_obs, logical_inds=None, logical_tags=None)

        Replace the logical observable(s) in the tensor network.

        :param logical_obs: Logical observable matrix (numpy.ndarray)
        :param logical_inds: (optional) List of logical index names
        :param logical_tags: (optional) List of logical tags

    .. method:: init_noise_model(noise_model, contract=False)

        Initialize or replace the noise model in the tensor network.

        :param noise_model: TensorNetwork representing the noise model
        :param contract: (bool, optional) Whether to contract the noise model immediately

    .. method:: optimize_path(optimize=None, batch_size=-1)

        Optimize the contraction path for the tensor network.

        :param optimize: Optimization options or None
        :param batch_size: (int, optional) Batch size for optimization (default: -1, no batching)
        :returns: Optimizer info object

    .. method:: flip_syndromes(values)

        Update the tensor network to represent a given syndrome.

        :param values: List of float values for the syndrome