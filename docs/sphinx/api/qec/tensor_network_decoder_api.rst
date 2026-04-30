.. class:: cudaq_qec.plugin.decoders.tensor_network_decoder.TensorNetworkDecoder

    A general class for tensor network decoders for quantum error correction codes.

    This decoder constructs a tensor network representation of a quantum code using its parity check matrix, logical observables, and noise model. The tensor network is based on the Tanner graph of the code and can be contracted to compute the probability that a logical observable has flipped, given a syndrome.

    The decoder supports both single-syndrome and batch decoding, and can run on CPU or GPU (using cuTensorNet if available).

    The Tensor Network Decoder is a Python-only implementation and it requires Python 3.11 or higher. C++ APIs are not available for this decoder.

    Due to the additional dependencies of the Tensor Network Decoder, you must
    specify the optional pip package when installing CUDA-Q QEC in order to use this
    decoder. Use `pip install cudaq-qec[tensor-network-decoder]` in order to use
    this decoder.
    
    The Tensor Network Decoder has the same GPU support as the `Quantum Low-Density Parity-Check Decoder <https://nvidia.github.io/cudaqx/components/qec/introduction.html#quantum-low-density-parity-check-decoder>`__.
    However, if you are using the V100 GPU (SM70), you will need to pin your
    cuTensor version to 2.2 by running `pip install cutensor_cu12==2.2`. Note
    that this GPU will not be supported by the Tensor Network Decoder when
    CUDA-Q 0.5.0 is released.

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

    .. method:: optimize_path(optimize=None, batch_size=-1)

        Optimize the contraction path for the tensor network.

        :param optimize: Optimization options or None
        :param batch_size: (int, optional) Batch size for optimization (default: -1, no batching)
        :returns: Optimizer info object

.. class:: cudaq_qec.plugins.decoders.tensor_network_decoder.NMOptimizer

    Differentiable noise-model optimizer built on top of :class:`TensorNetworkDecoder`.

    Fits a factorised per-error noise model to a syndrome dataset by
    backpropagating through a torch-backed tensor-network contraction.
    The noise probabilities are maintained as ``torch`` tensors with
    ``requires_grad=True`` so they can be updated with any ``torch.optim``
    optimizer.

    Requires Python 3.11 or higher and the same optional dependencies as
    :class:`TensorNetworkDecoder` (``pip install cudaq-qec[tensor-network-decoder]``).
    PyTorch must also be installed.

    .. note::
      Quick-start example (logit-space training; the loss has no ``log``
      guard, so direct probability training requires per-step clamping
      into ``[eps, 1 - eps]``)::

        import numpy as np
        import torch
        from cudaq_qec.plugins.decoders.tensor_network_decoder import (
            NMOptimizer, make_compiled_step,
        )

        H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.float64)
        logical = np.array([[1, 0, 1]], dtype=np.float64)
        priors = [0.1, 0.2, 0.3]

        opt = NMOptimizer(H, logical, priors, syndrome_data, obs_flips,
                          dtype="float64")
        logits = torch.logit(opt.noise_params[0].detach()).requires_grad_()
        adam = torch.optim.Adam([logits], lr=0.01)
        step = make_compiled_step(opt, logits, adam)
        for _ in range(100):
            step()

    :param H: Parity check matrix (numpy.ndarray), shape (num_checks, num_errors)
    :param logical_obs: Logical observable matrix (numpy.ndarray), shape (1, num_errors)
    :param noise_model: Initial per-error probabilities, list of floats in (0, 1).
                        Values outside ``[eps, 1 - eps]`` are clamped at
                        construction with a ``UserWarning``; non-finite values
                        raise ``ValueError``.  ``eps`` is ``1e-12`` for
                        ``"float64"`` and ``1e-6`` for ``"float32"``.
    :param syndrome_data: Observed syndromes, numpy.ndarray of shape (num_shots, num_checks)
    :param observable_flips: Observed logical flips, bool array of length num_shots
    :param check_inds: (optional) List of check index names; defaults track the parent decoder.
    :param error_inds: (optional) List of error index names; defaults track the parent decoder.
    :param logical_inds: (optional) List of logical index names; defaults track the parent decoder.
    :param logical_tags: (optional) List of logical tags; defaults track the parent decoder.
    :param dtype: (str, optional) ``"float32"`` (default) or ``"float64"``;
                  other values raise ``ValueError``.
    :param device: (str, optional) Torch device, e.g. ``"cpu"`` or ``"cuda"`` (default: ``"cuda"``)
    :param compile: (bool, optional, keyword-only) If ``True``, wrap the forward
                    and loss in :func:`torch.compile`. Most useful with
                    ``execute="codegen"``. Defaults to ``False``.
    :param execute: (str, optional, keyword-only) Forward backend.  ``"codegen"``
                    (default) partial-evaluates the contraction path into a flat
                    Python function with named locals; ``"unrolled"`` keeps an
                    interpretive einsum list; ``"opt_einsum"`` dispatches via
                    :func:`opt_einsum.contract_expression`.
    :param compile_mode: (str, optional, keyword-only) Forwarded to
                         :func:`torch.compile` (e.g. ``"reduce-overhead"``,
                         ``"default"``); ignored when ``compile=False``.
    :param dynamic_syndromes: (bool, optional, keyword-only) If ``True``
                              (default), syndromes are runtime arguments to the
                              compiled forward, so :meth:`update_dataset` reuses
                              the codegen/``torch.compile`` artifact when shapes
                              are unchanged.  ``False`` bakes syndromes into the
                              closure -- faster per call but every
                              :meth:`update_dataset` rebuilds the graph.  Only
                              affects ``execute="codegen"``.

    **Attributes**

    .. attribute:: noise_params

        ``list[torch.Tensor]`` — the learnable noise-probability tensors; pass
        directly to a ``torch.optim`` optimizer.

    .. attribute:: torch_device

        ``torch.device`` derived from the ``device`` constructor argument.
        Read-only.

    .. attribute:: observable_flips

        Bool ``torch.Tensor`` of logical flip outcomes for the current
        syndrome batch.  Assigning a new value also rebuilds the fused
        loss closure (the observable indices are baked into the codegen);
        prefer :meth:`update_dataset` when swapping syndromes and flips
        together.

    **Methods**

    .. method:: current_syndrome_args()

        Return the syndrome argument expected by the callable from
        :meth:`loss_fn`: the live tuple when ``dynamic_syndromes=True``,
        or ``()`` for static codegen (syndromes are closure-baked).
        Re-fetch each step so an intervening :meth:`update_dataset` is
        reflected.

        :returns: ``tuple[torch.Tensor, ...]``

    .. method:: cross_entropy_loss()

        Compute the cross-entropy loss between the predicted logical-flip
        probabilities and the observed ``observable_flips``.

        :returns: Scalar ``torch.Tensor`` (differentiable).

    .. method:: decoder_prediction()

        Run the forward pass and return per-shot probabilities.

        :returns: ``torch.Tensor`` of shape ``(num_shots, 2)`` where column 1
                  is ``P(logical flip | syndrome)``.

    .. method:: logical_error_rate()

        Fraction of shots where ``argmax`` of :meth:`decoder_prediction`
        disagrees with :attr:`observable_flips`.  Not differentiable
        (runs under :func:`torch.no_grad`).

        :returns: ``float`` in ``[0, 1]``.

    .. method:: loss_fn(from_logits=True)

        Return a compiled callable ``fn(params, syndrome_tuple) -> loss``
        suitable for use with external optimizers or ``torch.compile``.

        :param from_logits: If ``True`` (default), ``params`` are interpreted
                            as logits and passed through ``sigmoid`` before
                            contraction. If ``False``, ``params`` are
                            interpreted as probabilities already in ``[0, 1]``.
        :returns: Compiled loss function.

    .. method:: optimize_path(optimize=None, batch_size=-1)

        Cache a contraction path via quimb / opt_einsum and rebuild the
        compiled forward.  Pass e.g. ``cotengra.HyperOptimizer()`` to run a
        more expensive path search; ``None`` falls back to ``"auto"``.

        :param optimize: Optimization options (e.g. a ``cotengra.HyperOptimizer``)
                         or ``None``.
        :param batch_size: Accepted for signature compatibility; ignored.
        :returns: Contraction info object.

    .. method:: update_dataset(syndrome_data, observable_flips, enforce_shape=True)

        Swap in a new syndrome batch without rebuilding the tensor network.
        If ``dynamic_syndromes=True`` and the batch size is unchanged, the
        compiled contraction path is reused; a shape change triggers a full
        rebuild.

        :param syndrome_data: numpy.ndarray of shape (num_shots, num_checks)
        :param observable_flips: bool array of length num_shots
        :param enforce_shape: (bool, optional, default ``True``) Assert
                              per-tensor shapes match the existing layout
                              before patching in place.  A batch-size change
                              triggers a full rebuild regardless.

.. function:: cudaq_qec.plugins.decoders.tensor_network_decoder.make_compiled_step(optimizer, logits, torch_optimizer)

    Build a no-arg callable that runs one Adam step and returns the loss.

    The returned ``step()`` callable zeros gradients, evaluates the
    optimizer's fused ``loss_fn(from_logits=True)`` (sigmoid + contraction +
    cross-entropy), backpropagates, and steps ``torch_optimizer``. Intended
    for training in logit space; pair with :class:`NMOptimizer` constructed
    with ``compile=True`` for a ``torch.compile``-d variant.

    :param optimizer: An :class:`NMOptimizer` instance providing the fused
                      inner loss.
    :param logits: Trainable 1-D ``torch.Tensor`` of length
                   ``len(optimizer.error_inds)`` with ``requires_grad=True``.
    :param torch_optimizer: A ``torch.optim`` instance owning ``logits``.
    :returns: A no-arg callable that performs one optimization step and
              returns the scalar loss as a ``torch.Tensor``.