.. class:: pymatching

    A minimum-weight perfect matching (MWPM) decoder for matchable quantum error
    correction codes (such as the surface code), built on the open-source
    `PyMatching <https://github.com/oscarhiggott/PyMatching>`_ library. It is a
    CPU decoder: each syndrome bit becomes a detector node, and each error (column
    of the parity-check matrix) with one or two set entries becomes a (boundary)
    edge whose weight is derived from the error prior.

    .. note::
      It is required to create decoders with the `get_decoder` API from the
      CUDA-QX extension points API, such as

      .. tab:: Python

        .. code-block:: python

            import cudaq_qec as qec
            import numpy as np

            # Parity check matrix. Each column (error mechanism) must have one
            # or two set entries so the graph is matchable.
            H = np.array([[1, 1, 0],
                          [0, 1, 1]], dtype=np.uint8)

            dec = qec.get_decoder("pymatching", H,
                                  error_rate_vec=[0.1, 0.1, 0.1],
                                  merge_strategy="smallest_weight")

      .. tab:: C++

        .. code-block:: cpp

            #include "cudaq/qec/decoder.h"

            cudaqx::heterogeneous_map params;
            params.insert("merge_strategy", std::string("smallest_weight"));
            auto dec = cudaq::qec::get_decoder("pymatching", H, params);

    .. note::
      The `"pymatching"` decoder implements the :class:`cudaq_qec.Decoder`
      interface for Python and the :cpp:class:`cudaq::qec::decoder` interface for
      C++, so it supports all the methods in those respective classes.

    :param H: Parity check matrix. Each column must have one or two set entries
              (matchable graph). In Python, a ``scipy.sparse`` matrix or a dense
              NumPy ``uint8`` array may be passed.
    :param params: Heterogeneous map of parameters:

        - `error_rate_vec` (vector<double>): Per-error prior probabilities, one
          per column of ``H`` (length ``block_size``). Each value must lie in
          ``(0, 0.5]`` and sets the matching edge weight ``-log(p / (1 - p))``.
          When omitted, all edge weights default to ``1.0``.
        - `merge_strategy` (string): How to combine parallel edges that map to
          the same pair of detectors. One of ``"disallow"`` (default for the
          ``H``-only path), ``"independent"``, ``"smallest_weight"``,
          ``"keep_original"``, or ``"replace"``.
        - `O` (tensor, optional): A ``num_observables x block_size`` binary
          matrix. When provided, the decoder returns predicted observable flips
          (``decode_to_obs``) instead of a raw error vector, and
          ``merge_strategy`` defaults to ``"independent"`` to match PyMatching's
          detector-error-model construction.
