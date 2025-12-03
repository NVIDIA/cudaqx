.. class:: nv_qldpc_decoder

    A general purpose Quantum Low-Density Parity-Check Decoder (QLDPC)
    decoder based on GPU accelerated belief propagation (BP). Since belief
    propagation is an iterative method, decoding can be improved with a
    second-stage post-processing step. Optionally, ordered statistics decoding
    (OSD) can be chosen to perform the second stage of decoding.

    An [[n,k,d]] quantum error correction (QEC) code encodes k logical qubits
    into an n qubit data block, with a code distance d. Quantum low-density
    parity-check (QLDPC) codes are characterized by sparse parity-check matrices
    (or Tanner graphs), corresponding to a bounded number of parity checks per
    data qubit.

    Requires a CUDA-Q compatible GPU. See the `CUDA-Q GPU Compatibility
    List <https://nvidia.github.io/cuda-quantum/latest/using/install/local_installation.html#dependencies-and-compatibility>`_
    for a list of valid GPU configurations.

    References:
    `Decoding Across the Quantum LDPC Code Landscape <https://arxiv.org/pdf/2005.07016>`_

    .. note::
      It is required to create decoders with the `get_decoder` API from the CUDA-QX
      extension points API, such as

      .. tab:: Python

        .. code-block:: python

            import cudaq_qec as qec
            import numpy as np
            H = np.array([[1, 0, 0, 1, 0, 1, 1],
                          [0, 1, 0, 1, 1, 0, 1],
                          [0, 0, 1, 0, 1, 1, 1]], dtype=np.uint8) # sample 3x7 PCM
            opts = dict() # see below for options
            # Note: H must be in row-major order. If you use
            # `scipy.sparse.csr_matrix.todense()` to get the parity check
            # matrix, you must specify todense(order='C') to get a row-major
            # matrix.
            nvdec = qec.get_decoder('nv-qldpc-decoder', H, **opts)

      .. tab:: C++

        .. code-block:: cpp

            std::size_t block_size = 7;
            std::size_t syndrome_size = 3;
            cudaqx::tensor<uint8_t> H;

            std::vector<uint8_t> H_vec = {1, 0, 0, 1, 0, 1, 1, 
                                          0, 1, 0, 1, 1, 0, 1,
                                          0, 0, 1, 0, 1, 1, 1};
            H.copy(H_vec.data(), {syndrome_size, block_size});

            cudaqx::heterogeneous_map nv_custom_args;
            nv_custom_args.insert("use_osd", true);
            // See below for options

            auto nvdec = cudaq::qec::get_decoder("nv-qldpc-decoder", H, nv_custom_args);
      
    .. note::
      The `"nv-qldpc-decoder"` implements the :class:`cudaq_qec.Decoder`
      interface for Python and the :cpp:class:`cudaq::qec::decoder` interface
      for C++, so it supports all the methods in those respective classes.

    :param H: Parity check matrix (tensor format)
    :param params: Heterogeneous map of parameters:

        - `use_sparsity` (bool): Whether or not to use a sparse matrix solver
        - `error_rate` (double): Probability of an error (in 0-1 range) on a
          block data bit (defaults to 0.001)
        - `error_rate_vec` (double): Vector of length "block size" containing
          the probability of an error (in 0-1 range) on a block data bit (defaults
          to 0.001). This overrides `error_rate`.
        - `max_iterations` (int): Maximum number of BP iterations to perform
          (defaults to 30)
        - `n_threads` (int): Number of CUDA threads to use for the GPU decoder
          (defaults to smart selection based on parity matrix size)
        - `use_osd` (bool): Whether or not to use an OSD post processor if the
          initial BP algorithm fails to converge on a solution
        - `osd_method` (int): 1=OSD-0, 2=Exhaustive, 3=Combination Sweep
          (defaults to 1). Ignored unless `use_osd` is true.
        - `osd_order` (int): OSD postprocessor order (defaults to 0). Ref:
          `Decoding Across the Quantum LDPC Code Landscape <https://arxiv.org/pdf/2005.07016>`_

          - For `osd_method=2` (Exhaustive), the number of possible
            permutations searched after OSD-0 grows by 2^osd_order.
          - For `osd_method=3` (Combination Sweep), this is the λ parameter. All
            weight 1 permutations and the first λ bits worth of weight 2
            permutations are searched after OSD-0. This is (syndrome_length -
            block_size + λ * (λ - 1) / 2) additional permutations.
          - For other `osd_method` values, this is ignored.
        - `bp_batch_size` (int): Number of syndromes that will be decoded in
          parallel for the BP decoder (defaults to 1)
        - `osd_batch_size` (int): Number of syndromes that will be decoded in
          parallel for OSD (defaults to the number of concurrent threads supported
          by the hardware)
        - `iter_per_check` (int): Number of iterations between BP convergence checks
          (defaults to 1, and max is `max_iterations`). Introduced in 0.4.0.
        - `clip_value` (float): Value to clip the BP messages to. Should be a
          non-negative value (defaults to 0.0, which disables clipping). Introduced in
          0.4.0.
        - `bp_method` (int): Core BP algorithm to use (defaults to 0). Introduced in 0.4.0,
          expanded in 0.5.0:

          - 0: sum-product
          - 1: min-sum (introduced in 0.4.0)
          - 2: min-sum+mem (uniform memory strength, requires `use_sparsity=True`. Introduced in 0.5.0)
          - 3: min-sum+dmem (disordered memory strength, requires `use_sparsity=True`. Introduced in 0.5.0)
        - `composition` (int): Iteration strategy (defaults to 0). Introduced in 0.5.0:

          - 0: Standard (single run)
          - 1: Sequential relay (multiple gamma legs). Requires: `bp_method=3`, `use_sparsity=True`, and `srelay_config`
        - `scale_factor` (float): The scale factor to use for min-sum. Defaults to 1.0.
          When set to 0.0, the scale factor is dynamically computed based on the
          number of iterations. Introduced in 0.4.0.
        - `proc_float` (string): The processing float type to use. Defaults to
          "fp64". Valid values are "fp32" and "fp64". Introduced in 0.5.0.
        - `gamma0` (float): Memory strength parameter. Required for `bp_method=2`, and for
          `composition=1` (sequential relay). Introduced in 0.5.0.
        - `gamma_dist` (vector<float>): Gamma distribution interval [min, max] for disordered
          memory strength. Required for `bp_method=3` if `explicit_gammas` not provided.
          Introduced in 0.5.0.
        - `explicit_gammas` (vector<vector<float>>): Explicit gamma values for each variable node.
          For `bp_method=3` with `composition=0`, provide a 2D vector where each row has
          `block_size` columns. For `composition=1` (Sequential relay), provide `num_sets` rows
          (one per relay leg). Overrides `gamma_dist` if provided. Introduced in 0.5.0.
        - `srelay_config` (heterogeneous_map): Sequential relay configuration (required for
          `composition=1`). Contains the following parameters. Introduced in 0.5.0:

          - `pre_iter` (int): Number of pre-iterations to run before relay legs
          - `num_sets` (int): Number of relay sets (legs) to run
          - `stopping_criterion` (string): When to stop relay legs:

            - "All": Run all legs
            - "FirstConv": Stop relay after first convergence
            - "NConv": Stop after N convergences (requires `stop_nconv` parameter)
          - `stop_nconv` (int): Number of convergences to wait for before stopping
            (required only when `stopping_criterion="NConv"`)
        - `bp_seed` (int): Seed for random number generation used in `bp_method=3` (disordered
          memory BP). Optional parameter, defaults to 42 if not provided. Introduced in 0.5.0.
        - `opt_results` (heterogeneous_map): Optional results to return. This field can be
          left empty if no additional results are desired. Choices are:

          - `bp_llr_history` (int): Return the last `bp_llr_history` iterations
            of the BP LLR history. Minimum value is 0 and maximum value is
            max_iterations. The actual number of returned iterations might be fewer
            than `bp_llr_history` if BP converges before the requested number of
            iterations. Introduced in 0.4.0. Note: Not supported for `composition=1`.
          - `num_iter` (bool): If true, return the number of BP iterations run.
            Introduced in 0.5.0.

