CUDA-Q QEC Python API
******************************

.. automodule:: cudaq_qec

.. All classes below that are assigned from ``qecrt`` (the native C extension)
.. are documented with manual ``.. class::`` directives rather than
.. ``.. autoclass::``.  In docs-gen mode ``qecrt`` is a ``MagicMock``, so
.. ``.. autoclass::`` would emit "alias of <MagicMock …>" for every such class.

Code
=============

.. class:: Code

    Base class for all quantum error correction codes.

    Subclass and decorate with ``@qec.code(name)`` to register a custom
    QEC code with the CUDA-QX runtime.

    .. method:: get_parity() -> numpy.ndarray

        Full parity check matrix (X rows first, then Z).

    .. method:: get_parity_x() -> numpy.ndarray

        X-type parity check matrix.

    .. method:: get_parity_z() -> numpy.ndarray

        Z-type parity check matrix.

    .. method:: get_stabilizer_schedule_x() -> numpy.ndarray

        X-stabilizer schedule matrix; entry 0 = no support, entry *k* ≥ 1 =
        interaction at timestep *k*. Defaults to the X PCM.

    .. method:: get_stabilizer_schedule_z() -> numpy.ndarray

        Z-stabilizer counterpart of :meth:`get_stabilizer_schedule_x`.

    .. method:: get_pauli_observables_matrix() -> numpy.ndarray

        Matrix of all Pauli observables (X rows first, then Z).

    .. method:: get_observables_x() -> numpy.ndarray

        X-type logical observable matrix.

    .. method:: get_observables_z() -> numpy.ndarray

        Z-type logical observable matrix.

    .. method:: get_stabilizers() -> list

        Stabilizer generators as a list of Pauli strings.

    .. method:: contains_operation(op) -> bool

        Return ``True`` if this code has a registered kernel for *op*.

    .. method:: get_operation_one_qubit(op)

        CUDA-Q kernel for the one-qubit logical operation *op*.

    .. method:: get_operation_two_qubit(op)

        CUDA-Q kernel for the two-qubit logical operation *op*.

    .. method:: get_stabilizer_round()

        CUDA-Q kernel for one stabilizer measurement round.

    .. method:: get_num_data_qubits() -> int

        Total number of physical data qubits.

    .. method:: get_num_ancilla_qubits() -> int

        Total number of ancilla qubits.

    .. method:: get_num_ancilla_x_qubits() -> int

        Number of X-type ancilla qubits.

    .. method:: get_num_ancilla_z_qubits() -> int

        Number of Z-type ancilla qubits.

    .. method:: get_num_x_stabilizers() -> int

        Number of X-type stabilizers.

    .. method:: get_num_z_stabilizers() -> int

        Number of Z-type stabilizers.

Surface code layout
===================

.. _qec_stabilizer_grid_python:

The rotated surface code exposes a grid helper for stabilizer and data-qubit
indexing. In Python it is available as :class:`cudaq_qec.stabilizer_grid` (call
``cudaq_qec.stabilizer_grid(distance)``). The C++ type is
:cpp:class:`cudaq::qec::surface_code::stabilizer_grid` (:ref:`API <qec_stabilizer_grid_cpp>`).

.. ``stabilizer_grid`` is assigned from the native C extension (``qecrt``), which
.. is replaced by a ``MagicMock`` in docs-gen mode. ``.. autoclass::`` would
.. therefore emit "alias of <MagicMock ...>", so the class is documented manually.

.. class:: stabilizer_grid(distance: int, orientation: str = 'ZH')

    Grid helper for stabilizer and data-qubit indexing in the rotated surface
    code, following the layout convention of `arXiv:2311.10687
    <https://arxiv.org/abs/2311.10687>`_.

    Grid sites are stored in row-major order (left to right, top to bottom).
    For a distance-3 code the ``grid_length`` is 4, giving a 4×4 grid whose
    occupied sites form the familiar rotated-surface-code plaquette layout.

    :param distance: Code distance *d*; sets ``grid_length = d + 1``.
    :param orientation: One of ``'ZH'`` (default), ``'ZV'``, ``'XH'``, ``'XV'``.
        Controls which Pauli type occupies the horizontal/vertical boundaries.

    .. attribute:: distance
        :type: int

        Code distance *d* (number of data qubits per dimension).

    .. attribute:: orientation
        :type: sc_orientation

        Orientation enum value (``ZH``, ``ZV``, ``XH``, or ``XV``).
        The constructor also accepts the orientation as a ``str``.

    .. attribute:: grid_length
        :type: int

        Side length of the square stabilizer grid (``distance + 1``).

    .. attribute:: roles
        :type: list

        Flattened row-major list of ``surface_role`` enum values,
        one per grid site.

    .. attribute:: x_stab_coords
        :type: list[vec2d]

        X-stabilizer index → 2-D grid coordinate (``vec2d`` with ``.row``
        and ``.col`` attributes).

    .. attribute:: z_stab_coords
        :type: list[vec2d]

        Z-stabilizer index → 2-D grid coordinate.

    .. attribute:: data_coords
        :type: list[vec2d]

        Data-qubit index → 2-D coordinate (offset system from stabilizers).

    .. attribute:: x_stabilizers
        :type: list[list[int]]

        X stabilizers as lists of data-qubit indices they act on
        (weight 2 or 4 per stabilizer).

    .. attribute:: z_stabilizers
        :type: list[list[int]]

        Z stabilizers as lists of data-qubit indices they act on.

    .. attribute:: x_stab_indices
        :type: dict[tuple[int, int], int]

        Inverse map: 2-D coordinate → X-stabilizer index.

    .. attribute:: z_stab_indices
        :type: dict[tuple[int, int], int]

        Inverse map: 2-D coordinate → Z-stabilizer index.

    .. attribute:: data_indices
        :type: dict[tuple[int, int], int]

        Inverse map: 2-D coordinate → data-qubit index.

    .. method:: format_stabilizer_grid() -> str

        Return a string rendering of the full ``grid_length × grid_length``
        grid, including empty sites.

    .. method:: format_stabilizer_coords() -> str

        Return a string rendering of occupied stabilizer sites only.

    .. method:: format_stabilizer_indices() -> str

        Return a string rendering with X/Z stabilizer indices (``X0``,
        ``Z1``, …).

    .. method:: format_data_grid() -> str

        Return a string rendering of data-qubit positions and indices.

    .. method:: format_stabilizers() -> str

        Return the stabilizers in sparse Pauli format.

    .. method:: get_spin_op_stabilizers() -> list

        Return the stabilizers as a list of ``cudaq.spin_op_term`` objects.

    .. method:: get_spin_op_observables() -> list

        Return ``[X_logical, Z_logical]`` as ``cudaq.spin_op_term`` objects.
        The pair commutes with all stabilizers and the two observables
        anticommute with each other for every orientation.

    .. method:: get_cnot_schedule_x() -> numpy.ndarray

        Return the X-stabilizer CNOT schedule as a
        ``(num_x_stabilizers, distance²)`` uint8 array.  Entry 0 means the
        ancilla has no support on that data qubit; entry *k* ∈ {1…4} is the
        CNOT timestep within the stabilizer round.  Row ordering matches the
        rows of ``code.get_parity_x()``.

    .. method:: get_cnot_schedule_z() -> numpy.ndarray

        Z-stabilizer counterpart of :meth:`get_cnot_schedule_x`.

    .. method:: get_cnot_schedule_pairs_x() -> list[int]

        Return the X-stabilizer CNOT schedule as a flat list of
        *(stabilizer_index, data_index)* pairs ordered by timestep within
        each stabilizer.

    .. method:: get_cnot_schedule_pairs_z() -> list[int]

        Z-stabilizer counterpart of :meth:`get_cnot_schedule_pairs_x`.

Detector Error Model
====================

.. class:: DetectorErrorModel

    Detector error model (DEM) for a QEC circuit.

    A DEM encodes which errors flip which detectors and is used by the
    decoder to predict observable flips. Create one with
    :func:`dem_from_memory_circuit` or :func:`dem_from_stim_text`.

    .. attribute:: detector_error_matrix
        :type: numpy.ndarray

        2-D ``uint8`` array; entry ``[i, j]`` is 1 if detector *i* is
        triggered by error mechanism *j*.

    .. attribute:: observables_flips_matrix
        :type: numpy.ndarray

        2-D ``uint8`` array; entry ``[i, j]`` is 1 if Pauli observable
        *i* is flipped by error mechanism *j*.

    .. attribute:: error_rates
        :type: list[float]

        Probability assigned to each error mechanism; length equals the
        number of columns of ``detector_error_matrix``.

    .. attribute:: error_ids
        :type: list[int]

        Error mechanism IDs. Mechanisms sharing the same ID are
        correlated (at most one can occur per shot); different IDs are
        independent.

    .. method:: num_detectors() -> int

        Number of detectors.

    .. method:: num_error_mechanisms() -> int

        Number of error mechanisms (columns of the DEM matrix).

    .. method:: num_observables() -> int

        Number of Pauli observables.

    .. method:: canonicalize_for_rounds(num_syndromes_per_round, remove_zero_syndrome_errors=False)

        Merge columns that share the same detector–observable signature,
        composing error rates. Assigns fresh unique error IDs (input
        correlation structure is discarded).

        :param num_syndromes_per_round: Number of syndrome bits per round.
        :param remove_zero_syndrome_errors: Drop columns that trigger no
            detector. Default ``False`` preserves observable-flip
            probability.

    .. method:: canonicalize_for_rounds_with_boundary(num_syndromes_per_round, num_boundary_syndromes, remove_zero_syndrome_errors=False)

        Boundary-aware variant for memory-experiment DEMs whose first and
        last detector layers are narrower than interior layers.

        :param num_syndromes_per_round: Interior round width.
        :param num_boundary_syndromes: Width of boundary rounds.
        :param remove_zero_syndrome_errors: See
            :meth:`canonicalize_for_rounds`.

.. class:: DecoderContext

    Lazy handle returned by :func:`decoder_context_from_memory_circuit`.

    Stores raw circuit analysis; each ``*_component`` method
    canonicalizes the requested stabilizer type and returns a
    ``(dem, m2d, m2o)`` tuple.

    .. attribute:: num_measurements
        :type: int

        Total number of measurements per shot.

    .. method:: x_component() -> tuple

        Canonicalize X-stabilizer detectors.

        :returns: ``(dem, m2d, m2o)`` — the canonicalized
            :class:`DetectorErrorModel` and measurement-to-detector /
            measurement-to-observable index lists.

    .. method:: z_component() -> tuple

        Canonicalize Z-stabilizer detectors.

        :returns: ``(dem, m2d, m2o)`` — same layout as
            :meth:`x_component`.

    .. method:: full_component() -> tuple

        Canonicalize both stabilizer types with boundary awareness.

        :returns: ``(dem, m2d, m2o)`` — same layout as
            :meth:`x_component`.

.. autofunction:: cudaq_qec.dem_from_memory_circuit
.. autofunction:: cudaq_qec.x_dem_from_memory_circuit
.. autofunction:: cudaq_qec.z_dem_from_memory_circuit
.. autofunction:: cudaq_qec.decoder_context_from_memory_circuit
.. autofunction:: cudaq_qec.dem_from_stim_text

Decoder Interfaces
==================

.. class:: Decoder(H)

    Abstract base class for QEC decoders.

    Subclass and decorate with ``@qec.decoder(name)`` to register a
    custom decoder. Override ``decode_batch`` (preferred) or ``decode``.

    :param H: Parity check matrix — ``numpy.ndarray`` of ``uint8`` with
        shape ``(num_syndromes, num_qubits)`` or any ``scipy.sparse``
        format (CSR, CSC, COO, …).

    .. method:: decode(syndrome) -> DecoderResult

        Decode a single syndrome.

        :param syndrome: 1-D sequence of floats, one per syndrome bit.
        :returns: :class:`DecoderResult`.

    .. method:: decode_async(syndrome) -> AsyncDecoderResult

        Asynchronous variant of :meth:`decode`.

        :returns: :class:`AsyncDecoderResult`; call ``.get()`` to block
            until the result is ready.

    .. method:: decode_batch(syndrome) -> BatchDecoderResult

        Decode a batch of syndromes.

        :param syndrome: 2-D array of shape
            ``(num_shots, num_syndrome_bits)``.
        :returns: :class:`BatchDecoderResult`.

    .. method:: get_block_size() -> int

        Size of the code block (number of qubits).

    .. method:: get_syndrome_size() -> int

        Length of the syndrome vector.

    .. method:: get_version() -> str

        Version string of the decoder implementation.

.. class:: DecoderResult

    Single-shot decoder result returned by :meth:`Decoder.decode`.

    Also used as the construct-then-mutate return value in Python decoder
    plugins that override ``decode``:

    .. code-block:: python

        res = DecoderResult()
        res.converged = True
        res.result = np.zeros(n, dtype=np.float64)
        return res

    Supports ``len()`` (always 3) and unpacking into
    ``(converged, result, opt_results)``.

    .. attribute:: converged
        :type: bool

        ``True`` if the decoder converged to a valid correction chain.

    .. attribute:: result
        :type: numpy.ndarray

        1-D float array containing the decoded correction chain.
        A fresh array is allocated on each read.

    .. attribute:: opt_results

        Optional decoder-specific metadata (heterogeneous map); may be
        empty.

.. class:: BatchDecoderResult(result, converged, opt_results=None)

    Batched decoder result returned by :meth:`Decoder.decode_batch`.

    Output-only in normal use. Python decoder plugins that override
    ``decode_batch`` construct one to return:

    :param result: 2-D NumPy float array of shape
        ``(num_shots, correction_size)``.
    :param converged: 1-D NumPy bool array of length ``num_shots``.
    :param opt_results: Per-shot list of dicts (or ``None`` entries), or
        ``None``.

    Supports integer indexing and slicing (returns a
    :class:`DecoderResult` copy or a sliced :class:`BatchDecoderResult`)
    and iteration for compatibility with the previous
    ``list[DecoderResult]`` API. Prefer reading ``result``,
    ``converged``, and ``opt_results`` directly for batch workflows.

    .. attribute:: result
        :type: numpy.ndarray

        2-D float array of decoder outputs, one row per shot.

    .. attribute:: converged
        :type: numpy.ndarray

        1-D bool array; ``True`` per shot if the decoder converged.

    .. attribute:: opt_results
        :type: list

        Per-shot optional result dicts (or ``None`` entries).

.. class:: AsyncDecoderResult

    Future-like object returned by :meth:`Decoder.decode_async`.

    .. method:: get() -> DecoderResult

        Block until the asynchronous decode completes and return the
        :class:`DecoderResult`.

    .. method:: ready() -> bool

        Return ``True`` if the result is already available (non-blocking
        poll).

.. autofunction:: cudaq_qec.get_decoder

Built-in Decoders
=================

.. _nv_qldpc_decoder_api_python:

NVIDIA QLDPC Decoder
--------------------

.. include:: nv_qldpc_decoder_api.rst

Sliding Window Decoder
----------------------

.. include:: sliding_window_api.rst

.. _trt_decoder_api_python:

TensorRT Decoder
----------------

.. include:: trt_decoder_api.rst

.. _tensor_network_decoder_api_python:

Tensor Network Decoder
----------------------

.. include:: tensor_network_decoder_api.rst

Real-Time Decoding
==================

.. include:: python_realtime_decoding_api.rst


Common
=============

.. autofunction:: cudaq_qec.sample_memory_circuit
.. autofunction:: cudaq_qec.x_sample_memory_circuit
.. autofunction:: cudaq_qec.z_sample_memory_circuit

.. autofunction:: cudaq_qec.sample_code_capacity

.. _parity_check_matrix_utilities_python:

Parity Check Matrix Utilities
=============================

.. autofunction:: cudaq_qec.generate_random_pcm
.. autofunction:: cudaq_qec.generate_timelike_sparse_detector_matrix
.. autofunction:: cudaq_qec.get_pcm_for_rounds
.. autofunction:: cudaq_qec.get_sorted_pcm_column_indices
.. autofunction:: cudaq_qec.pcm_extend_to_n_rounds
.. autofunction:: cudaq_qec.pcm_is_sorted
.. autofunction:: cudaq_qec.pcm_to_sparse_vec
.. autofunction:: cudaq_qec.reorder_pcm_columns
.. autofunction:: cudaq_qec.shuffle_pcm_columns
.. autofunction:: cudaq_qec.simplify_pcm
.. autofunction:: cudaq_qec.sort_pcm_columns
