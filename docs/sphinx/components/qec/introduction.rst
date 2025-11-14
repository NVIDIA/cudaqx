CUDA-Q QEC - Quantum Error Correction Library
=============================================

Overview
--------
The ``cudaq-qec`` library provides a comprehensive framework for quantum
error correction research and development. It leverages GPU acceleration
for efficient syndrome decoding and error correction simulations (coming soon).

Core Components
----------------
``cudaq-qec`` is composed of two main interfaces - the :code:`cudaq::qec::code` and
:code:`cudaq::qec::decoder` types. These types are meant to be extended by developers
to provide new error correcting codes and new decoding strategies.

QEC Code Framework :code:`cudaq::qec::code`
-------------------------------------------

The :code:`cudaq::qec::code` class serves as the base class for all quantum error correcting codes in CUDA-Q QEC. It provides
a flexible extension point for implementing new codes and defines the core interface that all QEC codes must support.

The core abstraction here is that of a mapping or dictionary of logical operations to their
corresponding physical implementation in the error correcting code as CUDA-Q quantum kernels.

Class Structure
^^^^^^^^^^^^^^^

The code base class provides:

1. **Operation Enumeration**: Defines supported logical operations

   .. code-block:: cpp

       enum class operation {
           x,     // Logical X gate
           y,     // Logical Y gate
           z,     // Logical Z gate
           h,     // Logical Hadamard gate
           s,     // Logical S gate
           cx,    // Logical CNOT gate
           cy,    // Logical CY gate
           cz,    // Logical CZ gate
           stabilizer_round,  // Stabilizer measurement round
           prep0, // Prepare |0⟩ state
           prep1, // Prepare |1⟩ state
           prepp, // Prepare |+⟩ state
           prepm  // Prepare |-⟩ state
       };


2. **Patch Type**: Defines the structure of a logical qubit patch

   .. code-block:: cpp

       struct patch {
           cudaq::qview<> data;  // View of data qubits
           cudaq::qview<> ancx;  // View of X stabilizer ancilla qubits
           cudaq::qview<> ancz;  // View of Z stabilizer ancilla qubits
       };

   The `patch` type represents a logical qubit in quantum error correction codes. It contains:
   - `data`: A view of the data qubits in the patch
   - `ancx`: A view of the ancilla qubits used for X stabilizer measurements
   - `ancz`: A view of the ancilla qubits used for Z stabilizer measurements

   This structure is designed for use within CUDA-Q kernel code and provides a
   convenient way to access different qubit subsets within a logical qubit patch.


3. **Kernel Type Aliases**: Defines quantum kernel signatures

   .. code-block:: cpp

       using one_qubit_encoding = cudaq::qkernel<void(patch)>;
       using two_qubit_encoding = cudaq::qkernel<void(patch, patch)>;
       using stabilizer_round = cudaq::qkernel<std::vector<cudaq::measure_result>(
           patch, const std::vector<std::size_t>&, const std::vector<std::size_t>&)>;

4. **Protected Members**:

   - :code:`operation_encodings`: Maps operations to their quantum kernel implementations. The key is the ``operation`` enum and the value is a variant on the above kernel type aliases.
   - :code:`m_stabilizers`: Stores the code's stabilizer generators

Implementing a New Code
^^^^^^^^^^^^^^^^^^^^^^^

To implement a new quantum error correcting code:

1. **Create a New Class**:

   .. code-block:: cpp

       class my_code : public qec::code {
       protected:
           // Implement required virtual methods
       public:
           my_code(const heterogeneous_map& options);
       };

2. **Implement Required Virtual Methods**:

   .. code-block:: cpp

       // Number of physical data qubits
       std::size_t get_num_data_qubits() const override;

       // Total number of ancilla qubits
       std::size_t get_num_ancilla_qubits() const override;

       // Number of X-type ancilla qubits
       std::size_t get_num_ancilla_x_qubits() const override;

       // Number of Z-type ancilla qubits
       std::size_t get_num_ancilla_z_qubits() const override;

3. **Define Quantum Kernels**:

   Create CUDA-Q kernels for each logical operation:

   .. code-block:: cpp

       __qpu__ void x(patch p) {
           // Implement logical X
       }

       __qpu__ std::vector<cudaq::measure_result> stabilizer(patch p,
           const std::vector<std::size_t>& x_stabs,
           const std::vector<std::size_t>& z_stabs) {
           // Implement stabilizer measurements
       }

4. **Register Operations**:

   In the constructor, register quantum kernels for each operation:

   .. code-block:: cpp

        my_code::my_code(const heterogeneous_map& options) : code() {
            // Register operations
            operation_encodings.insert(
               std::make_pair(operation::x, x));
            operation_encodings.insert(
               std::make_pair(operation::stabilizer_round, stabilizer));

            // Define stabilizer generators
            m_stabilizers = qec::stabilizers({"XXXX", "ZZZZ"});
        }


   Note that in your constructor, you have access to user-provided ``options``. For
   example, if your code depends on an integer parameter called ``distance``, you can
   retrieve that from the user via

   .. code-block:: cpp

        my_code::my_code(const heterogeneous_map& options) : code() {
            // ... fill the map and stabilizers ...

            // Get the user-provided distance, or just
            // set to 3 if user did not provide one
            this->distance = options.get<int>("distance", /*defaultValue*/ 3);
        }

5. **Register Extension Point**:

   Add extension point registration:

   .. code-block:: cpp

       CUDAQ_EXTENSION_CUSTOM_CREATOR_FUNCTION(
           my_code,
           static std::unique_ptr<qec::code> create(
               const heterogeneous_map &options) {
               return std::make_unique<my_code>(options);
           }
       )

       CUDAQ_REGISTER_TYPE(my_code)

Example: Steane Code
^^^^^^^^^^^^^^^^^^^^^

The Steane [[7,1,3]] code provides a complete example implementation:

1. **Header Definition**:

   - Declares quantum kernels for all logical operations
   - Defines the code class with required virtual methods
   - Specifies 7 data qubits and 6 ancilla qubits (3 X-type, 3 Z-type)

2. **Implementation**:

   .. code-block:: cpp

       steane::steane(const heterogeneous_map &options) : code() {
           // Register all logical operations
           operation_encodings.insert(
               std::make_pair(operation::x, x));
           // ... register other operations ...

           // Define stabilizer generators
           m_stabilizers = qec::stabilizers({
               "XXXXIII", "IXXIXXI", "IIXXIXX",
               "ZZZZIII", "IZZIZZI", "IIZZIZZ"
           });
       }

3. **Quantum Kernels**:

   Implements fault-tolerant logical operations:

   .. code-block:: cpp

       __qpu__ void x(patch logicalQubit) {
           // Apply logical X to specific data qubits
           x(logicalQubit.data[4], logicalQubit.data[5],
             logicalQubit.data[6]);
       }

       __qpu__ std::vector<cudaq::measure_result> stabilizer(patch logicalQubit,
           const std::vector<std::size_t>& x_stabilizers,
           const std::vector<std::size_t>& z_stabilizers) {
           // Measure X stabilizers
           h(logicalQubit.ancx);
           // ... apply controlled-X gates ...
           h(logicalQubit.ancx);

           // Measure Z stabilizers
           // ... apply controlled-X gates ...

           // Return measurement results
           return mz(logicalQubit.ancz, logicalQubit.ancx);
       }

Implementing a New Code in Python
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

CUDA-Q QEC supports implementing quantum error correction codes in Python
using the :code:`@qec.code` decorator. This provides a more accessible way
to prototype and develop new codes.

1. **Create a New Python File**:

   Create a new file (e.g., :code:`my_steane.py`) with your code implementation:

   .. literalinclude:: ../../examples/qec/python/my_steane.py
      :language: python
      :start-after: [Begin Documentation1]
      :end-before: [End Documentation1]

2. **Define Quantum Kernels**:

   Implement the required quantum kernels using the :code:`@cudaq.kernel` decorator:

   .. literalinclude:: ../../examples/qec/python/my_steane.py
      :language: python
      :start-after: [Begin Documentation2]
      :end-before: [End Documentation2]

3. **Implement the Code Class**:

   Create a class decorated with :code:`@qec.code` that implements the required interface:

   .. literalinclude:: ../../examples/qec/python/my_steane.py
      :language: python
      :start-after: [Begin Documentation3]
      :end-before: [End Documentation3]

4. **Using the Code**:

   The code can now be used like any other CUDA-Q QEC code:

   .. literalinclude:: ../../examples/qec/python/my_steane_test.py
      :language: python
      :start-after: [Begin Documentation]

Key Points
^^^^^^^^^^^

* The :code:`@qec.code` decorator takes the name of the code as an argument
* Operation encodings are registered via the :code:`operation_encodings` dictionary
* Stabilizer generators are defined using the :code:`qec.Stabilizers` class
* The code must implement all required methods from the base class interface


Using the Code Framework
^^^^^^^^^^^^^^^^^^^^^^^^^

To use an implemented code:

.. tab:: Python

    .. code-block:: python

        # Create a code instance
        code = qec.get_code("steane")

        # Access stabilizer information
        stabilizers = code.get_stabilizers()
        parity = code.get_parity()

        # The code can now be used for various numerical
        # experiments - see section below.

.. tab:: C++

    .. code-block:: cpp

        // Create a code instance
        auto code = cudaq::qec::get_code("steane");

        // Access stabilizer information
        auto stabilizers = code->get_stabilizers();
        auto parity = code->get_parity();

        // The code can now be used for various numerical
        // experiments - see section below.


Pre-built QEC Codes
-------------------

CUDA-Q QEC provides several well-studied quantum error correction codes out of the box. Here's a detailed overview of each:

Steane Code
^^^^^^^^^^^

The Steane code is a ``[[7,1,3]]`` CSS (Calderbank-Shor-Steane) code that encodes
one logical qubit into seven physical qubits with a code distance of 3.

**Key Properties**:

* Data qubits: 7
* Encoded qubits: 1
* Code distance: 3
* Ancilla qubits: 6 (3 for X stabilizers, 3 for Z stabilizers)

**Stabilizer Generators**:

* X-type: ``["XXXXIII", "IXXIXXI", "IIXXIXX"]``
* Z-type: ``["ZZZZIII", "IZZIZZI", "IIZZIZZ"]``

The Steane code can correct any single-qubit error and detect up to two errors.
It is particularly notable for being the smallest CSS code that can implement a universal set of transversal gates.

Usage:

.. tab:: Python

    .. code-block:: python

        import cudaq_qec as qec

        # Create Steane code instance
        steane = qec.get_code("steane")

.. tab:: C++

    .. code-block:: cpp

        auto steane = cudaq::qec::get_code("steane");

Repetition Code
^^^^^^^^^^^^^^^
The repetition code is a simple [[n,1,n]] code that protects against
bit-flip (X) errors by encoding one logical qubit into n physical qubits, where n is the code distance.

**Key Properties**:

* Data qubits: n (distance)
* Encoded qubits: 1
* Code distance: n
* Ancilla qubits: n-1 (all for Z stabilizers)

**Stabilizer Generators**:

* For distance 3: ``["ZZI", "IZZ"]``
* For distance 5: ``["ZZIII", "IZZII", "IIZZI", "IIIZZ"]``

The repetition code is primarily educational as it can only correct
X errors. However, it serves as an excellent introduction to QEC concepts.

Usage:

.. tab:: Python

    .. code-block:: python

        import cudaq_qec as qec

        # Create distance-3 repetition code
        code = qec.get_code('repetition', distance=3)

        # Access stabilizers
        stabilizers = code.get_stabilizers()  # Returns ["ZZI", "IZZ"]

.. tab:: C++

    .. code-block:: cpp

        auto code = qec::get_code("repetition", {{"distance", 3}});

        // Access stabilizers
        auto stabilizers = code->get_stabilizers();


Decoder Framework :code:`cudaq::qec::decoder`
----------------------------------------------

The CUDA-Q QEC decoder framework provides an extensible system for implementing
quantum error correction decoders through the :code:`cudaq::qec::decoder` base class.

Class Structure
^^^^^^^^^^^^^^^

The decoder base class defines the core interface for syndrome decoding:

.. code-block:: cpp

    class decoder {
    protected:
        std::size_t block_size;       // For [n,k] code, this is n
        std::size_t syndrome_size;    // For [n,k] code, this is n-k
        tensor<uint8_t> H;            // Parity check matrix

    public:
        struct decoder_result {
            bool converged;                 // Decoder convergence status
            std::vector<float_t> result;    // Soft error probabilities
        };

        virtual decoder_result decode(
            const std::vector<float_t>& syndrome) = 0;

        virtual std::vector<decoder_result> decode_batch(
            const std::vector<std::vector<float_t>>& syndrome);
    };

Key Components:

* **Parity Check Matrix**: Defines the code structure via :code:`H`
* **Block Size**: Number of physical qubits in the code
* **Syndrome Size**: Number of stabilizer measurements
* **Decoder Result**: Contains convergence status and error probabilities
* **Multiple Decoding Modes**: Single syndrome or batch processing

Implementing a New Decoder in C++
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To implement a new decoder:

1. **Create Decoder Class**:

.. code-block:: cpp

    class my_decoder : public qec::decoder {
    private:
        // Decoder-specific members

    public:
        my_decoder(const tensor<uint8_t>& H,
                  const heterogeneous_map& params)
            : decoder(H) {
            // Initialize decoder
        }

        decoder_result decode(
            const std::vector<float_t>& syndrome) override {
            // Implement decoding logic
        }
    };

2. **Register Extension Point**:

.. code-block:: cpp

    CUDAQ_EXTENSION_CUSTOM_CREATOR_FUNCTION(
        my_decoder,
        static std::unique_ptr<decoder> create(
            const tensor<uint8_t>& H,
            const heterogeneous_map& params) {
            return std::make_unique<my_decoder>(H, params);
        }
    )

    CUDAQ_REGISTER_TYPE(my_decoder)

Example: Lookup Table Decoder
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Here's a simple lookup table decoder for the Steane code:

.. code-block:: cpp

    class single_error_lut : public decoder {
    private:
        std::map<std::string, std::size_t> single_qubit_err_signatures;

    public:
        single_error_lut(const tensor<uint8_t>& H,
                          const heterogeneous_map& params)
            : decoder(H) {
            // Build lookup table for single-qubit errors
            for (std::size_t qErr = 0; qErr < block_size; qErr++) {
                std::string err_sig(syndrome_size, '0');
                for (std::size_t r = 0; r < syndrome_size; r++) {
                    bool syndrome = 0;
                    for (std::size_t c = 0; c < block_size; c++)
                        syndrome ^= (c != qErr) && H.at({r, c});
                    err_sig[r] = syndrome ? '1' : '0';
                }
                single_qubit_err_signatures.insert({err_sig, qErr});
            }
        }

        decoder_result decode(
            const std::vector<float_t>& syndrome) override {
            decoder_result result{false,
                std::vector<float_t>(block_size, 0.0)};

            // Convert syndrome to string
            std::string syndrome_str(syndrome_size, '0');
            for (std::size_t i = 0; i < syndrome_size; i++)
                syndrome_str[i] = (syndrome[i] >= 0.5) ? '1' : '0';

            // Lookup error location
            auto it = single_qubit_err_signatures.find(syndrome_str);
            if (it != single_qubit_err_signatures.end()) {
                result.converged = true;
                result.result[it->second] = 1.0;
            }

            return result;
        }
    };

Implementing a Decoder in Python
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

CUDA-Q QEC supports implementing decoders in Python using the :code:`@qec.decoder` decorator:

1. **Create Decoder Class**:

.. code-block:: python

    @qec.decoder("my_decoder")
    class MyDecoder:
        def __init__(self, H, **kwargs):
            qec.Decoder.__init__(self, H)
            self.H = H
            # Initialize with optional kwargs

        def decode(self, syndrome):
            # Create result object
            result = qec.DecoderResult()

            # Implement decoding logic
            # ...

            # Set results
            result.converged = True
            result.result = [0.0] * self.block_size

            return result

2. **Using Custom Parameters**:

.. code-block:: python

    # Create decoder with custom parameters
    decoder = qec.get_decoder("my_decoder",
                            H=parity_check_matrix,
                            custom_param=42)

Key Features
^^^^^^^^^^^^^

* **Soft Decision Decoding**: Results are probabilities in [0,1]
* **Batch Processing**: Support for decoding multiple syndromes
* **Asynchronous Decoding**: Optional async interface for parallel processing
* **Custom Parameters**: Flexible configuration via heterogeneous_map
* **Python Integration**: First-class support for Python implementations

Usage Example
^^^^^^^^^^^^^^

.. tab:: Python

    .. code-block:: python

        import cudaq_qec as qec

        # Get a code instance
        steane = qec.get_code("steane")

        # Create decoder with code's parity matrix
        decoder = qec.get_decoder('single_error_lut', steane.get_parity())

        # Run stabilizer measurements
        syndromes, dataQubitResults = qec.sample_memory_circuit(steane, numShots=1, numRounds=1)

        # Decode a syndrome
        result = decoder.decode(syndromes[0])
        if result.converged:
            print("Error locations:",
                [i for i,p in enumerate(result.result) if p > 0.5])
            # No errors as we did not include a noise model and
            # thus prints:
            # Error locations: []

.. tab:: C++

    .. code-block:: cpp

        using namespace cudaq;

        // Get a code instance
        auto code = qec::get_code("steane");

        // Create decoder with code's parity matrix
        auto decoder = qec::get_decoder("single_error_lut",
                                code->get_parity());

        // Run stabilizer measurements
        auto [syndromes, dataQubitResults] = qec::sample_memory_circuit(*code, /*numShots*/numShots, /*numRounds*/ 1);

        // Decode syndrome
        auto result = decoder->decode(syndromes[0]);


Pre-built QEC Decoders
----------------------

CUDA-Q QEC provides pre-built decoders. Here's a detailed overview of each:

Quantum Low-Density Parity-Check Decoder
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Quantum Low-Density Parity-Check (QLDPC) decoder leverages GPU-accelerated belief propagation (BP) for efficient error correction. 
Since belief propagation is an iterative method which may not converge, decoding can be improved with a second-stage post-processing step. The `nv-qldpc-decoder`
API provides various post-processing options, which can be selected through its parameters.

The QLDPC decoder `nv-qldpc-decoder` requires a CUDA-Q compatible GPU. See the list below for dependencies and compatibility:
https://nvidia.github.io/cuda-quantum/latest/using/install/local_installation.html#dependencies-and-compatibility

The decoder is based on the following references:

* https://arxiv.org/pdf/2005.07016 
* https://github.com/quantumgizmos/ldpc


Usage:

.. tab:: Python

    .. code-block:: python

        import cudaq_qec as qec
        import numpy as np

        H_list = [
                    [1, 0, 0, 1, 0, 1, 1], 
                    [0, 1, 0, 1, 1, 0, 1],
                    [0, 0, 1, 0, 1, 1, 1]
                 ]

        H_np = np.array(H_list, dtype=np.uint8)

        decoder = qec.get_decoder("nv-qldpc-decoder", H_np)

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

        auto d1 = cudaq::qec::get_decoder("nv-qldpc-decoder", H, nv_custom_args);

        // Alternatively, configure the decoder without instantiating a heterogeneous_map 
        auto d2 = cudaq::qec::get_decoder("nv-qldpc-decoder", H, {{"use_osd", true}, {"bp_batch_size", 100}});

Tensor Network Decoder
^^^^^^^^^^^^^^^^^^^^^^

The ``tensor_network_decoder`` constructs a tensor network representation of a quantum code given its parity check matrix, logical observable(s), and noise model. It can decode individual syndromes or batches of syndromes, returning the probability that a logical observable has flipped.

Due to the additional dependencies of the Tensor Network Decoder, you must
specify the optional pip package when installing CUDA-Q QEC in order to use this
decoder. Use `pip install cudaq-qec[tensor-network-decoder]` in order to use
this decoder.

Key Steps:

1. **Define the parity check matrix**: This matrix encodes the structure of the quantum code. In the example, a simple [3,1] repetition code is used.

2. **Specify the logical observable**: This is typically a row vector indicating which qubits participate in the logical operator.

3. **Set the noise model**: The example uses a factorized noise model with independent bit-flip probability for each error mechanism.

4. **Instantiate the decoder**: Create a decoder object using ``qec.get_decoder("tensor_network_decoder", ...)`` with the code parameters.

5. **Decode syndromes**: Use the ``decode`` method for single syndromes or ``decode_batch`` for multiple syndromes.


Usage:

.. tab:: Python

    .. code-block:: python

        # This example demonstrates how to use the get_decoder("tensor_network_decoder", ...) API
        # from the ``cudaq_qec`` library to decode syndromes for a simple 
        # quantum error-correcting code using tensor networks.

        import cudaq_qec as qec
        import numpy as np

        # Define code parameters
        H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
        logical_obs = np.array([[1, 1, 1]], dtype=np.uint8)
        noise_model = [0.1, 0.1, 0.1]

        decoder = qec.get_decoder("tensor_network_decoder", H, logical_obs=logical_obs, noise_model=noise_model)

        # Decode a single syndrome
        syndrome = [0.0, 1.0]
        result = decoder.decode(syndrome)
        print(result.result)

        # Decode a batch of syndromes
        syndrome_batch = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
        batch_results = decoder.decode_batch(syndrome_batch)
        for res in batch_results:
            print(res.result)

.. tab:: C++

    The ``tensor_network_decoder`` is a Python-only implementation and it requires Python 3.11 or higher. C++ APIs are not available for this decoder.

Output:

The decoder returns the probability that the logical observable has flipped for each syndrome. This can be used to assess the performance of the code and the decoder under different error scenarios.

.. note::

    In general, the Tensor Network Decoder has the same GPU support as the
    `Quantum Low-Density Parity-Check Decoder <https://nvidia.github.io/cudaqx/components/qec/introduction.html#quantum-low-density-parity-check-decoder>`__.
    However, if you are using the V100 GPU (SM70), you will need to pin your
    cuTensor version to 2.2 by running `pip install cutensor_cu12==2.2`. Note
    that this GPU will not be supported by the Tensor Network Decoder when
    CUDA-Q 0.5.0 is released.


Numerical Experiments
---------------------

CUDA-Q QEC provides utilities for running numerical experiments with quantum error correction codes.

Conventions
^^^^^^^^^^^

To address vectors of qubits (`cudaq::qvector`), CUDAQ indexing starts from 0, and 0 corresponds
to the leftmost position when working with pauli strings (`cudaq::spin_op`). For example, applying a pauli X operator
to qubit 1 out of 7 would be `X_1 = IXIIIII`.

While implementing your own codes and decoders, you are free to follow any convention that is convenient to you. However,
to interact with the pre-built QEC codes and decoders within this library, the following conventions are used. All of these codes
are CSS codes, and so we separate :math:`X`-type and :math:`Z`-type errors. For example, an error vector for 3 qubits will
have 6 entries, 3 bits representing the presence of a bit-flip on each qubit, and 3 bits representing a phase-flip on each qubit.
An error vector representing a bit-flip on qubit 0, and a phase-flip on qubit 1 would look like `E = 100010`. This means that this
error vector is just two error vectors (`E_X, E_Z`) concatenated together (`E = E_X | E_Z`).

These errors are detected by stabilizers. :math:`Z`-stabilizers detect :math:`X`-type errors and vice versa. Thus we write our
CSS parity check matrices as

.. math::
  H_{CSS} = \begin{pmatrix}
   H_Z & 0 \\
   0 & H_X
   \end{pmatrix},

so that when we generate a syndrome vector by multiplying the parity check matrix by an error vector we get

.. math::
   \begin{align}
  S &= H \cdot E\\
  S_X &= H_Z \cdot E_x\\
  S_Z &= H_X \cdot E_Z.
  \end{align}

This means that for the concatenated syndrome vector `S = S_X | S_Z`, the first part, `S_X`, are syndrome bits triggered by `Z`
stabilizers detecting `X` errors. This is because the `Z` stabilizers like `ZZI` and `IZZ` anti-commute with `X` errors like
`IXI`.

The decoder prediction as to what error happened is `D = D_X | D_Z`. A successful error decoding does not require that `D = E`,
but that `D + E` is not a logical operator. There are a couple ways to check this.
For bitflip errors, we check that the residual error `R = D_X + E_X` is not `L_X`. Since `X` anticommutes
with `Z`, we can check that `L_Z(D_X + E_X) = 0`. This is because we just need to check if they have mutual support on an even
or odd number of qubits. We could also check that `R` is not a stabilizer.

Similar to the parity check matrix, the logical observables are also stored in a matrix as

.. math::
  L = \begin{pmatrix}
   L_Z & 0 \\
   0 & L_X
   \end{pmatrix},

so that when determining logical errors, we can do matrix multiplication

.. math::
   \begin{align}
  P &= L \cdot R\\
  P_X &= L_Z \cdot R_x\\
  P_Z &= L_X \cdot R_Z.
  \end{align}

Here we're using `P` as this can be stored in a Pauli frame tracker to track observable flips.

Each logical qubit has logical observables associated with it. Depending on what basis the data qubits are measured in, either the
`X` or `Z` logical observables can be measured. The data qubits which support the logical observable is contained the `qec::code` class as well.

To do a logical `Z(X)` measurement, measure out all of the data qubits in the `Z(X)` basis. Then check support on the appropriate
`Z(x)` observable.


Memory Circuit Experiments
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Memory circuit experiments test a QEC code's ability to preserve quantum information over time by:

1. Preparing an initial logical state
2. Performing multiple rounds of stabilizer measurements
3. Measuring data qubits to verify state preservation
4. Optionally applying noise during the process

Function Variants
~~~~~~~~~~~~~~~~~

.. tab:: Python

    .. code-block:: python

        import cudaq
        import cudaq_qec as qec

        # Use the stim backend for performance in QEC settings
        cudaq.set_target("stim")

        # Get a code instance
        code = qec.get_code("steane")

        # Basic memory circuit with |0⟩ state
        syndromes, measurements = qec.sample_memory_circuit(
            code,           # QEC code instance
            numShots=1000,  # Number of circuit executions
            numRounds=1     # Number of stabilizer rounds
        )

        # Memory circuit with custom initial state
        syndromes, measurements = qec.sample_memory_circuit(
            code,                     # QEC code instance
            op=qec.operation.prep1,   # Initial state
            numShots=1000,            # Number of shots
            numRounds=1               # Number of rounds
        )

        # Memory circuit with noise model
        noise = cudaq.NoiseModel()
        # Configure noise
        noise.add_all_qubit_channel("x", cudaq.Depolarization2(0.01), 1)
        syndromes, measurements = qec.sample_memory_circuit(
            code,             # QEC code instance
            numShots=1000,    # Number of shots
            numRounds=1,      # Number of rounds
            noise=noise       # Noise model
        )

.. tab:: C++

    .. code-block:: cpp

        // Basic memory circuit with |0⟩ state
        auto [syndromes, measurements] = qec::sample_memory_circuit(
            code,       // QEC code instance
            numShots,   // Number of circuit executions
            numRounds   // Number of stabilizer rounds
        );

        // Memory circuit with custom initial state
        auto [syndromes, measurements] = qec::sample_memory_circuit(
            code,               // QEC code instance
            operation::prep1,   // Initial state preparation
            numShots,           // Number of circuit executions
            numRounds           // Number of stabilizer rounds
        );

        // Memory circuit with noise model
        auto noise_model = cudaq::noise_model();
        noise_model.add_channel(...);  // Configure noise
        auto [syndromes, measurements] = qec::sample_memory_circuit(
            code,         // QEC code instance
            numShots,     // Number of circuit executions
            numRounds,    // Number of stabilizer rounds
            noise_model   // Noise model to apply
        );

Return Values
~~~~~~~~~~~~~

The functions return a tuple containing:

1. **Syndrome Measurements** (:code:`tensor<uint8_t>`):

   * Shape: :code:`(num_shots, num_rounds * syndrome_size)`
   * Contains stabilizer measurement results
   * Values are 0 or 1 representing measurement outcomes

2. **Data Measurements** (:code:`tensor<uint8_t>`):

   * Shape: :code:`(num_shots, block_size)`
   * Contains final data qubit measurements
   * Used to verify logical state preservation

Example Usage
~~~~~~~~~~~~~

Example of running a memory experiment:

.. tab:: Python

    .. code-block:: python

        import cudaq
        import cudaq_qec as qec

        # Use the stim backend for performance in QEC settings
        cudaq.set_target("stim")

        # Create code and decoder
        code = qec.get_code('steane')
        decoder = qec.get_decoder('single_error_lut',
                                  code.get_parity())

        # Configure noise
        noise = cudaq.NoiseModel()
        noise.add_all_qubit_channel("x", cudaq.Depolarization2(0.01), 1)

        # Run memory experiment
        syndromes, measurements = qec.sample_memory_circuit(
            code,
            op=qec.operation.prep0,
            numShots=1000,
            numRounds=10,
            noise=noise
        )

        # Analyze results
        for shot in range(1000):
            # Get syndrome for this shot
            syndrome = syndromes[shot].tolist()

            # Decode syndrome
            result = decoder.decode(syndrome)
            if result.converged:
                # Process correction
                pass

.. tab:: C++

    .. code-block:: cpp

        // Compile and run with:
        // nvq++ --enable-mlir --target=stim -lcudaq-qec example.cpp
        // ./a.out

        #include "cudaq.h"
        #include "cudaq/qec/decoder.h"
        #include "cudaq/qec/experiments.h"
        #include "cudaq/qec/noise_model.h"

        int main(){
          // Create a Steane code instance
          auto code = cudaq::qec::get_code("steane");

          // Configure noise model
          cudaq::noise_model noise;
          noise.add_all_qubit_channel("x", cudaq::depolarization2(0.1),
                              /*num_controls=*/1);

          // Run memory experiment
          auto [syndromes, data] = cudaq::qec::sample_memory_circuit(
              *code,                          // Code instance
              cudaq::qec::operation::prep0,   // Prepare |0⟩ state
              1000,                           // 1000 shots
              1,                              // 1 rounds
              noise                           // Apply noise
          );

          // Analyze results
          auto decoder = cudaq::qec::get_decoder("single_error_lut", code->get_parity());
          for (std::size_t shot = 0; shot < 1000; shot++) {
            // Get syndrome for this shot
            std::vector<cudaq::qec::float_t> syndrome(syndromes.shape()[1]);
            for (std::size_t i = 0; i < syndrome.size(); i++)
              syndrome[i] = syndromes.at({shot, i});

            // Decode syndrome
            auto results = decoder->decode(syndrome);
            // Process correction
            // ...
          }
        }

Additional Noise Models
~~~~~~~~~~~~~~~~~~~~~~~

.. tab:: Python

  .. code-block:: python

     noise = cudaq.NoiseModel()

     # Add multiple error channels
     noise.add_all_qubit_channel('h', cudaq.BitFlipChannel(0.001))

     # Specify two qubit errors
     noise.add_all_qubit_channel("x", cudaq.Depolarization2(p), 1)

.. tab:: C++

    .. code-block:: cpp

      cudaq::noise_model noise;

      // Add multiple error channels
      noise.add_all_qubit_channel(
          "x", cudaq::bit_flip_channel(/*probability*/ 0.01));

      // Specify two qubit errors
      noise.add_all_qubit_channel(
          "x", cudaq::depolarization2(/*probability*/ 0.01),
          /*numControls*/ 1);

