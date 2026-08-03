.. class:: trt_decoder

    A GPU-accelerated quantum error correction decoder based on NVIDIA TensorRT.
    This decoder leverages TensorRT's optimized inference engine to perform fast
    neural network-based decoding of quantum error correction syndromes.

    The TRT decoder supports loading pre-trained neural network models in ONNX
    format or directly loading pre-built TensorRT engine files for maximum
    performance. It automatically optimizes the model for the target GPU
    architecture and supports various precision modes (FP16, BF16, INT8, FP8)
    to balance accuracy and speed.

    Requires a CUDA-capable GPU and TensorRT installation. See the `CUDA-Q GPU
    Compatibility List
    <https://nvidia.github.io/cuda-quantum/latest/using/install/local_installation.html#dependencies-and-compatibility>`_
    for a list of valid GPU configurations.

    .. note::
      It is required to create decoders with the `get_decoder` API from the CUDA-QX
      extension points API, such as

      .. tab:: Python

        .. code-block:: python

            import cudaq_qec as qec
            import numpy as np
            
            # Create a simple parity check matrix (not used by the TRT decoder)
            H = np.array([[1, 0, 0, 1, 0, 1, 1],
                          [0, 1, 0, 1, 1, 0, 1],
                          [0, 0, 1, 0, 1, 1, 1]], dtype=np.uint8)
            
            # Option 1: Load from ONNX model (builds TRT engine)
            trt_dec = qec.get_decoder('trt_decoder', H,
                                      onnx_load_path='model.onnx',
                                      precision='fp16',
                                      engine_save_path='model.engine')
            
            # Option 2: Load pre-built TRT engine (faster startup)
            trt_dec = qec.get_decoder('trt_decoder', H,
                                      engine_load_path='model.engine')

      .. tab:: C++

        .. code-block:: cpp

            #include "cudaq/qec/decoder.h"
            
            std::size_t block_size = 7;
            std::size_t syndrome_size = 3;
            cudaqx::tensor<uint8_t> H;

            // Create a simple parity check matrix (not used by the TRT decoder)
            std::vector<uint8_t> H_vec = {1, 0, 0, 1, 0, 1, 1, 
                                          0, 1, 0, 1, 1, 0, 1,
                                          0, 0, 1, 0, 1, 1, 1};
            H.copy(H_vec.data(), {syndrome_size, block_size});
            
            // Option 1: Load from ONNX model (builds TRT engine)
            cudaqx::heterogeneous_map params1;
            params1.insert("onnx_load_path", "model.onnx");
            params1.insert("precision", "fp16");
            params1.insert("engine_save_path", "model.engine");
            auto trt_dec1 = cudaq::qec::get_decoder("trt_decoder", H, params1);
            
            // Option 2: Load pre-built TRT engine (faster startup)
            cudaqx::heterogeneous_map params2;
            params2.insert("engine_load_path", "model.engine");
            auto trt_dec2 = cudaq::qec::get_decoder("trt_decoder", H, params2);
      
    .. note::
      The `"trt_decoder"` implements the :class:`cudaq_qec.Decoder`
      interface for Python and the :cpp:class:`cudaq::qec::decoder` interface
      for C++, so it supports all the methods in those respective classes.

    .. note::
      The parity check matrix `H` is not used by the TRT decoder. The neural
      network model encodes the decoding logic, so the parity check matrix is
      only required to satisfy the decoder interface. You can pass any valid
      parity check matrix of appropriate dimensions.

    :param H: Parity check matrix (tensor format). Note: This parameter is not
              used by the TRT decoder but is required by the decoder interface.
    :param params: Heterogeneous map of parameters:

        **Required (choose one):**

        - `onnx_load_path` (string): Path to ONNX model file. The decoder will
          build a TensorRT engine from this model. Cannot be used together with
          `engine_load_path`.
        - `engine_load_path` (string): Path to pre-built TensorRT engine file.
          Provides faster initialization since the engine is already optimized.
          Cannot be used together with `onnx_load_path`.

        **Optional:**

        - `engine_save_path` (string): Path to save the built TensorRT engine.
          Only applicable when using `onnx_load_path`. Saving the engine allows
          for faster initialization in subsequent runs by using `engine_load_path`.
        - `precision` (string): Precision mode for inference (defaults to "best").
          Valid options:

          - "fp16": Use FP16 (half precision) - good balance of speed and accuracy
          - "bf16": Use BF16 (bfloat16) - available on newer GPUs (Ampere+)
          - "int8": Use INT8 quantization - fastest but requires calibration
          - "fp8": Use FP8 precision - available on Hopper GPUs
          - "tf32": Use TensorFloat-32 - available on Ampere+ GPUs
          - "noTF32": Disable TF32 and use standard FP32
          - "best": Let TensorRT automatically choose the best precision (default)

          Note: If the requested precision is not supported by the hardware, the
          decoder will fall back to FP32 with a warning.

        - `memory_workspace` (size_t): Memory workspace size in bytes for TensorRT
          engine building (defaults to 1GB = 1073741824 bytes). Larger workspaces
          may allow TensorRT to explore more optimization strategies.

