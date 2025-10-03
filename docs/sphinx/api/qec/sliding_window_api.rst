.. class:: sliding_window

    The Sliding Window Decoder is a wrapper around a standard decoder that
    introduces two key differences:
    
    1. **Sliding Window Decoding**: The decoding process is performed
    incrementally, one window at a time.  The window size is specified by the
    user. This allows decoding to begin before all syndromes have been received,
    potentially reducing overall latency in multi-round QEC codes.
    
    2. **Partial Syndrome Support**: Unlike standard decoders, the
    :code:`decode` function (and its variants like :code:`decode_batch`) can
    accept partial syndromes. If partial syndromes are provided, the return
    vector will be empty, the decoder will not complete the processing and
    remain in an intermediate state, awaiting future syndromes. The return
    vector is only non-empty once enough data has been provided to match the
    original syndrome size (calculated from the Parity Check Matrix).
    
    Sliding window decoders are advantageous in QEC codes subject to
    circuit-level noise across multiple syndrome extraction rounds. These
    decoders permit syndrome processing to begin before the complete syndrome
    measurement sequence is obtained, potentially reducing the overall decoding
    latency. However, this approach introduces a trade-off: the reduction in
    latency typically comes at the cost of increased logical error rates.
    Therefore, the viability of sliding window decoding depends critically on
    the specific code parameters, noise model, and latency requirements of the
    system under consideration.
    
    Sliding window decoding imposes only a single structural constraint on the
    parity check matrices: each syndrome extraction round must produce a
    constant number of syndrome measurements. Notably, the decoder makes no
    assumptions about temporal correlations or periodicity in the underlying
    noise process.

    References:
    `Toward Low-latency Iterative Decoding of QLDPC Codes Under Circuit-Level Noise <https://arxiv.org/abs/2403.18901>`_

    .. note::
      It is required to create decoders with the `get_decoder` API from the CUDA-QX
      extension points API, such as

      .. tab:: Python

        .. code-block:: python

            import cudaq
            import cudaq_qec as qec
            import numpy as np

            cudaq.set_target('stim')
            num_rounds = 5
            code = qec.get_code('surface_code', distance=num_rounds)
            noise = cudaq.NoiseModel()
            noise.add_all_qubit_channel("x", cudaq.Depolarization2(0.001), 1)
            statePrep = qec.operation.prep0
            dem = qec.z_dem_from_memory_circuit(code, statePrep, num_rounds, noise)
            inner_decoder_params = {'use_osd': True, 'max_iterations': 50}
            opts = {
                'error_rate_vec': np.array(dem.error_rates),
                'window_size': 1,
                'num_syndromes_per_round': dem.detector_error_matrix.shape[0] // num_rounds,
                'inner_decoder_name': 'single_error_lut',
                'inner_decoder_params': inner_decoder_params,
            }
            swdec = qec.get_decoder('sliding_window', dem.detector_error_matrix, **opts)

      .. tab:: C++

        .. code-block:: cpp

            #include "cudaq/qec/code.h"
            #include "cudaq/qec/decoder.h"
            #include "cudaq/qec/experiments.h"
            #include "common/NoiseModel.h"

            int main() {
                // Generate a Detector Error Model.
                int num_rounds = 5;
                auto code = cudaq::qec::get_code(
                    "surface_code", cudaqx::heterogeneous_map{{"distance", num_rounds}});
                cudaq::noise_model noise;
                noise.add_all_qubit_channel("x", cudaq::depolarization2(0.001), 1);
                auto statePrep = cudaq::qec::operation::prep0;
                auto dem = cudaq::qec::z_dem_from_memory_circuit(*code, statePrep, num_rounds,
                                                                noise);
                // Use the DEM to create a sliding window decoder.
                auto inner_decoder_params =
                    cudaqx::heterogeneous_map{{"use_osd", true}, {"max_iterations", 50}};
                auto opts = cudaqx::heterogeneous_map{
                    {"error_rate_vec", dem.error_rates},
                    {"window_size", 1},
                    {"num_syndromes_per_round",
                    dem.detector_error_matrix.shape()[0] / num_rounds},
                    {"inner_decoder_name", "single_error_lut"},
                    {"inner_decoder_params", inner_decoder_params}};
                auto swdec = cudaq::qec::get_decoder("sliding_window",
                                                    dem.detector_error_matrix, opts);

                return 0;
            }

    .. note::
      The `"sliding_window"` decoder implements the :class:`cudaq_qec.Decoder`
      interface for Python and the :cpp:class:`cudaq::qec::decoder` interface
      for C++, so it supports all the methods in those respective classes.

    :param H: Parity check matrix (tensor format)
    :param params: Heterogeneous map of parameters:

        - `error_rate_vec` (double): Vector of length "block size" containing
          the probability of an error (in 0-1 range). This vector is used to
          populate the `error_rate_vec` parameter for the inner decoder
          (automatically sliced correctly according to each window).
        - `window_size` (int): The number of rounds of syndrome data in each window. (Defaults to 1.)
        - `step_size` (int): The number of rounds to advance the window by each time. (Defaults to 1.)
        - `num_syndromes_per_round` (int): The number of syndromes per round. (Must be provided.)
        - `straddle_start_round` (bool): When forming a window, should error
          mechanisms that span the start round and any preceding rounds be included? (Defaults to False.)
        - `straddle_end_round` (bool): When forming a window, should error
          mechanisms that span the end round and any subsequent rounds be included? (Defaults to True.)
        - `inner_decoder_name` (string): The name of the inner decoder to use.
        - `inner_decoder_params` (Python dict or C++ `heterogeneous_map`): A
          dictionary of parameters to pass to the inner decoder.
