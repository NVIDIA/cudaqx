.. class:: sliding_window

    The sliding window decoder is a wrapper around a standard decoder that
    differs from standard decoders in two important ways.

    1. The actual decoding step is performed in a sliding window fashion, one
       window at a time. The window size is specified by the user. This allows
       the decoding process to begin before all of the syndromes have been
       received.

    2. Unlike the standard decoder interface, the :code:`decode` function (and
       its variants like :code:`decode_batch`) can accept partial syndromes. If
       a partial syndrome is provided, then the return vector will be empty, and
       the decoder object will be in an intermediate state (awaiting future
       syndromes to complete its processing). The only time the return vector
       will be non-empty is when the user has called :code:`decode` with enough
       data to fill the original syndrome size (as calculated from the original
       Parity Check Matrix).

    Sliding window decoders are useful for QEC codes with circuit-level noise
    that have multiple rounds. They allow decoding processing to begin prior to
    the full syndrome being available, which can - under the right conditions -
    reduce the latency for the final decoding call. However, note that this
    reduced latency can increase the logical error rate, so one needs to study
    their particular system carefully to ensure this trade-off is acceptable.

    The only structural requirement for Parity Check Matrices used by the
    Sliding Window decoder is that each round has a constant number of syndromes;
    no other assumptions about repeatable noise structure are used.

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
