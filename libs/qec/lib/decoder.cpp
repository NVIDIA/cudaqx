/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/qec/decoder.h"
#include "common/FmtCore.h"
#include "common/Logger.h"
#include "cuda-qx/core/library_utils.h"
#include "cudaq/qec/plugin_loader.h"
#include "cudaq/qec/version.h"
#include <cassert>
#include <dlfcn.h>
#include <filesystem>
#include <vector>

INSTANTIATE_REGISTRY(cudaq::qec::decoder, const cudaqx::tensor<uint8_t> &)
INSTANTIATE_REGISTRY(cudaq::qec::decoder, const cudaqx::tensor<uint8_t> &,
                     const cudaqx::heterogeneous_map &)

// Include decoder implementations AFTER registry instantiation
#include "decoders/sliding_window.h"

namespace cudaq::qec {

struct decoder::rt_impl {
  /// The number of measurement syndromes to be decoded per decode call (i.e.
  /// the number of columns in the D_sparse matrix)
  uint32_t num_msyn_per_decode = 0;

  /// The index of the next syndrome to be written in the msyn_buffer
  uint32_t msyn_buffer_index = 0;

  /// The buffer of measurement syndromes received from the client. Length is
  /// num_msyn_per_decode.
  std::vector<uint8_t> msyn_buffer;

  /// The current observable corrections. The length of this vector is the
  /// number of rows in the O_sparse matrix.
  std::vector<uint8_t> corrections;

  /// Persistent buffers to avoid dynamic memory allocation.
  std::vector<uint8_t> persistent_detector_buffer;
  std::vector<float_t> persistent_soft_detector_buffer;

  /// Whether to log decoder stats.
  bool should_log = false;

  /// A simple counter to distinguish log messages.
  uint32_t log_counter = 0;

  /// The id of the decoder (for instrumentation)
  uint32_t decoder_id = 0;

  bool is_sliding_window = false;

  /// The number of syndromes per round.  Only used for sliding window decoder.
  size_t num_syndromes_per_round = 0;

  /// Whether the first round detectors are included.  Only used for sliding
  /// window decoder.
  bool has_first_round_detectors = false;

  /// The current round.  Only used for sliding window decoder.
  uint32_t current_round = 0;
};

void decoder::rt_impl_deleter::operator()(rt_impl *p) const { delete p; }

decoder::decoder(const cudaqx::tensor<uint8_t> &H)
    : H(H), pimpl(std::unique_ptr<rt_impl, rt_impl_deleter>(new rt_impl())) {
  const auto H_shape = H.shape();
  assert(H_shape.size() == 2 && "H tensor must be of rank 2");
  syndrome_size = H_shape[0];
  block_size = H_shape[1];
  reset_decoder();
  pimpl->persistent_detector_buffer.resize(this->syndrome_size);
  pimpl->persistent_soft_detector_buffer.resize(this->syndrome_size);

  // We allow detailed logging of decoder stats via the CUDAQ_QEC_DEBUG_DECODER
  // environment variable or the CUDAQ_LOG_LEVEL=info environment variable. If
  // it is set with CUDAQ_LOG_LEVEL, it will be instrumented at the info level
  // just like any other message, but if it is set with CUDAQ_QEC_DEBUG_DECODER,
  // it will be instrumented as a simple printf.
  if (auto *ch = std::getenv("CUDAQ_QEC_DEBUG_DECODER"))
    pimpl->should_log = ch[0] == '1' || ch[0] == 'y' || ch[0] == 'Y';
}

// Provide a trivial implementation of for tensor<uint8_t> decode call. Child
// classes should override this if they never want to pass through floats.
decoder_result decoder::decode(const cudaqx::tensor<uint8_t> &syndrome) {
  // Check tensor is of order-1
  // If order >1, we could check that other modes are of dim = 1 such that
  // n x 1, or 1 x n tensors are still valid.
  if (syndrome.rank() != 1) {
    throw std::runtime_error("Decode requires rank-1 tensors");
  }
  std::vector<float_t> soft_syndrome(syndrome.shape()[0]);
  std::vector<uint8_t> vec_cast(syndrome.data(),
                                syndrome.data() + syndrome.shape()[0]);
  convert_vec_hard_to_soft(vec_cast, soft_syndrome);
  return decode(soft_syndrome);
}

// Provide a trivial implementation of the multi-syndrome decoder. Child classes
// should override this if they can do it more efficiently than this.
std::vector<decoder_result>
decoder::decode_batch(const std::vector<std::vector<float_t>> &syndrome) {
  std::vector<decoder_result> result;
  result.reserve(syndrome.size());
  for (auto &s : syndrome)
    result.push_back(decode(s));
  return result;
}

std::string decoder::get_version() const {
  std::stringstream ss;
  ss << "CUDA-Q QEC Base Decoder Interface " << cudaq::qec::getVersion() << " ("
     << cudaq::qec::getFullRepositoryVersion() << ")";
  return ss.str();
}

std::future<decoder_result>
decoder::decode_async(const std::vector<float_t> &syndrome) {
  return std::async(std::launch::async,
                    [this, syndrome] { return this->decode(syndrome); });
}

std::unique_ptr<decoder>
decoder::get(const std::string &name, const cudaqx::tensor<uint8_t> &H,
             const cudaqx::heterogeneous_map &param_map) {
  auto [mutex, registry] = get_registry();
  std::lock_guard<std::recursive_mutex> lock(mutex);
  auto iter = registry.find(name);
  if (iter == registry.end())
    throw std::runtime_error(
        "invalid decoder requested: " + name +
        ". Run with CUDAQ_LOG_LEVEL=info (environment variable) to see "
        "additional plugin diagnostics at startup.");
  return iter->second(H, param_map);
}

static uint32_t calculate_num_msyn_per_decode(
    const std::vector<std::vector<uint32_t>> &D_sparse) {
  uint32_t max_col = 0;
  for (const auto &row : D_sparse)
    for (const auto col : row)
      max_col = std::max(max_col, col);
  return max_col + 1;
}

static void
set_sparse_from_vec(const std::vector<int64_t> &vec_in,
                    std::vector<std::vector<uint32_t>> &sparse_out) {
  sparse_out.clear();
  bool first_of_row = true;
  for (auto elem : vec_in) {
    if (elem < 0) {
      first_of_row = true;
    } else {
      if (first_of_row) {
        sparse_out.emplace_back();
        first_of_row = false;
      }
      sparse_out.back().push_back(static_cast<uint32_t>(elem));
    }
  }
}

void decoder::set_O_sparse(const std::vector<std::vector<uint32_t>> &O_sparse) {
  this->O_sparse = O_sparse;
  this->pimpl->corrections.clear();
  this->pimpl->corrections.resize(O_sparse.size());
}

void decoder::set_O_sparse(const std::vector<int64_t> &O_sparse_vec_in) {
  set_sparse_from_vec(O_sparse_vec_in, this->O_sparse);
  this->pimpl->corrections.clear();
  this->pimpl->corrections.resize(O_sparse.size());
}

uint32_t decoder::get_num_msyn_per_decode() const {
  return pimpl->num_msyn_per_decode;
}

void decoder::set_decoder_id(uint32_t decoder_id) {
  pimpl->decoder_id = decoder_id;
}

uint32_t decoder::get_decoder_id() const { return pimpl->decoder_id; }

template <typename PimplType>
void set_D_sparse_common(decoder *decoder,
                         const std::vector<std::vector<uint32_t>> &D_sparse,
                         PimplType *pimpl) {
  auto *sw_decoder = dynamic_cast<sliding_window *>(decoder);

  if (sw_decoder != nullptr) {
    pimpl->is_sliding_window = true;
    pimpl->num_syndromes_per_round = sw_decoder->get_num_syndromes_per_round();
    // Check if first row is a first-round detector (single syndrome index)
    pimpl->has_first_round_detectors =
        (D_sparse.size() > 0 && D_sparse[0].size() == 1);
    pimpl->current_round = 0;
    pimpl->persistent_detector_buffer.resize(pimpl->num_syndromes_per_round);
    pimpl->persistent_soft_detector_buffer.resize(
        pimpl->num_syndromes_per_round);

  } else {
    pimpl->is_sliding_window = false;
  }

  pimpl->num_msyn_per_decode = calculate_num_msyn_per_decode(D_sparse);
  pimpl->msyn_buffer.clear();
  pimpl->msyn_buffer.resize(pimpl->num_msyn_per_decode);
  pimpl->msyn_buffer_index = 0;
}

void decoder::set_D_sparse(const std::vector<std::vector<uint32_t>> &D_sparse) {
  this->D_sparse = D_sparse;
  set_D_sparse_common(this, D_sparse, pimpl.get());
}

void decoder::set_D_sparse(const std::vector<int64_t> &D_sparse_vec_in) {
  set_sparse_from_vec(D_sparse_vec_in, this->D_sparse);
  set_D_sparse_common(this, this->D_sparse, pimpl.get());
}

bool decoder::enqueue_syndrome(const uint8_t *syndrome,
                               std::size_t syndrome_length) {
  if (pimpl->msyn_buffer_index + syndrome_length > pimpl->msyn_buffer.size()) {
    // CUDAQ_WARN("Syndrome buffer overflow. Syndrome will be ignored.");
    printf("Syndrome buffer overflow. Syndrome will be ignored.\n");
    return false;
  }

  pimpl->current_round++;
  bool did_decode = false;
  for (std::size_t i = 0; i < syndrome_length; i++) {
    pimpl->msyn_buffer[pimpl->msyn_buffer_index] = syndrome[i];
    pimpl->msyn_buffer_index++;
  }

  bool should_decode = false;
  if (!pimpl->is_sliding_window) {
    should_decode = (pimpl->msyn_buffer_index == pimpl->msyn_buffer.size());
  } else {
    should_decode =
        (pimpl->current_round >= 2) ||
        (pimpl->current_round == 1 && pimpl->has_first_round_detectors);
  }
  if (should_decode) {
    // These are just for logging. They are initialized in such a way to avoid
    // dynamic memory allocation if logging is disabled.
    std::vector<uint32_t> log_msyn;
    std::vector<uint32_t> log_detectors;
    std::vector<uint32_t> log_errors;
    std::vector<uint8_t> log_observable_corrections;
    // The four time points are used to measure the duration of each of 3 steps.
    std::chrono::time_point<std::chrono::high_resolution_clock> log_t0, log_t1,
        log_t2, log_t3;
    std::chrono::duration<double> log_dur1, log_dur2, log_dur3;

    const bool log_due_to_log_level =
        cudaq::details::should_log(cudaq::details::LogLevel::info);
    const bool should_log = pimpl->should_log || log_due_to_log_level;

    if (should_log) {
      log_t0 = std::chrono::high_resolution_clock::now();
      log_errors.reserve(syndrome_length);
      log_observable_corrections.resize(O_sparse.size());
    }

    // Decode now.
    if (!pimpl->is_sliding_window) {
      for (std::size_t i = 0; i < this->D_sparse.size(); i++) {
        pimpl->persistent_detector_buffer[i] = 0;
        for (auto col : this->D_sparse[i])
          pimpl->persistent_detector_buffer[i] ^= pimpl->msyn_buffer[col];
      }
    } else {
      // For sliding window decoder, syndrome_length must equal
      // num_syndromes_per_round
      assert(syndrome_length == pimpl->num_syndromes_per_round);
      if (pimpl->current_round == 1 && pimpl->has_first_round_detectors) {
        // First round: only compute first-round detectors (direct copy)
        for (std::size_t i = 0; i < pimpl->num_syndromes_per_round; i++) {
          pimpl->persistent_detector_buffer[i] = pimpl->msyn_buffer[i];
        }
      } else {
        // Buffer is full with 2 rounds: compute timelike detectors (XOR of two
        // rounds)
        std::size_t index =
            (pimpl->current_round - 2) * pimpl->num_syndromes_per_round;
        for (std::size_t i = 0; i < pimpl->num_syndromes_per_round; i++) {
          pimpl->persistent_detector_buffer[i] =
              pimpl->msyn_buffer[index + i] ^
              pimpl->msyn_buffer[index + i + pimpl->num_syndromes_per_round];
        }
      }
    }

    if (should_log) {
      log_msyn.reserve(pimpl->msyn_buffer.size());
      for (std::size_t d = 0, D = pimpl->msyn_buffer.size(); d < D; d++) {
        if (pimpl->msyn_buffer[d])
          log_msyn.push_back(d);
      }
      log_detectors.reserve(pimpl->persistent_detector_buffer.size());
      for (std::size_t d = 0, D = pimpl->persistent_detector_buffer.size();
           d < D; d++) {
        if (pimpl->persistent_detector_buffer[d])
          log_detectors.push_back(d);
      }
      log_t1 = std::chrono::high_resolution_clock::now();
    }
    // Send the data to the decoder.
    convert_vec_hard_to_soft(pimpl->persistent_detector_buffer,
                             pimpl->persistent_soft_detector_buffer);
    auto decoded_result = decode(pimpl->persistent_soft_detector_buffer);

    // If we didn't get a decoded result, just return
    if (pimpl->is_sliding_window) {
      if (decoded_result.result.size() == 0) {
        return false;
      }
    }

    if (should_log) {
      log_t2 = std::chrono::high_resolution_clock::now();
      for (std::size_t e = 0, E = decoded_result.result.size(); e < E; e++)
        if (decoded_result.result[e])
          log_errors.push_back(e);
    }
    // Process the results.
    // TODO - should this interrogate the decoded_result.converged flag?
    auto num_observables = O_sparse.size();
    // For each observable
    for (std::size_t i = 0; i < num_observables; i++) {
      // For each error that flips this observable
      for (auto col : O_sparse[i]) {
        // If the decoder predicted that this error occurred
        if (decoded_result.result[col]) {
          // Flip the correction for this observable
          pimpl->corrections[i] ^= 1;
          if (should_log)
            log_observable_corrections[i] ^= 1;
        }
      }
    }
    if (should_log) {
      log_t3 = std::chrono::high_resolution_clock::now();
      log_dur1 = log_t1 - log_t0;
      log_dur2 = log_t2 - log_t1;
      log_dur3 = log_t3 - log_t2;
      pimpl->log_counter++;
      auto s = fmt::format(
          "[DecoderStats][{}] Counter:{} DecoderId:{} InputMsyn:{} "
          "InputDetectors:{} Converged:{} Errors:{} "
          "ObservableCorrectionsThisCall:{} ObservableCorrectionsTotal:{} "
          "Dur1:{:.1f}us Dur2:{:.1f}us Dur3:{:.1f}us",
          static_cast<const void *>(this), pimpl->log_counter,
          pimpl->decoder_id, fmt::join(log_msyn, ","),
          fmt::join(log_detectors, ","), decoded_result.converged ? 1 : 0,
          fmt::join(log_errors, ","),
          fmt::join(log_observable_corrections, ","),
          fmt::join(std::vector<uint8_t>(pimpl->corrections.begin(),
                                         pimpl->corrections.end()),
                    ","),
          log_dur1.count() * 1e6, log_dur2.count() * 1e6,
          log_dur3.count() * 1e6);
      if (log_due_to_log_level)
        cudaq::info("{}", s);
      else
        printf("%s\n", s.c_str());
    }
    did_decode = true;
    // Prepare for more data.
    pimpl->msyn_buffer_index = 0;
    pimpl->current_round = 0;
  }
  return did_decode;
}

bool decoder::enqueue_syndrome(const std::vector<uint8_t> &syndrome) {
  return enqueue_syndrome(syndrome.data(), syndrome.size());
}

void decoder::clear_corrections() {
  pimpl->corrections.clear();
  pimpl->corrections.resize(O_sparse.size());
  const bool log_due_to_log_level =
      cudaq::details::should_log(cudaq::details::LogLevel::info);
  const bool should_log = pimpl->should_log || log_due_to_log_level;
  if (should_log) {
    pimpl->log_counter++;
    std::string s =
        fmt::format("[DecoderStats][{}] Counter:{} clear_corrections called",
                    static_cast<const void *>(this), pimpl->log_counter);
    if (log_due_to_log_level)
      cudaq::info("{}", s);
    else
      printf("%s\n", s.c_str());
  }
}

const uint8_t *decoder::get_obs_corrections() const {
  const bool log_due_to_log_level =
      cudaq::details::should_log(cudaq::details::LogLevel::info);
  const bool should_log = pimpl->should_log || log_due_to_log_level;
  if (should_log) {
    pimpl->log_counter++;
    std::string s =
        fmt::format("[DecoderStats][{}] Counter:{} get_obs_corrections called",
                    static_cast<const void *>(this), pimpl->log_counter);
    if (log_due_to_log_level)
      cudaq::info("{}", s);
    else
      printf("%s\n", s.c_str());
  }
  return pimpl->corrections.data();
}

std::size_t decoder::get_num_observables() const { return O_sparse.size(); }

void decoder::reset_decoder() {
  // Zero out all data that is considered "per-shot" memory.
  pimpl->msyn_buffer_index = 0;
  pimpl->current_round = 0;
  pimpl->msyn_buffer.clear();
  pimpl->msyn_buffer.resize(pimpl->num_msyn_per_decode);
  pimpl->corrections.clear();
  pimpl->corrections.resize(O_sparse.size());
  const bool log_due_to_log_level =
      cudaq::details::should_log(cudaq::details::LogLevel::info);
  const bool should_log = pimpl->should_log || log_due_to_log_level;
  if (should_log) {
    pimpl->log_counter++;
    std::string s =
        fmt::format("[DecoderStats][{}] Counter:{} reset_decoder called",
                    static_cast<const void *>(this), pimpl->log_counter);
    if (log_due_to_log_level)
      cudaq::info("{}", s);
    else
      printf("%s\n", s.c_str());
  }
}

std::unique_ptr<decoder> get_decoder(const std::string &name,
                                     const cudaqx::tensor<uint8_t> &H,
                                     const cudaqx::heterogeneous_map options) {
  return decoder::get(name, H, options);
}

// Constructor function for auto-loading plugins
__attribute__((constructor)) void load_decoder_plugins() {
  // Load plugins from the decoder-specific plugin directory
  std::filesystem::path libPath{cudaqx::__internal__::getCUDAQXLibraryPath(
      cudaqx::__internal__::CUDAQXLibraryType::QEC)};
  auto pluginPath = libPath.parent_path() / "decoder-plugins";
  load_plugins(pluginPath.string(), PluginType::DECODER);
}

// Destructor function to clean up only decoder plugins
__attribute__((destructor)) void cleanup_decoder_plugins() {
  // Clean up decoder-specific plugins
  cleanup_plugins(PluginType::DECODER);
}
} // namespace cudaq::qec
