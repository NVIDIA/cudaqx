/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/qec/decoder.h"
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

namespace cudaq::qec {

struct decoder::rt_impl {
  /// The number of syndromes per round (enables incremental detector
  /// computation)
  uint32_t num_syndromes_per_round = 0;

  /// The number of measurement syndromes to be decoded per decode call
  /// (for incremental mode: one round; for batch mode: full D_sparse columns)
  uint32_t num_msyn_per_decode = 0;

  /// Counter of total syndromes buffered but not yet processed.
  /// Used to detect complete rounds (when this is a multiple of
  /// num_msyn_per_decode). Gets decremented after each round is decoded. Not a
  /// direct buffer index.
  uint32_t num_syndromes_buffered_but_not_decoded = 0;

  /// The buffer of measurement syndromes received from the client.
  /// For incremental mode: size is calculated from max D_sparse column + 1
  /// This allows buffering multiple rounds while still decoding incrementally
  std::vector<uint8_t> msyn_buffer;

  /// Total buffer capacity (max column index in D_sparse + 1)
  uint32_t buffer_capacity = 0;

  /// Track which round we're on (0 = reference round)
  uint32_t current_round = 0;

  /// Circular buffer write position for the current round
  // Values are 0, num_msyn_per_decode * 2, num_msyn_per_decode * 3, etc. then
  // wrap around to 0.
  uint32_t current_round_buffer_offset = 0;

  /// Circular buffer position of the previous round (for incremental XOR)
  uint32_t prev_round_buffer_offset = 0;

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

  /// Whether D_sparse has first-round detectors (determined in set_D_sparse)
  bool has_first_round_detectors = false;
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
  auto &registry = get_registry();
  auto iter = registry.find(name);
  if (iter == registry.end())
    throw std::runtime_error("invalid decoder requested: " + name);
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

void decoder::set_D_sparse(const std::vector<std::vector<uint32_t>> &D_sparse) {
  this->D_sparse = D_sparse;

  // Analyze D_sparse structure (assumes well-formed D_sparse from generator):
  // 1. First-round detectors (if any) are always at the beginning
  // 2. All timelike detectors have the same stride (num_syndromes_per_round)
  
  // Check if first row is a first-round detector (single syndrome index)
  pimpl->has_first_round_detectors = (D_sparse.size() > 0 && D_sparse[0].size() == 1);
  
  // Find num_syndromes_per_round from first timelike detector
  // (skip first-round detectors if present, they're all at the beginning)
  pimpl->num_syndromes_per_round = 1; // Default fallback
  for (const auto& detector_syndrome_indices : D_sparse) {
    if (detector_syndrome_indices.size() >= 2) {
      // First timelike detector found: XORs syndromes from consecutive rounds
      // e.g., [0, 8] means XOR syndrome 0 (round 1) with syndrome 8 (round 2)
      // so num_syndromes_per_round = 8
      pimpl->num_syndromes_per_round = detector_syndrome_indices[1] - detector_syndrome_indices[0];
      break; // Found it, no need to continue
    }
  }

  // Calculate minimum buffer capacity from max column in D_sparse
  uint32_t min_capacity = calculate_num_msyn_per_decode(D_sparse);

  // Enable incremental mode: process one round at a time
  pimpl->num_msyn_per_decode = pimpl->num_syndromes_per_round;

  // Add one extra round to buffer capacity to guarantee no wraparound within
  // operations This eliminates all wraparound checks in hot loops (write and
  // detector computation)
  pimpl->buffer_capacity = min_capacity + pimpl->num_syndromes_per_round;

  // Allocate buffer to hold all syndromes plus extra round
  pimpl->msyn_buffer.clear();
  pimpl->msyn_buffer.resize(pimpl->buffer_capacity);

  pimpl->num_syndromes_buffered_but_not_decoded = 0;
  pimpl->current_round = 0;
  pimpl->current_round_buffer_offset = 0;
  pimpl->prev_round_buffer_offset = 0;
}

void decoder::set_D_sparse(const std::vector<int64_t> &D_sparse_vec_in) {
  set_sparse_from_vec(D_sparse_vec_in, this->D_sparse);

  // Analyze D_sparse structure (assumes well-formed D_sparse from generator):
  // 1. First-round detectors (if any) are always at the beginning
  // 2. All timelike detectors have the same stride (num_syndromes_per_round)
  
  // Check if first row is a first-round detector (single syndrome index)
  pimpl->has_first_round_detectors = (D_sparse.size() > 0 && D_sparse[0].size() == 1);
  
  // Find num_syndromes_per_round from first timelike detector
  // (skip first-round detectors if present, they're all at the beginning)
  pimpl->num_syndromes_per_round = 1; // Default fallback
  for (const auto& detector_syndrome_indices : D_sparse) {
    if (detector_syndrome_indices.size() >= 2) {
      // First timelike detector found: XORs syndromes from consecutive rounds
      // e.g., [0, 8] means XOR syndrome 0 (round 1) with syndrome 8 (round 2)
      // so num_syndromes_per_round = 8
      pimpl->num_syndromes_per_round = detector_syndrome_indices[1] - detector_syndrome_indices[0];
      break; // Found it, no need to continue
    }
  }

  // Calculate minimum buffer capacity from max column in D_sparse
  uint32_t min_capacity = calculate_num_msyn_per_decode(D_sparse);

  // Enable incremental mode: process one round at a time
  pimpl->num_msyn_per_decode = pimpl->num_syndromes_per_round;

  // Add one extra round to buffer capacity to guarantee no wraparound within
  // operations This eliminates all wraparound checks in hot loops (write and
  // detector computation)
  pimpl->buffer_capacity = min_capacity + pimpl->num_syndromes_per_round;

  // Allocate buffer to hold all syndromes plus extra round
  pimpl->msyn_buffer.clear();
  pimpl->msyn_buffer.resize(pimpl->buffer_capacity);

  pimpl->num_syndromes_buffered_but_not_decoded = 0;
  pimpl->current_round = 0;
  pimpl->current_round_buffer_offset = 0;
  pimpl->prev_round_buffer_offset = 0;
}

bool decoder::enqueue_syndrome(const uint8_t *syndrome,
                               std::size_t syndrome_length) {
  // position_in_round represents how many syndromes of the current round have
  // already been buffered but not yet decoded Values range from 0 to
  // num_msyn_per_decode - 1.
  uint32_t position_in_round = pimpl->num_syndromes_buffered_but_not_decoded %
                               pimpl->num_msyn_per_decode;

  // Check if this write would overwrite the previous round
  // We need to preserve prev_round_buffer for XOR computation, so the maximum
  // safe write from the start of the current round is buffer_capacity minus one
  // round
  uint32_t max_safe_from_round_start =
      pimpl->buffer_capacity - pimpl->num_syndromes_per_round;
  if (position_in_round + syndrome_length > max_safe_from_round_start) {
    // CUDAQ_WARN("Syndrome data too large - would overwrite previous round.
    // Data will be ignored.");
    printf("Syndrome data too large - would overwrite previous round. Data "
           "will be ignored.\n");
    return false;
  }
  bool did_decode = false;
  // Buffer the incoming syndromes
  // No wraparound check needed: buffer is sized to guarantee operations never
  // wrap mid-execution
  uint32_t write_start = pimpl->current_round_buffer_offset + position_in_round;
  for (std::size_t i = 0; i < syndrome_length; i++) {
    pimpl->msyn_buffer[write_start + i] = syndrome[i];
  }
  pimpl->num_syndromes_buffered_but_not_decoded += syndrome_length;

  // Process all complete rounds that are now available
  while ((pimpl->num_syndromes_buffered_but_not_decoded %
          pimpl->num_msyn_per_decode) == 0 &&
         pimpl->num_syndromes_buffered_but_not_decoded > 0) {
    pimpl->current_round++;

    // First round (round 1): skip decoding (store as reference)
    // UNLESS there are first-round detectors that need immediate decoding
    // (first-round detector check is done once in set_D_sparse)
    if (pimpl->current_round == 1 && !pimpl->has_first_round_detectors) {
      // Previous round stays at current position for next round's XOR
      pimpl->prev_round_buffer_offset = pimpl->current_round_buffer_offset;
      // Advance to next round position in circular buffer
      pimpl->current_round_buffer_offset += pimpl->num_msyn_per_decode;
      if (pimpl->current_round_buffer_offset >= pimpl->buffer_capacity)
        pimpl->current_round_buffer_offset -= pimpl->buffer_capacity;
      pimpl->num_syndromes_buffered_but_not_decoded -=
          pimpl->num_msyn_per_decode; // Decrement for next iteration
      continue;                       // Skip to next round
    }

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

    // Compute detectors based on whether first-round detectors exist
    if (pimpl->has_first_round_detectors) {
      // When first-round detectors exist, must use D_sparse for all detectors
      // because first-round detectors reference only one syndrome (not two)
      for (std::size_t i = 0; i < this->D_sparse.size(); i++) {
        pimpl->persistent_detector_buffer[i] = 0;
        for (auto col : this->D_sparse[i])
          pimpl->persistent_detector_buffer[i] ^= pimpl->msyn_buffer[col];
      }
    } else {
      // Pure timelike detectors: use incremental XOR (current ⊕ previous round)
      // Using circular buffer offsets - no D_sparse access needed
      // No wraparound checks needed: buffer is sized to guarantee operations never wrap
      for (std::size_t i = 0; i < pimpl->num_syndromes_per_round; i++) {
        pimpl->persistent_detector_buffer[i] =
            pimpl->msyn_buffer[pimpl->prev_round_buffer_offset + i] ^
            pimpl->msyn_buffer[pimpl->current_round_buffer_offset + i];
      }
    }

    // Update offsets for next round: current becomes previous, advance current
    pimpl->prev_round_buffer_offset = pimpl->current_round_buffer_offset;
    pimpl->current_round_buffer_offset += pimpl->num_msyn_per_decode;
    if (pimpl->current_round_buffer_offset >= pimpl->buffer_capacity)
      pimpl->current_round_buffer_offset -= pimpl->buffer_capacity;
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

    // Decrement counter for next iteration of while loop
    pimpl->num_syndromes_buffered_but_not_decoded -= pimpl->num_msyn_per_decode;
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
  pimpl->num_syndromes_buffered_but_not_decoded = 0;
  pimpl->msyn_buffer.clear();
  pimpl->msyn_buffer.resize(pimpl->buffer_capacity);

  // Reset incremental computation state
  pimpl->current_round = 0;
  pimpl->current_round_buffer_offset = 0;
  pimpl->prev_round_buffer_offset = 0;

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
