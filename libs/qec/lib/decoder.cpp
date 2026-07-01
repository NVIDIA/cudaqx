/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/qec/decoder.h"
#include "common/FmtCore.h"
#include "cuda-qx/core/library_utils.h"
#include "cudaq/qec/device_affinity.h"
#include "cudaq/qec/plugin_loader.h"
#include "cudaq/qec/version.h"
#include "cudaq/runtime/logger/logger.h"
#include <cassert>
#include <cctype>
#include <cstring>
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <filesystem>
#include <thread>
#include <vector>
#if defined(__linux__)
#include <cerrno>
#include <fstream>
#include <linux/mempolicy.h>
#include <sched.h>
#include <sstream>
#include <sys/syscall.h>
#include <unistd.h>
#endif
#include "hardware_affinity.h"

namespace {

inline cudaq::qec::detail_affinity::mempolicy_mode
to_affinity_mode(cudaq::qec::mempolicy_mode m) {
  return m == cudaq::qec::mempolicy_mode::bind
             ? cudaq::qec::detail_affinity::mempolicy_mode::bind
             : cudaq::qec::detail_affinity::mempolicy_mode::preferred;
}

// RAII: sets the calling thread's CUDA current device, restores on destruction.
// target < 0 = no-op. Throws std::runtime_error if target is out of range.
struct CudaDeviceGuard {
  int prev_ = -1;
  bool active_ = false;

  explicit CudaDeviceGuard(int target) {
    if (target < 0)
      return;
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || target >= count)
      throw std::runtime_error(
          "cuda_device_id " + std::to_string(target) +
          " out of range (device_count=" + std::to_string(count) + ")");
    if (cudaGetDevice(&prev_) != cudaSuccess) {
      prev_ = -1;
      return;
    } // can't safely switch
    if (prev_ != target) {
      cudaError_t e = cudaSetDevice(target);
      if (e != cudaSuccess)
        throw std::runtime_error("CudaDeviceGuard: cudaSetDevice(" +
                                 std::to_string(target) +
                                 ") failed: " + cudaGetErrorString(e));
      active_ = true;
    }
  }
  ~CudaDeviceGuard() {
    if (active_)
      cudaSetDevice(prev_);
  }
  CudaDeviceGuard(const CudaDeviceGuard &) = delete;
  CudaDeviceGuard &operator=(const CudaDeviceGuard &) = delete;
};

#if defined(__linux__)
// RAII: binds the calling thread to a NUMA node and restores on destruction.
// node < 0 = no-op. Uses the shared persistent-bind primitive for the set half
// and remembers prior affinity for the restore.
struct NumaGuard {
  bool affinity_set_ = false;
  bool has_prev_affinity_ = false;
  bool mempol_set_ = false;
  cpu_set_t prev_set_{};

  explicit NumaGuard(
      int node, cudaq::qec::detail_affinity::mempolicy_mode mode =
                    cudaq::qec::detail_affinity::mempolicy_mode::preferred) {
    if (node < 0)
      return;
    CPU_ZERO(&prev_set_);
    has_prev_affinity_ =
        (sched_getaffinity(0, sizeof(prev_set_), &prev_set_) == 0);
    affinity_set_ = has_prev_affinity_;
    mempol_set_ = (node < static_cast<int>(sizeof(unsigned long) * 8));
    cudaq::qec::detail_affinity::bind_this_thread_to_numa_node(node, mode);
  }

  ~NumaGuard() {
    if (mempol_set_)
      if (syscall(SYS_set_mempolicy, MPOL_DEFAULT, nullptr, 0UL) != 0)
        cudaq::qec::detail_affinity::affinity_warn(
            "NumaGuard restore: failed to reset thread mempolicy: " +
            std::string(std::strerror(errno)) + "; thread may remain bound");
    if (affinity_set_ && has_prev_affinity_)
      if (sched_setaffinity(0, sizeof(prev_set_), &prev_set_) != 0)
        cudaq::qec::detail_affinity::affinity_warn(
            "NumaGuard restore: failed to reset thread affinity: " +
            std::string(std::strerror(errno)) + "; thread may remain pinned");
  }

  NumaGuard(const NumaGuard &) = delete;
  NumaGuard &operator=(const NumaGuard &) = delete;
};
#else
struct NumaGuard {
  explicit NumaGuard(
      int node, cudaq::qec::detail_affinity::mempolicy_mode mode =
                    cudaq::qec::detail_affinity::mempolicy_mode::preferred) {
    (void)mode;
    if (node < 0)
      return;
    static bool warned = false;
    if (!warned) {
      std::cerr << "[cudaq-qec] numa_node_id ignored: NUMA binding is only "
                   "supported on Linux.\n";
      warned = true;
    }
  }
};
#endif

} // anonymous namespace

INSTANTIATE_REGISTRY(cudaq::qec::decoder, const cudaq::qec::decoder_init &,
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

decoder::decoder(cudaq::qec::sparse_binary_matrix H)
    : H(std::move(H)),
      pimpl(std::unique_ptr<rt_impl, rt_impl_deleter>(new rt_impl())) {
  syndrome_size = this->H.num_rows();
  block_size = this->H.num_cols();
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

int numa_node_for_cuda_device(int cuda_device_id) {
  if (cuda_device_id < 0)
    return -1;
#if defined(__linux__)
  char busid[32] = {0};
  if (cudaDeviceGetPCIBusId(busid, sizeof(busid), cuda_device_id) !=
      cudaSuccess) {
    cudaq::qec::detail_affinity::affinity_info(
        "numa_node_id auto-derive for cuda_device_id " +
        std::to_string(cuda_device_id) +
        " could not read the PCI bus id; NUMA binding skipped");
    return -1;
  }
  for (char *c = busid; *c; ++c)
    *c = static_cast<char>(std::tolower(static_cast<unsigned char>(*c)));
  std::ifstream f(std::string("/sys/bus/pci/devices/") + busid + "/numa_node");
  int node = -1;
  if (!f.is_open() || !(f >> node) || node < 0) {
    cudaq::qec::detail_affinity::affinity_info(
        "numa_node_id auto-derive for cuda_device_id " +
        std::to_string(cuda_device_id) +
        " resolved to no node (single-node host or unresolved topology); "
        "NUMA binding skipped");
    return -1;
  }
  return node;
#else
  return -1;
#endif
}

void decoder::set_hardware_params(const cudaqx::heterogeneous_map &params) {
  cuda_device_id_ = cudaq::qec::read_cuda_device_id(params);
  numa_node_id_ = cudaq::qec::read_numa_node_id(params);
  mempolicy_ = cudaq::qec::read_mempolicy(params);
  cpu_affinity_ = cudaq::qec::read_cpu_affinity(params);
  // Soft-derive: user pinned a GPU but not a node -> use the GPU's node (or -1
  // if unresolved).
  if (numa_node_id_ < 0 && cuda_device_id_ >= 0)
    numa_node_id_ = numa_node_for_cuda_device(cuda_device_id_);
}

int decoder::bind_current_thread() {
  if (cuda_device_id_ >= 0) {
    // Persistent set on this thread (CudaDeviceGuard restores on scope exit, so
    // set directly here and let it stick).
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || cuda_device_id_ >= count)
      throw std::runtime_error(
          "cuda_device_id " + std::to_string(cuda_device_id_) +
          " out of range or CUDA unavailable in bind_current_thread");
    cudaError_t e = cudaSetDevice(cuda_device_id_);
    if (e != cudaSuccess)
      throw std::runtime_error("bind_current_thread: cudaSetDevice(" +
                               std::to_string(cuda_device_id_) +
                               ") failed: " + cudaGetErrorString(e));
  }
  cudaq::qec::detail_affinity::bind_this_thread_to_numa_node(
      numa_node_id_, to_affinity_mode(mempolicy_));
  bound_persistently_ = true;
  if (numa_node_id_ >= 0) {
    // Best-effort confirmation: warn if the calling thread is not actually
    // running on the requested node's CPUs (e.g. locked cpuset in a container).
#if defined(__linux__)
    cpu_set_t want, have;
    CPU_ZERO(&want);
    CPU_ZERO(&have);
    if (cudaq::qec::detail_affinity::build_node_cpuset(numa_node_id_, want) &&
        sched_getaffinity(0, sizeof(have), &have) == 0) {
      bool on_node = false;
      for (int c = 0; c < CPU_SETSIZE; ++c)
        if (CPU_ISSET(c, &have) && CPU_ISSET(c, &want)) {
          on_node = true;
          break;
        }
      if (!on_node)
        cudaq::qec::detail_affinity::affinity_warn(
            "bind_current_thread: numa_node_id " +
            std::to_string(numa_node_id_) +
            " requested but the calling thread could not be pinned to it");
    }
#endif
  }
  if (!cpu_affinity_.empty())
    cudaq::qec::detail_affinity::set_thread_cpu_affinity(cpu_affinity_);
  return numa_node_id_;
}

decoder_result
decoder::decode_on_pinned_thread(const std::vector<float_t> &syndrome) {
  decoder_result result;
  std::exception_ptr err;
  std::thread worker([&] {
    try {
      bind_current_thread();
      result = decode(syndrome);
    } catch (...) {
      err = std::current_exception();
    }
  });
  worker.join();
  if (err)
    std::rethrow_exception(err); // surface an OOB-device throw to the caller
  return result;
}

void decoder::warn_if_device_mismatch() {
  if (cuda_device_id_ < 0 || device_mismatch_warned_)
    return;
  int current = -1;
  if (cudaGetDevice(&current) == cudaSuccess && current != cuda_device_id_) {
    cudaq::qec::detail_affinity::affinity_warn(
        "decoder pinned to cuda_device_id " + std::to_string(cuda_device_id_) +
        " but decoding on current device " + std::to_string(current) +
        "; call bind_current_thread() on this thread, or use "
        "decode_batch/decode_async");
    device_mismatch_warned_ = true;
  }
}

// Provide a trivial implementation of for tensor<uint8_t> decode call. Child
// classes should override this if they never want to pass through floats.
decoder_result decoder::decode(const cudaqx::tensor<uint8_t> &syndrome) {
  warn_if_device_mismatch();
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
  // Apply affinity once for the whole batch (not per-syndrome) to avoid
  // repeated syscall overhead on the hot path.
  // If the caller already bound this thread via bind_current_thread(), the
  // per-call guards are redundant set/restore syscalls — skip them.
  CudaDeviceGuard dev(bound_persistently_ ? -1 : cuda_device_id_);
  NumaGuard numa(bound_persistently_ ? -1 : numa_node_id_,
                 to_affinity_mode(mempolicy_));
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
  // Capture by value: avoids a data race if the decoder is destroyed before
  // the future resolves.
  const int cuda_id = cuda_device_id_;
  const int numa_id = numa_node_id_;
  const cudaq::qec::mempolicy_mode mempolicy = mempolicy_;
  return std::async(std::launch::async,
                    [this, syndrome, cuda_id, numa_id, mempolicy] {
                      CudaDeviceGuard dev(cuda_id);
                      NumaGuard numa(numa_id, to_affinity_mode(mempolicy));
                      return this->decode(syndrome);
                    });
}

std::unique_ptr<decoder>
decoder::get(const std::string &name, const decoder_init &init,
             const cudaqx::heterogeneous_map &param_map) {
  auto [mutex, registry] = get_registry();
  std::lock_guard<std::recursive_mutex> lock(mutex);
  auto iter = registry.find(name);
  if (iter == registry.end())
    throw std::runtime_error(
        "invalid decoder requested: " + name +
        ". Run with CUDAQ_LOG_LEVEL=info (environment variable) to see "
        "additional plugin diagnostics at startup.");
  // Guards during construction so allocations land on the right hardware.
  // Restored before this function returns; decode-time affinity is re-applied
  // per call in decode_batch() and decode_async().
  const int dev = cudaq::qec::read_cuda_device_id(param_map);
  int node = cudaq::qec::read_numa_node_id(param_map);
  if (node < 0 && dev >= 0)
    node = numa_node_for_cuda_device(dev);
  CudaDeviceGuard ctor_dev(dev);
  NumaGuard ctor_numa(node); // preferred mode at construction
  auto d = iter->second(init, param_map);
  d->set_hardware_params(param_map);
  return d;
}

namespace details {

dem_default_values dem_defaults_for_missing_keys(
    const std::function<bool(const std::string &)> &contains_user_key,
    const detector_error_model &dem) {
  dem_default_values out;
  if (!contains_user_key("O") && dem.num_observables() > 0)
    out.O = &dem.observables_flips_matrix;
  if (!contains_user_key("error_rate_vec"))
    out.error_rate_vec = &dem.error_rates;
  return out;
}

} // namespace details

static uint32_t calculate_num_msyn_per_decode(
    const std::vector<std::vector<uint32_t>> &D_sparse) {
  uint32_t max_col = 0;
  for (const auto &row : D_sparse)
    for (const auto col : row)
      max_col = std::max(max_col, col);
  return max_col + 1;
}

static void
validate_sparse_column_indices(const std::vector<std::vector<uint32_t>> &sparse,
                               std::size_t max_col, const char *name) {
  for (std::size_t row = 0; row < sparse.size(); ++row) {
    for (const auto col : sparse[row]) {
      if (col >= max_col) {
        throw std::invalid_argument(
            fmt::format("{} column index {} out of range [0, {}) at row {}",
                        name, col, max_col, row));
      }
    }
  }
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
  validate_sparse_column_indices(this->O_sparse, block_size, "O_sparse");
  this->pimpl->corrections.clear();
  this->pimpl->corrections.resize(O_sparse.size());
}

void decoder::set_O_sparse(const std::vector<int64_t> &O_sparse_vec_in) {
  set_sparse_from_vec(O_sparse_vec_in, this->O_sparse);
  validate_sparse_column_indices(this->O_sparse, block_size, "O_sparse");
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
    if (D_sparse.size() != decoder->get_syndrome_size()) {
      throw std::invalid_argument(
          fmt::format("D_sparse row count ({}) must match syndrome_size ({})",
                      D_sparse.size(), decoder->get_syndrome_size()));
    }
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
    std::vector<uint32_t> log_observables;
    std::vector<uint8_t> log_observable_corrections;
    // The four time points are used to measure the duration of each of 3 steps.
    std::chrono::time_point<std::chrono::high_resolution_clock> log_t0, log_t1,
        log_t2, log_t3;
    std::chrono::duration<double> log_dur1, log_dur2, log_dur3;

    const bool log_due_to_log_level =
        cudaq::detail::should_log(cudaq::detail::LogLevel::info);
    const bool should_log = pimpl->should_log || log_due_to_log_level;

    if (should_log) {
      log_t0 = std::chrono::high_resolution_clock::now();
      log_errors.reserve(syndrome_length);
      log_observables.reserve(O_sparse.size());
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
    warn_if_device_mismatch();
    auto decoded_result = decode(pimpl->persistent_soft_detector_buffer);

    // If we didn't get a decoded result, just return
    if (pimpl->is_sliding_window) {
      if (decoded_result.result.size() == 0) {
        return false;
      }
    }
    // Process the results.
    // TODO - should this interrogate the decoded_result.converged flag?
    const auto result_type = get_result_type();
    const auto num_observables = get_num_observables();
    const char *result_type_str = nullptr;
    const char *result_type_name = nullptr;
    std::size_t expected_result_size = 0;
    switch (result_type) {
    case decode_result_type::decode_to_errs:
      result_type_str = "errs";
      result_type_name = "decode_to_errs";
      expected_result_size = block_size;
      break;
    case decode_result_type::decode_to_obs:
      result_type_str = "obs";
      result_type_name = "decode_to_obs";
      expected_result_size = num_observables;
      break;
    }
    if (!result_type_name)
      throw std::runtime_error(
          fmt::format("Unsupported decoder result type ({})",
                      static_cast<int>(result_type)));
    if ((!pimpl->is_sliding_window &&
         decoded_result.result.size() != expected_result_size) ||
        (pimpl->is_sliding_window && !decoded_result.result.empty() &&
         decoded_result.result.size() != expected_result_size)) {
      throw std::runtime_error(fmt::format(
          "Decoder result size ({}) does not match expected size ({}) for "
          "result type {}",
          decoded_result.result.size(), expected_result_size,
          result_type_name));
    }

    // Flip an observable correction and mirror it into the per-call log so the
    // logged flips stay faithful to the applied corrections.
    auto flip_correction = [&](std::size_t i) {
      pimpl->corrections[i] ^= 1;
      if (should_log)
        log_observable_corrections[i] ^= 1;
    };

    if (should_log)
      log_t2 = std::chrono::high_resolution_clock::now();

    switch (result_type) {
    case decode_result_type::decode_to_obs:
      // Observable-frame path: decoder already projected to observables via its
      // internal "O" matrix; use the result directly.
      for (std::size_t i = 0; i < num_observables; i++)
        if (decoded_result.result[i]) {
          if (should_log)
            log_observables.push_back(i);
          flip_correction(i);
        }
      break;
    case decode_result_type::decode_to_errs:
      // Error-frame path: decoder returns a block-sized error vector; project
      // to observables via O_sparse.
      if (should_log)
        for (std::size_t e = 0, E = decoded_result.result.size(); e < E; e++)
          if (decoded_result.result[e])
            log_errors.push_back(e);
      // For each observable, flip its correction once for each predicted error
      // that flips it (net parity over O_sparse[i]).
      for (std::size_t i = 0; i < num_observables; i++)
        for (auto col : O_sparse[i])
          if (decoded_result.result[col])
            flip_correction(i);
      break;
    }
    if (should_log) {
      log_t3 = std::chrono::high_resolution_clock::now();
      log_dur1 = log_t1 - log_t0;
      log_dur2 = log_t2 - log_t1;
      log_dur3 = log_t3 - log_t2;
      pimpl->log_counter++;
      auto s = fmt::format(
          "[DecoderStats][{}] Counter:{} DecoderId:{} InputMsyn:{} "
          "InputDetectors:{} Converged:{} ResultType:{} Errors:{} "
          "Observables:{} "
          "ObservableCorrectionsThisCall:{} ObservableCorrectionsTotal:{} "
          "Dur1:{:.1f}us Dur2:{:.1f}us Dur3:{:.1f}us",
          static_cast<const void *>(this), pimpl->log_counter,
          pimpl->decoder_id, fmt::join(log_msyn, ","),
          fmt::join(log_detectors, ","), decoded_result.converged ? 1 : 0,
          result_type_str, fmt::join(log_errors, ","),
          fmt::join(log_observables, ","),
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
      cudaq::detail::should_log(cudaq::detail::LogLevel::info);
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
      cudaq::detail::should_log(cudaq::detail::LogLevel::info);
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
      cudaq::detail::should_log(cudaq::detail::LogLevel::info);
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
                                     const decoder_init &init,
                                     const cudaqx::heterogeneous_map options) {
  return decoder::get(name, init, options);
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
