/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cuda-qx/core/heterogeneous_map.h"
#include "cuda-qx/core/tensor.h"
#include "cudaq/utils/extension_point.h"
#include <future>
#include <optional>
#include <vector>

namespace cudaq::qec {

#if defined(CUDAQX_QEC_FLOAT_TYPE)
using float_t = CUDAQX_QEC_FLOAT_TYPE;
#else
using float_t = double;
#endif

/// @brief Validates that all keys in a heterogeneous map are found in a list of
/// acceptable types
/// @param config The heterogeneous map to validate
/// @param acceptable_types List of acceptable key types
/// @return Vector of invalid keys (empty if all keys are valid)
inline std::vector<std::string>
validate_config_parameters(const cudaqx::heterogeneous_map &config,
                           const std::vector<std::string> &acceptable_types) {
  std::vector<std::string> invalid_types;
  for (const auto &[key, _] : config) {
    if (std::find(acceptable_types.begin(), acceptable_types.end(), key) ==
        acceptable_types.end()) {
      invalid_types.push_back(key);
    }
  }
  return invalid_types;
}

/// @brief Decoder results
struct decoder_result {
  /// @brief Whether or not the decoder converged.
  bool converged = false;

  /// @brief Vector of length `block_size` with soft probabilities of errors in
  /// each index.
  std::vector<float_t> result;

  /// @brief Optional additional results from the decoder stored in a
  /// heterogeneous map. For equality comparison, this field is treated as a
  /// boolean flag - two decoder_results are considered equal only if both have
  /// empty opt_results (either std::nullopt or an empty map). If either result
  /// has non-empty opt_results, they are considered not equal.
  std::optional<cudaqx::heterogeneous_map> opt_results;

  // Manually define the equality operator
  bool operator==(const decoder_result &other) const {
    // First compare the non-optional fields
    if (std::tie(converged, result) !=
        std::tie(other.converged, other.result)) {
      return false;
    }
    // If both do not have opt_results or both have empty maps, then they are
    // equal
    bool this_empty = !opt_results.has_value() || opt_results->size() == 0;
    bool other_empty =
        !other.opt_results.has_value() || other.opt_results->size() == 0;
    if (this_empty && other_empty) {
      return true;
    }
    // Otherwise, they are not equal
    return false;
  }

  // Manually define the inequality operator
  bool operator!=(const decoder_result &other) const {
    return !(*this == other);
  }
};

/// @brief Return type for asynchronous decoding results
class async_decoder_result {
public:
  std::future<cudaq::qec::decoder_result> fut;

  /// @brief Construct an async_decoder_result from a std::future.
  /// @param f A rvalue reference to a std::future containing a decoder_result.
  async_decoder_result(std::future<cudaq::qec::decoder_result> &&f)
      : fut(std::move(f)) {}

  async_decoder_result(async_decoder_result &&other) noexcept
      : fut(std::move(other.fut)) {}

  async_decoder_result &operator=(async_decoder_result &&other) noexcept {
    if (this != &other) {
      fut = std::move(other.fut);
    }

    return *this;
  }

  /// @brief Block until the decoder result is ready and retrieve it.
  /// Wait until the underlying future is ready and then
  /// return the stored decoder result.
  /// @return The decoder_result obtained from the asynchronous operation.
  cudaq::qec::decoder_result get() { return fut.get(); }

  /// @brief Check if the asynchronous result is ready.
  /// @return `true` if the future is ready, `false` otherwise.
  bool ready() {
    return fut.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
  }
};

/// @brief The `decoder` base class should be subclassed by specific decoder
/// implementations. The `heterogeneous_map` provides a placeholder for
/// arbitrary constructor parameters that can be unique to each specific
/// decoder.
class decoder
    : public cudaq::extension_point<decoder, const cudaqx::tensor<uint8_t> &,
                                    const cudaqx::heterogeneous_map &> {
public:
  decoder() = delete;

  /// @brief Constructor
  /// @param H Decoder's parity check matrix represented as a tensor. The tensor
  /// is required be rank 2 and must be of dimensions \p syndrome_size x
  /// \p block_size.
  /// will use the same \p H.
  decoder(const cudaqx::tensor<uint8_t> &H);

  /// @brief Decode a single syndrome
  /// @param syndrome A vector of syndrome measurements where the floating point
  /// value is the probability that the syndrome measurement is a |1>. The
  /// length of the syndrome vector should be equal to \p syndrome_size.
  /// @returns Vector of length \p block_size with soft probabilities of errors
  /// in each index.
  virtual decoder_result decode(const std::vector<float_t> &syndrome) = 0;

  /// @brief Decode a single syndrome
  /// @param syndrome An order-1 tensor of syndrome measurements where a 1 bit
  /// represents that the syndrome measurement is a |1>. The
  /// length of the syndrome vector should be equal to \p syndrome_size.
  /// @returns Vector of length \p block_size of errors in each index.
  virtual decoder_result decode(const cudaqx::tensor<uint8_t> &syndrome);

  /// @brief Decode a single syndrome
  /// @param syndrome A vector of syndrome measurements where the floating point
  /// value is the probability that the syndrome measurement is a |1>.
  /// @returns std::future of a vector of length `block_size` with soft
  /// probabilities of errors in each index.
  virtual std::future<decoder_result>
  decode_async(const std::vector<float_t> &syndrome);

  /// @brief Decode multiple independent syndromes (may be done in serial or
  /// parallel depending on the specific implementation)
  /// @param syndrome A vector of `N` syndrome measurements where the floating
  /// point value is the probability that the syndrome measurement is a |1>.
  /// @returns 2-D vector of size `N` x `block_size` with soft probabilities of
  /// errors in each index.
  virtual std::vector<decoder_result>
  decode_batch(const std::vector<std::vector<float_t>> &syndrome);

  /// @brief This `get` overload supports default values.
  static std::unique_ptr<decoder>
  get(const std::string &name, const cudaqx::tensor<uint8_t> &H,
      const cudaqx::heterogeneous_map &param_map = cudaqx::heterogeneous_map());

  std::size_t get_block_size() { return block_size; }
  std::size_t get_syndrome_size() { return syndrome_size; }

  /// @brief Destructor
  virtual ~decoder() {}

  /// @brief Get the version of the decoder. Subclasses that are not part of the
  /// standard GitHub repo should override this to provide a more tailored
  /// version string.
  /// @return A string containing the version of the decoder
  virtual std::string get_version() const;

protected:
  /// @brief For a classical `[n,k]` code, this is `n`.
  std::size_t block_size = 0;

  /// @brief For a classical `[n,k]` code, this is `n-k`
  std::size_t syndrome_size = 0;

  /// @brief The decoder's parity check matrix
  cudaqx::tensor<uint8_t> H;
};

/// @brief Convert a vector of soft probabilities to a vector of hard
/// probabilities.
/// @param in Soft probability input vector in range [0.0, 1.0]
/// @param out Hard probability output vector containing only 0/false or 1/true.
/// @param thresh Values >= thresh are assigned 1/true and all others are
/// assigned 0/false.
template <typename t_soft, typename t_hard,
          typename std::enable_if<std::is_floating_point<t_soft>::value &&
                                      (std::is_integral<t_hard>::value ||
                                       std::is_same<t_hard, bool>::value),
                                  int>::type = 0>
inline void convert_vec_soft_to_hard(const std::vector<t_soft> &in,
                                     std::vector<t_hard> &out,
                                     t_soft thresh = 0.5) {
  out.resize(in.size());
  for (std::size_t i = 0; i < in.size(); i++)
    out[i] = static_cast<t_hard>(in[i] >= thresh ? 1 : 0);
}

/// @brief Convert a vector of soft probabilities to a tensor<uint8_t> of hard
/// probabilities. Tensor must be uninitialized, or initialized to a rank-1
/// tensor for equal dim as the vector.
/// @param in Soft probability input vector in range [0.0, 1.0]
/// @param out Hard probability output tensor containing only 0/false or 1/true.
/// @param thresh Values >= thresh are assigned 1/true and all others are
/// assigned 0/false.
template <typename t_soft, typename t_hard,
          typename std::enable_if<std::is_floating_point<t_soft>::value &&
                                      (std::is_integral<t_hard>::value ||
                                       std::is_same<t_hard, bool>::value),
                                  int>::type = 0>
inline void convert_vec_soft_to_tensor_hard(const std::vector<t_soft> &in,
                                            cudaqx::tensor<t_hard> &out,
                                            t_soft thresh = 0.5) {
  if (out.shape().empty())
    out = cudaqx::tensor<t_hard>({in.size()});
  if (out.rank() != 1)
    throw std::runtime_error(
        "Vector to tensor conversion requires rank-1 tensor");
  if (out.shape()[0] != in.size())
    throw std::runtime_error(
        "Vector to tensor conversion requires tensor dim == vector length");
  auto raw_ptr = out.data();
  for (size_t i = 0; i < in.size(); ++i)
    raw_ptr[i] = static_cast<t_hard>(in[i] >= thresh ? 1 : 0);
}

/// @brief Convert a vector of hard probabilities to a vector of soft
/// probabilities.
/// @param in Hard probability input vector containing only 0/false or 1/true.
/// @param out Soft probability output vector in the range [0.0, 1.0]
/// @param true_val The soft probability value assigned when the input is 1
/// (default to 1.0)
/// @param false_val The soft probability value assigned when the input is 0
/// (default to 0.0)
template <typename t_soft, typename t_hard,
          typename std::enable_if<std::is_floating_point<t_soft>::value &&
                                      (std::is_integral<t_hard>::value ||
                                       std::is_same<t_hard, bool>::value),
                                  int>::type = 0>
inline void convert_vec_hard_to_soft(const std::vector<t_hard> &in,
                                     std::vector<t_soft> &out,
                                     const t_soft true_val = 1.0,
                                     const t_soft false_val = 0.0) {
  out.resize(in.size());
  for (std::size_t i = 0; i < in.size(); i++)
    out[i] = static_cast<t_soft>(in[i] ? true_val : false_val);
}

/// @brief Convert a 2D vector of soft probabilities to a 2D vector of hard
/// probabilities.
/// @param in Soft probability input vector in range [0.0, 1.0]
/// @param out Hard probability output vector containing only 0/false or 1/true.
/// @param thresh Values >= thresh are assigned 1/true and all others are
/// assigned 0/false.
template <typename t_soft, typename t_hard,
          typename std::enable_if<std::is_floating_point<t_soft>::value &&
                                      (std::is_integral<t_hard>::value ||
                                       std::is_same<t_hard, bool>::value),
                                  int>::type = 0>
inline void convert_vec_soft_to_hard(const std::vector<std::vector<t_soft>> &in,
                                     std::vector<std::vector<t_hard>> &out,
                                     t_soft thresh = 0.5) {
  std::size_t row_index = 0;
  out.resize(in.size());
  for (auto &r : in) {
    auto &out_row = out[row_index++];
    out_row.resize(r.size());
    for (std::size_t c = 0; c < r.size(); c++)
      out_row[c] = static_cast<t_hard>(r[c] >= thresh ? 1 : 0);
  }
}

/// @brief Convert a 2D vector of hard probabilities to a 2D vector of soft
/// probabilities.
/// @param in Hard probability input vector containing only 0/false or 1/true.
/// @param out Soft probability output vector in the range [0.0, 1.0]
/// @param true_val The soft probability value assigned when the input is 1
/// (default to 1.0)
/// @param false_val The soft probability value assigned when the input is 0
/// (default to 0.0)
template <typename t_soft, typename t_hard,
          typename std::enable_if<std::is_floating_point<t_soft>::value &&
                                      (std::is_integral<t_hard>::value ||
                                       std::is_same<t_hard, bool>::value),
                                  int>::type = 0>
inline void convert_vec_hard_to_soft(const std::vector<std::vector<t_hard>> &in,
                                     std::vector<std::vector<t_soft>> &out,
                                     const t_soft true_val = 1.0,
                                     const t_soft false_val = 0.0) {
  out.clear();
  out.reserve(in.size());
  for (auto &r : in) {
    std::vector<t_soft> out_row;
    out_row.reserve(r.size());
    for (auto c : r)
      out_row.push_back(static_cast<t_soft>(c ? true_val : false_val));
    out.push_back(std::move(out_row));
  }
}

std::unique_ptr<decoder>
get_decoder(const std::string &name, const cudaqx::tensor<uint8_t> &H,
            const cudaqx::heterogeneous_map options = {});
} // namespace cudaq::qec
