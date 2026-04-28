/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <cuda_runtime.h>
#include <iosfwd>
#include <string>
#include <vector>

namespace cudaq::qec::realtime::experimental {

/// @brief Per-kernel resource usage captured from a CUDA graph.
struct kernel_resource_info {
  std::string name; ///< Kernel symbol name (demangled if available).
  dim3 grid_dim;    ///< Grid dimensions from the graph node.
  dim3 block_dim;   ///< Block dimensions from the graph node.
  std::size_t static_shmem = 0;  ///< Static shared memory per block (bytes).
  std::size_t dynamic_shmem = 0; ///< Dynamic shared memory per block (bytes).
  std::size_t local_mem = 0;     ///< Local memory per thread (bytes).
  std::size_t const_mem = 0; ///< Constant memory used by the kernel (bytes).
  int num_regs = 0;          ///< Registers per thread.
  int max_threads_per_block = 0; ///< Hardware max threads for this kernel.
};

/// @brief Aggregate resource usage for a CUDA graph.
struct graph_resource_info {
  std::size_t total_nodes = 0;
  std::size_t kernel_nodes = 0;
  std::size_t memcpy_nodes = 0;
  std::size_t host_nodes = 0;
  std::size_t other_nodes = 0;
  std::vector<kernel_resource_info> kernels;
};

/// @brief Walk a captured CUDA graph and return per-kernel resource usage.
/// @param graph  A captured (not-yet-destroyed) CUDA graph handle.
/// @returns An empty graph_resource_info if @p graph is null or traversal
///          fails, otherwise populated aggregate + per-kernel info.
///
/// @warning This routine uses the CUDA driver API
/// (@c cuGraphKernelNodeGetParams, @c cuFuncGetAttribute, @c cuFuncGetName)
/// to introspect kernels launched by external libraries such as TensorRT.
/// Those calls perturb the primary CUDA context state and can interfere
/// with DOCA / GPU-RoCE setup on the FPGA bridge path.  Callers that
/// share a CUDA context with DOCA-based transports must NOT invoke this
/// function.
graph_resource_info collect_graph_resources(cudaGraph_t graph);

/// @brief Pretty-print graph resource usage to an output stream.
/// @param os   Output stream (e.g. @c std::cout).
/// @param info Collected info from @c collect_graph_resources.
void print_graph_resources(std::ostream &os, const graph_resource_info &info);

} // namespace cudaq::qec::realtime::experimental
