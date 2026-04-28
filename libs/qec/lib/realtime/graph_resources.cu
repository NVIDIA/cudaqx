/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/qec/realtime/graph_resources.h"

#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cxxabi.h>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

namespace cudaq::qec::realtime::experimental {

namespace {

std::string demangle_symbol(const char *mangled) {
  if (!mangled)
    return "<unknown>";
  int status = 0;
  char *out = abi::__cxa_demangle(mangled, nullptr, nullptr, &status);
  std::string name = (status == 0 && out) ? std::string(out) : mangled;
  std::free(out);
  return name;
}

} // namespace

graph_resource_info collect_graph_resources(cudaGraph_t graph) {
  graph_resource_info result{};
  if (!graph)
    return result;

  std::size_t num_nodes = 0;
  if (cudaGraphGetNodes(graph, nullptr, &num_nodes) != cudaSuccess ||
      num_nodes == 0)
    return result;

  std::vector<cudaGraphNode_t> nodes(num_nodes);
  if (cudaGraphGetNodes(graph, nodes.data(), &num_nodes) != cudaSuccess)
    return result;

  result.total_nodes = num_nodes;

  for (auto node : nodes) {
    cudaGraphNodeType type;
    if (cudaGraphNodeGetType(node, &type) != cudaSuccess)
      continue;

    switch (type) {
    case cudaGraphNodeTypeKernel:
      ++result.kernel_nodes;
      break;
    case cudaGraphNodeTypeMemcpy:
      ++result.memcpy_nodes;
      continue;
    case cudaGraphNodeTypeHost:
      ++result.host_nodes;
      continue;
    default:
      ++result.other_nodes;
      continue;
    }

    kernel_resource_info info{};

    // Try runtime API first (works for kernels launched via <<<>>>).
    cudaKernelNodeParams params{};
    if (cudaGraphKernelNodeGetParams(node, &params) == cudaSuccess) {
      info.grid_dim = params.gridDim;
      info.block_dim = params.blockDim;
      info.dynamic_shmem = params.sharedMemBytes;

      const char *mangled = nullptr;
      if (params.func)
        cudaFuncGetName(&mangled, params.func);
      info.name = demangle_symbol(mangled);

      cudaFuncAttributes attr{};
      if (params.func &&
          cudaFuncGetAttributes(&attr, params.func) == cudaSuccess) {
        info.static_shmem = attr.sharedSizeBytes;
        info.local_mem = attr.localSizeBytes;
        info.const_mem = attr.constSizeBytes;
        info.num_regs = attr.numRegs;
        info.max_threads_per_block = attr.maxThreadsPerBlock;
      }
    } else {
      // Fall back to driver API for TRT-internal kernels launched via
      // cuLaunchKernel.  WARNING: these driver calls perturb CUDA context
      // state in ways that interfere with DOCA/Hololink GPU-RoCE setup, so
      // callers that share a CUDA context with DOCA-based transports must
      // NOT invoke this function.
      CUDA_KERNEL_NODE_PARAMS drv_params{};
      if (cuGraphKernelNodeGetParams(reinterpret_cast<CUgraphNode>(node),
                                     &drv_params) == CUDA_SUCCESS) {
        info.grid_dim =
            dim3(drv_params.gridDimX, drv_params.gridDimY, drv_params.gridDimZ);
        info.block_dim = dim3(drv_params.blockDimX, drv_params.blockDimY,
                              drv_params.blockDimZ);
        info.dynamic_shmem = drv_params.sharedMemBytes;

        CUfunction func = drv_params.func;
        if (func) {
          const char *raw_name = nullptr;
          if (cuFuncGetName(&raw_name, func) == CUDA_SUCCESS)
            info.name = demangle_symbol(raw_name);

          int regs = 0;
          if (cuFuncGetAttribute(&regs, CU_FUNC_ATTRIBUTE_NUM_REGS, func) ==
              CUDA_SUCCESS)
            info.num_regs = regs;

          int sshmem = 0;
          if (cuFuncGetAttribute(&sshmem, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
                                 func) == CUDA_SUCCESS)
            info.static_shmem = static_cast<std::size_t>(sshmem);

          int lmem = 0;
          if (cuFuncGetAttribute(&lmem, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,
                                 func) == CUDA_SUCCESS)
            info.local_mem = static_cast<std::size_t>(lmem);

          int cmem = 0;
          if (cuFuncGetAttribute(&cmem, CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES,
                                 func) == CUDA_SUCCESS)
            info.const_mem = static_cast<std::size_t>(cmem);

          int max_threads = 0;
          if (cuFuncGetAttribute(&max_threads,
                                 CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                                 func) == CUDA_SUCCESS)
            info.max_threads_per_block = max_threads;
        }
        if (info.name.empty())
          info.name = "<unknown-driver-kernel>";
      } else {
        info.name = "<introspection-failed>";
      }
    }

    result.kernels.push_back(std::move(info));
  }

  return result;
}

void print_graph_resources(std::ostream &os, const graph_resource_info &g) {
  os << "[GraphResources] total_nodes=" << g.total_nodes
     << " kernels=" << g.kernel_nodes << " memcpy=" << g.memcpy_nodes
     << " host=" << g.host_nodes << " other=" << g.other_nodes << "\n";

  std::size_t total_regs_per_launch = 0;
  std::size_t total_shmem_per_launch = 0;
  std::size_t total_threads = 0;

  for (std::size_t i = 0; i < g.kernels.size(); ++i) {
    const auto &k = g.kernels[i];
    std::size_t blocks =
        static_cast<std::size_t>(k.grid_dim.x) * k.grid_dim.y * k.grid_dim.z;
    std::size_t threads_per_block =
        static_cast<std::size_t>(k.block_dim.x) * k.block_dim.y * k.block_dim.z;
    std::size_t launch_threads = blocks * threads_per_block;
    std::size_t launch_regs =
        launch_threads * static_cast<std::size_t>(k.num_regs);
    std::size_t launch_shmem = blocks * (k.static_shmem + k.dynamic_shmem);

    total_regs_per_launch += launch_regs;
    total_shmem_per_launch += launch_shmem;
    total_threads += launch_threads;

    os << "  [" << i << "] " << k.name << "\n"
       << "      grid=(" << k.grid_dim.x << "," << k.grid_dim.y << ","
       << k.grid_dim.z << ") block=(" << k.block_dim.x << "," << k.block_dim.y
       << "," << k.block_dim.z << ")"
       << " threads=" << launch_threads << "\n"
       << "      regs/thread=" << k.num_regs << " local/thread=" << k.local_mem
       << "B"
       << " shmem/block=" << (k.static_shmem + k.dynamic_shmem)
       << "B (static=" << k.static_shmem << " dynamic=" << k.dynamic_shmem
       << ")"
       << " max_threads_per_block=" << k.max_threads_per_block << "\n";
  }

  os << "  Total launch: threads=" << total_threads
     << " regs=" << total_regs_per_launch << " shmem=" << total_shmem_per_launch
     << "B\n";
}

} // namespace cudaq::qec::realtime::experimental
