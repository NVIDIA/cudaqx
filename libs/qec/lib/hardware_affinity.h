/*******************************************************************************
 * Copyright (c) 2025 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Lib-private (NOT installed): hardware affinity primitives shared by the base
// decoder and the realtime session. Never include from a public header.
#pragma once

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(__linux__)
#include <cerrno>
#include <fstream>
#include <linux/mempolicy.h>
#include <sched.h>
#include <sstream>
#include <string>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace cudaq::qec::detail_affinity {

// Decoupled diagnostics: plain C, no cudaq dependency, so this header can be
// reused independently.
inline void affinity_warn(const std::string &msg) {
  std::fprintf(stderr, "[cudaq-qec affinity] WARNING: %s\n", msg.c_str());
}
// INFO/debug: only shown when CUDAQ_QEC_AFFINITY_DEBUG is set (a legitimate
// no-op like a single-node auto-derive must not spam; it is debug-loggable, not
// a failure).
inline void affinity_info(const std::string &msg) {
  if (std::getenv("CUDAQ_QEC_AFFINITY_DEBUG") != nullptr)
    std::fprintf(stderr, "[cudaq-qec affinity] INFO: %s\n", msg.c_str());
}

#if defined(__linux__)

enum class mempolicy_mode { preferred, bind };
inline int mempolicy_syscall_mode(mempolicy_mode m) {
  return (m == mempolicy_mode::bind) ? MPOL_BIND : MPOL_PREFERRED;
}

// Parse /sys/devices/system/node/node<N>/cpulist e.g. "0-7,16-23".
// Returns false if the file is missing or empty.
inline bool build_node_cpuset(int node, cpu_set_t &out) {
  std::ifstream f("/sys/devices/system/node/node" + std::to_string(node) +
                  "/cpulist");
  if (!f.is_open())
    return false;
  std::string list;
  std::getline(f, list);
  if (list.empty())
    return false;
  std::stringstream ss(list);
  std::string range;
  bool any = false;
  while (std::getline(ss, range, ',')) {
    auto dash = range.find('-');
    int lo = std::stoi(range.substr(0, dash));
    int hi =
        (dash == std::string::npos) ? lo : std::stoi(range.substr(dash + 1));
    for (int c = lo; c <= hi; ++c) {
      if (c < CPU_SETSIZE) { // guard against OOB on >1024-CPU machines
        CPU_SET(c, &out);
        any = true;
      }
    }
  }
  return any;
}

// Persistently bind the CALLING thread's CPU affinity + memory policy to a NUMA
// node. No restore. node < 0 = no-op. Only pins CPU affinity when prior
// affinity is readable (avoids permanent pinning in a locked-cpuset container).
// Throws if node cannot be encoded in the mempolicy nodemask (node >= 64).
// Warns (does not throw) if the OS declines an otherwise well-formed request.
inline void
bind_this_thread_to_numa_node(int node,
                              mempolicy_mode mode = mempolicy_mode::preferred) {
  if (node < 0)
    return;
  if (node >= static_cast<int>(sizeof(unsigned long) * 8))
    throw std::runtime_error("numa_node_id " + std::to_string(node) +
                             " exceeds the maximum encodable NUMA node (" +
                             std::to_string(sizeof(unsigned long) * 8 - 1) +
                             "); cannot bind memory policy");
  cpu_set_t prev;
  CPU_ZERO(&prev);
  const bool can_restore = (sched_getaffinity(0, sizeof(prev), &prev) == 0);
  cpu_set_t node_set;
  CPU_ZERO(&node_set);
  if (can_restore && build_node_cpuset(node, node_set)) {
    if (sched_setaffinity(0, sizeof(node_set), &node_set) != 0)
      affinity_warn(
          "numa_node_id " + std::to_string(node) +
          " requested but sched_setaffinity could not pin the thread: " +
          std::strerror(errno) + "; running unpinned (locked cpuset?)");
  } else {
    affinity_warn("numa_node_id " + std::to_string(node) +
                  " requested but its CPU list could not be read (or prior "
                  "affinity unreadable); CPU affinity left unpinned");
  }
  unsigned long nodemask = 1UL << node;
  if (syscall(SYS_set_mempolicy, mempolicy_syscall_mode(mode), &nodemask,
              static_cast<unsigned long>(sizeof(nodemask) * 8)) != 0)
    affinity_warn("numa_node_id " + std::to_string(node) +
                  " requested but set_mempolicy failed: " +
                  std::strerror(errno) + "; memory not bound to node");
}

// Migrate an already-allocated region onto a NUMA node. The ring buffers are
// calloc'd (already faulted) on the setup thread, so MPOL_MF_MOVE is required
// to relocate the pages. node < 0 / null / 0 bytes = no-op.
// Throws if node cannot be encoded in the mempolicy nodemask (node >= 64).
// Warns (does not throw) if the OS declines an otherwise well-formed request.
inline void
bind_region_to_numa_node(void *p, std::size_t bytes, int node,
                         mempolicy_mode mode = mempolicy_mode::preferred) {
  if (node < 0 || p == nullptr || bytes == 0)
    return;
  if (node >= static_cast<int>(sizeof(unsigned long) * 8))
    throw std::runtime_error(
        "numa_node_id " + std::to_string(node) +
        " exceeds the maximum encodable NUMA node; cannot migrate region");
  unsigned long nodemask = 1UL << node;
  if (syscall(SYS_mbind, p, static_cast<unsigned long>(bytes),
              mempolicy_syscall_mode(mode), &nodemask,
              static_cast<unsigned long>(sizeof(nodemask) * 8),
              MPOL_MF_MOVE) != 0)
    affinity_warn("mbind of region to numa_node_id " + std::to_string(node) +
                  " failed: " + std::strerror(errno) +
                  "; pages not migrated (missing CAP_SYS_NICE?)");
}

// Query the calling thread's current memory-policy mode (MPOL_* constant).
// Returns -1 if the query fails.
inline int current_thread_mempolicy_mode() {
  int mode = -1;
  if (syscall(SYS_get_mempolicy, &mode, nullptr, 0UL, nullptr, 0UL) != 0)
    return -1;
  return mode;
}

// Pin the CALLING thread's CPU affinity to an explicit list of core ids.
// No restore. cores.empty() = no-op. Throws std::invalid_argument if any core
// id is out of range. Warns (does not throw) if the OS declines an otherwise
// well-formed request.
inline void set_thread_cpu_affinity(const std::vector<int> &cores) {
  if (cores.empty())
    return;
  for (int c : cores)
    if (c < 0 || c >= CPU_SETSIZE)
      throw std::invalid_argument("cpu_affinity core id " + std::to_string(c) +
                                  " is out of range [0, " +
                                  std::to_string(CPU_SETSIZE) + ")");
  cpu_set_t set;
  CPU_ZERO(&set);
  for (int c : cores)
    CPU_SET(c, &set);
  if (sched_setaffinity(0, sizeof(set), &set) != 0)
    affinity_warn("cpu_affinity requested but sched_setaffinity failed: " +
                  std::string(std::strerror(errno)) +
                  "; thread left unpinned (locked cpuset?)");
}

// Query the calling thread's current CPU affinity set as a sorted list of core
// ids. Returns an empty vector if the query fails.
inline std::vector<int> current_thread_cpuset() {
  std::vector<int> out;
  cpu_set_t set;
  CPU_ZERO(&set);
  if (sched_getaffinity(0, sizeof(set), &set) != 0)
    return out;
  for (int c = 0; c < CPU_SETSIZE; ++c)
    if (CPU_ISSET(c, &set))
      out.push_back(c);
  return out;
}

#else // non-Linux: no-ops

enum class mempolicy_mode { preferred, bind };
inline void
bind_this_thread_to_numa_node(int node,
                              mempolicy_mode = mempolicy_mode::preferred) {
  if (node < 0)
    return;
  static bool warned = false;
  if (!warned) {
    affinity_warn(
        "numa_node_id ignored: NUMA binding is only supported on Linux");
    warned = true;
  }
}
inline void
bind_region_to_numa_node(void *, std::size_t, int,
                         mempolicy_mode = mempolicy_mode::preferred) {}
inline int current_thread_mempolicy_mode() { return -1; }
inline void set_thread_cpu_affinity(const std::vector<int> &) {}
inline std::vector<int> current_thread_cpuset() { return {}; }

#endif

} // namespace cudaq::qec::detail_affinity
