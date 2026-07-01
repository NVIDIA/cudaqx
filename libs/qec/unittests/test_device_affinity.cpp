/*******************************************************************************
 * Copyright (c) 2025 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cuda-qx/core/heterogeneous_map.h"
#include "cudaq/qec/device_affinity.h"
#include <cstdlib>
#include <gtest/gtest.h>

#if defined(__linux__)
#include "hardware_affinity.h"
#include <linux/mempolicy.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif
#include <vector>

using cudaq::qec::read_cuda_device_id;
using cudaq::qec::read_numa_node_id;
using cudaqx::heterogeneous_map;

TEST(DeviceAffinity, ReadAbsentReturnsMinusOne) {
  heterogeneous_map m;
  EXPECT_EQ(read_cuda_device_id(m), -1);
  EXPECT_EQ(read_numa_node_id(m), -1);
}

TEST(DeviceAffinity, ReadIntStorage) { // YAML path: std::optional<int> -> int
  heterogeneous_map m;
  m.insert("cuda_device_id", 3);
  m.insert("numa_node_id", 1);
  EXPECT_EQ(read_cuda_device_id(m), 3);
  EXPECT_EQ(read_numa_node_id(m), 1);
}

TEST(DeviceAffinity,
     ReadSizeTStorage) { // kwargs path: Python int -> std::size_t
  heterogeneous_map m;
  m.insert("cuda_device_id", std::size_t{2});
  EXPECT_EQ(read_cuda_device_id(m), 2);
}

TEST(DeviceAffinity, ReadNegativeThrows) {
  heterogeneous_map m;
  m.insert("cuda_device_id", -1);
  EXPECT_THROW(read_cuda_device_id(m), std::runtime_error);
}

#if defined(__linux__)
TEST(HardwareAffinity, MempolicyDefaultIsPreferredNotBind) {
  namespace da = cudaq::qec::detail_affinity;
  da::bind_this_thread_to_numa_node(0);
  EXPECT_EQ(da::current_thread_mempolicy_mode(), MPOL_PREFERRED);
  syscall(SYS_set_mempolicy, MPOL_DEFAULT, nullptr, 0UL);
}
TEST(HardwareAffinity, MempolicyBindWhenRequested) {
  namespace da = cudaq::qec::detail_affinity;
  da::bind_this_thread_to_numa_node(0, da::mempolicy_mode::bind);
  EXPECT_EQ(da::current_thread_mempolicy_mode(), MPOL_BIND);
  syscall(SYS_set_mempolicy, MPOL_DEFAULT, nullptr, 0UL);
}
TEST(HardwareAffinity, MempolicyRejectsUnknownStringThrows) {
  cudaqx::heterogeneous_map p;
  p.insert("mempolicy", std::string("bnid"));
  EXPECT_THROW(cudaq::qec::read_mempolicy(p), std::runtime_error);
}
TEST(HardwareAffinity, NumaNode64ThrowsOnMempolicyBind) {
  namespace da = cudaq::qec::detail_affinity;
  EXPECT_THROW(da::bind_this_thread_to_numa_node(64), std::runtime_error);
}
TEST(HardwareAffinity, CpuAffinityPinsToExactCores) {
  namespace da = cudaq::qec::detail_affinity;
  cpu_set_t saved;
  CPU_ZERO(&saved);
  sched_getaffinity(0, sizeof(saved), &saved);
  da::set_thread_cpu_affinity({0, 2});
  auto cpus = da::current_thread_cpuset();
  EXPECT_EQ(cpus, (std::vector<int>{0, 2}));
  sched_setaffinity(0, sizeof(saved), &saved);
}
TEST(HardwareAffinity, CpuAffinityEmptyIsNoop) {
  namespace da = cudaq::qec::detail_affinity;
  cpu_set_t before;
  CPU_ZERO(&before);
  sched_getaffinity(0, sizeof(before), &before);
  da::set_thread_cpu_affinity({});
  cpu_set_t after;
  CPU_ZERO(&after);
  sched_getaffinity(0, sizeof(after), &after);
  EXPECT_TRUE(CPU_EQUAL(&before, &after));
}
TEST(HardwareAffinity, CpuAffinityOutOfRangeCoreThrows) {
  namespace da = cudaq::qec::detail_affinity;
  EXPECT_THROW(da::set_thread_cpu_affinity({CPU_SETSIZE}),
               std::invalid_argument);
  EXPECT_THROW(da::set_thread_cpu_affinity({-1}), std::invalid_argument);
}
TEST(HardwareAffinity, BindThreadPinsAffinityToNodeCpus) {
  namespace da = cudaq::qec::detail_affinity;
  cpu_set_t node0;
  CPU_ZERO(&node0);
  if (!da::build_node_cpuset(0, node0))
    GTEST_SKIP() << "no node0 cpulist";
  cpu_set_t saved;
  CPU_ZERO(&saved);
  sched_getaffinity(0, sizeof(saved), &saved);
  da::bind_this_thread_to_numa_node(0);
  cpu_set_t have;
  CPU_ZERO(&have);
  sched_getaffinity(0, sizeof(have), &have);
  for (int c = 0; c < CPU_SETSIZE; ++c)
    if (CPU_ISSET(c, &have))
      EXPECT_TRUE(CPU_ISSET(c, &node0)) << "cpu " << c << " not on node 0";
  sched_setaffinity(0, sizeof(saved), &saved);
  syscall(SYS_set_mempolicy, MPOL_DEFAULT, nullptr, 0UL);
}
TEST(HardwareAffinity, NegativeNodeIsNoop) {
  namespace da = cudaq::qec::detail_affinity;
  cpu_set_t before;
  CPU_ZERO(&before);
  sched_getaffinity(0, sizeof(before), &before);
  da::bind_this_thread_to_numa_node(-1);
  cpu_set_t after;
  CPU_ZERO(&after);
  sched_getaffinity(0, sizeof(after), &after);
  EXPECT_TRUE(CPU_EQUAL(&before, &after));
  EXPECT_EQ(da::current_thread_mempolicy_mode(), MPOL_DEFAULT);
}
TEST(HardwareAffinity, BindRegionSetsPolicyOnBuffer) {
  namespace da = cudaq::qec::detail_affinity;
  const size_t bytes = 4096;
  void *p = std::calloc(1, bytes);
  ASSERT_NE(p, nullptr);
  da::bind_region_to_numa_node(p, bytes, 0); // preferred, node 0
  int mode = -1;
  long rc =
      syscall(SYS_get_mempolicy, &mode, nullptr, 0UL, p, 1UL /*MPOL_F_ADDR*/);
  if (rc == 0)
    EXPECT_TRUE(mode == MPOL_PREFERRED || mode == MPOL_DEFAULT)
        << "region policy after preferred-bind should be PREFERRED (or DEFAULT "
           "if unsupported)";
  da::bind_region_to_numa_node(p, bytes,
                               -1); // negative node -> no-op, no crash
  std::free(p);
}
TEST(HardwareAffinity, BindRegionNode64Throws) {
  namespace da = cudaq::qec::detail_affinity;
  void *p = std::calloc(1, 4096);
  ASSERT_NE(p, nullptr);
  EXPECT_THROW(da::bind_region_to_numa_node(p, 4096, 64), std::runtime_error);
  std::free(p);
}
#endif
