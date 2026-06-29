/*******************************************************************************
 * Copyright (c) 2025 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/qec/device_affinity.h"
#include <gtest/gtest.h>

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
