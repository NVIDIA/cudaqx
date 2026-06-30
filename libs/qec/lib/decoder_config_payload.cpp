/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/qec/decoder_config_payload.h"

#include <utility>

namespace cudaq::qec {

namespace {
decoder_config_payload_publisher g_publisher;
} // namespace

void set_decoder_config_payload_publisher(
    decoder_config_payload_publisher publisher) {
  g_publisher = std::move(publisher);
}

void publish_decoder_config_payload(const std::string &yaml) {
  if (g_publisher)
    g_publisher(yaml);
}

} // namespace cudaq::qec
