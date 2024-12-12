/*******************************************************************************
 * Copyright (c) 2024 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "plugin_exports.h"

extern "C" const char *get_exported_symbols() {
  static std::string exported_symbols;
  if (exported_symbols.empty()) {
    std::ostringstream oss;
    auto symbols = SymbolRegistry::instance().get_symbols();
    for (size_t i = 0; i < symbols.size(); ++i) {
      oss << symbols[i];
      if (i != symbols.size() - 1) {
        oss << ",";
      }
    }
    exported_symbols = oss.str();
  }
  return exported_symbols.c_str();
}