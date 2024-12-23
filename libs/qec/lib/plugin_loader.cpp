/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/qec/plugin_loader.h"
#include <filesystem>
#include <iostream>

namespace fs = std::filesystem;

static std::map<std::string, PluginHandle> &get_plugin_handles() {
  static std::map<std::string, PluginHandle> plugin_handles;
  return plugin_handles;
}

// Function to load plugins from a directory based on their type
void load_plugins(const std::string &plugin_dir, PluginType type) {
  if (!fs::exists(plugin_dir)) {
    std::cerr << "WARNING: Plugin directory does not exist: " << plugin_dir
              << std::endl;
    return;
  }
  for (const auto &entry : fs::directory_iterator(plugin_dir)) {
    if (entry.path().extension() == ".so") {

      void *handle = dlopen(entry.path().c_str(), RTLD_NOW);

      if (!handle) {
        std::cerr << "ERROR: Failed to load plugin: " << entry.path()
                  << " Error: " << dlerror() << std::endl;
      } else {
        get_plugin_handles().emplace(entry.path().filename().string(),
                                     PluginHandle{handle, type, false});
      }
    }
  }
}

void cleanup_plugins(PluginType type) {
  for (auto &[key, plugin] : get_plugin_handles()) {
    if (plugin.type == type) {
      if (plugin.handle && !plugin.is_closed) {
        dlclose(plugin.handle);
        plugin.is_closed = true;
      } else {
        std::cerr << "WARNING: Invalid or null handle for plugin: " << key
                  << "\n";
      }
    }
  }
}
