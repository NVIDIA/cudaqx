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

std::map<std::string, PluginHandle> &get_plugin_handles() {
  static std::map<std::string, PluginHandle> plugin_handles;
  return plugin_handles;
}

// Function to load plugins from a directory based on their type
void load_plugins(const std::string &plugin_dir, PluginType type) {
  for (const auto &entry : fs::directory_iterator(plugin_dir)) {
    if (entry.path().extension() == ".so") {

      void *handle = dlopen(entry.path().c_str(), RTLD_NOW);

      if (!handle) {
        std::cerr << "ERROR: Failed to load plugin: " << entry.path()
                  << " Error: " << dlerror() << std::endl;
      } else {
        get_plugin_handles().emplace(entry.path().filename().string(),
                                     PluginHandle{handle, type});
      }
    }
  }
}

// Function to clean up plugins based on their type
void cleanup_plugins(PluginType type) {
  for (auto it = get_plugin_handles().begin();
       it != get_plugin_handles().end();) {
    // Only erase if the type matches with the target
    if (it->second.type == type) {
      if (it->second.handle)
        dlclose(it->second.handle);
      // Erase from the map to avoid double dlclose()
      // Go the next iterator in the map
      it = get_plugin_handles().erase(it);
    } else {
      ++it; // Only increment if the item wasn't erased
    }
  }
}