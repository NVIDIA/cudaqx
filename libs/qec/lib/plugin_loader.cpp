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
#include <set>

namespace fs = std::filesystem;

static std::set<void *> closed_handles; // Track already-closed handles

static std::map<std::string, PluginHandle> &get_plugin_handles() {
  static std::map<std::string, PluginHandle> plugin_handles;
  return plugin_handles;
}

inline bool is_handle_closed(void *handle) {
  return closed_handles.find(handle) != closed_handles.end();
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
                                     PluginHandle{handle, type});
      }
    }
  }
}

void cleanup_plugins(PluginType type) {
  for (const auto &[key, plugin] : get_plugin_handles()) {
    if (plugin.type == type) {
      if (plugin.handle) {
        if (!is_handle_closed(plugin.handle)) {
          dlclose(plugin.handle);
          closed_handles.insert(plugin.handle);
        }
      } else {
        std::cerr << "WARNING: Invalid or null handle for plugin: " << key
                  << "\n";
      }
    }
  }
}
