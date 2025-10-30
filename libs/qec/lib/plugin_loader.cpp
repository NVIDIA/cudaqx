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

namespace cudaq::qec {

static std::map<std::string, PluginHandle> &get_plugin_handles() {
  // Formerly this code was like this:
  // static std::map<std::string, PluginHandle> plugin_handles;
  // return plugin_handles;
  // But that created a double free error when the program exited.
  // This was because the static destructor for the map was called first,
  // and then the cleanup_plugins function was called which tried to access the
  // already destroyed map. This new approach does create a small memory leak,
  // but it prevents the double free error.
  static auto *plugin_handles = new std::map<std::string, PluginHandle>();
  return *plugin_handles; // Dereference pointer to return reference
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
      void *raw_handle = dlopen(entry.path().c_str(), RTLD_NOW);
      if (raw_handle) {
        get_plugin_handles().emplace(
            entry.path().filename().string(),
            PluginHandle{std::unique_ptr<void, PluginDeleter>(raw_handle,
                                                              PluginDeleter()),
                         type});
      } else {
        std::cerr << "ERROR: Failed to load plugin: " << entry.path()
                  << " Error: " << dlerror() << std::endl;
      }
    }
  }
}

// Function to clean up the plugin handles
void cleanup_plugins(PluginType type) {
  auto &handles = get_plugin_handles();
  auto it = handles.begin();
  while (it != handles.end()) {
    if (it->second.type == type) {
      it = handles.erase(it); // dlclose is handled by the custom deleter
    } else {
      ++it;
    }
  }
}

} // namespace cudaq::qec
