/*******************************************************************************
 * Copyright (c) 2024 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#ifndef DECODER_PLUGINS_LOADER_H
#define DECODER_PLUGINS_LOADER_H

#include <dlfcn.h>
#include <filesystem>
#include <iostream>

namespace fs = std::filesystem;

/// @brief Load all shared libraries (.so) from the specified directory.
/// @param directory The directory where the shared libraries are located.
bool load_plugins(const std::string &directory) {
  bool success = true;
  for (const auto &entry : fs::directory_iterator(directory)) {
    if (entry.path().extension() == ".so") {
      std::cout << "Loading plugin: " << entry.path() << std::endl;

      // Open the shared library
      void *handle = dlopen(entry.path().c_str(), RTLD_NOW);
      if (!handle) {
        std::cerr << "Failed to load plugin: " << entry.path()
                  << "Error: " << dlerror() << std::endl;
        success = false;
      } else {
        // Close the shared library
        dlclose(handle);
      }
    }
  }
  return success;
}

#endif // DECODER_PLUGINS_LOADER_H