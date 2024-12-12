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
#include <memory>
#include <string>
#include <vector>

#include "cudaq.h"
#include "cudaq/qec/decoder.h"

namespace fs = std::filesystem;

// Type definition for the creator function
using CreatorFunction = std::unique_ptr<cudaq::qec::decoder> (*)(
    const cudaqx::tensor<uint8_t> &, const cudaqx::heterogeneous_map &);

/**
 * @brief Struct to manage loaded decoders.
 * This struct holds the name, handle, and creator function of a decoder plugin.
 */
struct DecoderPlugin {
  std::string name;
  void *handle;
  CreatorFunction creator;
};

/**
 * @brief Factory class to manage and create decoder plugins.
 * This class handles the loading of decoder plugins from shared libraries,
 * and provides functionality to create decoders and retrieve plugin names.
 */
class DecoderFactory {
private:
  std::vector<DecoderPlugin> plugins;

public:
  ~DecoderFactory() {
    for (auto &plugin : plugins) {
      if (plugin.handle) {
        dlclose(plugin.handle);
      }
    }
  }

  /**
   * @brief Load plugins from a specified directory.
   * This function scans the given directory for shared library files (.so)
   * and loads them as decoder plugins.
   * @param directory The directory to scan for plugins.
   */
  void load_plugins(const std::string &directory) {
    for (const auto &entry : fs::directory_iterator(directory)) {
      // scan the directory and load all .so files
      if (entry.path().extension() == ".so") {
        std::cout << "Loading plugin: " << entry.path() << "\n";

        void *handle = dlopen(entry.path().c_str(), RTLD_LAZY);
        if (!handle) {
          std::cerr << "Failed to load " << entry.path() << ": " << dlerror()
                    << "\n";
          continue;
        }

        dlerror(); // Clear errors
        using SymbolListFunction = const char *(*)();
        SymbolListFunction get_symbols = reinterpret_cast<SymbolListFunction>(
            dlsym(handle, "get_exported_symbols"));
        const char *dlsym_error = dlerror();
        if (dlsym_error) {
          std::cerr << "dlsym failed to get 'get_exported_symbols': "
                    << dlsym_error << "\n";
          dlclose(handle);
          continue;
        }

        // Extract the list of symbols and split by comma if there are multiple
        std::string symbols = get_symbols();
        std::vector<std::string> symbol_list;
        std::stringstream ss(symbols);
        std::string item;
        while (std::getline(ss, item, ',')) {
          symbol_list.push_back(item);
        }

        for (const auto &symbol : symbol_list) {
          std::cout << "Looking for function: " << symbol << "\n";
          CreatorFunction creator =
              reinterpret_cast<CreatorFunction>(dlsym(handle, symbol.c_str()));
          const char *dlsym_error = dlerror();
          if (!dlsym_error) {
            plugins.push_back({symbol, handle, creator});
            std::cout << "Successfully loaded symbol: " << symbol << "\n";
          } else {
            std::cerr << "dlsym failed to get symbol' " << symbol
                      << "': " << dlsym_error << "\n";
          }
        }
      }
    }
  }

  /**
   * @brief Create a decoder using a specified plugin.
   * This function creates a decoder by invoking the creator function of the
   * specified plugin.
   * @param plugin_name The name of the plugin to use.
   * @param H The tensor to pass to the creator function.
   * @param params The heterogeneous map to pass to the creator function.
   * @return A unique pointer to the created decoder.
   * @throws std::runtime_error if the plugin is not found.
   */
  std::unique_ptr<cudaq::qec::decoder>
  create_decoder(const std::string &plugin_name,
                 const cudaqx::tensor<uint8_t> &H,
                 const cudaqx::heterogeneous_map &params) {
    for (auto &plugin : plugins) {
      if (plugin.name == plugin_name) {
        return plugin.creator(H, params);
      }
    }
    throw std::runtime_error("Decoder " + plugin_name + " not found.");
  }

  /**
   * @brief Returns a vector of string of available decoders
   * @return A unique pointer to the created decoder.
   */
  std::vector<std::string> get_all_plugin_names() {
    std::vector<std::string> plugin_names;
    for (auto &plugin : plugins) {
      plugin_names.push_back(plugin.name);
    }
    return plugin_names;
  }
};

#endif // DECODER_PLUGINS_LOADER_H