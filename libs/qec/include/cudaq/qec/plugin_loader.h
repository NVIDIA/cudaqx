/*******************************************************************************
 * Copyright (c) 2024 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#ifndef PLUGIN_LOADER_H
#define PLUGIN_LOADER_H

#include <dlfcn.h>
#include <map>
#include <string>

/// @brief Enum to define different types of plugins
enum class PluginType {
  DECODER, // Decoder plugins
  CODE     // QEC codes plugins
           // Add other plugin types here as needed
};

/// @brief A struct to store plugin handle with its type
struct PluginHandle {
  void *handle; // Pointer to the shared library handle. This is the result of
                // dlopen() function.
  PluginType type; // Type of the plugin (e.g., decoder, code, etc)
};

/// @brief Function to load plugins from a directory based on type
/// @param plugin_dir The directory where the plugins are located
/// @param type The type of plugins to load. Only plugins of this type will be
/// loaded.
void load_plugins(const std::string &plugin_dir, PluginType type);

/// @brief Function to clean up loaded plugins of a specific type
/// @param type The type of plugins to clean up. Only plugins of this type will
/// be cleaned up.
void cleanup_plugins(PluginType type);

/// @brief Checks whether a plugin handle has already been closed
/// @param handle The handle of the plugin to check
/// @return Returns true if the handle has been closed, false otherwise
inline bool is_handle_closed(void *handle);

#endif // PLUGIN_LOADER_H
