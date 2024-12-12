/*******************************************************************************
 * Copyright (c) 2024 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#ifndef PLUGIN_EXPORTS_H
#define PLUGIN_EXPORTS_H

#include <iostream>
#include <sstream>
#include <string>
#include <vector>

/**
 * @brief Singleton class to store the list of symbols.
 * This class ensures that there is only one instance of the symbol registry
 * which can be accessed globally.
 */
class SymbolRegistry {
public:
  static SymbolRegistry &instance() {
    static SymbolRegistry registry;
    return registry;
  }

  void register_symbol(const std::string &symbol) { symbols.push_back(symbol); }

  std::vector<std::string> get_symbols() const { return symbols; }

private:
  std::vector<std::string> symbols;

  // Private constructor to prevent instantiation
  SymbolRegistry() = default;
  // Delete copy constructor and assignment operator to prevent copying.
  SymbolRegistry(const SymbolRegistry &) = delete;
  SymbolRegistry &operator=(const SymbolRegistry &) = delete;
};

/**
 * @brief Macro to register a symbol at compile-time.
 * @param symbol The symbol to be registered.
 */
#define REGISTER_DECODER(symbol)                                               \
  static bool symbol##_registered = []() {                                     \
    SymbolRegistry::instance().register_symbol(#symbol);                       \
    return true;                                                               \
  }()

/**
 * @brief Expose a C-style function to return a list of symbols.
 * This function returns a comma-separated list of the registered symbols.
 * @return A C-style string containing the list of registered symbols.
 */
extern "C" const char *get_exported_symbols();

#endif // PLUGIN_EXPORTS_H
