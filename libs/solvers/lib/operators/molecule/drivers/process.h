/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <signal.h>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <utility>

namespace cudaqx {
/// @brief Launches a new process and returns its process ID.
///
/// This function creates a new process with the specified name.
/// It then returns the process ID of the newly created process.
///
/// @param processName The name of the process to launch. This should be a
///                    null-terminated string containing the path to the
///                    executable or a command name that can be found in the
///                    system's PATH.
///
/// @return The process ID (pid_t) of the newly created process on success,
///         or -1 on failure. Also returns any potential error message from the
///         launched process.
std::pair<pid_t, std::string> launchProcess(const char *processName);
} // namespace cudaqx
