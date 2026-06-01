/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "process.h"

#include <stdexcept>
#include <string>
#include <thread>

namespace cudaqx {

/// @brief RAII guard to unlink a temporary file on scope exit.
struct TempFileGuard {
  const char *path;
  TempFileGuard(const char *p) : path(p) {}
  ~TempFileGuard() {
    if (path)
      unlink(path);
  }
  // Non-copyable
  TempFileGuard(const TempFileGuard &) = delete;
  TempFileGuard &operator=(const TempFileGuard &) = delete;
};

std::pair<pid_t, std::string> launchProcess(const char *command) {
  // Create temporary files for storing stdout and stderr.
  // mkstemp atomically creates and opens the file, preventing symlink attacks.
  char tempStdout[] = "/tmp/stdout_XXXXXX";
  char tempStderr[] = "/tmp/stderr_XXXXXX";

  int fdOut = mkstemp(tempStdout);
  if (fdOut == -1) {
    throw std::runtime_error("Failed to create temporary stdout file");
  }

  int fdErr = mkstemp(tempStderr);
  if (fdErr == -1) {
    close(fdOut);
    unlink(tempStdout);
    throw std::runtime_error("Failed to create temporary stderr file");
  }

  // Close the FDs immediately — we only needed mkstemp for atomic file
  // creation. The shell redirect will reopen by filename, but since we created
  // the files atomically, no symlink attack is possible on the initial create.
  close(fdOut);
  close(fdErr);

  // Ensure temporary files are cleaned up on all exit paths.
  TempFileGuard guardOut(tempStdout);
  TempFileGuard guardErr(tempStderr);

  // Construct command to redirect both stdout and stderr to temporary files.
  std::string argString = std::string(command) + " 1>" + tempStdout + " 2>" +
                          tempStderr + " & echo $!";

  // Launch the process
  FILE *pipe = popen(argString.c_str(), "r");
  if (!pipe) {
    throw std::runtime_error("Error launching process: " +
                             std::string(command));
  }

  // Read PID
  char buffer[128];
  std::string pidStr;
  while (fgets(buffer, sizeof(buffer), pipe) != nullptr)
    pidStr += buffer;
  pclose(pipe);

  std::this_thread::sleep_for(std::chrono::milliseconds(100));

  // Read any error output
  std::string errorOutput;
  FILE *errorFile = fopen(tempStderr, "r");
  if (errorFile) {
    while (fgets(buffer, sizeof(buffer), errorFile) != nullptr) {
      errorOutput += buffer;
    }
    fclose(errorFile);
  }

  // Convert PID string to integer
  pid_t pid = 0;
  try {
    pid = std::stoi(pidStr);
  } catch (const std::exception &e) {
    throw std::runtime_error("Failed to get process ID: " + errorOutput);
  }

  return std::make_pair(pid, errorOutput);
}
} // namespace cudaqx
