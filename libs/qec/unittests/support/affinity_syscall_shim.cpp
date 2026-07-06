/*******************************************************************************
 * Copyright (c) 2025 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
// Test-only LD_PRELOAD interposer: counts the placement-related syscalls so a
// test can assert a bound decode loop issues none (and an unbound one does).
// Counts four syscalls: sched_setaffinity / sched_getaffinity (glibc symbols)
// and SYS_set_mempolicy / SYS_get_mempolicy (issued through glibc's syscall(2)
// wrapper, so that wrapper is interposed too). Not linked into the library.
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <atomic>
#include <cerrno>
#include <cstdarg>
#include <cstdlib>
#include <dlfcn.h>
#include <sched.h>
#include <sys/syscall.h>
#include <unistd.h>

namespace {
std::atomic<long> g_setaff{0};
std::atomic<long> g_getaff{0};
std::atomic<long> g_setmem{0};
std::atomic<long> g_getmem{0};

// dlsym may itself invoke interposed functions; the thread-local flag breaks
// that recursion (the inner call sees "resolving" and reports ENOSYS instead
// of re-entering dlsym forever).
using syscall_fn = long (*)(long, long, long, long, long, long, long);
syscall_fn real_syscall_lazy() {
  static std::atomic<syscall_fn> cached{nullptr};
  syscall_fn fn = cached.load(std::memory_order_acquire);
  if (fn)
    return fn;
  static thread_local bool resolving = false;
  if (resolving)
    return nullptr;
  resolving = true;
  fn = reinterpret_cast<syscall_fn>(dlsym(RTLD_NEXT, "syscall"));
  resolving = false;
  if (fn)
    cached.store(fn, std::memory_order_release);
  return fn;
}
} // namespace

extern "C" {

// Interpose the glibc symbols; forward to the real ones via RTLD_NEXT.
int sched_setaffinity(pid_t pid, size_t cpusetsize, const cpu_set_t *mask) {
  static int (*real)(pid_t, size_t, const cpu_set_t *) = nullptr;
  if (!real)
    real = reinterpret_cast<int (*)(pid_t, size_t, const cpu_set_t *)>(
        dlsym(RTLD_NEXT, "sched_setaffinity"));
  g_setaff.fetch_add(1, std::memory_order_relaxed);
  return real(pid, cpusetsize, mask);
}

int sched_getaffinity(pid_t pid, size_t cpusetsize, cpu_set_t *mask) {
  static int (*real)(pid_t, size_t, cpu_set_t *) = nullptr;
  if (!real)
    real = reinterpret_cast<int (*)(pid_t, size_t, cpu_set_t *)>(
        dlsym(RTLD_NEXT, "sched_getaffinity"));
  g_getaff.fetch_add(1, std::memory_order_relaxed);
  return real(pid, cpusetsize, mask);
}

// The mempolicy calls have no glibc wrappers; the library reaches them through
// syscall(2), so interpose that. Counts ATTEMPTS (even ones seccomp rejects).
long syscall(long number, ...) {
  va_list ap;
  va_start(ap, number);
  long a0 = va_arg(ap, long), a1 = va_arg(ap, long), a2 = va_arg(ap, long);
  long a3 = va_arg(ap, long), a4 = va_arg(ap, long), a5 = va_arg(ap, long);
  va_end(ap);
  switch (number) {
  case SYS_set_mempolicy:
    g_setmem.fetch_add(1, std::memory_order_relaxed);
    break;
  case SYS_get_mempolicy:
    g_getmem.fetch_add(1, std::memory_order_relaxed);
    // Fault injection: simulate a seccomp profile that blocks get_mempolicy
    // while allowing set_mempolicy (the un-restorable-policy scenario).
    // getenv per call so a test can toggle it around a single decode.
    if (std::getenv("CUDAQX_SHIM_FAIL_GET_MEMPOLICY")) {
      errno = EPERM;
      return -1;
    }
    break;
  case SYS_sched_setaffinity:
    g_setaff.fetch_add(1, std::memory_order_relaxed);
    break;
  case SYS_sched_getaffinity:
    g_getaff.fetch_add(1, std::memory_order_relaxed);
    break;
  }
  syscall_fn real = real_syscall_lazy();
  if (!real) { // recursion during resolve; refuse rather than loop
    errno = ENOSYS;
    return -1;
  }
  return real(number, a0, a1, a2, a3, a4, a5);
}

// Read/reset hooks the test resolves (weakly) from the preloaded shim.
// Legacy pair kept for existing tests: count == sched_setaffinity count.
long cudaqx_affinity_syscall_count() {
  return g_setaff.load(std::memory_order_relaxed);
}
void cudaqx_affinity_syscall_reset() {
  g_setaff.store(0, std::memory_order_relaxed);
  g_getaff.store(0, std::memory_order_relaxed);
  g_setmem.store(0, std::memory_order_relaxed);
  g_getmem.store(0, std::memory_order_relaxed);
}
// Per-syscall counters.
long cudaqx_sched_setaffinity_count() {
  return g_setaff.load(std::memory_order_relaxed);
}
long cudaqx_sched_getaffinity_count() {
  return g_getaff.load(std::memory_order_relaxed);
}
long cudaqx_set_mempolicy_count() {
  return g_setmem.load(std::memory_order_relaxed);
}
long cudaqx_get_mempolicy_count() {
  return g_getmem.load(std::memory_order_relaxed);
}
}
