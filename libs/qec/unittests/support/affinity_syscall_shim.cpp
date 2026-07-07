/*******************************************************************************
 * Copyright (c) 2025 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
// Test-only LD_PRELOAD interposer: counts the placement-related syscalls so a
// test can assert a bound decode loop issues none (and an unbound one does).
// Counts five syscalls: sched_setaffinity / sched_getaffinity (glibc symbols)
// and SYS_set_mempolicy / SYS_get_mempolicy / SYS_mbind (issued through
// glibc's syscall(2) wrapper, so that wrapper is interposed too). Also counts
// opens of /sys/devices/system/node/* (the per-guard cpulist read) via the
// openat/openat64 glibc symbols. Not linked into the library.
//
// Fault injection (getenv per call, so a test can toggle around one decode):
//   CUDAQX_SHIM_FAIL_SETAFFINITY    sched_setaffinity  -> EPERM, -1
//   CUDAQX_SHIM_FAIL_GETAFFINITY    sched_getaffinity  -> EPERM, -1
//   CUDAQX_SHIM_FAIL_SET_MEMPOLICY  SYS_set_mempolicy  -> EPERM, -1
//   CUDAQX_SHIM_FAIL_GET_MEMPOLICY  SYS_get_mempolicy  -> EPERM, -1
//   CUDAQX_SHIM_FAIL_MBIND          SYS_mbind          -> EPERM, -1
//   CUDAQX_SHIM_FAIL_ALL_PLACEMENT  all of the above   -> EPERM, -1
// Every counter counts ATTEMPTS: the count is incremented before injection.
// The sysfs open counters have NO fault injection.
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <atomic>
#include <cerrno>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fcntl.h>
#include <sched.h>
#include <sys/syscall.h>
#include <unistd.h>

namespace {
std::atomic<long> g_setaff{0};
std::atomic<long> g_getaff{0};
std::atomic<long> g_setmem{0};
std::atomic<long> g_getmem{0};
std::atomic<long> g_mbind{0};
std::atomic<long> g_node_sysfs_open{0};

// True when the call must be failed: either its specific injection variable or
// the blanket CUDAQX_SHIM_FAIL_ALL_PLACEMENT is set. getenv per call so a test
// can set/unset around a single decode.
bool fail_injected(const char *specific) {
  return std::getenv(specific) != nullptr ||
         std::getenv("CUDAQX_SHIM_FAIL_ALL_PLACEMENT") != nullptr;
}

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

// openat's optional 4th argument exists only for these flags.
bool open_needs_mode(int flags) {
  if (flags & O_CREAT)
    return true;
#ifdef O_TMPFILE
  if ((flags & O_TMPFILE) == O_TMPFILE)
    return true;
#endif
  return false;
}

void count_node_sysfs_path(const char *pathname) {
  if (pathname && std::strstr(pathname, "/sys/devices/system/node/"))
    g_node_sysfs_open.fetch_add(1, std::memory_order_relaxed);
}
} // namespace

extern "C" {

// Interpose the glibc symbols; forward to the real ones via RTLD_NEXT.
// Counts first (attempts), then applies fault injection.
int sched_setaffinity(pid_t pid, size_t cpusetsize, const cpu_set_t *mask) {
  static int (*real)(pid_t, size_t, const cpu_set_t *) = nullptr;
  if (!real)
    real = reinterpret_cast<int (*)(pid_t, size_t, const cpu_set_t *)>(
        dlsym(RTLD_NEXT, "sched_setaffinity"));
  g_setaff.fetch_add(1, std::memory_order_relaxed);
  if (fail_injected("CUDAQX_SHIM_FAIL_SETAFFINITY")) {
    errno = EPERM;
    return -1;
  }
  return real(pid, cpusetsize, mask);
}

int sched_getaffinity(pid_t pid, size_t cpusetsize, cpu_set_t *mask) {
  static int (*real)(pid_t, size_t, cpu_set_t *) = nullptr;
  if (!real)
    real = reinterpret_cast<int (*)(pid_t, size_t, cpu_set_t *)>(
        dlsym(RTLD_NEXT, "sched_getaffinity"));
  g_getaff.fetch_add(1, std::memory_order_relaxed);
  if (fail_injected("CUDAQX_SHIM_FAIL_GETAFFINITY")) {
    errno = EPERM;
    return -1;
  }
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
    if (fail_injected("CUDAQX_SHIM_FAIL_SET_MEMPOLICY")) {
      errno = EPERM;
      return -1;
    }
    break;
  case SYS_get_mempolicy:
    g_getmem.fetch_add(1, std::memory_order_relaxed);
    // Fault injection: simulate a seccomp profile that blocks get_mempolicy
    // while allowing set_mempolicy (the un-restorable-policy scenario).
    if (fail_injected("CUDAQX_SHIM_FAIL_GET_MEMPOLICY")) {
      errno = EPERM;
      return -1;
    }
    break;
  case SYS_mbind:
    g_mbind.fetch_add(1, std::memory_order_relaxed);
    if (fail_injected("CUDAQX_SHIM_FAIL_MBIND")) {
      errno = EPERM;
      return -1;
    }
    break;
  case SYS_sched_setaffinity:
    g_setaff.fetch_add(1, std::memory_order_relaxed);
    if (fail_injected("CUDAQX_SHIM_FAIL_SETAFFINITY")) {
      errno = EPERM;
      return -1;
    }
    break;
  case SYS_sched_getaffinity:
    g_getaff.fetch_add(1, std::memory_order_relaxed);
    if (fail_injected("CUDAQX_SHIM_FAIL_GETAFFINITY")) {
      errno = EPERM;
      return -1;
    }
    break;
  }
  syscall_fn real = real_syscall_lazy();
  if (!real) { // recursion during resolve; refuse rather than loop
    errno = ENOSYS;
    return -1;
  }
  return real(number, a0, a1, a2, a3, a4, a5);
}

// Count opens of the node-topology sysfs tree (build_node_cpuset's per-guard
// /sys/devices/system/node/node<N>/cpulist read). On this platform libstdc++'s
// ifstream is stdio-based (__basic_file wraps FILE*), so the file is opened
// through glibc's fopen -- whose INTERNAL open bypasses every interposable
// open/openat symbol (measured: openat/openat64 alone counted 0, adding
// open/open64 still counted 0). fopen/fopen64 are therefore interposed too;
// the open* spellings are kept so the counter stays robust across libstdc++
// builds that call them directly. Pass-through only: NO fault injection.
FILE *fopen(const char *pathname, const char *mode) {
  static FILE *(*real)(const char *, const char *) = nullptr;
  if (!real)
    real = reinterpret_cast<FILE *(*)(const char *, const char *)>(
        dlsym(RTLD_NEXT, "fopen"));
  count_node_sysfs_path(pathname);
  return real(pathname, mode);
}

FILE *fopen64(const char *pathname, const char *mode) {
  static FILE *(*real)(const char *, const char *) = nullptr;
  if (!real)
    real = reinterpret_cast<FILE *(*)(const char *, const char *)>(
        dlsym(RTLD_NEXT, "fopen64"));
  count_node_sysfs_path(pathname);
  return real(pathname, mode);
}

int open(const char *pathname, int flags, ...) {
  static int (*real)(const char *, int, ...) = nullptr;
  if (!real)
    real = reinterpret_cast<int (*)(const char *, int, ...)>(
        dlsym(RTLD_NEXT, "open"));
  count_node_sysfs_path(pathname);
  if (open_needs_mode(flags)) {
    va_list ap;
    va_start(ap, flags);
    mode_t mode = va_arg(ap, mode_t);
    va_end(ap);
    return real(pathname, flags, mode);
  }
  return real(pathname, flags);
}

int open64(const char *pathname, int flags, ...) {
  static int (*real)(const char *, int, ...) = nullptr;
  if (!real)
    real = reinterpret_cast<int (*)(const char *, int, ...)>(
        dlsym(RTLD_NEXT, "open64"));
  count_node_sysfs_path(pathname);
  if (open_needs_mode(flags)) {
    va_list ap;
    va_start(ap, flags);
    mode_t mode = va_arg(ap, mode_t);
    va_end(ap);
    return real(pathname, flags, mode);
  }
  return real(pathname, flags);
}

int openat(int dirfd, const char *pathname, int flags, ...) {
  static int (*real)(int, const char *, int, ...) = nullptr;
  if (!real)
    real = reinterpret_cast<int (*)(int, const char *, int, ...)>(
        dlsym(RTLD_NEXT, "openat"));
  count_node_sysfs_path(pathname);
  if (open_needs_mode(flags)) {
    va_list ap;
    va_start(ap, flags);
    mode_t mode = va_arg(ap, mode_t);
    va_end(ap);
    return real(dirfd, pathname, flags, mode);
  }
  return real(dirfd, pathname, flags);
}

int openat64(int dirfd, const char *pathname, int flags, ...) {
  static int (*real)(int, const char *, int, ...) = nullptr;
  if (!real)
    real = reinterpret_cast<int (*)(int, const char *, int, ...)>(
        dlsym(RTLD_NEXT, "openat64"));
  count_node_sysfs_path(pathname);
  if (open_needs_mode(flags)) {
    va_list ap;
    va_start(ap, flags);
    mode_t mode = va_arg(ap, mode_t);
    va_end(ap);
    return real(dirfd, pathname, flags, mode);
  }
  return real(dirfd, pathname, flags);
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
  g_mbind.store(0, std::memory_order_relaxed);
  g_node_sysfs_open.store(0, std::memory_order_relaxed);
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
long cudaqx_mbind_count() { return g_mbind.load(std::memory_order_relaxed); }
long cudaqx_node_sysfs_open_count() {
  return g_node_sysfs_open.load(std::memory_order_relaxed);
}
}
