// Test-only LD_PRELOAD interposer: counts sched_setaffinity calls so a test can
// assert a bound decode loop issues none. Not linked into the library.
#define _GNU_SOURCE
#include <atomic>
#include <dlfcn.h>
#include <sched.h>

namespace {
std::atomic<long> g_count{0};
}

extern "C" {

// Interpose the glibc symbol; forward to the real one via RTLD_NEXT.
int sched_setaffinity(pid_t pid, size_t cpusetsize, const cpu_set_t *mask) {
  static int (*real)(pid_t, size_t, const cpu_set_t *) = nullptr;
  if (!real)
    real = reinterpret_cast<int (*)(pid_t, size_t, const cpu_set_t *)>(
        dlsym(RTLD_NEXT, "sched_setaffinity"));
  g_count.fetch_add(1, std::memory_order_relaxed);
  return real(pid, cpusetsize, mask);
}

// Read/reset hooks the test resolves (weakly) from the preloaded shim.
long cudaqx_affinity_syscall_count() {
  return g_count.load(std::memory_order_relaxed);
}
void cudaqx_affinity_syscall_reset() {
  g_count.store(0, std::memory_order_relaxed);
}
}
