/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#ifdef CUDAQ_REALTIME_ROOT

#include "cudaq/qec/decoder.h"
#include "cudaq/realtime/daemon/dispatcher/cudaq_realtime.h"
#include "cudaq/realtime/daemon/dispatcher/dispatch_kernel_launch.h"
#include "cudaq/realtime/daemon/dispatcher/host_dispatcher.h"

#include <cstdint>
#include <memory>
#include <thread>
#include <vector>

namespace cudaq::qec::realtime {

/// @brief Per-process realtime decoding session, dual-mode.
///
/// A session is **homogeneous**: at `initialize()` it inspects the decoders and
/// runs one of two dispatch modes, chosen by
/// `decoder::supports_graph_dispatch()`:
///
///   - DEVICE mode (every decoder supports graph dispatch -- e.g. the Relay BP
///     GPU decoder).  This is the per-round `GRAPH_LAUNCH` enqueue + shared
///     `DEVICE_CALL` get_corrections / reset_decoder design: a persistent GPU
///     DEVICE_LOOP dispatcher services the DEVICE_CALL entries while a CPU
///     HOST_LOOP monitor launches the per-decoder captured graphs.  Both share
///     a pinned-mapped ring via `shared_ring_mode=1`.  Requires a non-null
///     `device_launch_fn`.
///
///   - HOST mode (no decoder supports graph dispatch -- e.g. PyMatching, a CPU
///     decoder).  All three RPCs (enqueue_syndromes, get_corrections,
///     reset_decoder) are `CUDAQ_DISPATCH_HOST_CALL` handlers invoked inline by
///     the CPU HOST_LOOP using the two-ring callback ABI
///     `cudaq_host_rpc_fn_t(const void *rx, void *tx, size_t)`.  The ring is
///     plain host memory (no GPU required at runtime); the device-visible
///     pointers alias the host backings so `rpc_producer` is mode-agnostic.
///
/// A *mixed* session (some decoders support graph dispatch and some do not) is
/// rejected: the host loop resolves a slot to a function table entry by
/// `function_id` alone, so a `GRAPH_LAUNCH` enqueue and a `HOST_CALL` enqueue
/// sharing `kEnqueueSyndromesFunctionId` would collide.  One decoder per
/// session is the supported (and tested) configuration; homogeneous multi-CPU
/// sessions also work.  See the throw in `classify_mode()`.
///
/// Ring layout (both modes): producer writes RPCHeader + payload into the RX
/// backing, the dispatcher writes RPCResponse + result into the TX backing.
/// RX and TX are separate physical allocations.
///
/// Constructed with a reference to a vector of realized decoder instances --
/// the same vector held by `realtime_decoding.cpp::g_decoders` in the
/// production path, or a one-element vector in a unit test.  The session keeps
/// a non-owning reference so it can call `release_decode_graph()` (DEVICE mode)
/// on each captured graph at finalize time.
///
/// The class is marked `default`-visible so its constructor / destructor /
/// `initialize` / `finalize` symbols cross the `cudaq-qec-realtime-decoding`
/// shared-library boundary (the library is built with `-fvisibility=hidden`).
class __attribute__((visibility("default"))) qec_realtime_session {
public:
  /// @brief Construct a session over the given realized decoders.
  /// @param decoders Reference must outlive this session.  The session calls
  ///                 `supports_graph_dispatch()` on each non-null entry at
  ///                 `initialize()` time to choose its dispatch mode, and (in
  ///                 DEVICE mode) `capture_decode_graph()` /
  ///                 `release_decode_graph()`.
  /// @param device_launch_fn Function pointer passed to
  ///                 `cudaq_dispatcher_set_launch_fn` in DEVICE mode. Typically
  ///                 `&cudaq_launch_dispatch_kernel_regular` from libcudaq-
  ///                 realtime-dispatch.  Passed in (rather than referenced
  ///                 directly) so this shared library stays free of references
  ///                 to symbols that live only in static archives linked by the
  ///                 final executable.  May be null for a HOST-mode (CPU
  ///                 decoder) session; `initialize()` throws if a DEVICE-mode
  ///                 decoder set is given without it.
  explicit qec_realtime_session(
      std::vector<std::unique_ptr<cudaq::qec::decoder>> &decoders,
      cudaq_dispatch_launch_fn_t device_launch_fn = nullptr);

  ~qec_realtime_session();

  qec_realtime_session(const qec_realtime_session &) = delete;
  qec_realtime_session &operator=(const qec_realtime_session &) = delete;
  qec_realtime_session(qec_realtime_session &&) = delete;
  qec_realtime_session &operator=(qec_realtime_session &&) = delete;

  /// @brief Bring up the ring + dispatcher(s) for the selected mode.
  /// Idempotent: a second call is a no-op.  Throws `std::runtime_error` on any
  /// failure (mixed decoder set, DEVICE mode without device_launch_fn, decoder
  /// lacks graph dispatch in DEVICE mode, CUDA allocation failure, plugin
  /// failed to populate device entries, a second concurrent HOST-mode session,
  /// ...).
  void initialize();

  /// @brief Tear down dispatcher(s), release captured graphs (DEVICE mode),
  /// free ring.  Idempotent.  Safe to call from a destructor.  In DEVICE mode
  /// must be called BEFORE the decoders vector is cleared.
  void finalize();

  /// @brief True if `initialize()` has completed and `finalize()` has not.
  bool initialized() const { return initialized_; }

  /// @brief True if this session runs in DEVICE (GPU graph-dispatch) mode.
  /// Only meaningful after `initialize()`.
  bool device_mode() const { return device_mode_; }

  // ---- Accessors used by rpc_producer.cpp (and by tests). ----------------
  // In HOST mode the `_dev` pointers alias the `_host` backings (host memory),
  // so the producer's address-as-flag publish works unchanged in both modes.

  volatile std::uint64_t *rx_flags_host() const { return rx_flags_host_; }
  volatile std::uint64_t *tx_flags_host() const { return tx_flags_host_; }
  std::uint8_t *rx_data_host() const { return rx_data_host_; }
  std::uint8_t *rx_data_dev() const { return rx_data_dev_; }
  std::uint8_t *tx_data_host() const { return tx_data_host_; }
  std::uint8_t *tx_data_dev() const { return tx_data_dev_; }

  std::size_t num_slots() const { return num_slots_; }
  std::size_t slot_size() const { return slot_size_; }

  /// @brief (DEVICE mode) CUDA stream backing the HOST_LOOP worker for the
  /// decoder at index `decoder_id`.  Returns nullptr in HOST mode, out of
  /// range, or for a decoder that did not capture a graph.
  cudaStream_t host_worker_stream(std::size_t decoder_id) const {
    return decoder_id < host_worker_streams_.size()
               ? host_worker_streams_[decoder_id]
               : nullptr;
  }

  /// @brief (DEVICE mode) Number of decoders that captured a CUDA graph.
  std::size_t num_decoders_with_graph() const {
    return num_decoders_with_graph_;
  }

private:
  // Inspect decoders_ and set device_mode_.  Throws on an empty set, a mixed
  // (graph + non-graph) set, or a DEVICE-mode set without device_launch_fn_.
  void classify_mode();

  // ---- DEVICE-mode internals ----
  void capture_decoder_graphs();
  void start_device_loop();

  // ---- shared internals (branch on device_mode_) ----
  // allocate_ring_buffer() computes slot_size_, allocates rx/tx flags + data
  // (pinned-mapped in DEVICE mode, host memory in HOST mode), and fully
  // populates ringbuffer_.
  void allocate_ring_buffer();
  // populate_function_table() builds the shared function table: N GRAPH_LAUNCH
  // + 2 DEVICE_CALL entries in DEVICE mode; 3 HOST_CALL entries in HOST mode.
  void populate_function_table();
  // start_host_loop() launches the CPU HOST_LOOP thread.  In DEVICE mode it
  // wires the per-decoder graph workers + GraphIOContext mailbox; in HOST mode
  // it runs the inline HOST_CALL handlers (no worker pool).
  void start_host_loop();

  // Signal shutdown, join host thread, stop device dispatcher (if any), free
  // worker streams + per-worker storage.
  void stop_loops();

  // ---- References / external state ----
  std::vector<std::unique_ptr<cudaq::qec::decoder>> &decoders_;
  cudaq_dispatch_launch_fn_t device_launch_fn_ = nullptr;

  // ---- Lifetime / mode ----
  bool initialized_ = false;
  bool device_mode_ = false;

  // ---- Ring buffer (raw pointers; _dev aliases _host in HOST mode) ----
  static constexpr std::size_t kDefaultNumSlots = 8;
  std::size_t num_slots_ = kDefaultNumSlots;
  std::size_t slot_size_ = 0;
  volatile std::uint64_t *rx_flags_host_ = nullptr;
  volatile std::uint64_t *rx_flags_dev_ = nullptr;
  volatile std::uint64_t *tx_flags_host_ = nullptr;
  volatile std::uint64_t *tx_flags_dev_ = nullptr;
  std::uint8_t *rx_data_host_ = nullptr;
  std::uint8_t *rx_data_dev_ = nullptr;
  std::uint8_t *tx_data_host_ = nullptr;
  std::uint8_t *tx_data_dev_ = nullptr;
  cudaq_ringbuffer_t ringbuffer_{};

  // ---- Function table ----
  // DEVICE mode: pinned-mapped (host + device same UVA).  HOST mode: plain host
  // allocation (host_fn pointers are host code addresses); _dev aliases _host.
  std::size_t function_table_count_ = 0;
  cudaq_function_entry_t *function_table_host_ = nullptr;
  cudaq_function_entry_t *function_table_dev_ = nullptr;
  std::uint32_t get_corrections_fn_id_ = 0;
  std::uint32_t reset_decoder_fn_id_ = 0;

  // ---- DEVICE_LOOP wiring (DEVICE mode only) ----
  cudaq_dispatch_manager_t *device_manager_ = nullptr;
  cudaq_dispatcher_t *device_dispatcher_ = nullptr;
  std::uint64_t *device_stats_dev_ = nullptr;
  // Pinned-mapped shutdown flag shared with both dispatchers (DEVICE mode).
  int *shutdown_flag_host_ = nullptr;
  int *shutdown_flag_dev_ = nullptr;

  // ---- HOST_LOOP wiring (both modes) ----
  cudaq_host_dispatch_loop_ctx_t host_ctx_{};
  std::thread host_loop_thread_;
  std::uint64_t host_stats_counter_ = 0;
  // Plain (non-pinned) shutdown flag for HOST mode (no device kernel shares
  // it).
  int shutdown_flag_ = 0;

  // ---- DEVICE-mode HOST_LOOP graph workers ----
  std::vector<cudaq_host_dispatch_worker_t> host_workers_;
  std::vector<cudaStream_t> host_worker_streams_;
  std::uint64_t *host_idle_mask_storage_ = nullptr;
  std::uint64_t *host_live_dispatched_storage_ = nullptr;
  int *host_inflight_slot_tags_ = nullptr;
  cudaq::realtime::GraphIOContext *io_ctxs_host_ = nullptr;
  cudaq::realtime::GraphIOContext *io_ctxs_dev_ = nullptr;

  // ---- Graph state (DEVICE mode only) ----
  std::vector<void *> captured_graphs_;
  std::size_t num_decoders_with_graph_ = 0;
};

} // namespace cudaq::qec::realtime

#endif // CUDAQ_REALTIME_ROOT
