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

/// @brief Per-process realtime decoding session.
///
/// Owns the ring buffer (separate RX + TX physical backings, both shared
/// between dispatchers via `shared_ring_mode=1`), the HOST_LOOP CPU
/// dispatcher (for per-round GRAPH_LAUNCH enqueue RPCs), and the DEVICE_LOOP
/// persistent GPU dispatcher (for DEVICE_CALL get_corrections /
/// reset_decoder RPCs).
///
/// Ring layout: producer writes RPCHeader + payload into the RX backing,
/// dispatcher writes RPCResponse + result into the TX backing.  RX and TX
/// have separate physical allocations (matches the Hololink configuration
/// where `rx_data != tx_data`); both rings are still shared between the
/// device kernel and the host monitor.  Under `shared_ring_mode=1`, when
/// the device kernel sees an RX slot whose function_id resolves to a
/// GRAPH_LAUNCH entry it skips (does not drop) so the host monitor can
/// pick it up.  `skip_tx_markers=1` keeps the host monitor from writing
/// the in-flight TX sentinel; the captured graph writes the final
/// `tx_flag_value` directly (Hololink-compatible).
///
/// Constructed with a reference to a vector of realized decoder instances --
/// the same vector held by `realtime_decoding.cpp::g_decoders` in the
/// production path, or a one-element vector in the contract test.  The
/// session keeps a non-owning pointer to that vector so it can call
/// `release_decode_graph()` on each captured graph at finalize time.
///
/// Public surface:
///   - constructor takes a reference to the decoders vector
///   - `initialize()` -- captures graphs, allocates ring, builds function
///     table, starts both dispatchers.  Idempotent.
///   - `finalize()` -- reverse order.  Idempotent.  Must be called BEFORE
///     the decoders vector is cleared (the session calls
///     `release_decode_graph()` on each decoder it captured a graph for).
///   - accessors for `rpc_producer` to reach the ring + slot geometry.
///
/// Hardware floor: same as `nv-qldpc-decoder` (DEVICE_LOOP works on Pascal+ /
/// sm_60+; cooperative-grid persistent kernels are not Hopper-specific).
///
/// The class is marked `default`-visible so its constructor / destructor /
/// `initialize` / `finalize` symbols cross the `cudaq-qec-realtime-decoding`
/// shared-library boundary (the library is built with `-fvisibility=hidden`,
/// which would otherwise hide them).  Consumers are: this library's own
/// `rpc_producer.cpp` (accessor calls), the contract test
/// (`test_realtime_qldpc_graph_decoding.cpp` -- direct construction), and
/// `realtime_decoding.cpp::configure_decoders` (Step 9 -- destruction of
/// the global session unique_ptr).
class __attribute__((visibility("default"))) qec_realtime_session {
public:
  /// @brief Construct a session over the given realized decoders.
  /// @param decoders Reference must outlive this session.  The session calls
  ///                 `supports_graph_dispatch()` and `capture_decode_graph()`
  ///                 on each non-null entry at `initialize()` time, and
  ///                 `release_decode_graph()` at `finalize()` time.
  /// @param device_launch_fn Function pointer passed to
  ///                 `cudaq_dispatcher_set_launch_fn`.  Typically
  ///                 `&cudaq_launch_dispatch_kernel_regular` from libcudaq-
  ///                 realtime-dispatch.  Passed in (rather than referenced
  ///                 directly) so this shared library stays free of
  ///                 references to symbols that live in static archives
  ///                 only; the caller -- the final executable -- is the one
  ///                 that links libcudaq-realtime-dispatch.a, so the symbol
  ///                 resolves cleanly in that exe and is passed as a
  ///                 function pointer.
  qec_realtime_session(
      std::vector<std::unique_ptr<cudaq::qec::decoder>> &decoders,
      cudaq_dispatch_launch_fn_t device_launch_fn);

  ~qec_realtime_session();

  qec_realtime_session(const qec_realtime_session &) = delete;
  qec_realtime_session &operator=(const qec_realtime_session &) = delete;
  qec_realtime_session(qec_realtime_session &&) = delete;
  qec_realtime_session &operator=(qec_realtime_session &&) = delete;

  /// @brief Bring up the shared ring + both dispatchers.  Idempotent: a
  /// second call is a no-op.  Throws `std::runtime_error` on any failure
  /// (decoder lacks graph dispatch, duplicate enqueue function_id, CUDA
  /// allocation failure, plugin failed to populate device entries, ...).
  void initialize();

  /// @brief Tear down both dispatchers, release captured graphs, free ring.
  /// Idempotent.  Safe to call from a destructor.
  void finalize();

  /// @brief True if `initialize()` has completed and `finalize()` has not.
  bool initialized() const { return initialized_; }

  // ---- Accessors used by rpc_producer.cpp (and by the contract test). ----

  /// @brief Pinned-mapped host pointer to the rx flag array (num_slots()
  /// uint64_t entries).  Producer writes the DEVICE-visible RX slot address
  /// here to publish a request.
  volatile std::uint64_t *rx_flags_host() const { return rx_flags_host_; }

  /// @brief Pinned-mapped host pointer to the tx flag array.  Producer reads
  /// for back-pressure (slot is free when both rx_flags[i]==0 and
  /// tx_flags[i]==0); the consumer (GPU) is the writer.
  volatile std::uint64_t *tx_flags_host() const { return tx_flags_host_; }

  /// @brief Pinned-mapped host pointer to the RX data region
  /// (`num_slots() * slot_size()` bytes).  Producer writes RPCHeader +
  /// payload into `rx_data_host()[slot * slot_size()]`.  Separate physical
  /// backing from `tx_data_host()`; the RX and TX rings have the same
  /// slot stride but live at different addresses (matches the Hololink
  /// `rx_data != tx_data` configuration).
  std::uint8_t *rx_data_host() const { return rx_data_host_; }

  /// @brief Device-visible (UVA) pointer to the same RX backing as
  /// `rx_data_host()`.  Producer writes the device address into rx_flags
  /// so the GPU consumer can dereference directly.
  std::uint8_t *rx_data_dev() const { return rx_data_dev_; }

  /// @brief Pinned-mapped host pointer to the TX data region.  Dispatcher
  /// (host monitor or device kernel) writes RPCResponse + result bytes
  /// into `tx_data_host()[slot * slot_size()]`; producer polls the
  /// matching `tx_flags[slot]` then reads the result body from here.
  std::uint8_t *tx_data_host() const { return tx_data_host_; }

  /// @brief Device-visible (UVA) pointer to the same TX backing as
  /// `tx_data_host()`.  Wired into `ringbuffer_.tx_data` so both the
  /// device dispatcher and the host monitor's `GraphIOContext` mailbox
  /// see the same backing.
  std::uint8_t *tx_data_dev() const { return tx_data_dev_; }

  std::size_t num_slots() const { return num_slots_; }
  std::size_t slot_size() const { return slot_size_; }

  /// @brief Number of HOST_LOOP worker streams.  Always equals
  /// `num_decoders_with_graph()` today (one CUDA stream per decoder so
  /// per-decoder GRAPH_LAUNCH RPCs serialize on their owning stream).
  std::size_t num_host_workers() const { return host_workers_.size(); }

  /// @brief Get the CUDA stream backing the HOST_LOOP worker for the
  /// decoder at index `decoder_id` (the index into the configured
  /// `decoders_` vector -- not a packed worker slot).  The producer syncs
  /// this stream after an enqueue ACK to ensure the launched graph has
  /// actually retired before the caller treats the round as done
  /// (cudaDeviceSynchronize is unsafe -- it would also wait on the
  /// persistent DEVICE_LOOP dispatcher kernel, which doesn't exit until
  /// `finalize()` flips shutdown_flag).
  ///
  /// Returns nullptr if (a) `decoder_id` is out of range, (b) the session
  /// is not initialized, or (c) the corresponding decoder did not capture
  /// a graph (null entry or supports_graph_dispatch() returned false).
  /// Callers should treat a nullptr return as "no sync needed" -- the
  /// underlying decoder either doesn't exist or runs on the legacy
  /// direct-call path.
  cudaStream_t host_worker_stream(std::size_t decoder_id) const {
    return decoder_id < host_worker_streams_.size()
               ? host_worker_streams_[decoder_id]
               : nullptr;
  }

  /// @brief Number of decoders that successfully captured a CUDA graph at
  /// initialize() time -- this is also the number of GRAPH_LAUNCH entries
  /// at the front of the function table.
  std::size_t num_decoders_with_graph() const {
    return num_decoders_with_graph_;
  }

private:
  // Internal: walks decoders, captures each graph, registers the per-decoder
  // enqueue function_id, and appends a GRAPH_LAUNCH entry to function table
  // slot 0..(N-1).  Sets `num_decoders_with_graph_` to N.
  void capture_decoder_graphs();

  // Internal: builds the three (or N+2) entries of the shared function
  // table.  N GRAPH_LAUNCH entries from `capture_decoder_graphs()`, then
  // one DEVICE_CALL for `get_corrections` and one DEVICE_CALL for
  // `reset_decoder`.
  void populate_function_table();

  // Internal: computes slot_size_ from the trio's largest body across all
  // captured decoders, picks a stable default num_slots_, then allocates
  // rx/tx flags + the shared rx==tx data backing.
  void allocate_ring_buffer();

  // Internal: cudaq_dispatch_manager + dispatcher + start (DEVICE_LOOP).
  void start_device_loop();

  // Internal: cudaq_host_dispatch_loop_ctx_t + launch background thread
  // (HOST_LOOP).  Allocates one GraphIOContext per worker (pinned-mapped)
  // and wires host_ctx_.io_ctxs_host / .io_ctxs_dev so the captured
  // graph mailbox carries a GraphIOContext* with RX/TX slot pointers
  // (see [host_api.bs lines 1058-1083]).
  void start_host_loop();

  // Internal: signal shutdown, join host thread, stop device dispatcher.
  void stop_loops();

  // ---- References / external state ----
  std::vector<std::unique_ptr<cudaq::qec::decoder>> &decoders_;
  cudaq_dispatch_launch_fn_t device_launch_fn_ = nullptr;

  // ---- Lifetime ----
  bool initialized_ = false;

  // ---- Graph state ----
  // captured_graphs_[i] is the void* returned by decoders_[i]->capture_-
  // decode_graph() (or nullptr if that decoder was null or didn't support
  // graph dispatch and the session was relaxed -- today the latter throws).
  std::vector<void *> captured_graphs_;
  // Total non-null GRAPH_LAUNCH entries; equals captured_graphs_.size()
  // less any nulls.  Sets the GRAPH_LAUNCH count in the function table.
  std::size_t num_decoders_with_graph_ = 0;

  // ---- Ring buffer ----
  // RX and TX are separate physical pinned-mapped backings (the spec's
  // "rx_data != tx_data" Hololink-aligned design).  Both rings still
  // share the same slot stride and slot count; both rings are still
  // shared between the device dispatcher and the host monitor via
  // `shared_ring_mode=1`.
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

  // ---- Function table (pinned mapped: host + device same UVA) ----
  std::size_t function_table_count_ = 0;
  cudaq_function_entry_t *function_table_host_ = nullptr;
  cudaq_function_entry_t *function_table_dev_ = nullptr;
  // Cached for convenience (also stored in function_table_host_[1..2]).
  std::uint32_t get_corrections_fn_id_ = 0;
  std::uint32_t reset_decoder_fn_id_ = 0;

  // ---- DEVICE_LOOP wiring (public C API) ----
  cudaq_dispatch_manager_t *device_manager_ = nullptr;
  cudaq_dispatcher_t *device_dispatcher_ = nullptr;
  std::uint64_t *device_stats_dev_ = nullptr;
  // Pinned-mapped shutdown flag shared with both dispatchers.  Stored as a
  // plain int* (not volatile): the cuda-qx side never spin-reads it -- the
  // host dispatcher reads it as a cuda::std::atomic<int> with acquire
  // ordering, and the device kernel reads it through a volatile int* that the
  // device API re-qualifies -- so we only need to WRITE it correctly (atomic
  // release store + a full fence; see stop_loops()).
  int *shutdown_flag_host_ = nullptr;
  int *shutdown_flag_dev_ = nullptr;

  // ---- HOST_LOOP wiring (low-level ctx) ----
  cudaq_host_dispatch_loop_ctx_t host_ctx_{};
  std::vector<cudaq_host_dispatch_worker_t> host_workers_;
  std::vector<cudaStream_t> host_worker_streams_;
  std::uint64_t *host_idle_mask_storage_ = nullptr;
  std::uint64_t *host_live_dispatched_storage_ = nullptr;
  int *host_inflight_slot_tags_ = nullptr;
  std::uint64_t host_stats_counter_ = 0;
  std::thread host_loop_thread_;
  // Pinned-mapped GraphIOContext array (one per worker).  The host
  // monitor populates `[i]` before each `cudaGraphLaunch(workers[i])`
  // with rx_slot / tx_slot / tx_flag / tx_flag_value; the captured
  // graph reads it via the UVA device pointer (the mailbox stores
  // `&io_ctxs_dev_[i]`).  Lifetime tied to the session.
  cudaq::realtime::GraphIOContext *io_ctxs_host_ = nullptr;
  cudaq::realtime::GraphIOContext *io_ctxs_dev_ = nullptr;
};

} // namespace cudaq::qec::realtime

#endif // CUDAQ_REALTIME_ROOT
