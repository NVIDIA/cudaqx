/*******************************************************************************
 * Copyright (c) 2025 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include "cudaq/qec/decoder_pool.h"

#include <condition_variable>
#include <future>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <thread>

namespace cudaq::qec {

// One persistent, GPU/NUMA-pinned worker owning a single decoder and a job
// queue. The decoder is constructed and bound on this worker's own thread.
struct decoder_pool::worker {
  struct job {
    std::vector<std::vector<float_t>> syndromes; // owned copy (crosses threads)
    std::promise<std::vector<decoder_result>> result;
  };

  pool_decoder_spec spec;
  std::mutex m;
  std::condition_variable cv;
  bool stop = false;
  std::queue<std::unique_ptr<job>> jobs;
  std::thread thread;

  explicit worker(pool_decoder_spec s) : spec(std::move(s)) {
    thread = std::thread([this] { run(); });
  }
  ~worker() {
    {
      std::lock_guard<std::mutex> lk(m);
      stop = true;
    }
    cv.notify_all();
    if (thread.joinable())
      thread.join();
  }

  std::future<std::vector<decoder_result>>
  submit(std::vector<std::vector<float_t>> syndromes) {
    auto j = std::make_unique<job>();
    j->syndromes = std::move(syndromes);
    auto fut = j->result.get_future();
    {
      std::lock_guard<std::mutex> lk(m);
      jobs.push(std::move(j));
    }
    cv.notify_one();
    return fut;
  }

  void run() {
    // Construct + pin on this worker thread so resources land on the target
    // GPU.
    auto dec = decoder::get(spec.name, spec.H, spec.options);
    dec->bind_current_thread();
    for (;;) {
      std::unique_ptr<job> j;
      {
        std::unique_lock<std::mutex> lk(m);
        cv.wait(lk, [this] { return stop || !jobs.empty(); });
        if (stop && jobs.empty())
          return;
        j = std::move(jobs.front());
        jobs.pop();
      }
      try {
        j->result.set_value(dec->decode_batch(j->syndromes));
      } catch (...) {
        j->result.set_exception(std::current_exception());
      }
    }
  }
};

decoder_pool::decoder_pool(std::vector<pool_decoder_spec> specs) {
  workers_.reserve(specs.size());
  for (auto &s : specs) {
    int id = s.id;
    workers_.push_back(std::make_unique<worker>(std::move(s)));
    by_id_[id] = workers_.back().get();
  }
}

decoder_pool::~decoder_pool() = default; // worker dtors stop + join

std::unordered_map<int, std::vector<decoder_result>> decoder_pool::decode_all(
    const std::unordered_map<int, std::vector<std::vector<float_t>>> &work) {
  std::unordered_map<int, std::future<std::vector<decoder_result>>> futures;
  for (const auto &[id, syndromes] : work) {
    auto it = by_id_.find(id);
    if (it == by_id_.end())
      throw std::runtime_error("decoder_pool::decode_all: no decoder with id " +
                               std::to_string(id));
    futures.emplace(id, it->second->submit(syndromes));
  }
  std::unordered_map<int, std::vector<decoder_result>> results;
  for (auto &[id, fut] : futures)
    results.emplace(id, fut.get()); // rethrows any worker exception
  return results;
}

} // namespace cudaq::qec
