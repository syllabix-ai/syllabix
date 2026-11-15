#pragma once
#include "backend.hpp"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

namespace pk {

// One pooled CPU backend: its own ggml backend, graph allocator and pending
// input list (all inside Backend), plus a mutex that serializes compute on it.
struct PoolSlot {
    Backend backend;
    std::mutex mu;
    std::atomic<uint64_t> runs{0};
    PoolSlot(int n_threads) : backend(n_threads, /*cpu_only=*/true) {}
};

// A fixed set of K CPU backends that concurrent requests borrow one at a time.
//
// The library starts no worker threads for the pool. Callers keep their own
// threads: a request takes a PoolLease, which sets a thread-local route so that
// pk::run_graph on that thread uses the leased backend. When all K backends
// are leased, the next caller waits.
//
// Each backend has its own ggml thread team of `threads_each` threads. ggml's
// spin barriers degrade when the machine is oversubscribed, so
// backends * threads_each should not exceed the physical core count.
//
// The pool never owns weights. Weights are realized once per loader on the
// process-global backend; a CPU weight buffer is valid for every CPU backend.
class BackendPool {
public:
    BackendPool(int backends, int threads_each);
    // Blocks until every leased backend has been returned.
    ~BackendPool();

    BackendPool(const BackendPool&) = delete;
    BackendPool& operator=(const BackendPool&) = delete;

    int size() const { return (int)slots_.size(); }
    int threads_each() const { return threads_each_; }

    // Bookkeeping, safe from any thread.
    int in_use() const;
    int peak_in_use() const;
    uint64_t total_runs() const;
    // Sum of the graph allocator buffer sizes of all backends (bytes). This is
    // the extra memory the pool costs; size K with it.
    size_t working_set_bytes() const;

private:
    friend class PoolLease;
    PoolSlot* acquire();
    void release(PoolSlot* s);

    int threads_each_;
    std::vector<std::unique_ptr<PoolSlot>> slots_;
    std::vector<PoolSlot*> free_;
    mutable std::mutex mu_;
    std::condition_variable cv_;
    int in_use_ = 0;
    int peak_ = 0;
};

// RAII borrow of one backend for the calling thread. While it lives,
// pk::run_graph on this thread runs on the leased backend. It is reentrant: if
// the thread already holds a lease (from any pool), the new lease is inert and
// the existing route stays, so a request that calls another leased entry point
// cannot deadlock. A null pool gives an inert lease (the global backend is
// used, as without a pool). The lease keeps the pool alive.
class PoolLease {
public:
    explicit PoolLease(std::shared_ptr<BackendPool> pool);
    ~PoolLease();
    PoolLease(const PoolLease&) = delete;
    PoolLease& operator=(const PoolLease&) = delete;
    bool active() const { return slot_ != nullptr; }

private:
    std::shared_ptr<BackendPool> pool_;
    PoolSlot* slot_ = nullptr;
};

// The pool backend this thread is routed to, or nullptr (use the global one).
PoolSlot* current_route();

} // namespace pk
