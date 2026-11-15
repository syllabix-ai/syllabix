#include "backend_pool.hpp"

#include <algorithm>

namespace pk {

namespace {
thread_local PoolSlot* t_route = nullptr;
} // namespace

PoolSlot* current_route() { return t_route; }

BackendPool::BackendPool(int backends, int threads_each)
    : threads_each_(std::max(1, threads_each)) {
    const int k = std::max(1, backends);
    slots_.reserve((size_t)k);
    for (int i = 0; i < k; ++i) {
        slots_.push_back(std::make_unique<PoolSlot>(threads_each_));
        free_.push_back(slots_.back().get());
    }
}

BackendPool::~BackendPool() {
    std::unique_lock<std::mutex> lk(mu_);
    cv_.wait(lk, [this] { return in_use_ == 0; });
}

PoolSlot* BackendPool::acquire() {
    std::unique_lock<std::mutex> lk(mu_);
    cv_.wait(lk, [this] { return !free_.empty(); });
    PoolSlot* s = free_.back();
    free_.pop_back();
    ++in_use_;
    peak_ = std::max(peak_, in_use_);
    return s;
}

void BackendPool::release(PoolSlot* s) {
    {
        std::lock_guard<std::mutex> lk(mu_);
        free_.push_back(s);
        --in_use_;
    }
    cv_.notify_all();
}

int BackendPool::in_use() const {
    std::lock_guard<std::mutex> lk(mu_);
    return in_use_;
}

int BackendPool::peak_in_use() const {
    std::lock_guard<std::mutex> lk(mu_);
    return peak_;
}

uint64_t BackendPool::total_runs() const {
    uint64_t n = 0;
    for (const auto& s : slots_) n += s->runs.load(std::memory_order_relaxed);
    return n;
}

size_t BackendPool::working_set_bytes() const {
    size_t n = 0;
    for (const auto& s : slots_) n += s->backend.graph_alloc_bytes();
    return n;
}

PoolLease::PoolLease(std::shared_ptr<BackendPool> pool) {
    if (!pool || t_route != nullptr) return;  // inert: no pool, or already routed
    pool_ = std::move(pool);
    slot_ = pool_->acquire();
    t_route = slot_;
}

PoolLease::~PoolLease() {
    if (!slot_) return;
    t_route = nullptr;
    pool_->release(slot_);
}

} // namespace pk
