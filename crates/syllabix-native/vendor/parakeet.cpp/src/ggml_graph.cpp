#include "ggml_graph.hpp"
#include "backend.hpp"
#include "backend_pool.hpp"
#include "common.hpp"
#include "ggml.h"
#include <atomic>
#include <memory>
#include <mutex>

namespace pk {

// Process-global compute-thread override. 0 == unset (use the default thread
// count). Atomic so it is safe to set from one thread and read from the compute
// path; the benchmark CLI sets it once before any inference.
static std::atomic<int> g_num_threads{0};

void set_num_threads(int n) { g_num_threads.store(n < 0 ? 0 : n, std::memory_order_relaxed); }
int  num_threads()          { return g_num_threads.load(std::memory_order_relaxed); }

namespace {
// Process-global persistent backend. Created on first use with the configured
// thread count and reused across every graph computation. Holding it here (vs.
// per-Model) keeps the migration minimal: the components call pk::run_graph,
// which routes through this single Backend, so the per-call ggml_init/ggml_free
// churn is eliminated without threading a Backend handle through every layer.
// Task 2+ will fuse graphs on top of the same Backend.
std::unique_ptr<Backend> g_backend;
int g_backend_threads = 0;
// Serializes access to the process-global Backend and guards its lazy creation.
// The shared Backend (one gallocr + pending-input list, not re-entrant) is not
// safe for concurrent compute, so run_graph serializes it. Callers that need
// real parallelism lease a backend from a pool (backend_pool.hpp); a thread
// with a lease never takes this lock for compute.
// Recursive: run_graph holds it while the build lambda runs, and a build lambda
// that touches a loader whose weights are not yet realized calls
// global_backend() (through ensure_weights_realized) on the same thread.
std::recursive_mutex g_backend_mutex;
} // namespace

// Default ggml compute-thread count when --threads is unset. On this class of
// many-core desktop CPUs (e.g. the 20-core box this was tuned on, and rf-detr's
// dual-CCD finding) using ALL cores REGRESSES throughput: a single-CCD-sized
// count parallelises the now-fused encoder graph while avoiding cross-CCD cache
// traffic and threadpool contention. Sweep on the fused 110m offline path
// (LibriSpeech): t=1 25.6, t=2 41.6, t=4 60.9, t=8 75.4, t=16 70.9, t=20 43.4
// RTFx -> 8 is the clear winner (+24% over 16, +74% over 20, +24% over the old
// default of 4). The unit tests pass an explicit per-call n_threads (4) and so
// are unaffected by this default.
constexpr int kDefaultThreads = 8;

int effective_threads() {
    const int g = g_num_threads.load(std::memory_order_relaxed);
    return g > 0 ? g : kDefaultThreads;
}

// Caller holds g_backend_mutex.
static Backend& global_backend_locked() {
    // Lazy create (reset-safe: shutdown_backend() can free it, and a later call
    // recreates it).
    if (!g_backend) {
        const int g = g_num_threads.load(std::memory_order_relaxed);
        const int n = g > 0 ? g : kDefaultThreads;
        g_backend = std::make_unique<Backend>(n);
        g_backend_threads = n;
    }
    // Honor a late --threads override: keep the live backend's thread count in
    // sync with the global override (the override is set once before inference).
    const int g = g_num_threads.load(std::memory_order_relaxed);
    if (g > 0 && g != g_backend_threads) {
        g_backend->set_n_threads(g);
        g_backend_threads = g;
    }
    return *g_backend;
}

Backend& global_backend() {
    // Taken for the lazy create and the late thread-count sync. Concurrent
    // requests on pooled backends still ask the global backend for its device
    // name, so this must be safe to call from several threads.
    std::lock_guard<std::recursive_mutex> lock(g_backend_mutex);
    return global_backend_locked();
}

int backend_thread_count() {
    std::lock_guard<std::recursive_mutex> lock(g_backend_mutex);
    return g_backend_threads;
}

void shutdown_backend() {
    // Free the process-global backend explicitly. Required for GPU backends: the
    // backend (and its gallocr's device buffer) must be released while the CUDA/
    // etc. driver is still alive. Relying on static destruction frees it during
    // process exit, AFTER the driver's atexit handler, which aborts with
    // "driver shutting down". Call from main() before returning (model objects,
    // which hold their own device weight buffers, must already be destroyed).
    std::lock_guard<std::recursive_mutex> lock(g_backend_mutex);
    g_backend.reset();
    g_backend_threads = 0;
}

bool run_graph(size_t /*mem_bytes*/, int n_threads,
               const std::function<ggml_tensor*(ggml_context*)>& build,
               std::vector<float>& out) {
    // A thread that leased a pool backend computes there: its own gallocr, its
    // own thread count (set when the pool was built), its own mutex. The global
    // thread override and the per-call n_threads do not apply.
    if (PoolSlot* slot = current_route()) {
        std::lock_guard<std::mutex> slot_lock(slot->mu);
        slot->runs.fetch_add(1, std::memory_order_relaxed);
        return slot->backend.compute(build, out);
    }
    std::lock_guard<std::recursive_mutex> lock(g_backend_mutex);
    Backend& be = global_backend_locked();
    // When no global override is set, honor the caller's per-call n_threads (the
    // historical behavior, used by the unit tests) for THIS call only. The
    // backend is process-global, so a count left behind would slow every later
    // graph: a Silero VAD pass (1 thread per tiny chunk) used to leave the ASR
    // decode single-threaded. The previous count is restored after the compute.
    // A positive global override already pinned the backend's thread count in
    // global_backend_locked() and wins over the per-call value.
    const int g = g_num_threads.load(std::memory_order_relaxed);
    const int prev_threads = g_backend_threads;
    const bool scoped = g <= 0 && n_threads > 0 && n_threads != prev_threads;
    if (scoped) {
        be.set_n_threads(n_threads);
        g_backend_threads = n_threads;
    }

    // Backend::compute builds the graph in a no_alloc context, allocates via the
    // persistent gallocr, pushes inputs AFTER alloc, computes, and reads back.
    struct Restore {
        Backend& be; bool on; int prev;
        ~Restore() { if (on) { be.set_n_threads(prev); g_backend_threads = prev; } }
    } restore{be, scoped, prev_threads};
    return be.compute(build, out);
}

} // namespace pk
