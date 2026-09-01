#include "fdp_passthru.h"

#include <liburing.h>
#include <linux/nvme_ioctl.h>

#include <atomic>
#include <cerrno>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <time.h>
#include <unistd.h>
#include <vector>
#include <sys/syscall.h>

namespace proposal { namespace fdp {
namespace {

constexpr uint8_t kNvmeAdminIdentify = 0x06;
constexpr uint8_t kNvmeCmdFlush = 0x00;
constexpr uint8_t kNvmeCmdWrite = 0x01;
constexpr uint8_t kNvmeCmdRead = 0x02;
constexpr uint32_t kIdentifyCnsNamespace = 0x00;
constexpr uint32_t kIdentifyCnsController = 0x01;
constexpr uint32_t kNvmeIdentifyBytes = 4096;
constexpr uint32_t kCtrattFdpsBit = 1u << 19;
constexpr uint32_t kNvmeRwFua = 1u << 14;

uint32_t load_le32(const uint8_t* p) {
    return static_cast<uint32_t>(p[0]) |
           (static_cast<uint32_t>(p[1]) << 8) |
           (static_cast<uint32_t>(p[2]) << 16) |
           (static_cast<uint32_t>(p[3]) << 24);
}

uint64_t load_le64(const uint8_t* p) {
    return static_cast<uint64_t>(load_le32(p)) |
           (static_cast<uint64_t>(load_le32(p + 4)) << 32);
}

struct AlignedFree {
    void operator()(void* p) const noexcept { std::free(p); }
};

using AlignedPtr = std::unique_ptr<void, AlignedFree>;

AlignedPtr alloc_aligned(size_t alignment, size_t bytes) {
    void* p = nullptr;
    const int rc = ::posix_memalign(&p, alignment, bytes);
    if (rc != 0) {
        throw std::runtime_error(
            std::string("posix_memalign failed: ") + std::strerror(rc));
    }
    std::memset(p, 0, bytes);
    return AlignedPtr(p);
}

std::string errno_string(int e) {
    return std::strerror(e);
}

std::string neg_errno_string(int ret) {
    return std::strerror(-ret);
}

uint64_t env_u64_default(const char* name, uint64_t fallback) {
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') return fallback;
    errno = 0;
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(value, &end, 0);
    if (errno != 0 || end == value || *end != '\0') {
        throw std::runtime_error(std::string("invalid integer env ") + name);
    }
    return static_cast<uint64_t>(parsed);
}

bool env_bool_default(const char* name, bool fallback) {
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') return fallback;
    if (std::strcmp(value, "1") == 0 || std::strcmp(value, "true") == 0 ||
        std::strcmp(value, "True") == 0 || std::strcmp(value, "yes") == 0) {
        return true;
    }
    if (std::strcmp(value, "0") == 0 || std::strcmp(value, "false") == 0 ||
        std::strcmp(value, "False") == 0 || std::strcmp(value, "no") == 0) {
        return false;
    }
    throw std::runtime_error(std::string("invalid boolean env ") + name);
}

uint64_t monotonic_raw_ns() {
    timespec ts{};
    if (::clock_gettime(CLOCK_MONOTONIC_RAW, &ts) != 0) {
        throw std::runtime_error("clock_gettime(CLOCK_MONOTONIC_RAW) failed");
    }
    return static_cast<uint64_t>(ts.tv_sec) * 1000000000ull +
           static_cast<uint64_t>(ts.tv_nsec);
}

void prepare_uring_cmd_sqe(io_uring_sqe* sqe, int fd, bool fixed_file,
                           uint32_t cmd_op, const nvme_uring_cmd& cmd,
                           uint64_t user_data) {
    std::memset(sqe, 0, sizeof(*sqe));
    io_uring_prep_rw(IORING_OP_URING_CMD, sqe, fixed_file ? 0 : fd,
                     nullptr, 0, 0);
    if (fixed_file) sqe->flags |= IOSQE_FIXED_FILE;
    sqe->cmd_op = cmd_op;
    sqe->user_data = user_data;
    std::memcpy(sqe->cmd, &cmd, sizeof(cmd));
}

}  // namespace

struct UcmdRing::Impl {
    io_uring ring{};
    bool initialized = false;
    bool batching = false;
    bool file_registered = false;
    bool use_completion_thread = false;
    bool use_sqpoll = false;
    bool blocking_completion = false;
    bool profile_runtime = false;
    std::atomic<bool> stop_completion{false};
    std::atomic<uint32_t> in_flight{0};
    std::atomic<int64_t> completion_tid{-1};
    std::atomic<uint64_t> completion_waits{0};
    std::atomic<uint64_t> completed_cqes{0};
    std::thread completion_thread;
    mutable std::mutex state_mutex;
    std::condition_variable state_cv;
    UcmdRingBatchStats active_stats;
    UcmdRingBatchStats last_stats;
    std::string first_error;
};

UcmdRing::UcmdRing()
    : impl_(new Impl()), queue_depth_(0), next_user_data_(1) {}

UcmdRing::~UcmdRing() {
    close();
    delete impl_;
}

void UcmdRing::init(uint32_t queue_depth, const std::string& name,
                    int registered_fd, bool completion_thread) {
    if (queue_depth == 0) {
        throw std::invalid_argument("UcmdRing::init: queue_depth must be > 0");
    }
    close();

    io_uring_params params{};
    params.flags = IORING_SETUP_SQE128 | IORING_SETUP_CQE32;
#ifdef IORING_SETUP_SUBMIT_ALL
    params.flags |= IORING_SETUP_SUBMIT_ALL;
#endif
#ifdef IORING_SETUP_CLAMP
    params.flags |= IORING_SETUP_CLAMP;
#endif
#ifdef IORING_SETUP_CQSIZE
    params.flags |= IORING_SETUP_CQSIZE;
    params.cq_entries = queue_depth * 4;
#endif
#ifdef IORING_SETUP_SINGLE_ISSUER
    if (completion_thread) params.flags |= IORING_SETUP_SINGLE_ISSUER;
#endif
#ifdef IORING_SETUP_NO_SQARRAY
    params.flags |= IORING_SETUP_NO_SQARRAY;
#endif
    const bool use_sqpoll = completion_thread && env_bool_default(
        "PROPOSAL_FDP_SQPOLL", true);
    if (use_sqpoll) {
#ifdef IORING_SETUP_SQPOLL
        params.flags |= IORING_SETUP_SQPOLL;
        params.sq_thread_idle = static_cast<uint32_t>(env_u64_default(
            "PROPOSAL_FDP_SQPOLL_IDLE_MS", 2000));
#else
        throw std::runtime_error("IORING_SETUP_SQPOLL is unavailable");
#endif
    }

    const int ret = io_uring_queue_init_params(queue_depth, &impl_->ring,
                                               &params);
    if (ret < 0) {
        throw std::runtime_error(
            "io_uring_queue_init_params(" + name + ") failed: " +
            neg_errno_string(ret));
    }
    impl_->initialized = true;
    queue_depth_ = queue_depth;
    name_ = name;

    io_uring_probe* probe = io_uring_get_probe_ring(&impl_->ring);
    if (probe == nullptr) {
        close();
        throw std::runtime_error("io_uring_get_probe_ring(" + name + ") failed");
    }
    const bool supported = io_uring_opcode_supported(
        probe, IORING_OP_URING_CMD);
    io_uring_free_probe(probe);
    if (!supported) {
        close();
        throw std::runtime_error(
            "io_uring ring " + name + " does not support URING_CMD");
    }

    if (registered_fd >= 0) {
        int fds[1] = {registered_fd};
        const int file_ret = io_uring_register_files(&impl_->ring, fds, 1);
        if (file_ret < 0) {
            close();
            throw std::runtime_error(
                "io_uring_register_files(" + name + ") failed: " +
                neg_errno_string(file_ret));
        }
        impl_->file_registered = true;
    }

    if (completion_thread) {
        impl_->use_completion_thread = true;
        impl_->use_sqpoll = use_sqpoll;
        impl_->blocking_completion = env_bool_default(
            "PROPOSAL_FDP_BLOCKING_COMPLETION", false);
        impl_->profile_runtime = env_bool_default(
            "PROPOSAL_FDP_PERF_DIAG", false);
        impl_->stop_completion.store(false, std::memory_order_release);
        impl_->completion_thread = std::thread(&UcmdRing::completion_loop, this);
    }
}

void UcmdRing::close() {
    if (impl_ == nullptr) return;
    if (impl_->use_completion_thread) {
        impl_->stop_completion.store(true, std::memory_order_release);
        impl_->state_cv.notify_all();
        if (impl_->completion_thread.joinable()) {
            impl_->completion_thread.join();
        }
    }
    if (impl_->initialized) {
        if (impl_->file_registered) {
            (void)io_uring_unregister_files(&impl_->ring);
        }
        io_uring_queue_exit(&impl_->ring);
    }
    impl_->ring = io_uring{};
    impl_->initialized = false;
    impl_->batching = false;
    impl_->file_registered = false;
    impl_->use_completion_thread = false;
    impl_->use_sqpoll = false;
    impl_->blocking_completion = false;
    impl_->profile_runtime = false;
    impl_->stop_completion.store(false, std::memory_order_release);
    impl_->in_flight.store(0, std::memory_order_release);
    impl_->completion_tid.store(-1, std::memory_order_release);
    impl_->completion_waits.store(0, std::memory_order_release);
    impl_->completed_cqes.store(0, std::memory_order_release);
    {
        std::lock_guard<std::mutex> lock(impl_->state_mutex);
        impl_->active_stats = UcmdRingBatchStats{};
        impl_->last_stats = UcmdRingBatchStats{};
        impl_->first_error.clear();
    }
    queue_depth_ = 0;
    name_.clear();
    next_user_data_ = 1;
}

bool UcmdRing::is_open() const {
    return impl_ != nullptr && impl_->initialized;
}

bool UcmdRing::is_batching() const {
    return impl_ != nullptr && impl_->batching;
}

bool UcmdRing::has_completion_thread() const {
    return impl_ != nullptr && impl_->use_completion_thread;
}

void UcmdRing::begin_batch() {
    if (!is_open() || !impl_->use_completion_thread) {
        throw std::runtime_error("UcmdRing::begin_batch requires a data ring");
    }
    if (impl_->batching) {
        throw std::runtime_error("UcmdRing::begin_batch: batch already open");
    }
    if (impl_->in_flight.load(std::memory_order_acquire) != 0 ||
        io_uring_sq_ready(&impl_->ring) != 0) {
        throw std::runtime_error("UcmdRing::begin_batch: pending commands exist");
    }
    std::lock_guard<std::mutex> lock(impl_->state_mutex);
    impl_->active_stats = UcmdRingBatchStats{};
    impl_->active_stats.queue_depth = queue_depth_;
    impl_->active_stats.sqpoll = impl_->use_sqpoll;
    impl_->active_stats.blocking_completion = impl_->blocking_completion;
    if (impl_->profile_runtime) {
        impl_->active_stats.batch_begin_mono_raw_ns = monotonic_raw_ns();
    }
    impl_->first_error.clear();
    impl_->batching = true;
}

uint32_t UcmdRing::end_batch() {
    if (!is_open() || !impl_->batching) {
        throw std::runtime_error("UcmdRing::end_batch: batch is not open");
    }
    std::unique_lock<std::mutex> lock(impl_->state_mutex);
    if (impl_->profile_runtime) {
        impl_->active_stats.end_wait_begin_mono_raw_ns = monotonic_raw_ns();
        impl_->active_stats.last_submit_mono_raw_ns =
            impl_->active_stats.end_wait_begin_mono_raw_ns;
    }
    impl_->state_cv.wait(lock, [&] {
        return impl_->active_stats.commands == impl_->active_stats.submitted &&
               impl_->in_flight.load(std::memory_order_acquire) == 0;
    });
    UcmdRingBatchStats& stats = impl_->active_stats;
    if (impl_->profile_runtime) {
        stats.batch_end_mono_raw_ns = monotonic_raw_ns();
    }
    impl_->last_stats = stats;
    const uint32_t completed = stats.commands;
    const std::string error = impl_->first_error;
    impl_->batching = false;
    lock.unlock();
    if (!error.empty()) {
        throw std::runtime_error("UcmdRing::end_batch(" + name_ + "): " + error);
    }
    return completed;
}

UcmdRingBatchStats UcmdRing::last_batch_stats() const {
    if (impl_ == nullptr) return UcmdRingBatchStats{};
    std::lock_guard<std::mutex> lock(impl_->state_mutex);
    return impl_->last_stats;
}

UcmdRingRuntimeStats UcmdRing::runtime_stats() const {
    UcmdRingRuntimeStats out;
    if (impl_ == nullptr) return out;
    out.name = name_;
    out.ring_fd = impl_->initialized ? impl_->ring.ring_fd : -1;
    out.completion_tid = impl_->completion_tid.load(std::memory_order_acquire);
    out.sqpoll = impl_->use_sqpoll;
    out.blocking_completion = impl_->blocking_completion;
    out.completion_waits = impl_->completion_waits.load(std::memory_order_relaxed);
    out.completed_cqes = impl_->completed_cqes.load(std::memory_order_relaxed);
    return out;
}

void UcmdRing::wait_for_sq_space() {
    while (io_uring_sq_space_left(&impl_->ring) == 0) {
        int ret = 0;
        if (impl_->use_completion_thread) {
            ret = io_uring_sqring_wait(&impl_->ring);
        }
        if (!impl_->use_completion_thread || ret == -EAGAIN ||
            ret == -EINTR || ret == -EINVAL) {
            std::this_thread::yield();
        } else if (ret < 0) {
            throw std::runtime_error("io_uring_sqring_wait(" + name_ +
                                     ") failed: " + neg_errno_string(ret));
        }
    }
}

void UcmdRing::submit_async(int fd, uint32_t cmd_op,
                            const nvme_uring_cmd& cmd, const char* where) {
    wait_for_sq_space();
    io_uring_sqe* sqe = io_uring_get_sqe(&impl_->ring);
    if (sqe == nullptr) {
        throw std::runtime_error(std::string(where) + ": no free SQE");
    }
    prepare_uring_cmd_sqe(sqe, fd, impl_->file_registered, cmd_op, cmd,
                          next_user_data_++);

    std::lock_guard<std::mutex> lock(impl_->state_mutex);
    int ret = 0;
    do {
        ret = io_uring_submit(&impl_->ring);
        ++impl_->active_stats.submit_calls;
        if (ret < 0) {
            throw std::runtime_error(std::string(where) +
                                     ": io_uring_submit failed: " +
                                     neg_errno_string(ret));
        }
        if (ret == 0) std::this_thread::yield();
    } while (ret == 0);

    if (impl_->profile_runtime && impl_->active_stats.submitted == 0) {
        impl_->active_stats.first_submit_mono_raw_ns = monotonic_raw_ns();
    }
    ++impl_->active_stats.submitted;
    if (impl_->file_registered) ++impl_->active_stats.registered_file_submits;
    impl_->in_flight.fetch_add(1, std::memory_order_release);
    if (impl_->blocking_completion) {
        impl_->state_cv.notify_all();
    }
}

void UcmdRing::submit_sync(int fd, uint32_t cmd_op,
                           const nvme_uring_cmd& cmd, const char* where) {
    wait_for_sq_space();
    io_uring_sqe* sqe = io_uring_get_sqe(&impl_->ring);
    if (sqe == nullptr) {
        throw std::runtime_error(std::string(where) + ": no free SQE");
    }
    prepare_uring_cmd_sqe(sqe, fd, impl_->file_registered, cmd_op, cmd,
                          next_user_data_++);
    const int submit_ret = io_uring_submit_and_wait(&impl_->ring, 1);
    if (submit_ret < 0) {
        throw std::runtime_error(std::string(where) +
                                 ": io_uring_submit_and_wait failed: " +
                                 neg_errno_string(submit_ret));
    }
    if (submit_ret == 0) {
        throw std::runtime_error(std::string(where) +
                                 ": io_uring submitted zero commands");
    }
    io_uring_cqe* cqe = nullptr;
    const int wait_ret = io_uring_wait_cqe(&impl_->ring, &cqe);
    if (wait_ret < 0 || cqe == nullptr) {
        throw std::runtime_error(std::string(where) +
                                 ": io_uring_wait_cqe failed");
    }
    const int result = cqe->res;
    io_uring_cqe_seen(&impl_->ring, cqe);
    if (result < 0) {
        throw std::runtime_error(std::string(where) +
                                 ": NVMe command failed: " +
                                 errno_string(-result));
    }
    if (result != 0) {
        throw std::runtime_error(std::string(where) +
                                 ": NVMe completion status " +
                                 std::to_string(result));
    }
}

void UcmdRing::submit_nvme(int fd, uint32_t cmd_op,
                           const nvme_uring_cmd& cmd, const char* where) {
    if (!is_open()) {
        throw std::runtime_error(std::string(where) + ": ring is not open");
    }
    if (!impl_->use_completion_thread) {
        submit_sync(fd, cmd_op, cmd, where);
        return;
    }
    const bool implicit_batch = !impl_->batching;
    if (implicit_batch) begin_batch();
    submit_async(fd, cmd_op, cmd, where);
    if (implicit_batch) (void)end_batch();
}

void UcmdRing::completion_loop() {
    impl_->completion_tid.store(
        static_cast<int64_t>(::syscall(SYS_gettid)),
        std::memory_order_release);
    std::vector<io_uring_cqe*> cqes(queue_depth_ * 4);
    while (!impl_->stop_completion.load(std::memory_order_acquire) ||
           impl_->in_flight.load(std::memory_order_acquire) != 0) {
        if (impl_->blocking_completion &&
            impl_->in_flight.load(std::memory_order_acquire) == 0) {
            std::unique_lock<std::mutex> idle_lock(impl_->state_mutex);
            impl_->state_cv.wait(idle_lock, [&] {
                return impl_->stop_completion.load(std::memory_order_acquire) ||
                       impl_->in_flight.load(std::memory_order_acquire) != 0;
            });
            continue;
        }
        if (impl_->blocking_completion) {
            io_uring_cqe* first = nullptr;
            if (impl_->profile_runtime) {
                impl_->completion_waits.fetch_add(1, std::memory_order_relaxed);
            }
            const int wait_ret = io_uring_wait_cqe(&impl_->ring, &first);
            if (wait_ret < 0) {
                std::lock_guard<std::mutex> lock(impl_->state_mutex);
                if (impl_->first_error.empty()) {
                    impl_->first_error = name_ +
                        ": io_uring_wait_cqe failed: " +
                        neg_errno_string(wait_ret);
                }
                impl_->active_stats.commands = impl_->active_stats.submitted;
                impl_->in_flight.store(0, std::memory_order_release);
                impl_->state_cv.notify_all();
                break;
            }
        }
        const uint32_t got = io_uring_peek_batch_cqe(
            &impl_->ring, cqes.data(), static_cast<unsigned>(cqes.size()));
        if (got == 0) {
            std::this_thread::yield();
            continue;
        }
        std::lock_guard<std::mutex> lock(impl_->state_mutex);
        const bool first_completion = impl_->active_stats.commands == 0;
        for (uint32_t i = 0; i < got; ++i) {
            io_uring_cqe* cqe = cqes[i];
            if (cqe == nullptr || cqe->user_data == 0) continue;
            ++impl_->active_stats.commands;
            impl_->in_flight.fetch_sub(1, std::memory_order_release);
            if (cqe->res < 0 && impl_->first_error.empty()) {
                impl_->first_error = name_ + ": NVMe command failed: " +
                                     errno_string(-cqe->res);
            } else if (cqe->res != 0 && impl_->first_error.empty()) {
                impl_->first_error = name_ + ": NVMe completion status " +
                                     std::to_string(cqe->res);
            }
        }
        if (impl_->profile_runtime) {
            if (first_completion) {
                impl_->active_stats.first_completion_mono_raw_ns =
                    monotonic_raw_ns();
            }
            if (
                impl_->active_stats.commands == impl_->active_stats.submitted &&
                impl_->in_flight.load(std::memory_order_acquire) == 0
            ) {
                impl_->active_stats.last_completion_mono_raw_ns =
                    monotonic_raw_ns();
            }
            impl_->completed_cqes.fetch_add(got, std::memory_order_relaxed);
        }
        io_uring_cq_advance(&impl_->ring, got);
        impl_->state_cv.notify_all();
    }
}

NvmeDevice::NvmeDevice() : fd_(-1) {}

NvmeDevice::~NvmeDevice() {
    close();
}

void NvmeDevice::open_device(const std::string& path) {
    close();
    if (path.empty()) {
        throw std::invalid_argument("PROPOSAL_FDP_DEVICE is empty");
    }
    fd_ = ::open(path.c_str(), O_RDWR | O_CLOEXEC);
    if (fd_ < 0) {
        throw std::runtime_error("open " + path + " failed: " +
                                 errno_string(errno));
    }
    path_ = path;
}

void NvmeDevice::close() {
    if (fd_ >= 0) {
        (void)::close(fd_);
        fd_ = -1;
    }
    path_.clear();
}

NamespaceInfo NvmeDevice::identify(UcmdRing& admin_ring,
                                   uint32_t expected_nsid) {
    if (!is_open()) {
        throw std::runtime_error("NvmeDevice::identify: device is not open");
    }
    if (expected_nsid == 0) {
        throw std::invalid_argument("PROPOSAL_FDP_NSID must be > 0");
    }
    const uint64_t env_lba_size = env_u64_default("PROPOSAL_FDP_LBA_SIZE", 0);
    const uint64_t env_namespace_nlb = env_u64_default(
        "PROPOSAL_FDP_NAMESPACE_NLB", 0);
    if (env_lba_size != 0 || env_namespace_nlb != 0) {
        if (env_lba_size == 0 || env_namespace_nlb == 0 ||
            env_lba_size > UINT32_MAX ||
            (env_lba_size & (env_lba_size - 1)) != 0) {
            throw std::runtime_error("invalid FDP namespace geometry env");
        }
        NamespaceInfo out;
        out.nsid = expected_nsid;
        out.lba_size = static_cast<uint32_t>(env_lba_size);
        out.namespace_nlb = env_namespace_nlb;
        out.fdp_supported = env_bool_default(
            "PROPOSAL_FDP_ASSUME_FDP_SUPPORTED", true);
        out.fdp_pid_enabled = env_bool_default(
            "PROPOSAL_FDP_ASSUME_FDP_PID_ENABLED", false);
        return out;
    }

    AlignedPtr ns_buf = alloc_aligned(4096, kNvmeIdentifyBytes);
    nvme_uring_cmd cmd{};
    cmd.opcode = kNvmeAdminIdentify;
    cmd.nsid = expected_nsid;
    cmd.addr = reinterpret_cast<uint64_t>(ns_buf.get());
    cmd.data_len = kNvmeIdentifyBytes;
    cmd.cdw10 = kIdentifyCnsNamespace;
    admin_ring.submit_nvme(fd_, NVME_URING_CMD_ADMIN, cmd,
                           "identify_namespace");

    const uint8_t* ns = static_cast<const uint8_t*>(ns_buf.get());
    const uint64_t nsze = load_le64(ns);
    const uint8_t flbas = ns[26];
    const uint8_t lbaf_index =
        static_cast<uint8_t>((flbas & 0x0f) | ((flbas & 0x60) >> 1));
    const size_t lbaf_off = 128 + static_cast<size_t>(lbaf_index) * 4;
    const uint8_t lbads = ns[lbaf_off + 2];
    if (lbaf_index >= 64 || lbads < 9 || lbads > 20 || nsze == 0) {
        throw std::runtime_error("identify_namespace returned invalid geometry");
    }

    AlignedPtr ctrl_buf = alloc_aligned(4096, kNvmeIdentifyBytes);
    cmd = nvme_uring_cmd{};
    cmd.opcode = kNvmeAdminIdentify;
    cmd.addr = reinterpret_cast<uint64_t>(ctrl_buf.get());
    cmd.data_len = kNvmeIdentifyBytes;
    cmd.cdw10 = kIdentifyCnsController;
    admin_ring.submit_nvme(fd_, NVME_URING_CMD_ADMIN, cmd,
                           "identify_controller");

    NamespaceInfo out;
    out.nsid = expected_nsid;
    out.lba_size = 1u << lbads;
    out.namespace_nlb = nsze;
    out.fdp_supported =
        (load_le32(static_cast<const uint8_t*>(ctrl_buf.get()) + 96) &
         kCtrattFdpsBit) != 0;
    return out;
}

void validate_aligned_buffer(const void* buffer, uint64_t bytes,
                             uint32_t lba_size, const char* where) {
    if (buffer == nullptr || lba_size == 0 ||
        (lba_size & (lba_size - 1)) != 0) {
        throw std::invalid_argument(std::string(where) + ": invalid buffer");
    }
    if ((reinterpret_cast<uintptr_t>(buffer) & 4095u) != 0) {
        throw std::invalid_argument(std::string(where) +
                                    ": buffer is not 4KiB-aligned");
    }
    if (bytes == 0 || (bytes % lba_size) != 0) {
        throw std::invalid_argument(std::string(where) +
                                    ": byte length is not LBA-aligned");
    }
}

void NvmeDevice::read_lba(UcmdRing& ring, uint32_t nsid, void* buffer,
                          uint64_t slba, uint32_t nlb,
                          uint32_t lba_size) {
    if (nlb == 0 || nlb > 65536u) {
        throw std::invalid_argument("read_lba: invalid nlb");
    }
    const uint64_t bytes = static_cast<uint64_t>(nlb) * lba_size;
    validate_aligned_buffer(buffer, bytes, lba_size, "read_lba");
    nvme_uring_cmd cmd{};
    cmd.opcode = kNvmeCmdRead;
    cmd.nsid = nsid;
    cmd.addr = reinterpret_cast<uint64_t>(buffer);
    cmd.data_len = static_cast<uint32_t>(bytes);
    cmd.cdw10 = static_cast<uint32_t>(slba);
    cmd.cdw11 = static_cast<uint32_t>(slba >> 32);
    cmd.cdw12 = (nlb - 1u) & 0xffffu;
    ring.submit_nvme(fd_, NVME_URING_CMD_IO, cmd, "read_lba");
}

void NvmeDevice::write_lba(UcmdRing& ring, uint32_t nsid,
                           const void* buffer, uint64_t slba, uint32_t nlb,
                           uint32_t lba_size, bool fua, uint16_t pid_hint) {
    (void)pid_hint;
    if (nlb == 0 || nlb > 65536u) {
        throw std::invalid_argument("write_lba: invalid nlb");
    }
    const uint64_t bytes = static_cast<uint64_t>(nlb) * lba_size;
    validate_aligned_buffer(buffer, bytes, lba_size, "write_lba");
    nvme_uring_cmd cmd{};
    cmd.opcode = kNvmeCmdWrite;
    cmd.nsid = nsid;
    cmd.addr = reinterpret_cast<uint64_t>(buffer);
    cmd.data_len = static_cast<uint32_t>(bytes);
    cmd.cdw10 = static_cast<uint32_t>(slba);
    cmd.cdw11 = static_cast<uint32_t>(slba >> 32);
    cmd.cdw12 = ((nlb - 1u) & 0xffffu) |
                (fua ? (kNvmeRwFua << 16) : 0u);
    ring.submit_nvme(fd_, NVME_URING_CMD_IO, cmd, "write_lba");
}

void NvmeDevice::flush(UcmdRing& ring, uint32_t nsid) {
    nvme_uring_cmd cmd{};
    cmd.opcode = kNvmeCmdFlush;
    cmd.nsid = nsid;
    ring.submit_nvme(fd_, NVME_URING_CMD_IO, cmd, "flush");
}

}}  // namespace proposal::fdp
