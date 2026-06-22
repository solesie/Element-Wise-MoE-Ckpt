#include "fdp_passthru.h"

#include <liburing.h>
#include <linux/nvme_ioctl.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <sys/stat.h>
#include <unordered_map>
#include <unistd.h>
#include <vector>

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

uint16_t load_le16(const uint8_t* p) {
    return static_cast<uint16_t>(p[0]) |
           (static_cast<uint16_t>(p[1]) << 8);
}

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

bool env_bool_default(const char* name, bool default_value) {
    const char* v = std::getenv(name);
    if (v == nullptr || v[0] == '\0') return default_value;
    if (std::strcmp(v, "1") == 0 || std::strcmp(v, "true") == 0 ||
        std::strcmp(v, "True") == 0 || std::strcmp(v, "yes") == 0) {
        return true;
    }
    if (std::strcmp(v, "0") == 0 || std::strcmp(v, "false") == 0 ||
        std::strcmp(v, "False") == 0 || std::strcmp(v, "no") == 0) {
        return false;
    }
    throw std::runtime_error(std::string("invalid boolean env ") + name);
}

bool env_u64(const char* name, uint64_t* out) {
    const char* v = std::getenv(name);
    if (v == nullptr || v[0] == '\0') return false;
    errno = 0;
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(v, &end, 0);
    if (errno != 0 || end == v || *end != '\0') {
        throw std::runtime_error(std::string("invalid integer env ") +
                                 name + "=" + v);
    }
    *out = static_cast<uint64_t>(parsed);
    return true;
}

uint64_t steady_now_ns() {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());
}

void prepare_uring_cmd_sqe(struct io_uring_sqe* sqe, int fd, uint32_t cmd_op,
                           const nvme_uring_cmd& cmd, uint64_t user_data) {
    std::memset(sqe, 0, sizeof(*sqe));
    io_uring_prep_rw(IORING_OP_URING_CMD, sqe, fd, nullptr, 0, 0);
    sqe->cmd_op = cmd_op;
    sqe->user_data = user_data;
    std::memcpy(sqe->cmd, &cmd, sizeof(cmd));
}

}  // namespace

struct UcmdRing::Impl {
    io_uring ring;
    bool initialized = false;
    bool batching = false;
    bool track_where = false;
    uint32_t batch_completed = 0;
    uint32_t batch_submitted = 0;
    uint32_t in_flight = 0;
    uint64_t batch_first_submit_ns = 0;
    uint64_t batch_last_cqe_ns = 0;
    UcmdRingBatchStats last_batch_stats;
    std::vector<io_uring_cqe*> cqe_batch;
    std::unordered_map<uint64_t, std::string> pending_where;
    std::string first_error;
};

UcmdRing::UcmdRing()
    : impl_(new Impl()), queue_depth_(0), next_user_data_(1) {
    std::memset(&impl_->ring, 0, sizeof(impl_->ring));
}

UcmdRing::~UcmdRing() {
    close();
    delete impl_;
}

void UcmdRing::init(uint32_t queue_depth, const std::string& name) {
    if (queue_depth == 0) {
        throw std::invalid_argument("UcmdRing::init: queue_depth must be > 0");
    }
    close();
    io_uring_params params;
    std::memset(&params, 0, sizeof(params));
    params.flags = IORING_SETUP_SQE128 | IORING_SETUP_CQE32;
    const int ret = io_uring_queue_init_params(queue_depth, &impl_->ring,
                                               &params);
    if (ret < 0) {
        throw std::runtime_error(
            "io_uring_queue_init_params(" + name + ") failed: "
            + neg_errno_string(ret));
    }
    impl_->initialized = true;
    impl_->track_where = env_bool_default(
        "PROPOSAL_FDP_TRACK_CQE_WHERE", false);
    impl_->cqe_batch.resize(queue_depth);
    if (impl_->track_where) {
        impl_->pending_where.reserve(queue_depth);
    }
    queue_depth_ = queue_depth;
    name_ = name;

    io_uring_probe* probe = io_uring_get_probe_ring(&impl_->ring);
    if (probe == nullptr) {
        close();
        throw std::runtime_error(
            "io_uring_get_probe_ring(" + name + ") failed");
    }
    const bool supported = io_uring_opcode_supported(
        probe, IORING_OP_URING_CMD);
    io_uring_free_probe(probe);
    if (!supported) {
        close();
        throw std::runtime_error(
            "io_uring ring " + name + " does not support IORING_OP_URING_CMD");
    }
}

void UcmdRing::close() {
    if (impl_ != nullptr && impl_->initialized) {
        io_uring_queue_exit(&impl_->ring);
        std::memset(&impl_->ring, 0, sizeof(impl_->ring));
        impl_->initialized = false;
    }
    if (impl_ != nullptr) {
        impl_->batching = false;
        impl_->batch_completed = 0;
        impl_->batch_submitted = 0;
        impl_->in_flight = 0;
        impl_->batch_first_submit_ns = 0;
        impl_->batch_last_cqe_ns = 0;
        impl_->last_batch_stats = UcmdRingBatchStats{};
        impl_->track_where = false;
        impl_->cqe_batch.clear();
        impl_->pending_where.clear();
        impl_->first_error.clear();
    }
    queue_depth_ = 0;
    name_.clear();
}

bool UcmdRing::is_open() const {
    return impl_ != nullptr && impl_->initialized;
}

bool UcmdRing::is_batching() const {
    return impl_ != nullptr && impl_->batching;
}

void UcmdRing::begin_batch() {
    if (!is_open()) {
        throw std::runtime_error("UcmdRing::begin_batch: ring is not open");
    }
    if (impl_->batching) {
        throw std::runtime_error("UcmdRing::begin_batch: batch already open");
    }
    if (!impl_->pending_where.empty() || impl_->in_flight != 0 ||
        io_uring_sq_ready(&impl_->ring) != 0) {
        throw std::runtime_error("UcmdRing::begin_batch: pending commands exist");
    }
    impl_->batching = true;
    impl_->batch_completed = 0;
    impl_->batch_submitted = 0;
    impl_->in_flight = 0;
    impl_->batch_first_submit_ns = 0;
    impl_->batch_last_cqe_ns = 0;
    impl_->last_batch_stats = UcmdRingBatchStats{};
    impl_->pending_where.clear();
    if (impl_->track_where) {
        impl_->pending_where.reserve(queue_depth_);
    }
    impl_->first_error.clear();
}

uint32_t UcmdRing::drain_batch_fragment() {
    if (!is_open()) {
        throw std::runtime_error(
            "UcmdRing::drain_batch_fragment: ring is not open");
    }
    const uint32_t submitted = submit_ready_sqes("drain_batch_fragment");
    reap_available_cqes();
    if (submitted == 0 && impl_->in_flight > 0) {
        wait_one_cqe();
    }
    if (!impl_->first_error.empty()) {
        while (impl_->in_flight > 0) {
            wait_one_cqe();
        }
        const std::string error = impl_->first_error;
        impl_->pending_where.clear();
        impl_->first_error.clear();
        throw std::runtime_error(
            "UcmdRing::drain_batch_fragment(" + name_ + "): " + error);
    }
    return submitted;
}

uint32_t UcmdRing::submit_ready_sqes(const char* where) {
    const uint32_t ready = static_cast<uint32_t>(
        io_uring_sq_ready(&impl_->ring));
    if (ready == 0) {
        return 0;
    }

    const uint64_t submit_start_ns = steady_now_ns();
    const int submit_ret = io_uring_submit(&impl_->ring);
    if (submit_ret < 0) {
        impl_->pending_where.clear();
        impl_->first_error =
            std::string(where == nullptr ? name_.c_str() : where) +
            ": io_uring_submit failed: " + neg_errno_string(submit_ret);
        impl_->batching = false;
        throw std::runtime_error(
            "UcmdRing::submit_ready_sqes(" + name_ + "): " +
            impl_->first_error);
    }
    if (submit_ret < static_cast<int>(ready)) {
        std::ostringstream oss;
        oss << "submitted " << submit_ret << " of " << ready
            << " ready commands";
        impl_->first_error = oss.str();
        impl_->pending_where.clear();
        impl_->batching = false;
        throw std::runtime_error(
            "UcmdRing::submit_ready_sqes(" + name_ + "): " +
            impl_->first_error);
    }
    impl_->in_flight += static_cast<uint32_t>(submit_ret);
    impl_->batch_submitted += static_cast<uint32_t>(submit_ret);
    if (impl_->batch_first_submit_ns == 0) {
        impl_->batch_first_submit_ns = submit_start_ns;
    }
    return static_cast<uint32_t>(submit_ret);
}

uint32_t UcmdRing::reap_available_cqes() {
    uint32_t count = 0;
    while (impl_->in_flight > 0) {
        if (impl_->cqe_batch.empty()) {
            break;
        }
        const uint32_t want = std::min<uint32_t>(
            impl_->in_flight,
            static_cast<uint32_t>(impl_->cqe_batch.size()));
        const uint32_t got = io_uring_peek_batch_cqe(
            &impl_->ring, impl_->cqe_batch.data(), want);
        if (got == 0) {
            break;
        }
        uint64_t last_seen_ns = 0;
        for (uint32_t i = 0; i < got; ++i) {
            io_uring_cqe* cqe = impl_->cqe_batch[i];
            if (cqe == nullptr) {
                continue;
            }

            std::string where = name_;
            if (impl_->track_where) {
                auto it = impl_->pending_where.find(cqe->user_data);
                if (it != impl_->pending_where.end()) {
                    where = it->second;
                    impl_->pending_where.erase(it);
                }
            }
            const int res = cqe->res;
            if (impl_->in_flight > 0) {
                --impl_->in_flight;
            }
            ++impl_->batch_completed;
            ++count;

            if (res < 0 && impl_->first_error.empty()) {
                impl_->first_error = where + ": NVMe uring command failed: " +
                                     errno_string(-res);
            } else if (res != 0 && impl_->first_error.empty()) {
                std::ostringstream oss;
                oss << where << ": NVMe completion status " << res;
                impl_->first_error = oss.str();
            }
            last_seen_ns = steady_now_ns();
        }
        io_uring_cq_advance(&impl_->ring, got);
        if (last_seen_ns != 0) {
            impl_->batch_last_cqe_ns = last_seen_ns;
        }
    }
    return count;
}

uint32_t UcmdRing::wait_one_cqe() {
    if (impl_->in_flight == 0) {
        return 0;
    }
    io_uring_cqe* cqe = nullptr;
    const int wait_ret = io_uring_wait_cqe(&impl_->ring, &cqe);
    if (wait_ret < 0) {
        if (impl_->first_error.empty()) {
            impl_->first_error =
                "io_uring_wait_cqe failed: " + neg_errno_string(wait_ret);
        }
        return 0;
    }
    if (cqe == nullptr) {
        return 0;
    }

    std::string where = name_;
    if (impl_->track_where) {
        auto it = impl_->pending_where.find(cqe->user_data);
        if (it != impl_->pending_where.end()) {
            where = it->second;
            impl_->pending_where.erase(it);
        }
    }
    const int res = cqe->res;
    io_uring_cqe_seen(&impl_->ring, cqe);
    impl_->batch_last_cqe_ns = steady_now_ns();
    if (impl_->in_flight > 0) {
        --impl_->in_flight;
    }
    ++impl_->batch_completed;

    if (res < 0 && impl_->first_error.empty()) {
        impl_->first_error = where + ": NVMe uring command failed: " +
                             errno_string(-res);
    } else if (res != 0 && impl_->first_error.empty()) {
        std::ostringstream oss;
        oss << where << ": NVMe completion status " << res;
        impl_->first_error = oss.str();
    }
    return 1;
}

void UcmdRing::wait_for_batch_room() {
    if (!impl_->batching || queue_depth_ == 0) {
        return;
    }
    const uint32_t burst_room = std::max<uint32_t>(
        1u, std::min<uint32_t>(256u,
                               std::max<uint32_t>(1u, queue_depth_ / 2u)));
    reap_available_cqes();
    if (impl_->in_flight + io_uring_sq_ready(&impl_->ring) <=
        queue_depth_ - burst_room) {
        return;
    }
    submit_ready_sqes("wait_for_batch_room");
    reap_available_cqes();
    while (impl_->in_flight + io_uring_sq_ready(&impl_->ring) >
           queue_depth_ - burst_room) {
        if (impl_->in_flight == 0) {
            break;
        }
        wait_one_cqe();
        reap_available_cqes();
        if (!impl_->first_error.empty()) {
            break;
        }
    }
}

uint32_t UcmdRing::end_batch() {
    if (!is_open()) {
        throw std::runtime_error("UcmdRing::end_batch: ring is not open");
    }
    if (!impl_->batching) {
        throw std::runtime_error("UcmdRing::end_batch: batch is not open");
    }
    submit_ready_sqes("end_batch");
    reap_available_cqes();
    while (impl_->in_flight > 0) {
        wait_one_cqe();
        reap_available_cqes();
    }
    const uint32_t total = impl_->batch_completed;
    UcmdRingBatchStats stats;
    stats.commands = total;
    stats.submitted = impl_->batch_submitted;
    stats.first_submit_ns = impl_->batch_first_submit_ns;
    stats.last_cqe_ns = impl_->batch_last_cqe_ns;
    if (stats.first_submit_ns != 0 &&
        stats.last_cqe_ns >= stats.first_submit_ns) {
        stats.submit_lifetime_ns =
            stats.last_cqe_ns - stats.first_submit_ns;
    }
    impl_->last_batch_stats = stats;
    const std::string first_error = impl_->first_error;
    impl_->batching = false;
    impl_->batch_completed = 0;
    impl_->batch_submitted = 0;
    impl_->in_flight = 0;
    impl_->batch_first_submit_ns = 0;
    impl_->batch_last_cqe_ns = 0;
    impl_->pending_where.clear();
    impl_->first_error.clear();
    if (!first_error.empty()) {
        throw std::runtime_error(
            "UcmdRing::end_batch(" + name_ + "): " + first_error);
    }
    return total;
}

UcmdRingBatchStats UcmdRing::last_batch_stats() const {
    if (impl_ == nullptr) {
        return UcmdRingBatchStats{};
    }
    return impl_->last_batch_stats;
}

void UcmdRing::submit_nvme(int fd, uint32_t cmd_op,
                           const nvme_uring_cmd& cmd,
                           const char* where) {
    if (!is_open()) {
        throw std::runtime_error(std::string(where) + ": ring is not open");
    }
    if (impl_->batching) {
        wait_for_batch_room();
        if (!impl_->first_error.empty()) {
            while (impl_->in_flight > 0) {
                wait_one_cqe();
            }
            const std::string first_error = impl_->first_error;
            impl_->batching = false;
            impl_->batch_completed = 0;
            impl_->in_flight = 0;
            impl_->pending_where.clear();
            impl_->first_error.clear();
            throw std::runtime_error(
                "UcmdRing::submit_nvme(" + name_ + "): " + first_error);
        }
    }

    io_uring_sqe* sqe = io_uring_get_sqe(&impl_->ring);
    if (sqe == nullptr) {
        if (impl_->batching) {
            drain_batch_fragment();
            sqe = io_uring_get_sqe(&impl_->ring);
            if (sqe == nullptr && impl_->in_flight > 0) {
                wait_one_cqe();
                sqe = io_uring_get_sqe(&impl_->ring);
            }
        }
        if (sqe == nullptr) {
            throw std::runtime_error(std::string(where) + ": no free SQE");
        }
    }
    const uint64_t user_data = next_user_data_++;
    prepare_uring_cmd_sqe(sqe, fd, cmd_op, cmd, user_data);

    if (impl_->batching) {
        if (impl_->track_where) {
            impl_->pending_where.emplace(
                user_data, where == nullptr ? name_ : std::string(where));
        }
        const uint32_t ready = static_cast<uint32_t>(
            io_uring_sq_ready(&impl_->ring));
        const uint32_t submit_burst = std::max<uint32_t>(
            1u, std::min<uint32_t>(
                    256u, std::max<uint32_t>(1u, queue_depth_ / 2u)));
        if (ready >= submit_burst ||
            impl_->in_flight + ready >= queue_depth_) {
            submit_ready_sqes("submit_nvme_batch");
            reap_available_cqes();
        }
        return;
    }

    const int submit_ret = io_uring_submit_and_wait(&impl_->ring, 1);
    if (submit_ret < 0) {
        throw std::runtime_error(
            std::string(where) + ": io_uring_submit_and_wait failed: "
            + neg_errno_string(submit_ret));
    }
    if (submit_ret == 0) {
        throw std::runtime_error(
            std::string(where) + ": io_uring submitted zero commands");
    }

    io_uring_cqe* cqe = nullptr;
    const int wait_ret = io_uring_wait_cqe(&impl_->ring, &cqe);
    if (wait_ret < 0) {
        throw std::runtime_error(
            std::string(where) + ": io_uring_wait_cqe failed: "
            + neg_errno_string(wait_ret));
    }
    const int res = cqe->res;
    io_uring_cqe_seen(&impl_->ring, cqe);

    if (res < 0) {
        throw std::runtime_error(
            std::string(where) + ": NVMe uring command failed: "
            + errno_string(-res));
    }
    if (res != 0) {
        std::ostringstream oss;
        oss << where << ": NVMe completion status " << res;
        throw std::runtime_error(oss.str());
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
        throw std::runtime_error(
            "open " + path + " failed: " + errno_string(errno));
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
    uint64_t env_lba_size = 0;
    uint64_t env_namespace_nlb = 0;
    const bool have_env_lba =
        env_u64("PROPOSAL_FDP_LBA_SIZE", &env_lba_size);
    const bool have_env_nlb =
        env_u64("PROPOSAL_FDP_NAMESPACE_NLB", &env_namespace_nlb);
    if (have_env_lba || have_env_nlb) {
        if (!have_env_lba || !have_env_nlb) {
            throw std::runtime_error(
                "PROPOSAL_FDP_LBA_SIZE and PROPOSAL_FDP_NAMESPACE_NLB "
                "must be set together");
        }
        if (env_lba_size == 0 ||
            env_lba_size > static_cast<uint64_t>(UINT32_MAX) ||
            (env_lba_size & (env_lba_size - 1)) != 0) {
            throw std::runtime_error("invalid PROPOSAL_FDP_LBA_SIZE");
        }
        if (env_namespace_nlb == 0) {
            throw std::runtime_error("invalid PROPOSAL_FDP_NAMESPACE_NLB");
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
    nvme_uring_cmd cmd;
    std::memset(&cmd, 0, sizeof(cmd));
    cmd.opcode = kNvmeAdminIdentify;
    cmd.nsid = expected_nsid;
    cmd.addr = reinterpret_cast<uint64_t>(ns_buf.get());
    cmd.data_len = kNvmeIdentifyBytes;
    cmd.cdw10 = kIdentifyCnsNamespace;
    admin_ring.submit_nvme(fd_, NVME_URING_CMD_ADMIN, cmd,
                           "identify_namespace");

    const uint8_t* ns = static_cast<const uint8_t*>(ns_buf.get());
    const uint64_t nsze = load_le64(ns + 0);
    const uint8_t flbas = ns[26];
    const uint8_t lbaf_index =
        static_cast<uint8_t>((flbas & 0x0f) | ((flbas & 0x60) >> 1));
    if (lbaf_index >= 64) {
        throw std::runtime_error("identify_namespace: invalid FLBAS index");
    }
    const size_t lbaf_off = 128 + static_cast<size_t>(lbaf_index) * 4;
    const uint8_t lbads = ns[lbaf_off + 2];
    if (lbads < 9 || lbads > 20) {
        throw std::runtime_error("identify_namespace: unsupported LBA size");
    }
    const uint32_t lba_size = 1u << lbads;
    if (nsze == 0) {
        throw std::runtime_error("identify_namespace: namespace size is zero");
    }

    AlignedPtr ctrl_buf = alloc_aligned(4096, kNvmeIdentifyBytes);
    std::memset(&cmd, 0, sizeof(cmd));
    cmd.opcode = kNvmeAdminIdentify;
    cmd.nsid = 0;
    cmd.addr = reinterpret_cast<uint64_t>(ctrl_buf.get());
    cmd.data_len = kNvmeIdentifyBytes;
    cmd.cdw10 = kIdentifyCnsController;
    admin_ring.submit_nvme(fd_, NVME_URING_CMD_ADMIN, cmd,
                           "identify_controller");

    const uint8_t* ctrl = static_cast<const uint8_t*>(ctrl_buf.get());
    const uint32_t ctratt = load_le32(ctrl + 96);

    NamespaceInfo out;
    out.nsid = expected_nsid;
    out.lba_size = lba_size;
    out.namespace_nlb = nsze;
    out.fdp_supported = (ctratt & kCtrattFdpsBit) != 0;
    out.fdp_pid_enabled = false;
    return out;
}

void validate_aligned_buffer(const void* buffer, uint64_t bytes,
                             uint32_t lba_size, const char* where) {
    if (buffer == nullptr) {
        throw std::invalid_argument(std::string(where) + ": null buffer");
    }
    if (lba_size == 0 || (lba_size & (lba_size - 1)) != 0) {
        throw std::invalid_argument(std::string(where) + ": bad lba_size");
    }
    const uintptr_t addr = reinterpret_cast<uintptr_t>(buffer);
    if ((addr & 4095u) != 0) {
        throw std::invalid_argument(
            std::string(where) + ": buffer address is not 4KiB-aligned");
    }
    if (bytes == 0 || (bytes % lba_size) != 0) {
        throw std::invalid_argument(
            std::string(where) + ": byte length is not LBA-aligned");
    }
}

void NvmeDevice::read_lba(UcmdRing& ring, uint32_t nsid, void* buffer,
                          uint64_t slba, uint32_t nlb,
                          uint32_t lba_size) {
    if (nlb == 0 || nlb > 65536u) {
        throw std::invalid_argument("read_lba: nlb must be in [1, 65536]");
    }
    const uint64_t bytes = static_cast<uint64_t>(nlb) * lba_size;
    validate_aligned_buffer(buffer, bytes, lba_size, "read_lba");
    nvme_uring_cmd cmd;
    std::memset(&cmd, 0, sizeof(cmd));
    cmd.opcode = kNvmeCmdRead;
    cmd.nsid = nsid;
    cmd.addr = reinterpret_cast<uint64_t>(buffer);
    cmd.data_len = static_cast<uint32_t>(bytes);
    cmd.cdw10 = static_cast<uint32_t>(slba & 0xffffffffu);
    cmd.cdw11 = static_cast<uint32_t>(slba >> 32);
    cmd.cdw12 = (nlb - 1u) & 0xffffu;
    ring.submit_nvme(fd_, NVME_URING_CMD_IO, cmd, "read_lba");
}

void NvmeDevice::write_lba(UcmdRing& ring, uint32_t nsid,
                           const void* buffer, uint64_t slba, uint32_t nlb,
                           uint32_t lba_size, bool fua, uint16_t pid_hint) {
    (void)pid_hint;  // FDP placement directive wiring is a later hardening step.
    if (nlb == 0 || nlb > 65536u) {
        throw std::invalid_argument("write_lba: nlb must be in [1, 65536]");
    }
    const uint64_t bytes = static_cast<uint64_t>(nlb) * lba_size;
    validate_aligned_buffer(buffer, bytes, lba_size, "write_lba");
    nvme_uring_cmd cmd;
    std::memset(&cmd, 0, sizeof(cmd));
    cmd.opcode = kNvmeCmdWrite;
    cmd.nsid = nsid;
    cmd.addr = reinterpret_cast<uint64_t>(buffer);
    cmd.data_len = static_cast<uint32_t>(bytes);
    cmd.cdw10 = static_cast<uint32_t>(slba & 0xffffffffu);
    cmd.cdw11 = static_cast<uint32_t>(slba >> 32);
    cmd.cdw12 = ((nlb - 1u) & 0xffffu) |
                (fua ? (kNvmeRwFua << 16) : 0u);
    ring.submit_nvme(fd_, NVME_URING_CMD_IO, cmd, "write_lba");
}

void NvmeDevice::flush(UcmdRing& ring, uint32_t nsid) {
    nvme_uring_cmd cmd;
    std::memset(&cmd, 0, sizeof(cmd));
    cmd.opcode = kNvmeCmdFlush;
    cmd.nsid = nsid;
    ring.submit_nvme(fd_, NVME_URING_CMD_IO, cmd, "flush");
}

}}  // namespace proposal::fdp
