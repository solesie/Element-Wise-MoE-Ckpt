#include "fdp_store.h"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <functional>
#include <limits>
#include <mutex>
#include <shared_mutex>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <time.h>

#include <fcntl.h>
#include <pthread.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace proposal { namespace fdp {
namespace {

constexpr uint64_t kHeaderBlockBytes = 65536;
constexpr uint64_t kSharedMetadataMagic =
    make_magic8('F', 'D', 'P', 'M', 'M', 'A', 'P', '1');
constexpr uint32_t kSharedMetadataVersion = 1;

struct SharedMetadataControl {
    uint64_t magic = 0;
    uint32_t version = 0;
    uint32_t header_bytes = 0;
    uint64_t layout_fingerprint = 0;
    uint64_t metadata_image_bytes = 0;
    uint32_t world_size = 1;
    uint32_t rank_count = 1;
    pthread_mutex_t metadata_mutex;
    uint64_t active_iter = 0;
    uint32_t expected_processes = 1;
    uint32_t completed_processes = 0;
    uint32_t flush_started = 0;
    uint32_t flush_done = 1;
    uint32_t flush_owner_rank = 0;
    uint32_t error_rank = 0xffffffffu;
    uint64_t metadata_flush_count = 0;
    // reserved[0]: bitset of ranks that reported local writer completion for
    // active_iter.  Keep it in reserved storage so the mmap control layout
    // remains compatible with already-created objects.
    uint64_t reserved[16] = {};
};

uint32_t popcount_u64(uint64_t value) {
    uint32_t count = 0;
    while (value != 0) {
        value &= (value - 1);
        ++count;
    }
    return count;
}

const char* getenv_raw(const char* name) {
    const char* v = std::getenv(name);
    return (v == nullptr || v[0] == '\0') ? nullptr : v;
}

uint64_t parse_u64(const char* name, const char* value) {
    if (value == nullptr) {
        throw std::runtime_error(std::string("missing required env ") + name);
    }
    errno = 0;
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(value, &end, 0);
    if (errno != 0 || end == value || *end != '\0') {
        throw std::runtime_error(
            std::string("invalid integer env ") + name + "=" + value);
    }
    return static_cast<uint64_t>(parsed);
}

uint64_t env_u64_required(const char* name) {
    return parse_u64(name, getenv_raw(name));
}

uint64_t env_u64_default(const char* name, uint64_t default_value) {
    const char* v = getenv_raw(name);
    return v == nullptr ? default_value : parse_u64(name, v);
}

uint64_t monotonic_raw_ns() {
    timespec ts{};
    if (::clock_gettime(CLOCK_MONOTONIC_RAW, &ts) != 0) {
        throw std::runtime_error("clock_gettime(CLOCK_MONOTONIC_RAW) failed");
    }
    return static_cast<uint64_t>(ts.tv_sec) * 1000000000ull +
           static_cast<uint64_t>(ts.tv_nsec);
}

bool env_bool_default(const char* name, bool default_value) {
    const char* v = getenv_raw(name);
    if (v == nullptr) return default_value;
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

uint64_t checked_range_end(uint64_t start, uint64_t len,
                           const char* where) {
    if (len == 0) {
        throw std::runtime_error(std::string(where) + ": zero-length range");
    }
    if (start > std::numeric_limits<uint64_t>::max() - len) {
        throw std::runtime_error(std::string(where) + ": range overflow");
    }
    return start + len;
}

bool ranges_overlap(uint64_t a_start, uint64_t a_len,
                    uint64_t b_start, uint64_t b_len) {
    const uint64_t a_end = a_start + a_len;
    const uint64_t b_end = b_start + b_len;
    return a_start < b_end && b_start < a_end;
}

std::string sanitize_lock_component(const std::string& value) {
    std::string out;
    out.reserve(value.size());
    for (unsigned char ch : value) {
        if (std::isalnum(ch) || ch == '.' || ch == '-' || ch == '_') {
            out.push_back(static_cast<char>(ch));
        } else {
            out.push_back('_');
        }
    }
    return out.empty() ? "device" : out;
}

uint32_t checked_u32(uint64_t v, const char* name) {
    if (v > std::numeric_limits<uint32_t>::max()) {
        throw std::runtime_error(std::string(name) + " exceeds uint32_t");
    }
    return static_cast<uint32_t>(v);
}

uint32_t popcount_byte(uint8_t b) {
    return static_cast<uint32_t>(__builtin_popcount(
        static_cast<unsigned int>(b)));
}

size_t current_thread_token() {
    size_t token = std::hash<std::thread::id>{}(std::this_thread::get_id());
    return token == 0 ? 1 : token;
}

}  // namespace

AlignedBuffer::AlignedBuffer() : ptr_(nullptr), bytes_(0) {}

AlignedBuffer::~AlignedBuffer() {
    reset();
}

AlignedBuffer::AlignedBuffer(AlignedBuffer&& other) noexcept
    : ptr_(other.ptr_), bytes_(other.bytes_) {
    other.ptr_ = nullptr;
    other.bytes_ = 0;
}

AlignedBuffer& AlignedBuffer::operator=(AlignedBuffer&& other) noexcept {
    if (this != &other) {
        reset();
        ptr_ = other.ptr_;
        bytes_ = other.bytes_;
        other.ptr_ = nullptr;
        other.bytes_ = 0;
    }
    return *this;
}

void AlignedBuffer::allocate(size_t bytes, size_t alignment) {
    reset();
    if (bytes == 0) {
        throw std::invalid_argument("AlignedBuffer::allocate: zero bytes");
    }
    void* p = nullptr;
    const int rc = ::posix_memalign(&p, alignment, bytes);
    if (rc != 0) {
        throw std::runtime_error(
            std::string("posix_memalign failed: ") + std::strerror(rc));
    }
    ptr_ = p;
    bytes_ = bytes;
    std::memset(ptr_, 0, bytes_);
}

void AlignedBuffer::reset() {
    if (ptr_ != nullptr) {
        std::free(ptr_);
        ptr_ = nullptr;
        bytes_ = 0;
    }
}

FdpStore::FdpStore() = default;

FdpStore::~FdpStore() {
    close();
}

void FdpStore::load_config_from_env() {
    cfg_ = FdpEnvConfig();
    const char* device = getenv_raw("PROPOSAL_FDP_DEVICE");
    if (device == nullptr) {
        throw std::runtime_error("missing required env PROPOSAL_FDP_DEVICE");
    }
    cfg_.device_path = device;
    cfg_.nsid = checked_u32(env_u64_required("PROPOSAL_FDP_NSID"),
                            "PROPOSAL_FDP_NSID");
    cfg_.metadata_mmap = env_bool_default(
        "PROPOSAL_FDP_METADATA_MMAP", false);
    perf_diag_ = env_bool_default("PROPOSAL_FDP_PERF_DIAG", false);
    cfg_.metadata_start_lba =
        env_u64_required("PROPOSAL_FDP_METADATA_START_LBA");
    cfg_.data_start_lba = env_u64_required("PROPOSAL_FDP_DATA_START_LBA");
    cfg_.data_nlb = env_u64_required("PROPOSAL_FDP_DATA_NLB");

    if (const char* v = getenv_raw("PROPOSAL_FDP_METADATA_NLB")) {
        cfg_.metadata_nlb = parse_u64("PROPOSAL_FDP_METADATA_NLB", v);
        cfg_.metadata_nlb_explicit = true;
    }
    if (const char* v = getenv_raw("PROPOSAL_FDP_LOCK_PATH")) {
        global_lock_path_ = v;
    } else {
        std::ostringstream oss;
        oss << "/tmp/proposal_fdp_"
            << sanitize_lock_component(cfg_.device_path)
            << "_m" << cfg_.metadata_start_lba << ".lock";
        global_lock_path_ = oss.str();
    }
    if (const char* v = getenv_raw("PROPOSAL_FDP_MMAP_NAME")) {
        shared_mmap_name_ = v;
        if (shared_mmap_name_.empty() || shared_mmap_name_[0] != '/') {
            shared_mmap_name_.insert(shared_mmap_name_.begin(), '/');
        }
    } else {
        std::ostringstream oss;
        oss << "/proposal_fdp_"
            << sanitize_lock_component(cfg_.device_path)
            << "_m" << cfg_.metadata_start_lba
            << "_v" << kFormatVersion;
        shared_mmap_name_ = oss.str();
    }

    if (const char* v = getenv_raw("PROPOSAL_FDP_CHUNK_BYTES")) {
        cfg_.chunk_bytes = parse_u64("PROPOSAL_FDP_CHUNK_BYTES", v);
    } else {
        const uint64_t mb = env_u64_default("PROPOSAL_FDP_CHUNK_MB", 64);
        cfg_.chunk_bytes = mb * 1024ull * 1024ull;
    }

    cfg_.rank = checked_u32(env_u64_default("PROPOSAL_FDP_RANK", 0),
                            "PROPOSAL_FDP_RANK");
    cfg_.world_size = checked_u32(env_u64_default("PROPOSAL_FDP_WORLD_SIZE", 1),
                                  "PROPOSAL_FDP_WORLD_SIZE");
    cfg_.dp_rank = checked_u32(env_u64_default("PROPOSAL_FDP_DP_RANK", 0),
                               "PROPOSAL_FDP_DP_RANK");
    cfg_.dp_size = checked_u32(env_u64_default("PROPOSAL_FDP_DP_SIZE", 1),
                               "PROPOSAL_FDP_DP_SIZE");
    cfg_.pp_rank = checked_u32(env_u64_default("PROPOSAL_FDP_PP_RANK", 0),
                               "PROPOSAL_FDP_PP_RANK");
    cfg_.pp_size = checked_u32(env_u64_default("PROPOSAL_FDP_PP_SIZE", 1),
                               "PROPOSAL_FDP_PP_SIZE");
    cfg_.ep_rank = checked_u32(env_u64_default("PROPOSAL_FDP_EP_RANK", 0),
                               "PROPOSAL_FDP_EP_RANK");
    cfg_.ep_size = checked_u32(env_u64_default("PROPOSAL_FDP_EP_SIZE", 1),
                               "PROPOSAL_FDP_EP_SIZE");
    if (cfg_.dp_size == 0 || cfg_.pp_size == 0 || cfg_.ep_size == 0) {
        throw std::runtime_error("FDP topology sizes must be non-zero");
    }
    if (cfg_.dp_rank >= cfg_.dp_size ||
        cfg_.pp_rank >= cfg_.pp_size ||
        cfg_.ep_rank >= cfg_.ep_size) {
        throw std::runtime_error("FDP topology rank is out of range");
    }
    cfg_.expert_id_base = checked_u32(
        env_u64_default("PROPOSAL_FDP_EXPERT_ID_BASE", 0),
        "PROPOSAL_FDP_EXPERT_ID_BASE");
    cfg_.expert_ids_global = env_bool_default(
        "PROPOSAL_FDP_EXPERT_IDS_GLOBAL", false);

    cfg_.num_moe_layers = checked_u32(
        env_u64_default("PROPOSAL_FDP_NUM_MOE_LAYERS", 0),
        "PROPOSAL_FDP_NUM_MOE_LAYERS");
    cfg_.experts_per_layer = checked_u32(
        env_u64_default("PROPOSAL_FDP_EXPERTS_PER_LAYER", 0),
        "PROPOSAL_FDP_EXPERTS_PER_LAYER");
    if ((cfg_.num_moe_layers == 0) != (cfg_.experts_per_layer == 0)) {
        throw std::runtime_error(
            "PROPOSAL_FDP_NUM_MOE_LAYERS and "
            "PROPOSAL_FDP_EXPERTS_PER_LAYER must both be zero or both be set");
    }
    const uint64_t slots = static_cast<uint64_t>(cfg_.dp_size) *
                           static_cast<uint64_t>(cfg_.pp_size) *
                           static_cast<uint64_t>(cfg_.num_moe_layers) *
                           static_cast<uint64_t>(cfg_.experts_per_layer);
    if (slots > kMaxExpertSlots) {
        throw std::runtime_error("FDP expert slot count exceeds kMaxExpertSlots");
    }
    cfg_.expert_slots = static_cast<uint32_t>(slots);
    const uint64_t nefc_shards =
        static_cast<uint64_t>(cfg_.pp_size) *
        static_cast<uint64_t>(cfg_.dp_size) *
        static_cast<uint64_t>(cfg_.ep_size);
    if (nefc_shards == 0 || nefc_shards > kMaxExpertSlots) {
        throw std::runtime_error("FDP NEFC shard count is out of range");
    }
    cfg_.nefc_shard_count = static_cast<uint32_t>(nefc_shards);
    cfg_.nefc_shard_index = current_nefc_shard();

    cfg_.nefc_qd = checked_u32(env_u64_default("PROPOSAL_FDP_NEFC_QD", 64),
                               "PROPOSAL_FDP_NEFC_QD");
    cfg_.expert_qd = checked_u32(
        env_u64_default("PROPOSAL_FDP_EXPERT_QD", 64),
        "PROPOSAL_FDP_EXPERT_QD");
    cfg_.metadata_qd = checked_u32(
        env_u64_default("PROPOSAL_FDP_METADATA_QD", 8),
        "PROPOSAL_FDP_METADATA_QD");
    cfg_.max_io_kb = checked_u32(
        env_u64_default("PROPOSAL_FDP_MAX_IO_KB", 256),
        "PROPOSAL_FDP_MAX_IO_KB");
    if (cfg_.max_io_kb == 0) {
        throw std::runtime_error("PROPOSAL_FDP_MAX_IO_KB must be > 0");
    }

    cfg_.allow_format = env_bool_default("PROPOSAL_FDP_ALLOW_FORMAT", false);
    cfg_.force_format = env_bool_default("PROPOSAL_FDP_FORCE_FORMAT", false);
    cfg_.nvme_flush = env_bool_default("PROPOSAL_FDP_NVME_FLUSH", true);
    cfg_.write_fua = env_bool_default("PROPOSAL_FDP_WRITE_FUA", false);
}

void FdpStore::open_from_env() {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    close();
    load_config_from_env();

    try {
        device_.open_device(cfg_.device_path);
        metadata_ring_.init(cfg_.metadata_qd, "metadata_ring", device_.fd(),
                            false);
        ns_ = device_.identify(metadata_ring_, cfg_.nsid);

        if (!cfg_.metadata_nlb_explicit) {
            if (cfg_.metadata_start_lba >= cfg_.data_start_lba) {
                throw std::runtime_error(
                    "PROPOSAL_FDP_METADATA_NLB is required when metadata_start "
                    "is not before data_start");
            }
            cfg_.metadata_nlb = cfg_.data_start_lba - cfg_.metadata_start_lba;
        }

        superblock_nlb_ = div_round_up_u64(sizeof(SuperBlock), ns_.lba_size);
        metadata_image_lba_ = cfg_.metadata_start_lba + superblock_nlb_;
        if (cfg_.chunk_bytes == 0 ||
            (cfg_.chunk_bytes % ns_.lba_size) != 0) {
            throw std::runtime_error(
                "PROPOSAL_FDP_CHUNK_BYTES must be a positive multiple of "
                "device lba_size");
        }
        if ((kHeaderBlockBytes % ns_.lba_size) != 0) {
            throw std::runtime_error(
                "FDP header block size must be a multiple of device lba_size");
        }
        const uint64_t max_io_bytes =
            static_cast<uint64_t>(cfg_.max_io_kb) * 1024ull;
        max_io_nlb_ = checked_u32(max_io_bytes / ns_.lba_size,
                                  "PROPOSAL_FDP_MAX_IO_KB/LBA_SIZE");
        if (max_io_nlb_ == 0) {
            throw std::runtime_error(
                "PROPOSAL_FDP_MAX_IO_KB is smaller than one device LBA");
        }
        max_io_nlb_ = std::min<uint32_t>(max_io_nlb_, 65536u);
        if (cfg_.chunk_bytes < kHeaderBlockBytes) {
            throw std::runtime_error(
                "PROPOSAL_FDP_CHUNK_BYTES must be at least 64KiB");
        }
        chunk_lba_count_ = cfg_.chunk_bytes / ns_.lba_size;
        if (chunk_lba_count_ == 0) {
            throw std::runtime_error("chunk_lba_count is zero");
        }
        const uint64_t raw_chunk_count = cfg_.data_nlb / chunk_lba_count_;
        if (raw_chunk_count == 0) {
            throw std::runtime_error(
                "PROPOSAL_FDP_DATA_NLB is smaller than one chunk");
        }
        data_chunk_count_ = checked_u32(raw_chunk_count,
                                        "FDP data chunk count");

        layout_ = compute_metadata_layout(data_chunk_count_,
                                          cfg_.expert_slots,
                                          cfg_.nefc_shard_count);
        layout_.image_bytes = round_up_u64(layout_.image_bytes, ns_.lba_size);
        metadata_image_nlb_ = layout_.image_bytes / ns_.lba_size;

        validate_namespace_layout();

        nefc_data_ring_.init(cfg_.nefc_qd, "nefc_data_ring", device_.fd(),
                             true);
        expert_data_ring_.init(cfg_.expert_qd, "expert_data_ring", device_.fd(),
                               true);

        allocate_metadata_image();
        format_or_load();
        if (cfg_.metadata_mmap) {
            open_or_create_shared_metadata_mmap();
        }
        open_ = true;
    } catch (...) {
        close();
        throw;
    }
}

void FdpStore::close() {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    std::lock_guard<std::mutex> state_lock(batch_state_mutex_);
    if (batch_lock_.owns_lock()) {
        batch_owner_token_.store(0, std::memory_order_release);
        if (nefc_batch_ring_lock_.owns_lock()) {
            nefc_batch_ring_lock_.unlock();
        }
        if (expert_batch_ring_lock_.owns_lock()) {
            expert_batch_ring_lock_.unlock();
        }
        batch_lock_.unlock();
    }
    if (nefc_batch_lock_.owns_lock()) {
        nefc_batch_owner_token_.store(0, std::memory_order_release);
        if (nefc_batch_ring_lock_.owns_lock()) {
            nefc_batch_ring_lock_.unlock();
        }
        nefc_batch_lock_.unlock();
    }
    if (expert_batch_lock_.owns_lock()) {
        expert_batch_owner_token_.store(0, std::memory_order_release);
        expert_batch_read_only_ = false;
        if (expert_batch_ring_lock_.owns_lock()) {
            expert_batch_ring_lock_.unlock();
        }
        expert_batch_lock_.unlock();
    }
    while (global_lock_depth_ > 0) {
        try {
            release_global_metadata_lock();
        } catch (...) {
            global_lock_depth_ = 0;
            break;
        }
    }
    if (global_lock_fd_ >= 0) {
        ::close(global_lock_fd_);
        global_lock_fd_ = -1;
    }
    close_shared_metadata_mmap();
    open_ = false;
    nefc_data_ring_.close();
    expert_data_ring_.close();
    metadata_ring_.close();
    device_.close();
    superblock_buf_.reset();
    metadata_image_.reset();
    scratch_buf_.reset();
    ns_ = NamespaceInfo();
    layout_ = MetadataLayout();
    superblock_nlb_ = 0;
    metadata_image_lba_ = 0;
    metadata_image_nlb_ = 0;
    chunk_lba_count_ = 0;
    max_io_nlb_ = 0;
    data_chunk_count_ = 0;
    seq_ = 0;
    last_flush_iter_ = 0;
    allocated_chunks_ = 0;
    next_lfsy_handle_ = 1;
    lfsy_handles_.clear();
    batch_owned_buffers_.clear();
    nefc_batch_plans_.clear();
    expert_batch_plans_.clear();
    global_lock_path_.clear();
    shared_mmap_name_.clear();
    local_iter_ = 0;
    local_nonexpert_done_ = false;
    local_expert_done_ = false;
    local_process_done_reported_ = false;
}

bool FdpStore::is_open() const {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    return open_;
}

void FdpStore::validate_namespace_layout() {
    const uint64_t metadata_end = checked_range_end(
        cfg_.metadata_start_lba, cfg_.metadata_nlb, "metadata range");
    const uint64_t data_end = checked_range_end(
        cfg_.data_start_lba, cfg_.data_nlb, "data range");
    if (metadata_end > ns_.namespace_nlb) {
        throw std::runtime_error("metadata range exceeds namespace capacity");
    }
    if (data_end > ns_.namespace_nlb) {
        throw std::runtime_error("data range exceeds namespace capacity");
    }
    if (ranges_overlap(cfg_.metadata_start_lba, cfg_.metadata_nlb,
                       cfg_.data_start_lba, cfg_.data_nlb)) {
        throw std::runtime_error("metadata range overlaps data range");
    }
    const uint64_t required_metadata_nlb =
        superblock_nlb_ + metadata_image_nlb_;
    if (cfg_.metadata_nlb < required_metadata_nlb) {
        std::ostringstream oss;
        oss << "metadata range too small: need " << required_metadata_nlb
            << " LBAs, got " << cfg_.metadata_nlb;
        throw std::runtime_error(oss.str());
    }
}

void FdpStore::allocate_metadata_image() {
    const uint64_t superblock_bytes = superblock_nlb_ * ns_.lba_size;
    superblock_buf_.allocate(static_cast<size_t>(superblock_bytes), 4096);
    metadata_image_.allocate(static_cast<size_t>(layout_.image_bytes), 4096);
    scratch_buf_.allocate(static_cast<size_t>(
        std::max<uint64_t>(4096, ns_.lba_size)), 4096);
}

void FdpStore::format_or_load() {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    if (!device_.is_open()) {
        throw std::runtime_error("format_or_load: device is not open");
    }
    auto lock_global = [&]() {
        if (global_lock_path_.empty()) {
            return;
        }
        if (global_lock_fd_ < 0) {
            global_lock_fd_ = ::open(
                global_lock_path_.c_str(), O_CREAT | O_RDWR | O_CLOEXEC, 0600);
            if (global_lock_fd_ < 0) {
                throw std::runtime_error(
                    std::string("open FDP format lock failed: ") +
                    std::strerror(errno));
            }
        }
        if (::flock(global_lock_fd_, LOCK_EX) != 0) {
            throw std::runtime_error(
                std::string("flock FDP format lock failed: ") +
                std::strerror(errno));
        }
    };
    auto unlock_global = [&]() {
        if (global_lock_fd_ >= 0 &&
            ::flock(global_lock_fd_, LOCK_UN) != 0) {
            throw std::runtime_error(
                std::string("unlock FDP format lock failed: ") +
                std::strerror(errno));
        }
    };

    const bool rank0 = cfg_.rank == 0;
    for (uint32_t attempt = 0; attempt < 200; ++attempt) {
        bool done = false;
        lock_global();
        try {
            std::memset(superblock_buf_.data(), 0, superblock_buf_.bytes());
            raw_read_metadata(superblock_buf_.data(), cfg_.metadata_start_lba,
                              superblock_nlb_, "read_superblock");
            const SuperBlock* sb = superblock_buf_.as<SuperBlock>();
            const bool has_superblock = sb->magic == kSuperMagic;
            const bool force_here = cfg_.force_format && rank0 && !open_;
            if (has_superblock && !force_here) {
                try {
                    validate_loaded_superblock(*sb);
                    load_existing_metadata(*sb);
                    done = true;
                } catch (...) {
                    if (!(cfg_.force_format && !rank0)) {
                        throw;
                    }
                }
            } else if (force_here || (!has_superblock && cfg_.allow_format && rank0)) {
                initialize_fresh_metadata();
                seq_ = 1;
                last_flush_iter_ = 0;
                allocated_chunks_ = 0;
                update_metadata_header(0, false);
                write_superblock();
                write_metadata_image();
                if (cfg_.nvme_flush) {
                    device_.flush(metadata_ring_, ns_.nsid);
                }
                done = true;
            } else if (!has_superblock && !cfg_.allow_format && !cfg_.force_format) {
                throw std::runtime_error(
                    "FDP superblock magic not found and PROPOSAL_FDP_ALLOW_FORMAT=1 "
                    "is not set");
            }
            unlock_global();
        } catch (...) {
            unlock_global();
            throw;
        }
        if (done) {
            return;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    throw std::runtime_error(
        "FDP format_or_load: timed out waiting for rank0 to initialize metadata");
}

void FdpStore::initialize_fresh_metadata() {
    std::memset(metadata_image_.data(), 0, metadata_image_.bytes());
}

void FdpStore::validate_loaded_superblock(const SuperBlock& sb) const {
    if (sb.version != kFormatVersion) {
        throw std::runtime_error("FDP superblock version mismatch");
    }
    if (sb.lba_size != ns_.lba_size ||
        sb.namespace_nlb != ns_.namespace_nlb ||
        sb.chunk_bytes != cfg_.chunk_bytes ||
        sb.chunk_lba_count != chunk_lba_count_ ||
        sb.metadata_start_lba != cfg_.metadata_start_lba ||
        sb.metadata_nlb != cfg_.metadata_nlb ||
        sb.metadata_image_lba != metadata_image_lba_ ||
        sb.metadata_image_nlb != metadata_image_nlb_ ||
        sb.metadata_image_bytes != layout_.image_bytes ||
        sb.data_start_lba != cfg_.data_start_lba ||
        sb.data_nlb != cfg_.data_nlb ||
        sb.data_chunk_count != data_chunk_count_ ||
        sb.world_size != cfg_.world_size ||
        sb.num_moe_layers != cfg_.num_moe_layers ||
        sb.experts_per_layer != cfg_.experts_per_layer ||
        sb.expert_slots != cfg_.expert_slots ||
        sb.dp_size != cfg_.dp_size ||
        sb.pp_size != cfg_.pp_size ||
        sb.ep_size != cfg_.ep_size ||
        sb.nefc_shard_count != cfg_.nefc_shard_count) {
        throw std::runtime_error(
            "FDP superblock does not match current environment layout");
    }
}

void FdpStore::load_existing_metadata(const SuperBlock& sb) {
    seq_ = sb.seq;
    last_flush_iter_ = sb.last_flush_iter;
    allocated_chunks_ = sb.allocated_chunks;
    read_metadata_image();
    const MetadataHeader* hdr = metadata_header();
    if (hdr->magic != kMetadataMagic) {
        throw std::runtime_error("FDP metadata image magic mismatch");
    }
    if (hdr->version != kFormatVersion) {
        throw std::runtime_error("FDP metadata image version mismatch");
    }
    if (hdr->metadata_image_bytes != layout_.image_bytes ||
        hdr->data_chunk_count != data_chunk_count_ ||
        hdr->bitmap_bytes != layout_.bitmap_bytes ||
        hdr->bitmap_offset != layout_.bitmap_offset ||
        hdr->nefc_offset != layout_.nefc_offset ||
        hdr->nefc_shard_count != cfg_.nefc_shard_count ||
        hdr->lfcs_offset != layout_.lfcs_offset ||
        hdr->lfsy_offset != layout_.lfsy_offset) {
        throw std::runtime_error("FDP metadata image layout mismatch");
    }
    seq_ = hdr->seq;
    last_flush_iter_ = hdr->last_flush_iter;
    allocated_chunks_ = recalc_allocated_chunks();
    update_metadata_header(last_flush_iter_, false);
}

void FdpStore::write_superblock() {
    std::memset(superblock_buf_.data(), 0, superblock_buf_.bytes());
    SuperBlock* sb = superblock_buf_.as<SuperBlock>();
    sb->magic = kSuperMagic;
    sb->version = kFormatVersion;
    sb->struct_bytes = sizeof(SuperBlock);
    sb->lba_size = ns_.lba_size;
    sb->superblock_nlb = checked_u32(superblock_nlb_, "superblock_nlb");
    sb->namespace_nlb = ns_.namespace_nlb;
    sb->chunk_bytes = cfg_.chunk_bytes;
    sb->chunk_lba_count = chunk_lba_count_;
    sb->metadata_start_lba = cfg_.metadata_start_lba;
    sb->metadata_nlb = cfg_.metadata_nlb;
    sb->metadata_image_lba = metadata_image_lba_;
    sb->metadata_image_nlb = metadata_image_nlb_;
    sb->metadata_image_bytes = layout_.image_bytes;
    sb->data_start_lba = cfg_.data_start_lba;
    sb->data_nlb = cfg_.data_nlb;
    sb->data_chunk_count = data_chunk_count_;
    sb->bitmap_bytes = checked_u32(layout_.bitmap_bytes, "bitmap_bytes");
    sb->rank = cfg_.rank;
    sb->world_size = cfg_.world_size;
    sb->num_moe_layers = cfg_.num_moe_layers;
    sb->experts_per_layer = cfg_.experts_per_layer;
    sb->expert_slots = cfg_.expert_slots;
    sb->dp_rank = cfg_.dp_rank;
    sb->dp_size = cfg_.dp_size;
    sb->pp_rank = cfg_.pp_rank;
    sb->pp_size = cfg_.pp_size;
    sb->ep_rank = cfg_.ep_rank;
    sb->ep_size = cfg_.ep_size;
    sb->nefc_shard_count = cfg_.nefc_shard_count;
    sb->max_chunks_nefc = kMaxChunksNefc;
    sb->max_chunks_lfcs = kMaxChunksLfcs;
    sb->max_chunks_lfsy = kMaxChunksLfsy;
    sb->seq = seq_;
    sb->last_flush_iter = last_flush_iter_;
    sb->allocated_chunks = allocated_chunks_;
    sb->fdp_supported = ns_.fdp_supported ? 1 : 0;
    sb->fdp_pid_enabled = ns_.fdp_pid_enabled ? 1 : 0;

    raw_write_metadata(superblock_buf_.data(), cfg_.metadata_start_lba,
                       superblock_nlb_, "write_superblock");
}

void FdpStore::write_metadata_image() {
    raw_write_metadata(metadata_bytes(), metadata_image_lba_,
                       metadata_image_nlb_, "write_metadata_image");
}

void FdpStore::read_metadata_image() {
    std::memset(metadata_image_.data(), 0, metadata_image_.bytes());
    raw_read_metadata(metadata_image_.data(), metadata_image_lba_,
                      metadata_image_nlb_, "read_metadata_image");
}

uint64_t FdpStore::metadata_layout_fingerprint() const {
    uint64_t h = 1469598103934665603ull;
    auto mix = [&h](uint64_t v) {
        h ^= v;
        h *= 1099511628211ull;
    };
    mix(kFormatVersion);
    mix(ns_.lba_size);
    mix(ns_.namespace_nlb);
    mix(cfg_.metadata_start_lba);
    mix(cfg_.metadata_nlb);
    mix(metadata_image_lba_);
    mix(metadata_image_nlb_);
    mix(layout_.image_bytes);
    mix(cfg_.data_start_lba);
    mix(cfg_.data_nlb);
    mix(cfg_.chunk_bytes);
    mix(chunk_lba_count_);
    mix(data_chunk_count_);
    mix(cfg_.world_size);
    mix(cfg_.num_moe_layers);
    mix(cfg_.experts_per_layer);
    mix(cfg_.expert_slots);
    mix(cfg_.dp_size);
    mix(cfg_.pp_size);
    mix(cfg_.ep_size);
    mix(cfg_.nefc_shard_count);
    return h;
}

void FdpStore::open_or_create_shared_metadata_mmap() {
    if (!cfg_.metadata_mmap) {
        return;
    }
    const uint64_t control_bytes =
        round_up_u64(sizeof(SharedMetadataControl), 4096);
    shared_mmap_bytes_ = control_bytes + layout_.image_bytes;
    if (shared_mmap_bytes_ >
        static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        throw std::runtime_error("shared metadata mmap is too large");
    }

    if (global_lock_fd_ < 0) {
        global_lock_fd_ = ::open(
            global_lock_path_.c_str(), O_CREAT | O_RDWR | O_CLOEXEC, 0600);
        if (global_lock_fd_ < 0) {
            throw std::runtime_error(
                std::string("open FDP mmap init lock failed: ") +
                std::strerror(errno));
        }
    }
    if (::flock(global_lock_fd_, LOCK_EX) != 0) {
        throw std::runtime_error(
            std::string("flock FDP mmap init lock failed: ") +
            std::strerror(errno));
    }

    try {
        shared_mmap_fd_ = ::shm_open(
            shared_mmap_name_.c_str(), O_CREAT | O_RDWR, 0600);
        if (shared_mmap_fd_ < 0) {
            throw std::runtime_error(
                std::string("shm_open FDP metadata failed: ") +
                std::strerror(errno));
        }
        if (::ftruncate(shared_mmap_fd_,
                        static_cast<off_t>(shared_mmap_bytes_)) != 0) {
            throw std::runtime_error(
                std::string("ftruncate FDP metadata mmap failed: ") +
                std::strerror(errno));
        }
        shared_mmap_base_ = ::mmap(
            nullptr, static_cast<size_t>(shared_mmap_bytes_),
            PROT_READ | PROT_WRITE, MAP_SHARED, shared_mmap_fd_, 0);
        if (shared_mmap_base_ == MAP_FAILED) {
            shared_mmap_base_ = nullptr;
            throw std::runtime_error(
                std::string("mmap FDP metadata failed: ") +
                std::strerror(errno));
        }
        shared_control_ = shared_mmap_base_;
        shared_metadata_bytes_ =
            static_cast<uint8_t*>(shared_mmap_base_) + control_bytes;

        auto* ctrl =
            static_cast<SharedMetadataControl*>(shared_control_);
        const uint64_t fingerprint = metadata_layout_fingerprint();
        const bool valid =
            ctrl->magic == kSharedMetadataMagic &&
            ctrl->version == kSharedMetadataVersion &&
            ctrl->header_bytes == sizeof(SharedMetadataControl) &&
            ctrl->layout_fingerprint == fingerprint &&
            ctrl->metadata_image_bytes == layout_.image_bytes &&
            ctrl->world_size == cfg_.world_size;
        const bool reset_for_format = cfg_.force_format && cfg_.rank == 0;
        if (!valid || reset_for_format) {
            std::memset(shared_mmap_base_, 0,
                        static_cast<size_t>(shared_mmap_bytes_));
            pthread_mutexattr_t attr;
            if (::pthread_mutexattr_init(&attr) != 0) {
                throw std::runtime_error("pthread_mutexattr_init failed");
            }
            if (::pthread_mutexattr_setpshared(
                    &attr, PTHREAD_PROCESS_SHARED) != 0) {
                ::pthread_mutexattr_destroy(&attr);
                throw std::runtime_error(
                    "pthread_mutexattr_setpshared failed");
            }
            if (::pthread_mutex_init(&ctrl->metadata_mutex, &attr) != 0) {
                ::pthread_mutexattr_destroy(&attr);
                throw std::runtime_error("pthread_mutex_init failed");
            }
            ::pthread_mutexattr_destroy(&attr);

            ctrl->magic = kSharedMetadataMagic;
            ctrl->version = kSharedMetadataVersion;
            ctrl->header_bytes = sizeof(SharedMetadataControl);
            ctrl->layout_fingerprint = fingerprint;
            ctrl->metadata_image_bytes = layout_.image_bytes;
            ctrl->world_size = cfg_.world_size;
            ctrl->rank_count = cfg_.world_size;
            ctrl->active_iter = 0;
            ctrl->expected_processes = cfg_.world_size;
            ctrl->completed_processes = 0;
            ctrl->flush_started = 0;
            ctrl->flush_done = 1;
            ctrl->flush_owner_rank = 0;
            ctrl->error_rank = 0xffffffffu;
            ctrl->metadata_flush_count = 0;
            std::memcpy(shared_metadata_bytes_, metadata_image_.data(),
                        static_cast<size_t>(layout_.image_bytes));
        }

        const MetadataHeader* hdr =
            reinterpret_cast<const MetadataHeader*>(shared_metadata_bytes_);
        if (hdr->magic != kMetadataMagic ||
            hdr->version != kFormatVersion) {
            throw std::runtime_error(
                "FDP shared metadata image header mismatch");
        }
        seq_ = hdr->seq;
        last_flush_iter_ = hdr->last_flush_iter;
        allocated_chunks_ = recalc_allocated_chunks();
    } catch (...) {
        (void)::flock(global_lock_fd_, LOCK_UN);
        throw;
    }
    if (::flock(global_lock_fd_, LOCK_UN) != 0) {
        throw std::runtime_error(
            std::string("unlock FDP mmap init lock failed: ") +
            std::strerror(errno));
    }
}

void FdpStore::close_shared_metadata_mmap() {
    if (shared_mmap_base_ != nullptr) {
        ::munmap(shared_mmap_base_,
                 static_cast<size_t>(shared_mmap_bytes_));
    }
    if (shared_mmap_fd_ >= 0) {
        ::close(shared_mmap_fd_);
    }
    shared_mmap_fd_ = -1;
    shared_mmap_base_ = nullptr;
    shared_mmap_bytes_ = 0;
    shared_control_ = nullptr;
    shared_metadata_bytes_ = nullptr;
}

void FdpStore::update_metadata_header(uint64_t iter, bool bump_seq) {
    if (bump_seq) ++seq_;
    last_flush_iter_ = iter;
    MetadataHeader* hdr = metadata_header();
    std::memset(hdr, 0, sizeof(*hdr));
    hdr->magic = kMetadataMagic;
    hdr->version = kFormatVersion;
    hdr->header_bytes = kMetadataHeaderBytes;
    hdr->metadata_image_bytes = layout_.image_bytes;
    hdr->seq = seq_;
    hdr->last_flush_iter = last_flush_iter_;
    hdr->allocated_chunks = allocated_chunks_;
    hdr->rank = cfg_.rank;
    hdr->world_size = cfg_.world_size;
    hdr->num_moe_layers = cfg_.num_moe_layers;
    hdr->experts_per_layer = cfg_.experts_per_layer;
    hdr->expert_slots = cfg_.expert_slots;
    hdr->nefc_shard_count = cfg_.nefc_shard_count;
    hdr->data_chunk_count = data_chunk_count_;
    hdr->bitmap_bytes = checked_u32(layout_.bitmap_bytes, "bitmap_bytes");
    hdr->dp_size = cfg_.dp_size;
    hdr->pp_size = cfg_.pp_size;
    hdr->ep_size = cfg_.ep_size;
    hdr->bitmap_offset = layout_.bitmap_offset;
    hdr->nefc_offset = layout_.nefc_offset;
    hdr->lfcs_offset = layout_.lfcs_offset;
    hdr->lfsy_offset = layout_.lfsy_offset;
}

void FdpStore::acquire_global_metadata_lock() {
    global_lock_mutex_.lock();
    if (global_lock_depth_ > 0) {
        ++global_lock_depth_;
        return;
    }
    if (cfg_.metadata_mmap) {
        if (shared_control_ == nullptr) {
            global_lock_mutex_.unlock();
            throw std::runtime_error("FDP mmap metadata is not attached");
        }
        auto* ctrl = static_cast<SharedMetadataControl*>(shared_control_);
        const int rc = ::pthread_mutex_lock(&ctrl->metadata_mutex);
        if (rc != 0) {
            global_lock_mutex_.unlock();
            throw std::runtime_error(
                std::string("pthread_mutex_lock FDP metadata failed: ") +
                std::strerror(rc));
        }
        global_lock_depth_ = 1;
        allocated_chunks_ = recalc_allocated_chunks();
        const MetadataHeader* hdr = metadata_header();
        if (hdr->magic == kMetadataMagic) {
            seq_ = hdr->seq;
            last_flush_iter_ = hdr->last_flush_iter;
        }
        return;
    }
    if (global_lock_path_.empty()) {
        global_lock_mutex_.unlock();
        throw std::runtime_error("FDP global lock path is empty");
    }
    if (global_lock_fd_ < 0) {
        global_lock_fd_ = ::open(
            global_lock_path_.c_str(), O_CREAT | O_RDWR | O_CLOEXEC, 0600);
        if (global_lock_fd_ < 0) {
            global_lock_mutex_.unlock();
            throw std::runtime_error(
                std::string("open FDP global lock failed: ") +
                std::strerror(errno));
        }
    }
    if (::flock(global_lock_fd_, LOCK_EX) != 0) {
        global_lock_mutex_.unlock();
        throw std::runtime_error(
            std::string("flock FDP global lock failed: ") +
            std::strerror(errno));
    }
    global_lock_depth_ = 1;
    try {
        read_metadata_image();
        allocated_chunks_ = recalc_allocated_chunks();
        const MetadataHeader* hdr = metadata_header();
        if (hdr->magic == kMetadataMagic) {
            seq_ = hdr->seq;
            last_flush_iter_ = hdr->last_flush_iter;
        }
    } catch (...) {
        global_lock_depth_ = 0;
        (void)::flock(global_lock_fd_, LOCK_UN);
        global_lock_mutex_.unlock();
        throw;
    }
}

void FdpStore::release_global_metadata_lock() {
    if (global_lock_depth_ == 0) {
        return;
    }
    --global_lock_depth_;
    if (global_lock_depth_ != 0) {
        global_lock_mutex_.unlock();
        return;
    }
    if (cfg_.metadata_mmap) {
        auto* ctrl = static_cast<SharedMetadataControl*>(shared_control_);
        const int rc = ::pthread_mutex_unlock(&ctrl->metadata_mutex);
        global_lock_mutex_.unlock();
        if (rc != 0) {
            throw std::runtime_error(
                std::string("pthread_mutex_unlock FDP metadata failed: ") +
                std::strerror(rc));
        }
        return;
    }
    if (global_lock_fd_ >= 0 && ::flock(global_lock_fd_, LOCK_UN) != 0) {
        global_lock_mutex_.unlock();
        throw std::runtime_error(
            std::string("unlock FDP global lock failed: ") +
            std::strerror(errno));
    }
    global_lock_mutex_.unlock();
}

bool FdpStore::global_metadata_lock_held() const {
    return global_lock_depth_ > 0;
}

void FdpStore::persist_metadata_image_locked(uint64_t iter, bool bump_seq) {
    MetadataCommitPerfStats perf;
    if (perf_diag_) {
        perf.valid = true;
        perf.iter = iter;
        perf.owner_rank = cfg_.rank;
        perf.metadata_image_bytes = layout_.image_bytes;
        perf.nvme_flush_enabled = cfg_.nvme_flush;
        perf.start_mono_raw_ns = monotonic_raw_ns();
    }
    allocated_chunks_ = recalc_allocated_chunks();
    if (perf_diag_) perf.recalc_done_mono_raw_ns = monotonic_raw_ns();
    update_metadata_header(iter, bump_seq);
    if (perf_diag_) perf.header_done_mono_raw_ns = monotonic_raw_ns();
    write_metadata_image();
    if (perf_diag_) perf.image_write_done_mono_raw_ns = monotonic_raw_ns();
    if (cfg_.nvme_flush) {
        device_.flush(metadata_ring_, ns_.nsid);
    }
    if (perf_diag_) {
        perf.flush_done_mono_raw_ns = monotonic_raw_ns();
        perf.end_mono_raw_ns = perf.flush_done_mono_raw_ns;
        std::lock_guard<std::mutex> lock(metadata_perf_mutex_);
        last_metadata_commit_perf_ = perf;
    }
}

MetadataCommitPerfStats FdpStore::last_metadata_commit_perf() const {
    std::lock_guard<std::mutex> lock(metadata_perf_mutex_);
    return last_metadata_commit_perf_;
}

UcmdRingRuntimeStats FdpStore::nefc_ring_runtime_stats() const {
    return nefc_data_ring_.runtime_stats();
}

UcmdRingRuntimeStats FdpStore::expert_ring_runtime_stats() const {
    return expert_data_ring_.runtime_stats();
}

void FdpStore::flush_metadata(uint64_t iter) {
    std::unique_lock<std::shared_mutex> op_lock(op_mutex_);
    std::lock_guard<std::recursive_mutex> bitmap_lock(bitmap_mutex_);
    if (!open_) {
        throw std::runtime_error("flush_metadata: FDP session is not open");
    }
    const bool had_lock = global_metadata_lock_held();
    if (!had_lock) {
        acquire_global_metadata_lock();
    }
    try {
        persist_metadata_image_locked(iter, true);
    } catch (...) {
        if (!had_lock) {
            release_global_metadata_lock();
        }
        throw;
    }
    if (!had_lock) {
        release_global_metadata_lock();
    }
}

void FdpStore::begin_data_batch() {
    std::unique_lock<std::shared_mutex> lock(op_mutex_);
    std::lock_guard<std::mutex> state_lock(batch_state_mutex_);
    if (batch_lock_.owns_lock()) {
        throw std::runtime_error("begin_data_batch: batch already active");
    }
    if (nefc_batch_lock_.owns_lock() || expert_batch_lock_.owns_lock()) {
        throw std::runtime_error(
            "begin_data_batch: per-ring batch already active");
    }
    if (!open_) {
        throw std::runtime_error("begin_data_batch: FDP session is not open");
    }
    if (!cfg_.metadata_mmap) {
        acquire_global_metadata_lock();
    }
    nefc_batch_ring_lock_ =
        std::unique_lock<std::recursive_mutex>(nefc_ring_mutex_);
    expert_batch_ring_lock_ =
        std::unique_lock<std::recursive_mutex>(expert_ring_mutex_);
    batch_owned_buffers_.clear();
    nefc_data_ring_.begin_batch();
    try {
        expert_data_ring_.begin_batch();
    } catch (...) {
        (void)nefc_data_ring_.end_batch();
        nefc_batch_ring_lock_.unlock();
        expert_batch_ring_lock_.unlock();
        if (!cfg_.metadata_mmap) {
            release_global_metadata_lock();
        }
        throw;
    }
    batch_owner_token_.store(current_thread_token(), std::memory_order_release);
    batch_lock_ = std::move(lock);
}

std::vector<UcmdRingBatchStats> FdpStore::end_data_batch_stats() {
    std::lock_guard<std::mutex> state_lock(batch_state_mutex_);
    if (!batch_lock_.owns_lock()) {
        throw std::runtime_error("end_data_batch: no active batch");
    }
    if (!open_) {
        throw std::runtime_error("end_data_batch: FDP session is not open");
    }
    std::vector<UcmdRingBatchStats> out(2);
    std::exception_ptr first_error;
    if (cfg_.metadata_mmap) {
        try {
            prepare_io_plans(nefc_batch_plans_);
            prepare_io_plans(expert_batch_plans_);
        } catch (...) {
            first_error = std::current_exception();
        }
    } else {
        try {
            persist_metadata_image_locked(last_flush_iter_, true);
        } catch (...) {
            first_error = std::current_exception();
        }
        try {
            release_global_metadata_lock();
        } catch (...) {
            if (!first_error) {
                first_error = std::current_exception();
            }
        }
    }
    try {
        (void)nefc_data_ring_.end_batch();
        out[0] = nefc_data_ring_.last_batch_stats();
    } catch (...) {
        first_error = std::current_exception();
    }
    try {
        (void)expert_data_ring_.end_batch();
        out[1] = expert_data_ring_.last_batch_stats();
    } catch (...) {
        if (!first_error) {
            first_error = std::current_exception();
        }
    }
    if (first_error) {
        batch_owned_buffers_.clear();
        nefc_batch_plans_.clear();
        expert_batch_plans_.clear();
        batch_owner_token_.store(0, std::memory_order_release);
        nefc_batch_ring_lock_.unlock();
        expert_batch_ring_lock_.unlock();
        batch_lock_.unlock();
        std::rethrow_exception(first_error);
    }
    batch_owned_buffers_.clear();
    nefc_batch_plans_.clear();
    expert_batch_plans_.clear();
    batch_owner_token_.store(0, std::memory_order_release);
    nefc_batch_ring_lock_.unlock();
    expert_batch_ring_lock_.unlock();
    batch_lock_.unlock();
    if (first_error) {
        std::rethrow_exception(first_error);
    }
    return out;
}

std::vector<uint32_t> FdpStore::end_data_batch() {
    std::vector<UcmdRingBatchStats> stats = end_data_batch_stats();
    std::vector<uint32_t> out(2, 0);
    out[0] = stats.size() > 0 ? stats[0].commands : 0;
    out[1] = stats.size() > 1 ? stats[1].commands : 0;
    return out;
}

void FdpStore::begin_nefc_batch() {
    std::shared_lock<std::shared_mutex> lock(op_mutex_);
    std::lock_guard<std::mutex> state_lock(batch_state_mutex_);
    if (batch_lock_.owns_lock()) {
        throw std::runtime_error("begin_nefc_batch: global batch already active");
    }
    if (nefc_batch_lock_.owns_lock()) {
        throw std::runtime_error("begin_nefc_batch: NEFC batch already active");
    }
    if (!open_) {
        throw std::runtime_error("begin_nefc_batch: FDP session is not open");
    }
    if (!cfg_.metadata_mmap) {
        acquire_global_metadata_lock();
    }
    nefc_batch_ring_lock_ =
        std::unique_lock<std::recursive_mutex>(nefc_ring_mutex_);
    try {
        nefc_data_ring_.begin_batch();
    } catch (...) {
        nefc_batch_ring_lock_.unlock();
        if (!cfg_.metadata_mmap) {
            release_global_metadata_lock();
        }
        throw;
    }
    nefc_batch_owner_token_.store(current_thread_token(),
                                  std::memory_order_release);
    nefc_batch_lock_ = std::move(lock);
}

UcmdRingBatchStats FdpStore::end_nefc_batch_stats() {
    {
        std::lock_guard<std::mutex> state_lock(batch_state_mutex_);
        if (!nefc_batch_lock_.owns_lock()) {
            throw std::runtime_error("end_nefc_batch: no active NEFC batch");
        }
        if (!open_) {
            throw std::runtime_error("end_nefc_batch: FDP session is not open");
        }
    }
    std::exception_ptr first_error;
    if (cfg_.metadata_mmap) {
        try {
            prepare_io_plans(nefc_batch_plans_);
        } catch (...) {
            first_error = std::current_exception();
        }
    } else {
        try {
            persist_metadata_image_locked(last_flush_iter_, true);
        } catch (...) {
            first_error = std::current_exception();
        }
        try {
            release_global_metadata_lock();
        } catch (...) {
            if (!first_error) {
                first_error = std::current_exception();
            }
        }
    }
    try {
        (void)nefc_data_ring_.end_batch();
        const UcmdRingBatchStats stats = nefc_data_ring_.last_batch_stats();
        std::lock_guard<std::mutex> state_lock(batch_state_mutex_);
        nefc_batch_owner_token_.store(0, std::memory_order_release);
        nefc_batch_plans_.clear();
        nefc_batch_ring_lock_.unlock();
        nefc_batch_lock_.unlock();
        if (first_error) {
            std::rethrow_exception(first_error);
        }
        return stats;
    } catch (...) {
        std::lock_guard<std::mutex> state_lock(batch_state_mutex_);
        nefc_batch_owner_token_.store(0, std::memory_order_release);
        nefc_batch_plans_.clear();
        nefc_batch_ring_lock_.unlock();
        nefc_batch_lock_.unlock();
        throw;
    }
}

uint32_t FdpStore::end_nefc_batch() {
    return end_nefc_batch_stats().commands;
}

void FdpStore::begin_expert_batch_impl(bool read_only) {
    std::shared_lock<std::shared_mutex> lock(op_mutex_);
    std::lock_guard<std::mutex> state_lock(batch_state_mutex_);
    if (batch_lock_.owns_lock()) {
        throw std::runtime_error(
            "begin_expert_batch: global batch already active");
    }
    if (expert_batch_lock_.owns_lock()) {
        throw std::runtime_error(
            "begin_expert_batch: expert batch already active");
    }
    if (!open_) {
        throw std::runtime_error("begin_expert_batch: FDP session is not open");
    }
    const bool write_metadata_lock = !read_only && !cfg_.metadata_mmap;
    if (write_metadata_lock) {
        acquire_global_metadata_lock();
    }
    expert_batch_ring_lock_ =
        std::unique_lock<std::recursive_mutex>(expert_ring_mutex_);
    if (read_only) {
        // Match the writer's ring -> metadata order in mmap mode.  Holding
        // this lock through end_batch keeps chunk lists stable until every
        // queued DMA has completed.
        try {
            acquire_global_metadata_lock();
        } catch (...) {
            expert_batch_ring_lock_.unlock();
            throw;
        }
    }
    if (!read_only) {
        batch_owned_buffers_.clear();
    }
    try {
        expert_data_ring_.begin_batch();
    } catch (...) {
        expert_batch_ring_lock_.unlock();
        if (read_only || write_metadata_lock) {
            release_global_metadata_lock();
        }
        throw;
    }
    expert_batch_read_only_ = read_only;
    expert_batch_owner_token_.store(
        read_only ? 0 : current_thread_token(), std::memory_order_release);
    expert_batch_lock_ = std::move(lock);
}

void FdpStore::begin_expert_batch() {
    begin_expert_batch_impl(false);
}

void FdpStore::begin_expert_read_batch() {
    begin_expert_batch_impl(true);
}

UcmdRingBatchStats FdpStore::end_expert_batch_stats_impl(bool read_only) {
    {
        std::lock_guard<std::mutex> state_lock(batch_state_mutex_);
        if (!expert_batch_lock_.owns_lock()) {
            throw std::runtime_error("end_expert_batch: no active expert batch");
        }
        if (expert_batch_read_only_ != read_only) {
            throw std::runtime_error(
                read_only
                    ? "end_expert_read_batch: active batch is write-only"
                    : "end_expert_batch: active batch is read-only");
        }
        if (!open_) {
            throw std::runtime_error(
                "end_expert_batch: FDP session is not open");
        }
    }
    std::exception_ptr first_error;
    // A read batch only needs the ring completion fence.  In particular, it
    // must not prepare write plans or persist metadata during recovery.
    if (!read_only) {
        if (cfg_.metadata_mmap) {
            try {
                prepare_io_plans(expert_batch_plans_);
            } catch (...) {
                first_error = std::current_exception();
            }
        } else {
            try {
                persist_metadata_image_locked(last_flush_iter_, true);
            } catch (...) {
                first_error = std::current_exception();
            }
            try {
                release_global_metadata_lock();
            } catch (...) {
                if (!first_error) {
                    first_error = std::current_exception();
                }
            }
        }
    }
    UcmdRingBatchStats stats;
    try {
        (void)expert_data_ring_.end_batch();
        stats = expert_data_ring_.last_batch_stats();
    } catch (...) {
        if (!first_error) {
            first_error = std::current_exception();
        }
    }
    if (read_only) {
        try {
            release_global_metadata_lock();
        } catch (...) {
            if (!first_error) {
                first_error = std::current_exception();
            }
        }
    }
    {
        std::lock_guard<std::mutex> state_lock(batch_state_mutex_);
        if (!read_only) {
            batch_owned_buffers_.clear();
            expert_batch_plans_.clear();
        }
        expert_batch_owner_token_.store(0, std::memory_order_release);
        expert_batch_read_only_ = false;
        expert_batch_ring_lock_.unlock();
        expert_batch_lock_.unlock();
    }
    if (first_error) {
        std::rethrow_exception(first_error);
    }
    return stats;
}

UcmdRingBatchStats FdpStore::end_expert_batch_stats() {
    return end_expert_batch_stats_impl(false);
}

uint32_t FdpStore::end_expert_batch() {
    return end_expert_batch_stats().commands;
}

UcmdRingBatchStats FdpStore::end_expert_read_batch_stats() {
    return end_expert_batch_stats_impl(true);
}

uint32_t FdpStore::end_expert_read_batch() {
    return end_expert_read_batch_stats().commands;
}

void FdpStore::begin_checkpoint_iter(uint64_t iter) {
    if (!cfg_.metadata_mmap) {
        return;
    }
    if (!open_) {
        throw std::runtime_error(
            "begin_checkpoint_iter: FDP session is not open");
    }

    {
        std::lock_guard<std::mutex> local_lock(local_iter_mutex_);
        if (local_iter_ != iter) {
            local_iter_ = iter;
            local_nonexpert_done_ = false;
            local_expert_done_ = false;
            local_process_done_reported_ = false;
        }
    }

    acquire_global_metadata_lock();
    try {
        auto* ctrl = static_cast<SharedMetadataControl*>(shared_control_);
        if (ctrl == nullptr) {
            throw std::runtime_error(
                "begin_checkpoint_iter: FDP mmap metadata is not attached");
        }
        if (ctrl->active_iter != iter) {
            if (ctrl->flush_done == 0 && ctrl->completed_processes != 0) {
                std::ostringstream oss;
                oss << "begin_checkpoint_iter: previous iteration "
                    << ctrl->active_iter << " has incomplete metadata flush";
                throw std::runtime_error(oss.str());
            }
            ctrl->active_iter = iter;
            ctrl->expected_processes = cfg_.world_size;
            ctrl->completed_processes = 0;
            ctrl->flush_started = 0;
            ctrl->flush_done = 0;
            ctrl->flush_owner_rank = 0xffffffffu;
            ctrl->error_rank = 0xffffffffu;
            ctrl->reserved[0] = 0;
        }
        release_global_metadata_lock();
    } catch (...) {
        release_global_metadata_lock();
        throw;
    }
}

bool FdpStore::mark_process_done_locked(uint64_t iter) {
    auto* ctrl = static_cast<SharedMetadataControl*>(shared_control_);
    if (ctrl == nullptr) {
        throw std::runtime_error(
            "mark_process_done_locked: FDP mmap metadata is not attached");
    }
    if (ctrl->active_iter != iter) {
        std::ostringstream oss;
        oss << "mark_process_done_locked: active iter "
            << ctrl->active_iter << " does not match " << iter;
        throw std::runtime_error(oss.str());
    }
    if (cfg_.rank < 64 && ctrl->expected_processes <= 64) {
        const uint64_t rank_bit = (uint64_t{1} << cfg_.rank);
        if ((ctrl->reserved[0] & rank_bit) != 0) {
            return false;
        }
        ctrl->reserved[0] |= rank_bit;
        ctrl->completed_processes = popcount_u64(ctrl->reserved[0]);
    } else {
        if (ctrl->completed_processes >= ctrl->expected_processes) {
            return false;
        }
        ctrl->completed_processes += 1;
    }
    if (ctrl->completed_processes != ctrl->expected_processes ||
        ctrl->flush_started != 0) {
        return false;
    }

    ctrl->flush_started = 1;
    ctrl->flush_owner_rank = cfg_.rank;
    persist_metadata_image_locked(iter, true);
    ctrl->metadata_flush_count += 1;
    ctrl->flush_done = 1;
    return true;
}

bool FdpStore::mark_writer_done(uint64_t iter,
                                const std::string& writer_kind) {
    if (!cfg_.metadata_mmap) {
        return false;
    }
    if (!open_) {
        throw std::runtime_error(
            "mark_writer_done: FDP session is not open");
    }

    bool report_process_done = false;
    {
        std::lock_guard<std::mutex> local_lock(local_iter_mutex_);
        if (local_iter_ != iter) {
            if (iter < local_iter_) {
                return false;
            }
            local_iter_ = iter;
            local_nonexpert_done_ = false;
            local_expert_done_ = false;
            local_process_done_reported_ = false;
        }
        if (writer_kind == "nonexpert" || writer_kind == "nefc") {
            local_nonexpert_done_ = true;
        } else if (writer_kind == "expert") {
            local_expert_done_ = true;
        } else if (writer_kind == "all") {
            local_nonexpert_done_ = true;
            local_expert_done_ = true;
        } else {
            throw std::invalid_argument(
                "mark_writer_done: writer_kind must be nonexpert, nefc, expert, or all");
        }
        if (local_nonexpert_done_ && local_expert_done_) {
            local_process_done_reported_ = true;
            report_process_done = true;
        }
    }
    if (!report_process_done) {
        return false;
    }

    acquire_global_metadata_lock();
    try {
        const bool did_flush = mark_process_done_locked(iter);
        release_global_metadata_lock();
        return did_flush;
    } catch (...) {
        release_global_metadata_lock();
        throw;
    }
}

size_t FdpStore::write_nefc(const std::string& object_name,
                            int64_t pinned_addr,
                            const std::string& header,
                            uint64_t payload_bytes,
                            uint64_t total_padded_bytes,
                            uint64_t iter) {
    std::shared_lock<std::shared_mutex> op_lock;
    const size_t thread_token = current_thread_token();
    const bool owns_batch =
        batch_owner_token_.load(std::memory_order_acquire) == thread_token ||
        nefc_batch_owner_token_.load(std::memory_order_acquire) == thread_token;
    if (!owns_batch) {
        op_lock = std::shared_lock<std::shared_mutex>(op_mutex_);
    }
    (void)object_name;
    if (!open_) {
        throw std::runtime_error("write_nefc: FDP session is not open");
    }
    const uint64_t logical_bytes = kHeaderBlockBytes + payload_bytes;
    if (logical_bytes > total_padded_bytes) {
        throw std::invalid_argument("write_nefc: logical bytes exceed padded bytes");
    }
    if ((total_padded_bytes % ns_.lba_size) != 0) {
        throw std::invalid_argument("write_nefc: padded bytes are not LBA-aligned");
    }
    const uint64_t needed64 = div_round_up_u64(total_padded_bytes,
                                               cfg_.chunk_bytes);
    if (needed64 == 0 || needed64 > kMaxChunksNefc) {
        throw std::runtime_error("write_nefc: chunk count exceeds NEFC cap");
    }
    const uint32_t needed = static_cast<uint32_t>(needed64);

    stamp_header_into_pinned(pinned_addr, header);
    std::vector<FdpChunkId> chunks;
    const bool metadata_lock_here = cfg_.metadata_mmap || !owns_batch;
    if (metadata_lock_here) {
        acquire_global_metadata_lock();
    }
    try {
        chunks = allocate_chunks(needed);
        NefcEntry* entry = nefc_entry();
        const NefcEntry old = *entry;
        if (old.valid) {
            free_chunks(old.chunks, old.chunk_count);
        }
        std::memset(entry, 0, sizeof(*entry));
        entry->valid = 1;
        entry->rank = cfg_.rank;
        entry->dp_rank = cfg_.dp_rank;
        entry->pp_rank = cfg_.pp_rank;
        entry->ep_rank = cfg_.ep_rank;
        entry->shard_index = cfg_.nefc_shard_index;
        entry->iter = iter;
        entry->logical_bytes = logical_bytes;
        entry->chunk_count = needed;
        for (uint32_t i = 0; i < needed; ++i) {
            entry->chunks[i] = chunks[i];
        }
        if (owns_batch || cfg_.metadata_mmap) {
            update_metadata_header(iter, true);
        } else {
            persist_metadata_image_locked(iter, true);
        }
        if (metadata_lock_here) {
            release_global_metadata_lock();
        }
    } catch (...) {
        if (metadata_lock_here) {
            release_global_metadata_lock();
        }
        throw;
    }

    if (cfg_.metadata_mmap && owns_batch) {
        FdpIoPlan plan;
        plan.kind = FdpIoPlanKind::ChunkListWrite;
        plan.ring = FdpIoPlanRing::Nefc;
        plan.pinned_addr = pinned_addr;
        plan.total_padded_bytes = total_padded_bytes;
        plan.chunks = std::move(chunks);
        plan.where = "write_nefc";
        enqueue_io_plan(std::move(plan));
        return static_cast<size_t>(total_padded_bytes);
    }

    {
        std::lock_guard<std::recursive_mutex> ring_lock(nefc_ring_mutex_);
        write_chunk_list(nefc_data_ring_, pinned_addr, chunks,
                         total_padded_bytes, "write_nefc");
    }
    return static_cast<size_t>(total_padded_bytes);
}

size_t FdpStore::write_lfcs(const std::string& object_name,
                            int64_t pinned_addr,
                            const std::string& header,
                            uint64_t payload_bytes,
                            uint64_t total_padded_bytes,
                            uint32_t layer_id,
                            uint32_t expert_id,
                            uint64_t iter,
                            uint64_t generation) {
    std::shared_lock<std::shared_mutex> op_lock;
    const size_t thread_token = current_thread_token();
    const bool owns_batch =
        batch_owner_token_.load(std::memory_order_acquire) == thread_token ||
        expert_batch_owner_token_.load(std::memory_order_acquire) == thread_token;
    if (!owns_batch) {
        op_lock = std::shared_lock<std::shared_mutex>(op_mutex_);
    }
    (void)object_name;
    if (!open_) {
        throw std::runtime_error("write_lfcs: FDP session is not open");
    }
    const uint32_t slot = expert_slot(layer_id, expert_id);
    const uint32_t global_expert = global_expert_id(expert_id);
    const uint64_t logical_bytes = kHeaderBlockBytes + payload_bytes;
    if (logical_bytes > total_padded_bytes) {
        throw std::invalid_argument("write_lfcs: logical bytes exceed padded bytes");
    }
    if ((total_padded_bytes % ns_.lba_size) != 0) {
        throw std::invalid_argument("write_lfcs: padded bytes are not LBA-aligned");
    }
    const uint64_t needed64 = div_round_up_u64(total_padded_bytes,
                                               cfg_.chunk_bytes);
    if (needed64 == 0 || needed64 > kMaxChunksLfcs) {
        throw std::runtime_error("write_lfcs: chunk count exceeds LFCS cap");
    }
    const uint32_t needed = static_cast<uint32_t>(needed64);

    stamp_header_into_pinned(pinned_addr, header);
    std::vector<FdpChunkId> chunks;
    const bool metadata_lock_here = cfg_.metadata_mmap || !owns_batch;
    if (metadata_lock_here) {
        acquire_global_metadata_lock();
    }
    try {
        chunks = allocate_chunks(needed);
        LfcsEntry* lfcs = lfcs_entry(slot);
        LfsyEntry* lfsy = lfsy_entry(slot);
        const LfcsEntry old_lfcs = *lfcs;
        const LfsyEntry old_lfsy = *lfsy;
        if (old_lfcs.valid) {
            free_chunks(old_lfcs.chunks, old_lfcs.chunk_count);
        }
        if (old_lfsy.valid) {
            free_chunks(old_lfsy.chunks, old_lfsy.chunk_count);
        }
        erase_lfsy_handles_for_slot(slot);

        std::memset(lfcs, 0, sizeof(*lfcs));
        lfcs->valid = 1;
        lfcs->rank = cfg_.rank;
        lfcs->layer_id = layer_id;
        lfcs->expert_id = global_expert;
        lfcs->dp_rank = cfg_.dp_rank;
        lfcs->pp_rank = cfg_.pp_rank;
        lfcs->ep_rank = cfg_.ep_rank;
        lfcs->generation = generation;
        lfcs->iter = iter;
        lfcs->logical_bytes = logical_bytes;
        lfcs->chunk_count = needed;
        for (uint32_t i = 0; i < needed; ++i) {
            lfcs->chunks[i] = chunks[i];
        }

        std::memset(lfsy, 0, sizeof(*lfsy));
        if (owns_batch || cfg_.metadata_mmap) {
            update_metadata_header(iter, true);
        } else {
            persist_metadata_image_locked(iter, true);
        }
        if (metadata_lock_here) {
            release_global_metadata_lock();
        }
    } catch (...) {
        if (metadata_lock_here) {
            release_global_metadata_lock();
        }
        throw;
    }

    if (cfg_.metadata_mmap && owns_batch) {
        FdpIoPlan plan;
        plan.kind = FdpIoPlanKind::ChunkListWrite;
        plan.ring = FdpIoPlanRing::Expert;
        plan.pinned_addr = pinned_addr;
        plan.total_padded_bytes = total_padded_bytes;
        plan.chunks = std::move(chunks);
        plan.where = "write_lfcs";
        enqueue_io_plan(std::move(plan));
        return static_cast<size_t>(total_padded_bytes);
    }

    {
        std::lock_guard<std::recursive_mutex> ring_lock(expert_ring_mutex_);
        write_chunk_list(expert_data_ring_, pinned_addr, chunks,
                         total_padded_bytes, "write_lfcs");
    }
    return static_cast<size_t>(total_padded_bytes);
}

int64_t FdpStore::open_lfsy(const std::string& object_name,
                            const std::string& header,
                            uint32_t layer_id,
                            uint32_t expert_id,
                            int64_t start_iter,
                            uint64_t generation) {
    std::shared_lock<std::shared_mutex> op_lock;
    const size_t thread_token = current_thread_token();
    const bool owns_batch =
        batch_owner_token_.load(std::memory_order_acquire) == thread_token ||
        expert_batch_owner_token_.load(std::memory_order_acquire) == thread_token;
    if (!owns_batch) {
        op_lock = std::shared_lock<std::shared_mutex>(op_mutex_);
    }
    (void)object_name;
    if (!open_) {
        throw std::runtime_error("open_lfsy: FDP session is not open");
    }
    const uint32_t slot = expert_slot(layer_id, expert_id);
    const uint32_t global_expert = global_expert_id(expert_id);
    const uint64_t start_iter_u64 =
        start_iter < 0 ? 0 : static_cast<uint64_t>(start_iter);

    AlignedBuffer header_buf;
    header_buf.allocate(static_cast<size_t>(kHeaderBlockBytes), 4096);
    stamp_header_into_pinned(
        reinterpret_cast<int64_t>(header_buf.data()), header);

    std::vector<FdpChunkId> chunks;
    int64_t handle = 0;
    const bool metadata_lock_here = cfg_.metadata_mmap || !owns_batch;
    if (metadata_lock_here) {
        acquire_global_metadata_lock();
    }
    try {
        LfsyEntry* entry = lfsy_entry(slot);
        const LfsyEntry old = *entry;
        chunks = allocate_chunks(1);
        if (old.valid) {
            free_chunks(old.chunks, old.chunk_count);
        }
        erase_lfsy_handles_for_slot(slot);

        std::memset(entry, 0, sizeof(*entry));
        entry->valid = 1;
        entry->rank = cfg_.rank;
        entry->layer_id = layer_id;
        entry->expert_id = global_expert;
        entry->dp_rank = cfg_.dp_rank;
        entry->pp_rank = cfg_.pp_rank;
        entry->ep_rank = cfg_.ep_rank;
        entry->generation = generation;
        entry->start_iter = start_iter_u64;
        entry->logical_tail = kHeaderBlockBytes;
        entry->chunk_count = 1;
        entry->chunks[0] = chunks[0];

        handle = next_lfsy_handle_++;
        LfsyHandle h;
        h.slot = slot;
        h.layer_id = layer_id;
        h.expert_id = expert_id;
        h.generation = generation;
        h.last_iter = entry->start_iter;
        lfsy_handles_.emplace(handle, h);
        if (owns_batch || cfg_.metadata_mmap) {
            update_metadata_header(entry->start_iter, true);
        } else {
            persist_metadata_image_locked(entry->start_iter, true);
        }
        if (metadata_lock_here) {
            release_global_metadata_lock();
        }
    } catch (...) {
        if (metadata_lock_here) {
            release_global_metadata_lock();
        }
        throw;
    }

    if (cfg_.metadata_mmap && owns_batch) {
        FdpIoPlan plan;
        plan.kind = FdpIoPlanKind::ChunkListWrite;
        plan.ring = FdpIoPlanRing::Expert;
        plan.total_padded_bytes = kHeaderBlockBytes;
        plan.chunks = chunks;
        plan.where = "open_lfsy";
        plan.owned_buffer = std::move(header_buf);
        plan.pinned_addr = reinterpret_cast<int64_t>(plan.owned_buffer.data());
        enqueue_io_plan(std::move(plan));
        return handle;
    }

    {
        std::lock_guard<std::recursive_mutex> ring_lock(expert_ring_mutex_);
        const uint64_t header_nlb = kHeaderBlockBytes / ns_.lba_size;
        raw_write_data(expert_data_ring_, header_buf.data(),
                       data_chunk_lba(chunks[0]), header_nlb,
                       "open_lfsy");
        if (expert_data_ring_.is_batching()) {
            batch_owned_buffers_.push_back(std::move(header_buf));
        }
    }
    return handle;
}

size_t FdpStore::append_lfsy(int64_t handle,
                             int64_t pinned_addr,
                             uint64_t total_bytes,
                             uint64_t total_padded_bytes,
                             uint32_t layer_id,
                             uint32_t expert_id,
                             uint64_t iter,
                             uint64_t generation) {
    std::shared_lock<std::shared_mutex> op_lock;
    const size_t thread_token = current_thread_token();
    const bool owns_batch =
        batch_owner_token_.load(std::memory_order_acquire) == thread_token ||
        expert_batch_owner_token_.load(std::memory_order_acquire) == thread_token;
    if (!owns_batch) {
        op_lock = std::shared_lock<std::shared_mutex>(op_mutex_);
    }
    if (!open_) {
        throw std::runtime_error("append_lfsy: FDP session is not open");
    }
    if (total_padded_bytes == 0) {
        return 0;
    }
    if (total_bytes > total_padded_bytes) {
        throw std::invalid_argument("append_lfsy: total bytes exceed padded bytes");
    }
    if ((pinned_addr & 4095ll) != 0) {
        throw std::invalid_argument("append_lfsy: pinned ptr not 4KiB-aligned");
    }
    if ((total_padded_bytes % ns_.lba_size) != 0) {
        throw std::invalid_argument("append_lfsy: padded bytes are not LBA-aligned");
    }

    uint64_t old_tail = 0;
    uint64_t new_tail = 0;
    std::vector<FdpChunkId> chunks_for_write;
    const bool metadata_lock_here = cfg_.metadata_mmap || !owns_batch;
    if (metadata_lock_here) {
        acquire_global_metadata_lock();
    }
    try {
        LfsyHandle& h = checked_lfsy_handle(
            handle, layer_id, expert_id, generation);
        LfsyEntry* entry = lfsy_entry(h.slot);
        old_tail = entry->logical_tail;
        new_tail = old_tail + total_padded_bytes;
        if (new_tail < old_tail) {
            throw std::overflow_error("append_lfsy: logical tail overflow");
        }
        const uint64_t needed64 = div_round_up_u64(new_tail, cfg_.chunk_bytes);
        if (needed64 == 0 || needed64 > kMaxChunksLfsy) {
            throw std::runtime_error("append_lfsy: chunk count exceeds LFSY cap");
        }
        if (needed64 < entry->chunk_count) {
            throw std::runtime_error(
                "append_lfsy: corrupt logical tail/chunk_count");
        }

        if (needed64 > entry->chunk_count) {
            std::vector<FdpChunkId> added_chunks = allocate_chunks(
                static_cast<uint32_t>(needed64 - entry->chunk_count));
            for (FdpChunkId chunk : added_chunks) {
                entry->chunks[entry->chunk_count++] = chunk;
            }
        }

        entry->logical_tail = new_tail;
        h.last_iter = iter;
        chunks_for_write.assign(entry->chunks, entry->chunks + entry->chunk_count);
        if (owns_batch || cfg_.metadata_mmap) {
            update_metadata_header(iter, true);
        } else {
            persist_metadata_image_locked(iter, true);
        }
        if (metadata_lock_here) {
            release_global_metadata_lock();
        }
    } catch (...) {
        if (metadata_lock_here) {
            release_global_metadata_lock();
        }
        throw;
    }

    if (cfg_.metadata_mmap && owns_batch) {
        FdpIoPlan plan;
        plan.kind = FdpIoPlanKind::LfsyAppend;
        plan.ring = FdpIoPlanRing::Expert;
        plan.pinned_addr = pinned_addr;
        plan.total_padded_bytes = total_padded_bytes;
        plan.logical_offset = old_tail;
        plan.chunks = std::move(chunks_for_write);
        plan.where = "append_lfsy";
        enqueue_io_plan(std::move(plan));
        return static_cast<size_t>(total_padded_bytes);
    }

    {
        std::lock_guard<std::recursive_mutex> ring_lock(expert_ring_mutex_);
        const uint8_t* src = reinterpret_cast<const uint8_t*>(pinned_addr);
        uint64_t remaining = total_padded_bytes;
        uint64_t logical = old_tail;
        uint64_t src_off = 0;
        while (remaining > 0) {
            const uint64_t chunk_index = logical / cfg_.chunk_bytes;
            const uint64_t offset_in_chunk = logical % cfg_.chunk_bytes;
            const uint64_t chunk_left = cfg_.chunk_bytes - offset_in_chunk;
            const uint64_t len = std::min<uint64_t>(remaining, chunk_left);
            if ((offset_in_chunk % ns_.lba_size) != 0 ||
                (len % ns_.lba_size) != 0) {
                throw std::runtime_error("append_lfsy: unaligned append range");
            }
            const FdpChunkId chunk_id =
                chunks_for_write.at(static_cast<size_t>(chunk_index));
            const uint64_t slba =
                data_chunk_lba(chunk_id) + offset_in_chunk / ns_.lba_size;
            raw_write_data(expert_data_ring_, src + src_off, slba,
                           len / ns_.lba_size, "append_lfsy");
            logical += len;
            src_off += len;
            remaining -= len;
        }
    }
    return static_cast<size_t>(total_padded_bytes);
}

void FdpStore::fsync_handle(int64_t handle) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    if (!open_) {
        throw std::runtime_error("fsync_handle: FDP session is not open");
    }
    auto it = lfsy_handles_.find(handle);
    if (it == lfsy_handles_.end()) {
        throw std::runtime_error("fsync_handle: unknown LFSY handle");
    }
    flush_metadata(it->second.last_iter);
}

void FdpStore::close_lfsy(int64_t handle) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    lfsy_handles_.erase(handle);
}

std::string FdpStore::debug_read_nefc(uint64_t offset, uint64_t nbytes) {
    if (!open_) {
        throw std::runtime_error("debug_read_nefc: FDP session is not open");
    }
    const NefcEntry* entry = nefc_entry();
    if (!entry->valid) {
        throw std::runtime_error("debug_read_nefc: NEFC entry is not valid");
    }
    return read_chunk_bytes(nefc_data_ring_, entry->chunks, entry->chunk_count,
                            entry->logical_bytes, offset, nbytes,
                            "debug_read_nefc");
}

std::string FdpStore::debug_read_lfcs(uint32_t layer_id, uint32_t expert_id,
                                      uint64_t offset, uint64_t nbytes) {
    if (!open_) {
        throw std::runtime_error("debug_read_lfcs: FDP session is not open");
    }
    const uint32_t slot = expert_slot(layer_id, expert_id);
    const LfcsEntry* entry = lfcs_entry(slot);
    if (!entry->valid) {
        throw std::runtime_error("debug_read_lfcs: LFCS entry is not valid");
    }
    return read_chunk_bytes(expert_data_ring_, entry->chunks, entry->chunk_count,
                            entry->logical_bytes, offset, nbytes,
                            "debug_read_lfcs");
}

std::string FdpStore::debug_read_lfsy(uint32_t layer_id, uint32_t expert_id,
                                      uint64_t offset, uint64_t nbytes) {
    if (!open_) {
        throw std::runtime_error("debug_read_lfsy: FDP session is not open");
    }
    const uint32_t slot = expert_slot(layer_id, expert_id);
    const LfsyEntry* entry = lfsy_entry(slot);
    if (!entry->valid) {
        throw std::runtime_error("debug_read_lfsy: LFSY entry is not valid");
    }
    return read_chunk_bytes(expert_data_ring_, entry->chunks, entry->chunk_count,
                            entry->logical_tail, offset, nbytes,
                            "debug_read_lfsy");
}

size_t FdpStore::read_nefc_into(int64_t dst_addr, uint64_t dst_bytes) {
    return read_nefc_shard_into(cfg_.nefc_shard_index, dst_addr, dst_bytes);
}

size_t FdpStore::read_nefc_shard_into(uint32_t shard_index,
                                      int64_t dst_addr,
                                      uint64_t dst_bytes) {
    if (!open_) {
        throw std::runtime_error(
            "read_nefc_shard_into: FDP session is not open");
    }
    const NefcEntry* entry = nefc_entry(shard_index);
    if (!entry->valid) {
        throw std::runtime_error(
            "read_nefc_shard_into: NEFC entry is not valid");
    }
    return read_chunk_list_into(nefc_data_ring_, entry->chunks,
                                entry->chunk_count, entry->logical_bytes,
                                dst_addr, dst_bytes, "read_nefc_shard_into");
}

size_t FdpStore::read_lfcs_into(uint32_t layer_id, uint32_t expert_id,
                                int64_t dst_addr, uint64_t dst_bytes) {
    std::lock_guard<std::recursive_mutex> ring_lock(expert_ring_mutex_);
    if (!open_) {
        throw std::runtime_error("read_lfcs_into: FDP session is not open");
    }
    acquire_global_metadata_lock();
    size_t result = 0;
    try {
        const uint32_t slot = expert_slot(layer_id, expert_id);
        const LfcsEntry* entry = lfcs_entry(slot);
        if (!entry->valid) {
            throw std::runtime_error(
                "read_lfcs_into: LFCS entry is not valid");
        }
        result = read_chunk_list_into(
            expert_data_ring_, entry->chunks, entry->chunk_count,
            entry->logical_bytes, dst_addr, dst_bytes, "read_lfcs_into");
    } catch (...) {
        release_global_metadata_lock();
        throw;
    }
    release_global_metadata_lock();
    return result;
}

size_t FdpStore::read_lfsy_into(uint32_t layer_id, uint32_t expert_id,
                                int64_t dst_addr, uint64_t dst_bytes) {
    std::lock_guard<std::recursive_mutex> ring_lock(expert_ring_mutex_);
    if (!open_) {
        throw std::runtime_error("read_lfsy_into: FDP session is not open");
    }
    acquire_global_metadata_lock();
    size_t result = 0;
    try {
        const uint32_t slot = expert_slot(layer_id, expert_id);
        const LfsyEntry* entry = lfsy_entry(slot);
        if (!entry->valid) {
            throw std::runtime_error(
                "read_lfsy_into: LFSY entry is not valid");
        }
        result = read_chunk_list_into(
            expert_data_ring_, entry->chunks, entry->chunk_count,
            entry->logical_tail, dst_addr, dst_bytes, "read_lfsy_into");
    } catch (...) {
        release_global_metadata_lock();
        throw;
    }
    release_global_metadata_lock();
    return result;
}

void FdpStore::validate_metadata_range(uint64_t slba, uint64_t nlb,
                                       const char* where) const {
    const uint64_t end = checked_range_end(slba, nlb, where);
    const uint64_t metadata_end = cfg_.metadata_start_lba + cfg_.metadata_nlb;
    if (slba < cfg_.metadata_start_lba || end > metadata_end) {
        throw std::runtime_error(std::string(where) +
                                 ": LBA range outside metadata region");
    }
}

void FdpStore::raw_read_metadata(void* buffer, uint64_t slba, uint64_t nlb,
                                 const char* where) {
    validate_metadata_range(slba, nlb, where);
    uint8_t* ptr = static_cast<uint8_t*>(buffer);
    uint64_t left = nlb;
    uint64_t cur = slba;
    while (left > 0) {
        const uint32_t batch = static_cast<uint32_t>(
            std::min<uint64_t>(left, max_io_nlb_));
        device_.read_lba(metadata_ring_, ns_.nsid, ptr, cur, batch,
                         ns_.lba_size);
        ptr += static_cast<uint64_t>(batch) * ns_.lba_size;
        cur += batch;
        left -= batch;
    }
}

void FdpStore::raw_write_metadata(const void* buffer, uint64_t slba,
                                  uint64_t nlb, const char* where) {
    validate_metadata_range(slba, nlb, where);
    const uint8_t* ptr = static_cast<const uint8_t*>(buffer);
    uint64_t left = nlb;
    uint64_t cur = slba;
    while (left > 0) {
        const uint32_t batch = static_cast<uint32_t>(
            std::min<uint64_t>(left, max_io_nlb_));
        device_.write_lba(metadata_ring_, ns_.nsid, ptr, cur, batch,
                          ns_.lba_size, cfg_.write_fua);
        ptr += static_cast<uint64_t>(batch) * ns_.lba_size;
        cur += batch;
        left -= batch;
    }
}

void FdpStore::raw_write_data(UcmdRing& ring, const void* buffer,
                              uint64_t slba, uint64_t nlb,
                              const char* where) {
    const uint64_t end = checked_range_end(slba, nlb, where);
    const uint64_t data_end = cfg_.data_start_lba + cfg_.data_nlb;
    if (slba < cfg_.data_start_lba || end > data_end) {
        throw std::runtime_error(std::string(where) +
                                 ": LBA range outside data region");
    }
    const uint8_t* ptr = static_cast<const uint8_t*>(buffer);
    uint64_t left = nlb;
    uint64_t cur = slba;
    while (left > 0) {
        const uint32_t batch = static_cast<uint32_t>(
            std::min<uint64_t>(left, max_io_nlb_));
        device_.write_lba(ring, ns_.nsid, ptr, cur, batch, ns_.lba_size,
                          cfg_.write_fua);
        ptr += static_cast<uint64_t>(batch) * ns_.lba_size;
        cur += batch;
        left -= batch;
    }
}

void FdpStore::raw_read_data(UcmdRing& ring, void* buffer,
                             uint64_t slba, uint64_t nlb,
                             const char* where) {
    const uint64_t end = checked_range_end(slba, nlb, where);
    const uint64_t data_end = cfg_.data_start_lba + cfg_.data_nlb;
    if (slba < cfg_.data_start_lba || end > data_end) {
        throw std::runtime_error(std::string(where) +
                                 ": LBA range outside data region");
    }
    uint8_t* ptr = static_cast<uint8_t*>(buffer);
    uint64_t left = nlb;
    uint64_t cur = slba;
    while (left > 0) {
        const uint32_t batch = static_cast<uint32_t>(
            std::min<uint64_t>(left, max_io_nlb_));
        device_.read_lba(ring, ns_.nsid, ptr, cur, batch, ns_.lba_size);
        ptr += static_cast<uint64_t>(batch) * ns_.lba_size;
        cur += batch;
        left -= batch;
    }
}

uint8_t* FdpStore::metadata_bytes() {
    if (cfg_.metadata_mmap && shared_metadata_bytes_ != nullptr) {
        return shared_metadata_bytes_;
    }
    return static_cast<uint8_t*>(metadata_image_.data());
}

const uint8_t* FdpStore::metadata_bytes() const {
    if (cfg_.metadata_mmap && shared_metadata_bytes_ != nullptr) {
        return shared_metadata_bytes_;
    }
    return static_cast<const uint8_t*>(metadata_image_.data());
}

MetadataHeader* FdpStore::metadata_header() {
    return reinterpret_cast<MetadataHeader*>(metadata_bytes());
}

const MetadataHeader* FdpStore::metadata_header() const {
    return reinterpret_cast<const MetadataHeader*>(metadata_bytes());
}

uint8_t* FdpStore::bitmap_bytes() {
    return metadata_bytes() + layout_.bitmap_offset;
}

const uint8_t* FdpStore::bitmap_bytes() const {
    return metadata_bytes() + layout_.bitmap_offset;
}

NefcEntry* FdpStore::nefc_entry(uint32_t shard) {
    if (shard >= cfg_.nefc_shard_count) {
        throw std::out_of_range("nefc_entry: shard out of range");
    }
    return reinterpret_cast<NefcEntry*>(
        metadata_bytes() + layout_.nefc_offset +
        static_cast<uint64_t>(shard) * sizeof(NefcEntry));
}

const NefcEntry* FdpStore::nefc_entry(uint32_t shard) const {
    if (shard >= cfg_.nefc_shard_count) {
        throw std::out_of_range("nefc_entry: shard out of range");
    }
    return reinterpret_cast<const NefcEntry*>(
        metadata_bytes() + layout_.nefc_offset +
        static_cast<uint64_t>(shard) * sizeof(NefcEntry));
}

NefcEntry* FdpStore::nefc_entry() {
    return nefc_entry(cfg_.nefc_shard_index);
}

const NefcEntry* FdpStore::nefc_entry() const {
    return nefc_entry(cfg_.nefc_shard_index);
}

LfcsEntry* FdpStore::lfcs_entry(uint32_t slot) {
    if (slot >= cfg_.expert_slots) {
        throw std::out_of_range("lfcs_entry: expert slot out of range");
    }
    return reinterpret_cast<LfcsEntry*>(
        metadata_bytes() + layout_.lfcs_offset +
        static_cast<uint64_t>(slot) * sizeof(LfcsEntry));
}

const LfcsEntry* FdpStore::lfcs_entry(uint32_t slot) const {
    if (slot >= cfg_.expert_slots) {
        throw std::out_of_range("lfcs_entry: expert slot out of range");
    }
    return reinterpret_cast<const LfcsEntry*>(
        metadata_bytes() + layout_.lfcs_offset +
        static_cast<uint64_t>(slot) * sizeof(LfcsEntry));
}

LfsyEntry* FdpStore::lfsy_entry(uint32_t slot) {
    if (slot >= cfg_.expert_slots) {
        throw std::out_of_range("lfsy_entry: expert slot out of range");
    }
    return reinterpret_cast<LfsyEntry*>(
        metadata_bytes() + layout_.lfsy_offset +
        static_cast<uint64_t>(slot) * sizeof(LfsyEntry));
}

const LfsyEntry* FdpStore::lfsy_entry(uint32_t slot) const {
    if (slot >= cfg_.expert_slots) {
        throw std::out_of_range("lfsy_entry: expert slot out of range");
    }
    return reinterpret_cast<const LfsyEntry*>(
        metadata_bytes() + layout_.lfsy_offset +
        static_cast<uint64_t>(slot) * sizeof(LfsyEntry));
}

std::vector<FdpChunkId> FdpStore::allocate_chunks(uint32_t count) {
    std::lock_guard<std::recursive_mutex> lock(bitmap_mutex_);
    if (count == 0) {
        throw std::invalid_argument("allocate_chunks: count must be > 0");
    }
    std::vector<FdpChunkId> out;
    out.reserve(count);
    uint8_t* bm = bitmap_bytes();
    for (uint32_t chunk = 0; chunk < data_chunk_count_ &&
         out.size() < count; ++chunk) {
        const uint32_t byte_idx = chunk >> 3;
        const uint8_t bit = static_cast<uint8_t>(1u << (chunk & 7u));
        if ((bm[byte_idx] & bit) == 0) {
            bm[byte_idx] = static_cast<uint8_t>(bm[byte_idx] | bit);
            out.push_back(chunk);
            ++allocated_chunks_;
        }
    }
    if (out.size() != count) {
        free_chunks(out.data(), static_cast<uint32_t>(out.size()));
        throw std::runtime_error("allocate_chunks: no free FDP chunks");
    }
    return out;
}

void FdpStore::free_chunks(const FdpChunkId* chunks, uint32_t count) {
    std::lock_guard<std::recursive_mutex> lock(bitmap_mutex_);
    if (count == 0) return;
    if (chunks == nullptr) {
        throw std::invalid_argument("free_chunks: null chunks");
    }
    uint8_t* bm = bitmap_bytes();
    for (uint32_t i = 0; i < count; ++i) {
        const FdpChunkId chunk = chunks[i];
        if (chunk >= data_chunk_count_) {
            throw std::out_of_range("free_chunks: chunk id out of range");
        }
        const uint32_t byte_idx = chunk >> 3;
        const uint8_t bit = static_cast<uint8_t>(1u << (chunk & 7u));
        if ((bm[byte_idx] & bit) == 0) {
            throw std::runtime_error("free_chunks: double free");
        }
        bm[byte_idx] = static_cast<uint8_t>(bm[byte_idx] & ~bit);
        if (allocated_chunks_ > 0) --allocated_chunks_;
    }
}

uint64_t FdpStore::data_chunk_lba(FdpChunkId chunk_id) const {
    if (chunk_id >= data_chunk_count_) {
        throw std::out_of_range("data_chunk_lba: chunk id out of range");
    }
    return cfg_.data_start_lba +
           static_cast<uint64_t>(chunk_id) * chunk_lba_count_;
}

void FdpStore::write_chunk_list(UcmdRing& ring, int64_t pinned_addr,
                                const std::vector<FdpChunkId>& chunks,
                                uint64_t total_padded_bytes,
                                const char* where) {
    if (chunks.empty()) {
        throw std::invalid_argument(std::string(where) + ": empty chunks");
    }
    if ((pinned_addr & 4095ll) != 0) {
        throw std::invalid_argument(std::string(where) +
                                    ": pinned ptr not 4KiB-aligned");
    }
    if (total_padded_bytes == 0 ||
        (total_padded_bytes % ns_.lba_size) != 0) {
        throw std::invalid_argument(std::string(where) +
                                    ": padded bytes not LBA-aligned");
    }
    const uint64_t capacity =
        static_cast<uint64_t>(chunks.size()) * cfg_.chunk_bytes;
    if (total_padded_bytes > capacity) {
        throw std::invalid_argument(std::string(where) +
                                    ": chunk list too small");
    }
    const uint8_t* base = reinterpret_cast<const uint8_t*>(pinned_addr);
    uint64_t remaining = total_padded_bytes;
    for (size_t i = 0; i < chunks.size() && remaining > 0; ++i) {
        const uint64_t len = std::min<uint64_t>(cfg_.chunk_bytes, remaining);
        if ((len % ns_.lba_size) != 0) {
            throw std::invalid_argument(std::string(where) +
                                        ": chunk write length not LBA-aligned");
        }
        const uint64_t nlb = len / ns_.lba_size;
        const uint64_t slba = data_chunk_lba(chunks[i]);
        raw_write_data(ring, base + static_cast<uint64_t>(i) * cfg_.chunk_bytes,
                       slba, nlb, where);
        remaining -= len;
    }
}

void FdpStore::enqueue_io_plan(FdpIoPlan&& plan) {
    if (plan.ring == FdpIoPlanRing::Nefc) {
        nefc_batch_plans_.push_back(std::move(plan));
    } else {
        expert_batch_plans_.push_back(std::move(plan));
    }
}

void FdpStore::prepare_io_plan(const FdpIoPlan& plan) {
    UcmdRing& ring =
        plan.ring == FdpIoPlanRing::Nefc ? nefc_data_ring_ : expert_data_ring_;
    const char* where =
        plan.where.empty() ? "fdp_io_plan" : plan.where.c_str();
    if (plan.kind == FdpIoPlanKind::ChunkListWrite) {
        write_chunk_list(ring, plan.pinned_addr, plan.chunks,
                         plan.total_padded_bytes, where);
        return;
    }
    if (plan.kind == FdpIoPlanKind::LfsyAppend) {
        if ((plan.pinned_addr & 4095ll) != 0) {
            throw std::invalid_argument(
                std::string(where) + ": pinned ptr not 4KiB-aligned");
        }
        const uint8_t* src = reinterpret_cast<const uint8_t*>(plan.pinned_addr);
        uint64_t remaining = plan.total_padded_bytes;
        uint64_t logical = plan.logical_offset;
        uint64_t src_off = 0;
        while (remaining > 0) {
            const uint64_t chunk_index = logical / cfg_.chunk_bytes;
            const uint64_t offset_in_chunk = logical % cfg_.chunk_bytes;
            const uint64_t chunk_left = cfg_.chunk_bytes - offset_in_chunk;
            const uint64_t len = std::min<uint64_t>(remaining, chunk_left);
            if ((offset_in_chunk % ns_.lba_size) != 0 ||
                (len % ns_.lba_size) != 0) {
                throw std::runtime_error(
                    std::string(where) + ": unaligned append range");
            }
            if (chunk_index >= plan.chunks.size()) {
                throw std::runtime_error(
                    std::string(where) + ": chunk index exceeds plan");
            }
            const FdpChunkId chunk_id =
                plan.chunks[static_cast<size_t>(chunk_index)];
            const uint64_t slba =
                data_chunk_lba(chunk_id) + offset_in_chunk / ns_.lba_size;
            raw_write_data(ring, src + src_off, slba,
                           len / ns_.lba_size, where);
            logical += len;
            src_off += len;
            remaining -= len;
        }
        return;
    }
    throw std::runtime_error("unknown FDP IO plan kind");
}

void FdpStore::prepare_io_plans(std::vector<FdpIoPlan>& plans) {
    for (const FdpIoPlan& plan : plans) {
        prepare_io_plan(plan);
    }
}

std::string FdpStore::read_chunk_bytes(UcmdRing& ring,
                                       const FdpChunkId* chunks,
                                       uint32_t chunk_count,
                                       uint64_t logical_limit,
                                       uint64_t offset,
                                       uint64_t nbytes,
                                       const char* where) {
    if (nbytes == 0) {
        return std::string();
    }
    if (chunks == nullptr || chunk_count == 0) {
        throw std::invalid_argument(std::string(where) + ": empty chunks");
    }
    if (offset > logical_limit ||
        nbytes > logical_limit - offset) {
        throw std::out_of_range(std::string(where) +
                                ": requested range exceeds logical size");
    }
    if (nbytes > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        throw std::runtime_error(std::string(where) +
                                 ": requested range exceeds size_t");
    }

    std::string out(static_cast<size_t>(nbytes), '\0');
    uint64_t copied = 0;
    while (copied < nbytes) {
        const uint64_t logical = offset + copied;
        const uint64_t chunk_index = logical / cfg_.chunk_bytes;
        if (chunk_index >= chunk_count) {
            throw std::runtime_error(std::string(where) +
                                     ": chunk index exceeds entry chunk_count");
        }
        const uint64_t offset_in_chunk = logical % cfg_.chunk_bytes;
        const uint64_t chunk_left = cfg_.chunk_bytes - offset_in_chunk;
        const uint64_t len = std::min<uint64_t>(nbytes - copied, chunk_left);

        const uint64_t lba_prefix = offset_in_chunk % ns_.lba_size;
        const uint64_t aligned_offset = offset_in_chunk - lba_prefix;
        const uint64_t aligned_bytes =
            round_up_u64(lba_prefix + len, ns_.lba_size);
        if (aligned_offset + aligned_bytes > cfg_.chunk_bytes) {
            throw std::runtime_error(std::string(where) +
                                     ": aligned range crosses chunk boundary");
        }

        AlignedBuffer tmp;
        tmp.allocate(static_cast<size_t>(aligned_bytes), 4096);
        const FdpChunkId chunk_id =
            chunks[static_cast<uint32_t>(chunk_index)];
        const uint64_t slba =
            data_chunk_lba(chunk_id) + aligned_offset / ns_.lba_size;
        raw_read_data(ring, tmp.data(), slba,
                      aligned_bytes / ns_.lba_size, where);
        const char* src = static_cast<const char*>(tmp.data()) + lba_prefix;
        std::memcpy(&out[static_cast<size_t>(copied)], src,
                    static_cast<size_t>(len));
        copied += len;
    }
    return out;
}

size_t FdpStore::read_chunk_list_into(UcmdRing& ring,
                                      const FdpChunkId* chunks,
                                      uint32_t chunk_count,
                                      uint64_t logical_bytes,
                                      int64_t dst_addr,
                                      uint64_t dst_bytes,
                                      const char* where) {
    if (logical_bytes == 0) {
        return 0;
    }
    if (chunks == nullptr || chunk_count == 0) {
        throw std::invalid_argument(std::string(where) + ": empty chunks");
    }
    if ((dst_addr & 4095ll) != 0) {
        throw std::invalid_argument(std::string(where) +
                                    ": destination ptr not 4KiB-aligned");
    }
    const uint64_t read_bytes = round_up_u64(logical_bytes, ns_.lba_size);
    if (read_bytes > dst_bytes) {
        throw std::invalid_argument(std::string(where) +
                                    ": destination buffer too small");
    }
    if ((read_bytes % ns_.lba_size) != 0) {
        throw std::invalid_argument(std::string(where) +
                                    ": read length not LBA-aligned");
    }
    const uint64_t capacity =
        static_cast<uint64_t>(chunk_count) * cfg_.chunk_bytes;
    if (read_bytes > capacity) {
        throw std::invalid_argument(std::string(where) +
                                    ": chunk list too small");
    }

    uint8_t* base = reinterpret_cast<uint8_t*>(dst_addr);
    uint64_t remaining = read_bytes;
    for (uint32_t i = 0; i < chunk_count && remaining > 0; ++i) {
        const uint64_t len = std::min<uint64_t>(cfg_.chunk_bytes, remaining);
        if ((len % ns_.lba_size) != 0) {
            throw std::invalid_argument(std::string(where) +
                                        ": chunk read length not LBA-aligned");
        }
        const uint64_t slba = data_chunk_lba(chunks[i]);
        raw_read_data(ring, base + static_cast<uint64_t>(i) * cfg_.chunk_bytes,
                      slba, len / ns_.lba_size, where);
        remaining -= len;
    }
    return static_cast<size_t>(logical_bytes);
}

uint32_t FdpStore::expert_slot(uint32_t layer_id, uint32_t expert_id) const {
    if (cfg_.experts_per_layer == 0 || cfg_.expert_slots == 0) {
        throw std::runtime_error(
            "write_lfcs requires PROPOSAL_FDP_NUM_MOE_LAYERS and "
            "PROPOSAL_FDP_EXPERTS_PER_LAYER");
    }
    const uint32_t global_expert = global_expert_id(expert_id);
    if (global_expert >= cfg_.experts_per_layer) {
        throw std::out_of_range("expert_slot: expert_id out of range");
    }
    const uint64_t rank_base =
        (static_cast<uint64_t>(cfg_.dp_rank) * cfg_.pp_size + cfg_.pp_rank) *
        static_cast<uint64_t>(cfg_.num_moe_layers) *
        static_cast<uint64_t>(cfg_.experts_per_layer);
    const uint64_t slot = rank_base +
                          static_cast<uint64_t>(layer_id) *
                              cfg_.experts_per_layer +
                          global_expert;
    if (slot >= cfg_.expert_slots) {
        throw std::out_of_range("expert_slot: layer/expert out of range");
    }
    return static_cast<uint32_t>(slot);
}

uint32_t FdpStore::global_expert_id(uint32_t expert_id) const {
    if (cfg_.expert_ids_global || cfg_.ep_size <= 1) {
        return expert_id;
    }
    const uint64_t mapped =
        static_cast<uint64_t>(cfg_.expert_id_base) + expert_id;
    if (mapped > std::numeric_limits<uint32_t>::max()) {
        throw std::out_of_range("global_expert_id: mapped id overflow");
    }
    return static_cast<uint32_t>(mapped);
}

uint32_t FdpStore::current_nefc_shard() const {
    const uint64_t shard =
        static_cast<uint64_t>(cfg_.pp_rank) *
            static_cast<uint64_t>(cfg_.dp_size) *
            static_cast<uint64_t>(cfg_.ep_size) +
        static_cast<uint64_t>(cfg_.dp_rank) *
            static_cast<uint64_t>(cfg_.ep_size) +
        static_cast<uint64_t>(cfg_.ep_rank);
    if (shard > std::numeric_limits<uint32_t>::max()) {
        throw std::out_of_range("current_nefc_shard: shard overflow");
    }
    return static_cast<uint32_t>(shard);
}

void FdpStore::erase_lfsy_handles_for_slot(uint32_t slot) {
    for (auto it = lfsy_handles_.begin(); it != lfsy_handles_.end(); ) {
        if (it->second.slot == slot) {
            it = lfsy_handles_.erase(it);
        } else {
            ++it;
        }
    }
}

LfsyHandle& FdpStore::checked_lfsy_handle(int64_t handle,
                                          uint32_t layer_id,
                                          uint32_t expert_id,
                                          uint64_t generation) {
    auto it = lfsy_handles_.find(handle);
    if (it == lfsy_handles_.end()) {
        throw std::runtime_error("checked_lfsy_handle: unknown LFSY handle");
    }
    LfsyHandle& h = it->second;
    const uint32_t slot = expert_slot(layer_id, expert_id);
    const uint32_t global_expert = global_expert_id(expert_id);
    if (h.slot != slot || h.layer_id != layer_id ||
        h.expert_id != expert_id || h.generation != generation) {
        throw std::runtime_error("checked_lfsy_handle: stale LFSY handle");
    }
    LfsyEntry* entry = lfsy_entry(slot);
    if (!entry->valid || entry->generation != generation ||
        entry->layer_id != layer_id || entry->expert_id != global_expert) {
        throw std::runtime_error("checked_lfsy_handle: stale LFSY metadata");
    }
    return h;
}

void FdpStore::stamp_header_into_pinned(int64_t pinned_addr,
                                        const std::string& header) const {
    if (pinned_addr == 0) {
        throw std::invalid_argument("stamp_header_into_pinned: null pinned ptr");
    }
    if ((pinned_addr & 4095ll) != 0) {
        throw std::invalid_argument(
            "stamp_header_into_pinned: pinned ptr not 4KiB-aligned");
    }
    if (header.size() > kHeaderBlockBytes) {
        throw std::invalid_argument(
            "stamp_header_into_pinned: header exceeds header block");
    }
    char* dst = reinterpret_cast<char*>(pinned_addr);
    std::memcpy(dst, header.data(), header.size());
    if (header.size() < kHeaderBlockBytes) {
        std::memset(dst + header.size(), 0,
                    static_cast<size_t>(kHeaderBlockBytes - header.size()));
    }
}

uint64_t FdpStore::recalc_allocated_chunks() const {
    const uint8_t* bm = bitmap_bytes();
    uint64_t count = 0;
    for (uint64_t i = 0; i < layout_.bitmap_bytes; ++i) {
        count += popcount_byte(bm[i]);
    }
    const uint64_t excess_bits =
        layout_.bitmap_bytes * 8ull - data_chunk_count_;
    if (excess_bits > 0) {
        const uint8_t last = bm[layout_.bitmap_bytes - 1];
        const uint8_t mask = static_cast<uint8_t>(
            0xffu << (8u - static_cast<uint32_t>(excess_bits)));
        count -= popcount_byte(last & mask);
    }
    return count;
}

uint32_t FdpStore::debug_alloc_chunk() {
    std::lock_guard<std::recursive_mutex> lock(bitmap_mutex_);
    if (!open_) {
        throw std::runtime_error("debug_alloc_chunk: FDP session is not open");
    }
    uint8_t* bm = bitmap_bytes();
    for (uint32_t chunk = 0; chunk < data_chunk_count_; ++chunk) {
        const uint32_t byte_idx = chunk >> 3;
        const uint8_t bit = static_cast<uint8_t>(1u << (chunk & 7u));
        if ((bm[byte_idx] & bit) == 0) {
            bm[byte_idx] = static_cast<uint8_t>(bm[byte_idx] | bit);
            ++allocated_chunks_;
            return chunk;
        }
    }
    throw std::runtime_error("debug_alloc_chunk: no free FDP chunks");
}

void FdpStore::debug_free_chunk(uint32_t chunk_id) {
    std::lock_guard<std::recursive_mutex> lock(bitmap_mutex_);
    if (!open_) {
        throw std::runtime_error("debug_free_chunk: FDP session is not open");
    }
    if (chunk_id >= data_chunk_count_) {
        throw std::invalid_argument("debug_free_chunk: chunk id out of range");
    }
    uint8_t* bm = bitmap_bytes();
    const uint32_t byte_idx = chunk_id >> 3;
    const uint8_t bit = static_cast<uint8_t>(1u << (chunk_id & 7u));
    if ((bm[byte_idx] & bit) == 0) {
        throw std::runtime_error("debug_free_chunk: double free");
    }
    bm[byte_idx] = static_cast<uint8_t>(bm[byte_idx] & ~bit);
    --allocated_chunks_;
}

MetadataSummary FdpStore::debug_read_metadata_summary() {
    if (!open_) {
        throw std::runtime_error(
            "debug_read_metadata_summary: FDP session is not open");
    }
    AlignedBuffer tmp;
    tmp.allocate(static_cast<size_t>(layout_.image_bytes), 4096);
    raw_read_metadata(tmp.data(), metadata_image_lba_, metadata_image_nlb_,
                      "debug_read_metadata_summary");
    const MetadataHeader* hdr = tmp.as<MetadataHeader>();
    const uint8_t* base = static_cast<const uint8_t*>(tmp.data());
    MetadataSummary out;
    out.magic = hdr->magic;
    out.version = hdr->version;
    out.seq = hdr->seq;
    out.last_flush_iter = hdr->last_flush_iter;
    out.allocated_chunks = hdr->allocated_chunks;
    out.data_chunk_count = hdr->data_chunk_count;
    out.bitmap_bytes = hdr->bitmap_bytes;
    out.bitmap_offset = hdr->bitmap_offset;
    out.nefc_offset = hdr->nefc_offset;
    out.lfcs_offset = hdr->lfcs_offset;
    out.lfsy_offset = hdr->lfsy_offset;
    out.metadata_image_bytes = hdr->metadata_image_bytes;
    if (hdr->magic == kMetadataMagic &&
        hdr->nefc_offset + sizeof(NefcEntry) <= hdr->metadata_image_bytes) {
        const uint32_t nefc_count = std::max<uint32_t>(1, hdr->nefc_shard_count);
        out.nefc_shard_count = nefc_count;
        for (uint32_t shard = 0; shard < nefc_count; ++shard) {
            const uint64_t nefc_off =
                hdr->nefc_offset +
                static_cast<uint64_t>(shard) * sizeof(NefcEntry);
            if (nefc_off + sizeof(NefcEntry) > hdr->metadata_image_bytes) {
                break;
            }
            const NefcEntry* nefc = reinterpret_cast<const NefcEntry*>(
                base + nefc_off);
            if (nefc->valid) {
                ++out.nefc_valid_count;
            }
            if (shard == cfg_.nefc_shard_index) {
                out.nefc_valid = nefc->valid != 0;
                out.nefc_iter = nefc->iter;
                out.nefc_logical_bytes = nefc->logical_bytes;
                out.nefc_chunk_count = nefc->chunk_count;
            }
        }
    }
    if (hdr->magic == kMetadataMagic &&
        hdr->lfcs_offset < hdr->metadata_image_bytes &&
        hdr->lfsy_offset < hdr->metadata_image_bytes) {
        const uint32_t slots = hdr->expert_slots;
        for (uint32_t slot = 0; slot < slots; ++slot) {
            const uint64_t lfcs_off =
                hdr->lfcs_offset + static_cast<uint64_t>(slot) * sizeof(LfcsEntry);
            const uint64_t lfsy_off =
                hdr->lfsy_offset + static_cast<uint64_t>(slot) * sizeof(LfsyEntry);
            if (lfcs_off + sizeof(LfcsEntry) <= hdr->metadata_image_bytes) {
                const LfcsEntry* lfcs = reinterpret_cast<const LfcsEntry*>(
                    base + lfcs_off);
                if (lfcs->valid) {
                    ++out.lfcs_valid_count;
                    out.lfcs_total_chunks += lfcs->chunk_count;
                    out.lfcs_max_chunk_count =
                        std::max(out.lfcs_max_chunk_count, lfcs->chunk_count);
                }
            }
            if (lfsy_off + sizeof(LfsyEntry) <= hdr->metadata_image_bytes) {
                const LfsyEntry* lfsy = reinterpret_cast<const LfsyEntry*>(
                    base + lfsy_off);
                if (lfsy->valid) ++out.lfsy_valid_count;
            }
        }
    }
    return out;
}

NefcEntrySnapshot FdpStore::read_nefc_entry() {
    if (!open_) {
        throw std::runtime_error("read_nefc_entry: FDP session is not open");
    }
    const NefcEntry* entry = nefc_entry();
    NefcEntrySnapshot snap;
    snap.valid = entry->valid != 0;
    snap.shard_index = cfg_.nefc_shard_index;
    snap.rank = entry->rank;
    snap.dp_rank = entry->dp_rank;
    snap.pp_rank = entry->pp_rank;
    snap.ep_rank = entry->ep_rank;
    snap.iter = entry->iter;
    snap.logical_bytes = entry->logical_bytes;
    snap.chunk_count = entry->chunk_count;
    return snap;
}

std::vector<NefcEntrySnapshot> FdpStore::read_nefc_entries() {
    if (!open_) {
        throw std::runtime_error("read_nefc_entries: FDP session is not open");
    }
    std::vector<NefcEntrySnapshot> out;
    out.reserve(cfg_.nefc_shard_count);
    for (uint32_t shard = 0; shard < cfg_.nefc_shard_count; ++shard) {
        const NefcEntry* entry = nefc_entry(shard);
        NefcEntrySnapshot snap;
        snap.valid = entry->valid != 0;
        snap.shard_index = shard;
        snap.rank = entry->rank;
        snap.dp_rank = entry->dp_rank;
        snap.pp_rank = entry->pp_rank;
        snap.ep_rank = entry->ep_rank;
        snap.iter = entry->iter;
        snap.logical_bytes = entry->logical_bytes;
        snap.chunk_count = entry->chunk_count;
        out.push_back(std::move(snap));
    }
    return out;
}

std::vector<LfcsEntrySnapshot> FdpStore::read_lfcs_entries() {
    if (!open_) {
        throw std::runtime_error("read_lfcs_entries: FDP session is not open");
    }
    std::vector<LfcsEntrySnapshot> out;
    out.reserve(cfg_.expert_slots);
    for (uint32_t slot = 0; slot < cfg_.expert_slots; ++slot) {
        const LfcsEntry* entry = lfcs_entry(slot);
        LfcsEntrySnapshot snap;
        snap.valid = entry->valid != 0;
        snap.slot = slot;
        snap.rank = entry->rank;
        snap.layer_id = entry->layer_id;
        snap.expert_id = entry->expert_id;
        snap.dp_rank = entry->dp_rank;
        snap.pp_rank = entry->pp_rank;
        snap.ep_rank = entry->ep_rank;
        snap.generation = entry->generation;
        snap.iter = entry->iter;
        snap.logical_bytes = entry->logical_bytes;
        snap.chunk_count = entry->chunk_count;
        const uint32_t chunk_count =
            std::min<uint32_t>(entry->chunk_count, kMaxChunksLfcs);
        snap.chunks.reserve(chunk_count);
        for (uint32_t i = 0; i < chunk_count; ++i) {
            snap.chunks.push_back(entry->chunks[i]);
        }
        out.push_back(std::move(snap));
    }
    return out;
}

std::vector<LfsyEntrySnapshot> FdpStore::read_lfsy_entries() {
    if (!open_) {
        throw std::runtime_error("read_lfsy_entries: FDP session is not open");
    }
    std::vector<LfsyEntrySnapshot> out;
    out.reserve(cfg_.expert_slots);
    for (uint32_t slot = 0; slot < cfg_.expert_slots; ++slot) {
        const LfsyEntry* entry = lfsy_entry(slot);
        LfsyEntrySnapshot snap;
        snap.valid = entry->valid != 0;
        snap.slot = slot;
        snap.rank = entry->rank;
        snap.layer_id = entry->layer_id;
        snap.expert_id = entry->expert_id;
        snap.dp_rank = entry->dp_rank;
        snap.pp_rank = entry->pp_rank;
        snap.ep_rank = entry->ep_rank;
        snap.generation = entry->generation;
        snap.start_iter = entry->start_iter;
        snap.logical_tail = entry->logical_tail;
        snap.chunk_count = entry->chunk_count;
        const uint32_t chunk_count =
            std::min<uint32_t>(entry->chunk_count, kMaxChunksLfsy);
        snap.chunks.reserve(chunk_count);
        for (uint32_t i = 0; i < chunk_count; ++i) {
            snap.chunks.push_back(entry->chunks[i]);
        }
        out.push_back(std::move(snap));
    }
    return out;
}

std::vector<LfsyEntrySnapshot> FdpStore::debug_read_lfsy_entries() {
    if (!open_) {
        throw std::runtime_error(
            "debug_read_lfsy_entries: FDP session is not open");
    }
    AlignedBuffer tmp;
    tmp.allocate(static_cast<size_t>(layout_.image_bytes), 4096);
    raw_read_metadata(tmp.data(), metadata_image_lba_, metadata_image_nlb_,
                      "debug_read_lfsy_entries");
    const MetadataHeader* hdr = tmp.as<MetadataHeader>();
    if (hdr->magic != kMetadataMagic ||
        hdr->lfsy_offset >= hdr->metadata_image_bytes) {
        throw std::runtime_error("debug_read_lfsy_entries: bad metadata image");
    }

    const uint8_t* base = static_cast<const uint8_t*>(tmp.data());
    std::vector<LfsyEntrySnapshot> out;
    out.reserve(hdr->expert_slots);
    for (uint32_t slot = 0; slot < hdr->expert_slots; ++slot) {
        const uint64_t off =
            hdr->lfsy_offset + static_cast<uint64_t>(slot) * sizeof(LfsyEntry);
        if (off + sizeof(LfsyEntry) > hdr->metadata_image_bytes) {
            throw std::runtime_error(
                "debug_read_lfsy_entries: LFSY table exceeds metadata image");
        }
        const LfsyEntry* entry = reinterpret_cast<const LfsyEntry*>(base + off);
        LfsyEntrySnapshot snap;
        snap.valid = entry->valid != 0;
        snap.slot = slot;
        snap.rank = entry->rank;
        snap.layer_id = entry->layer_id;
        snap.expert_id = entry->expert_id;
        snap.generation = entry->generation;
        snap.start_iter = entry->start_iter;
        snap.logical_tail = entry->logical_tail;
        snap.chunk_count = entry->chunk_count;
        const uint32_t chunk_count =
            std::min<uint32_t>(entry->chunk_count, kMaxChunksLfsy);
        snap.chunks.reserve(chunk_count);
        for (uint32_t i = 0; i < chunk_count; ++i) {
            snap.chunks.push_back(entry->chunks[i]);
        }
        out.push_back(std::move(snap));
    }
    return out;
}

bool FdpStore::debug_raw_write_read_smoke() {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    if (!open_) {
        throw std::runtime_error(
            "debug_raw_write_read_smoke: FDP session is not open");
    }
    const uint64_t scratch_lba =
        cfg_.metadata_start_lba + cfg_.metadata_nlb - 1;
    const uint64_t metadata_used_end =
        metadata_image_lba_ + metadata_image_nlb_;
    if (scratch_lba < metadata_used_end) {
        throw std::runtime_error(
            "debug_raw_write_read_smoke: metadata range has no reserved "
            "scratch LBA");
    }

    uint8_t* buf = static_cast<uint8_t*>(scratch_buf_.data());
    std::memset(buf, 0, scratch_buf_.bytes());
    for (uint32_t i = 0; i < ns_.lba_size; ++i) {
        buf[i] = static_cast<uint8_t>((i * 131u + 17u) & 0xffu);
    }
    raw_write_metadata(buf, scratch_lba, 1, "debug_smoke_write");
    if (cfg_.nvme_flush) {
        device_.flush(metadata_ring_, ns_.nsid);
    }

    AlignedBuffer readback;
    readback.allocate(static_cast<size_t>(
        std::max<uint64_t>(4096, ns_.lba_size)), 4096);
    raw_read_metadata(readback.data(), scratch_lba, 1, "debug_smoke_read");
    return std::memcmp(buf, readback.data(), ns_.lba_size) == 0;
}

FdpStatsSnapshot FdpStore::stats() const {
    FdpStatsSnapshot out;
    out.backend = "fdp";
    out.device_path = cfg_.device_path;
    out.nsid = ns_.nsid;
    out.lba_size = ns_.lba_size;
    out.namespace_nlb = ns_.namespace_nlb;
    out.fdp_supported = ns_.fdp_supported;
    out.fdp_pid_enabled = ns_.fdp_pid_enabled;
    out.metadata_start_lba = cfg_.metadata_start_lba;
    out.metadata_nlb = cfg_.metadata_nlb;
    out.metadata_image_lba = metadata_image_lba_;
    out.metadata_image_nlb = metadata_image_nlb_;
    out.metadata_image_bytes = layout_.image_bytes;
    out.data_start_lba = cfg_.data_start_lba;
    out.data_nlb = cfg_.data_nlb;
    out.chunk_bytes = cfg_.chunk_bytes;
    out.chunk_lba_count = chunk_lba_count_;
    out.data_chunk_count = data_chunk_count_;
    out.allocated_chunks = allocated_chunks_;
    out.rank = cfg_.rank;
    out.world_size = cfg_.world_size;
    out.num_moe_layers = cfg_.num_moe_layers;
    out.experts_per_layer = cfg_.experts_per_layer;
    out.expert_slots = cfg_.expert_slots;
    out.dp_rank = cfg_.dp_rank;
    out.dp_size = cfg_.dp_size;
    out.pp_rank = cfg_.pp_rank;
    out.pp_size = cfg_.pp_size;
    out.ep_rank = cfg_.ep_rank;
    out.ep_size = cfg_.ep_size;
    out.nefc_shard_index = cfg_.nefc_shard_index;
    out.nefc_shard_count = cfg_.nefc_shard_count;
    out.expert_id_base = cfg_.expert_id_base;
    out.expert_ids_global = cfg_.expert_ids_global;
    out.nefc_qd = cfg_.nefc_qd;
    out.expert_qd = cfg_.expert_qd;
    out.metadata_qd = cfg_.metadata_qd;
    out.max_io_nlb = max_io_nlb_;
    out.max_io_bytes = static_cast<uint64_t>(max_io_nlb_) * ns_.lba_size;
    out.write_fua = cfg_.write_fua;
    out.metadata_mmap = cfg_.metadata_mmap;
    if (cfg_.metadata_mmap && shared_control_ != nullptr) {
        auto* ctrl = static_cast<const SharedMetadataControl*>(shared_control_);
        out.metadata_flush_count = ctrl->metadata_flush_count;
    }
    out.seq = seq_;
    out.last_flush_iter = last_flush_iter_;
    return out;
}

FdpStore& global_store() {
    static FdpStore store;
    return store;
}

}}  // namespace proposal::fdp
