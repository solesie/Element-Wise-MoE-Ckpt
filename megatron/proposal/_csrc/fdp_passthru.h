#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

struct nvme_uring_cmd;

namespace proposal { namespace fdp {

struct NamespaceInfo {
    uint32_t nsid = 0;
    uint32_t lba_size = 0;
    uint64_t namespace_nlb = 0;
    bool fdp_supported = false;
    bool fdp_pid_enabled = false;
};

struct UcmdRingBatchStats {
    uint32_t commands = 0;
    uint32_t submitted = 0;
    uint32_t queue_depth = 0;
    uint32_t submit_calls = 0;
    uint32_t registered_file_submits = 0;
    bool sqpoll = false;
    bool blocking_completion = false;
    uint64_t batch_begin_mono_raw_ns = 0;
    uint64_t first_submit_mono_raw_ns = 0;
    uint64_t last_submit_mono_raw_ns = 0;
    uint64_t end_wait_begin_mono_raw_ns = 0;
    uint64_t first_completion_mono_raw_ns = 0;
    uint64_t last_completion_mono_raw_ns = 0;
    uint64_t batch_end_mono_raw_ns = 0;
};

struct UcmdRingRuntimeStats {
    std::string name;
    int32_t ring_fd = -1;
    int64_t completion_tid = -1;
    bool sqpoll = false;
    bool blocking_completion = false;
    uint64_t completion_waits = 0;
    uint64_t completed_cqes = 0;
};

class UcmdRing {
public:
    UcmdRing();
    ~UcmdRing();

    UcmdRing(const UcmdRing&) = delete;
    UcmdRing& operator=(const UcmdRing&) = delete;

    void init(uint32_t queue_depth, const std::string& name,
              int registered_fd, bool completion_thread);
    void close();
    bool is_open() const;
    bool is_batching() const;
    bool has_completion_thread() const;
    uint32_t queue_depth() const { return queue_depth_; }
    const std::string& name() const { return name_; }

    void begin_batch();
    uint32_t end_batch();
    UcmdRingBatchStats last_batch_stats() const;
    UcmdRingRuntimeStats runtime_stats() const;

    void submit_nvme(int fd, uint32_t cmd_op, const nvme_uring_cmd& cmd,
                     const char* where);

private:
    void submit_async(int fd, uint32_t cmd_op, const nvme_uring_cmd& cmd,
                      const char* where);
    void submit_sync(int fd, uint32_t cmd_op, const nvme_uring_cmd& cmd,
                     const char* where);
    void wait_for_sq_space();
    void completion_loop();

    struct Impl;
    Impl* impl_;
    uint32_t queue_depth_;
    std::string name_;
    uint64_t next_user_data_;
};

class NvmeDevice {
public:
    NvmeDevice();
    ~NvmeDevice();

    NvmeDevice(const NvmeDevice&) = delete;
    NvmeDevice& operator=(const NvmeDevice&) = delete;

    void open_device(const std::string& path);
    void close();
    bool is_open() const { return fd_ >= 0; }
    int fd() const { return fd_; }
    const std::string& path() const { return path_; }

    NamespaceInfo identify(UcmdRing& admin_ring, uint32_t expected_nsid);

    void read_lba(UcmdRing& ring, uint32_t nsid, void* buffer,
                  uint64_t slba, uint32_t nlb, uint32_t lba_size);
    void write_lba(UcmdRing& ring, uint32_t nsid, const void* buffer,
                   uint64_t slba, uint32_t nlb, uint32_t lba_size,
                   bool fua = false, uint16_t pid_hint = 0);
    void flush(UcmdRing& ring, uint32_t nsid);

private:
    int fd_;
    std::string path_;
};

void validate_aligned_buffer(const void* buffer, uint64_t bytes,
                             uint32_t lba_size, const char* where);

}}  // namespace proposal::fdp
