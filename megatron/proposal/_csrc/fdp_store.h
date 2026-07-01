#pragma once

#include "fdp_metadata.h"
#include "fdp_passthru.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace proposal { namespace fdp {

struct FdpEnvConfig {
    std::string device_path;
    uint32_t nsid = 0;

    uint64_t metadata_start_lba = 0;
    uint64_t metadata_nlb = 0;
    bool metadata_nlb_explicit = false;
    uint64_t data_start_lba = 0;
    uint64_t data_nlb = 0;

    uint64_t chunk_bytes = kDefaultChunkBytes;
    uint32_t rank = 0;
    uint32_t world_size = 1;
    uint32_t num_moe_layers = 0;
    uint32_t experts_per_layer = 0;
    uint32_t expert_slots = 0;
    uint32_t dp_rank = 0;
    uint32_t dp_size = 1;
    uint32_t pp_rank = 0;
    uint32_t pp_size = 1;
    uint32_t ep_rank = 0;
    uint32_t ep_size = 1;
    uint32_t nefc_shard_index = 0;
    uint32_t nefc_shard_count = 1;
    uint32_t expert_id_base = 0;
    bool expert_ids_global = false;

    uint32_t nefc_qd = 32;
    uint32_t expert_qd = 64;
    uint32_t metadata_qd = 8;
    uint32_t max_io_kb = 256;

    bool allow_format = false;
    bool force_format = false;
    bool nvme_flush = true;
    bool write_fua = false;
    bool metadata_mmap = false;
};

struct FdpStatsSnapshot {
    std::string backend;
    std::string device_path;
    uint32_t nsid = 0;
    uint32_t lba_size = 0;
    uint64_t namespace_nlb = 0;
    bool fdp_supported = false;
    bool fdp_pid_enabled = false;

    uint64_t metadata_start_lba = 0;
    uint64_t metadata_nlb = 0;
    uint64_t metadata_image_lba = 0;
    uint64_t metadata_image_nlb = 0;
    uint64_t metadata_image_bytes = 0;

    uint64_t data_start_lba = 0;
    uint64_t data_nlb = 0;
    uint64_t chunk_bytes = 0;
    uint64_t chunk_lba_count = 0;
    uint32_t data_chunk_count = 0;
    uint64_t allocated_chunks = 0;

    uint32_t rank = 0;
    uint32_t world_size = 1;
    uint32_t num_moe_layers = 0;
    uint32_t experts_per_layer = 0;
    uint32_t expert_slots = 0;
    uint32_t dp_rank = 0;
    uint32_t dp_size = 1;
    uint32_t pp_rank = 0;
    uint32_t pp_size = 1;
    uint32_t ep_rank = 0;
    uint32_t ep_size = 1;
    uint32_t nefc_shard_index = 0;
    uint32_t nefc_shard_count = 1;
    uint32_t expert_id_base = 0;
    bool expert_ids_global = false;

    uint32_t nefc_qd = 0;
    uint32_t expert_qd = 0;
    uint32_t metadata_qd = 0;
    uint32_t max_io_nlb = 0;
    uint64_t max_io_bytes = 0;
    bool write_fua = false;
    bool metadata_mmap = false;
    uint64_t metadata_flush_count = 0;
    uint64_t seq = 0;
    uint64_t last_flush_iter = 0;
};

struct MetadataSummary {
    uint64_t magic = 0;
    uint32_t version = 0;
    uint64_t seq = 0;
    uint64_t last_flush_iter = 0;
    uint64_t allocated_chunks = 0;
    uint32_t data_chunk_count = 0;
    uint32_t bitmap_bytes = 0;
    uint64_t bitmap_offset = 0;
    uint64_t nefc_offset = 0;
    uint64_t lfcs_offset = 0;
    uint64_t lfsy_offset = 0;
    uint64_t metadata_image_bytes = 0;
    bool nefc_valid = false;
    uint64_t nefc_iter = 0;
    uint64_t nefc_logical_bytes = 0;
    uint32_t nefc_chunk_count = 0;
    uint32_t nefc_shard_count = 1;
    uint32_t nefc_valid_count = 0;
    uint32_t lfcs_valid_count = 0;
    uint32_t lfcs_total_chunks = 0;
    uint32_t lfcs_max_chunk_count = 0;
    uint32_t lfsy_valid_count = 0;
};

struct NefcEntrySnapshot {
    bool valid = false;
    uint32_t shard_index = 0;
    uint32_t rank = 0;
    uint32_t dp_rank = 0;
    uint32_t pp_rank = 0;
    uint32_t ep_rank = 0;
    uint64_t iter = 0;
    uint64_t logical_bytes = 0;
    uint32_t chunk_count = 0;
};

struct LfcsEntrySnapshot {
    bool valid = false;
    uint32_t slot = 0;
    uint32_t rank = 0;
    uint32_t layer_id = 0;
    uint32_t expert_id = 0;
    uint64_t generation = 0;
    uint64_t iter = 0;
    uint64_t logical_bytes = 0;
    uint32_t chunk_count = 0;
    std::vector<FdpChunkId> chunks;
};

struct LfsyEntrySnapshot {
    bool valid = false;
    uint32_t slot = 0;
    uint32_t rank = 0;
    uint32_t layer_id = 0;
    uint32_t expert_id = 0;
    uint64_t generation = 0;
    uint64_t start_iter = 0;
    uint64_t logical_tail = 0;
    uint32_t chunk_count = 0;
    std::vector<FdpChunkId> chunks;
};

struct LfsyHandle {
    uint32_t slot = 0;
    uint32_t layer_id = 0;
    uint32_t expert_id = 0;
    uint64_t generation = 0;
    uint64_t last_iter = 0;
};

class AlignedBuffer {
public:
    AlignedBuffer();
    ~AlignedBuffer();

    AlignedBuffer(const AlignedBuffer&) = delete;
    AlignedBuffer& operator=(const AlignedBuffer&) = delete;

    AlignedBuffer(AlignedBuffer&& other) noexcept;
    AlignedBuffer& operator=(AlignedBuffer&& other) noexcept;

    void allocate(size_t bytes, size_t alignment = 4096);
    void reset();
    void* data() { return ptr_; }
    const void* data() const { return ptr_; }
    size_t bytes() const { return bytes_; }

    template <typename T>
    T* as() { return reinterpret_cast<T*>(ptr_); }

    template <typename T>
    const T* as() const { return reinterpret_cast<const T*>(ptr_); }

private:
    void* ptr_;
    size_t bytes_;
};

enum class FdpIoPlanKind {
    ChunkListWrite,
    LfsyAppend,
};

enum class FdpIoPlanRing {
    Nefc,
    Expert,
};

struct FdpIoPlan {
    FdpIoPlanKind kind = FdpIoPlanKind::ChunkListWrite;
    FdpIoPlanRing ring = FdpIoPlanRing::Expert;
    int64_t pinned_addr = 0;
    uint64_t total_padded_bytes = 0;
    uint64_t logical_offset = 0;
    std::vector<FdpChunkId> chunks;
    AlignedBuffer owned_buffer;
    std::string where;
};

class FdpStore {
public:
    FdpStore();
    ~FdpStore();

    FdpStore(const FdpStore&) = delete;
    FdpStore& operator=(const FdpStore&) = delete;

    void open_from_env();
    void close();
    bool is_open() const;

    void format_or_load();
    void flush_metadata(uint64_t iter);
    void begin_data_batch();
    std::vector<uint32_t> end_data_batch();
    std::vector<UcmdRingBatchStats> end_data_batch_stats();
    void begin_nefc_batch();
    uint32_t end_nefc_batch();
    UcmdRingBatchStats end_nefc_batch_stats();
    void begin_expert_batch();
    uint32_t end_expert_batch();
    UcmdRingBatchStats end_expert_batch_stats();
    void begin_checkpoint_iter(uint64_t iter);
    bool mark_writer_done(uint64_t iter, const std::string& writer_kind);
    size_t write_nefc(const std::string& object_name,
                      int64_t pinned_addr,
                      const std::string& header,
                      uint64_t payload_bytes,
                      uint64_t total_padded_bytes,
                      uint64_t iter);
    size_t write_lfcs(const std::string& object_name,
                      int64_t pinned_addr,
                      const std::string& header,
                      uint64_t payload_bytes,
                      uint64_t total_padded_bytes,
                      uint32_t layer_id,
                      uint32_t expert_id,
                      uint64_t iter,
                      uint64_t generation);
    int64_t open_lfsy(const std::string& object_name,
                      const std::string& header,
                      uint32_t layer_id,
                      uint32_t expert_id,
                      int64_t start_iter,
                      uint64_t generation);
    size_t append_lfsy(int64_t handle,
                       int64_t pinned_addr,
                       uint64_t total_bytes,
                       uint64_t total_padded_bytes,
                       uint32_t layer_id,
                       uint32_t expert_id,
                       uint64_t iter,
                       uint64_t generation);
    void fsync_handle(int64_t handle);
    void close_lfsy(int64_t handle);

    FdpStatsSnapshot stats() const;
    MetadataSummary debug_read_metadata_summary();
    NefcEntrySnapshot read_nefc_entry();
    std::vector<NefcEntrySnapshot> read_nefc_entries();
    std::vector<LfcsEntrySnapshot> read_lfcs_entries();
    std::vector<LfsyEntrySnapshot> read_lfsy_entries();
    std::vector<LfsyEntrySnapshot> debug_read_lfsy_entries();
    bool debug_raw_write_read_smoke();
    size_t read_nefc_into(int64_t dst_addr, uint64_t dst_bytes);
    size_t read_nefc_shard_into(uint32_t shard_index,
                                int64_t dst_addr,
                                uint64_t dst_bytes);
    size_t read_lfcs_into(uint32_t layer_id, uint32_t expert_id,
                          int64_t dst_addr, uint64_t dst_bytes);
    size_t read_lfsy_into(uint32_t layer_id, uint32_t expert_id,
                          int64_t dst_addr, uint64_t dst_bytes);
    std::string debug_read_nefc(uint64_t offset, uint64_t nbytes);
    std::string debug_read_lfcs(uint32_t layer_id, uint32_t expert_id,
                                uint64_t offset, uint64_t nbytes);
    std::string debug_read_lfsy(uint32_t layer_id, uint32_t expert_id,
                                uint64_t offset, uint64_t nbytes);

    uint32_t debug_alloc_chunk();
    void debug_free_chunk(uint32_t chunk_id);

private:
    void load_config_from_env();
    void validate_namespace_layout();
    void allocate_metadata_image();
    void initialize_fresh_metadata();
    void load_existing_metadata(const SuperBlock& sb);
    void write_superblock();
    void write_metadata_image();
    void read_metadata_image();
    void open_or_create_shared_metadata_mmap();
    void close_shared_metadata_mmap();
    uint64_t metadata_layout_fingerprint() const;
    void prepare_io_plan(const FdpIoPlan& plan);
    void prepare_io_plans(std::vector<FdpIoPlan>& plans);
    void enqueue_io_plan(FdpIoPlan&& plan);
    bool mark_process_done_locked(uint64_t iter);
    void update_metadata_header(uint64_t iter, bool bump_seq);
    void acquire_global_metadata_lock();
    void release_global_metadata_lock();
    bool global_metadata_lock_held() const;
    void persist_metadata_image_locked(uint64_t iter, bool bump_seq);
    void validate_loaded_superblock(const SuperBlock& sb) const;
    void validate_metadata_range(uint64_t slba, uint64_t nlb,
                                 const char* where) const;
    void raw_read_metadata(void* buffer, uint64_t slba, uint64_t nlb,
                           const char* where);
    void raw_write_metadata(const void* buffer, uint64_t slba, uint64_t nlb,
                            const char* where);
    void raw_write_data(UcmdRing& ring, const void* buffer, uint64_t slba,
                        uint64_t nlb, const char* where);
    void raw_read_data(UcmdRing& ring, void* buffer, uint64_t slba,
                       uint64_t nlb, const char* where);
    uint64_t recalc_allocated_chunks() const;
    std::vector<FdpChunkId> allocate_chunks(uint32_t count);
    void free_chunks(const FdpChunkId* chunks, uint32_t count);
    uint64_t data_chunk_lba(FdpChunkId chunk_id) const;
    void write_chunk_list(UcmdRing& ring, int64_t pinned_addr,
                          const std::vector<FdpChunkId>& chunks,
                          uint64_t total_padded_bytes,
                          const char* where);
    std::string read_chunk_bytes(UcmdRing& ring,
                                 const FdpChunkId* chunks,
                                 uint32_t chunk_count,
                                 uint64_t logical_limit,
                                 uint64_t offset,
                                 uint64_t nbytes,
                                 const char* where);
    size_t read_chunk_list_into(UcmdRing& ring,
                                const FdpChunkId* chunks,
                                uint32_t chunk_count,
                                uint64_t logical_bytes,
                                int64_t dst_addr,
                                uint64_t dst_bytes,
                                const char* where);
    uint32_t expert_slot(uint32_t layer_id, uint32_t expert_id) const;
    uint32_t global_expert_id(uint32_t expert_id) const;
    uint32_t current_nefc_shard() const;
    void stamp_header_into_pinned(int64_t pinned_addr,
                                  const std::string& header) const;
    void erase_lfsy_handles_for_slot(uint32_t slot);
    LfsyHandle& checked_lfsy_handle(int64_t handle,
                                    uint32_t layer_id,
                                    uint32_t expert_id,
                                    uint64_t generation);

    uint8_t* metadata_bytes();
    const uint8_t* metadata_bytes() const;
    MetadataHeader* metadata_header();
    const MetadataHeader* metadata_header() const;
    uint8_t* bitmap_bytes();
    const uint8_t* bitmap_bytes() const;
    NefcEntry* nefc_entry(uint32_t shard);
    const NefcEntry* nefc_entry(uint32_t shard) const;
    NefcEntry* nefc_entry();
    const NefcEntry* nefc_entry() const;
    LfcsEntry* lfcs_entry(uint32_t slot);
    const LfcsEntry* lfcs_entry(uint32_t slot) const;
    LfsyEntry* lfsy_entry(uint32_t slot);
    const LfsyEntry* lfsy_entry(uint32_t slot) const;

    FdpEnvConfig cfg_;
    NamespaceInfo ns_;
    MetadataLayout layout_;

    NvmeDevice device_;
    UcmdRing nefc_data_ring_;
    UcmdRing expert_data_ring_;
    UcmdRing metadata_ring_;

    AlignedBuffer superblock_buf_;
    AlignedBuffer metadata_image_;
    AlignedBuffer scratch_buf_;

    uint64_t superblock_nlb_ = 0;
    uint64_t metadata_image_lba_ = 0;
    uint64_t metadata_image_nlb_ = 0;
    uint64_t chunk_lba_count_ = 0;
    uint32_t max_io_nlb_ = 0;
    uint32_t data_chunk_count_ = 0;
    uint64_t seq_ = 0;
    uint64_t last_flush_iter_ = 0;
    uint64_t allocated_chunks_ = 0;
    int64_t next_lfsy_handle_ = 1;
    std::unordered_map<int64_t, LfsyHandle> lfsy_handles_;
    std::vector<AlignedBuffer> batch_owned_buffers_;
    std::vector<FdpIoPlan> nefc_batch_plans_;
    std::vector<FdpIoPlan> expert_batch_plans_;
    mutable std::recursive_mutex mutex_;
    mutable std::shared_mutex op_mutex_;
    mutable std::recursive_mutex bitmap_mutex_;
    mutable std::recursive_mutex nefc_ring_mutex_;
    mutable std::recursive_mutex expert_ring_mutex_;
    mutable std::mutex batch_state_mutex_;
    std::unique_lock<std::shared_mutex> batch_lock_;
    std::shared_lock<std::shared_mutex> nefc_batch_lock_;
    std::shared_lock<std::shared_mutex> expert_batch_lock_;
    std::unique_lock<std::recursive_mutex> nefc_batch_ring_lock_;
    std::unique_lock<std::recursive_mutex> expert_batch_ring_lock_;
    std::atomic<size_t> batch_owner_token_{0};
    std::atomic<size_t> nefc_batch_owner_token_{0};
    std::atomic<size_t> expert_batch_owner_token_{0};
    std::string global_lock_path_;
    int global_lock_fd_ = -1;
    uint32_t global_lock_depth_ = 0;
    mutable std::recursive_mutex global_lock_mutex_;
    std::string shared_mmap_name_;
    int shared_mmap_fd_ = -1;
    void* shared_mmap_base_ = nullptr;
    uint64_t shared_mmap_bytes_ = 0;
    void* shared_control_ = nullptr;
    uint8_t* shared_metadata_bytes_ = nullptr;
    uint64_t local_iter_ = 0;
    bool local_nonexpert_done_ = false;
    bool local_expert_done_ = false;
    bool local_process_done_reported_ = false;
    mutable std::mutex local_iter_mutex_;
    bool open_ = false;
};

FdpStore& global_store();

}}  // namespace proposal::fdp
