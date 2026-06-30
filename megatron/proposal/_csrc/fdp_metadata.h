#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>

namespace proposal { namespace fdp {

using FdpChunkId = uint32_t;

constexpr uint64_t make_magic8(char a, char b, char c, char d,
                               char e, char f, char g, char h) {
    return (static_cast<uint64_t>(static_cast<unsigned char>(a)) << 0) |
           (static_cast<uint64_t>(static_cast<unsigned char>(b)) << 8) |
           (static_cast<uint64_t>(static_cast<unsigned char>(c)) << 16) |
           (static_cast<uint64_t>(static_cast<unsigned char>(d)) << 24) |
           (static_cast<uint64_t>(static_cast<unsigned char>(e)) << 32) |
           (static_cast<uint64_t>(static_cast<unsigned char>(f)) << 40) |
           (static_cast<uint64_t>(static_cast<unsigned char>(g)) << 48) |
           (static_cast<uint64_t>(static_cast<unsigned char>(h)) << 56);
}

constexpr uint64_t kSuperMagic = make_magic8('F', 'D', 'P', 'C', 'P', 'S', 'B', '1');
constexpr uint64_t kMetadataMagic = make_magic8('F', 'D', 'P', 'C', 'P', 'M', 'D', '1');
constexpr uint32_t kFormatVersion = 2;
constexpr uint64_t kDefaultChunkBytes = 64ull * 1024ull * 1024ull;
constexpr uint32_t kMetadataHeaderBytes = 4096;

constexpr uint32_t kMaxChunksNefc = 4096;
constexpr uint32_t kMaxChunksLfcs = 32;
constexpr uint32_t kMaxChunksLfsy = 96;
constexpr uint32_t kMaxExpertSlots = 65536;

inline uint64_t div_round_up_u64(uint64_t n, uint64_t d) {
    return d == 0 ? 0 : (n + d - 1) / d;
}

inline uint64_t round_up_u64(uint64_t n, uint64_t align) {
    return div_round_up_u64(n, align) * align;
}

#pragma pack(push, 1)

struct SuperBlock {
    uint64_t magic;
    uint32_t version;
    uint32_t struct_bytes;
    uint32_t lba_size;
    uint32_t superblock_nlb;

    uint64_t namespace_nlb;
    uint64_t chunk_bytes;
    uint64_t chunk_lba_count;

    uint64_t metadata_start_lba;
    uint64_t metadata_nlb;
    uint64_t metadata_image_lba;
    uint64_t metadata_image_nlb;
    uint64_t metadata_image_bytes;

    uint64_t data_start_lba;
    uint64_t data_nlb;
    uint32_t data_chunk_count;
    uint32_t bitmap_bytes;

    uint32_t rank;
    uint32_t world_size;
    uint32_t num_moe_layers;
    uint32_t experts_per_layer;
    uint32_t expert_slots;

    uint32_t dp_rank;
    uint32_t dp_size;
    uint32_t pp_rank;
    uint32_t pp_size;
    uint32_t ep_rank;
    uint32_t ep_size;
    uint32_t nefc_shard_count;
    uint32_t reserved_topology0;

    uint32_t max_chunks_nefc;
    uint32_t max_chunks_lfcs;
    uint32_t max_chunks_lfsy;

    uint64_t seq;
    uint64_t last_flush_iter;
    uint64_t allocated_chunks;

    uint8_t fdp_supported;
    uint8_t fdp_pid_enabled;
    uint8_t reserved0[6];
    uint64_t reserved1[32];
};

struct MetadataHeader {
    uint64_t magic;
    uint32_t version;
    uint32_t header_bytes;

    uint64_t metadata_image_bytes;
    uint64_t seq;
    uint64_t last_flush_iter;
    uint64_t allocated_chunks;

    uint32_t rank;
    uint32_t world_size;
    uint32_t num_moe_layers;
    uint32_t experts_per_layer;
    uint32_t expert_slots;
    uint32_t nefc_shard_count;
    uint32_t data_chunk_count;
    uint32_t bitmap_bytes;
    uint32_t dp_size;
    uint32_t pp_size;
    uint32_t ep_size;

    uint64_t bitmap_offset;
    uint64_t nefc_offset;
    uint64_t lfcs_offset;
    uint64_t lfsy_offset;
    uint64_t reserved1[24];
};

struct NefcEntry {
    uint8_t valid;
    uint8_t reserved0[3];
    uint32_t rank;
    uint32_t dp_rank;
    uint32_t pp_rank;
    uint32_t ep_rank;
    uint32_t shard_index;
    uint64_t iter;
    uint64_t logical_bytes;
    uint32_t chunk_count;
    uint32_t reserved1;
    FdpChunkId chunks[kMaxChunksNefc];
};

struct LfcsEntry {
    uint8_t valid;
    uint8_t reserved0[3];
    uint32_t rank;
    uint32_t layer_id;
    uint32_t expert_id;
    uint64_t generation;
    uint64_t iter;
    uint64_t logical_bytes;
    uint32_t chunk_count;
    uint32_t reserved1;
    FdpChunkId chunks[kMaxChunksLfcs];
};

struct LfsyEntry {
    uint8_t valid;
    uint8_t reserved0[3];
    uint32_t rank;
    uint32_t layer_id;
    uint32_t expert_id;
    uint64_t generation;
    uint64_t start_iter;
    uint64_t logical_tail;
    uint32_t chunk_count;
    uint32_t reserved1;
    FdpChunkId chunks[kMaxChunksLfsy];
};

#pragma pack(pop)

struct MetadataLayout {
    uint64_t image_bytes = 0;
    uint64_t bitmap_offset = 0;
    uint64_t bitmap_bytes = 0;
    uint64_t nefc_offset = 0;
    uint32_t nefc_count = 1;
    uint64_t lfcs_offset = 0;
    uint64_t lfsy_offset = 0;
};

inline MetadataLayout compute_metadata_layout(uint32_t data_chunk_count,
                                              uint32_t expert_slots,
                                              uint32_t nefc_count = 1) {
    MetadataLayout out;
    out.bitmap_offset = kMetadataHeaderBytes;
    out.bitmap_bytes = div_round_up_u64(data_chunk_count, 8);
    out.nefc_count = nefc_count == 0 ? 1 : nefc_count;

    uint64_t cursor = out.bitmap_offset + out.bitmap_bytes;
    cursor = round_up_u64(cursor, 64);
    out.nefc_offset = cursor;
    cursor += static_cast<uint64_t>(out.nefc_count) * sizeof(NefcEntry);

    cursor = round_up_u64(cursor, 64);
    out.lfcs_offset = cursor;
    cursor += static_cast<uint64_t>(expert_slots) * sizeof(LfcsEntry);

    cursor = round_up_u64(cursor, 64);
    out.lfsy_offset = cursor;
    cursor += static_cast<uint64_t>(expert_slots) * sizeof(LfsyEntry);

    out.image_bytes = round_up_u64(cursor, kMetadataHeaderBytes);
    return out;
}

inline uint32_t checked_chunk_count(uint64_t data_nlb,
                                    uint64_t chunk_lba_count) {
    const uint64_t n = data_nlb / chunk_lba_count;
    if (n > std::numeric_limits<uint32_t>::max()) {
        return std::numeric_limits<uint32_t>::max();
    }
    return static_cast<uint32_t>(n);
}

}}  // namespace proposal::fdp
