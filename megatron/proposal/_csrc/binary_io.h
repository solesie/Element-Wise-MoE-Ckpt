// binary_io: declarations for the proposal CP file-write hot paths.
//
// All direct ``::write()`` / ``::open()`` / ``::fsync()`` / ``::close()``
// / ``::rename()`` calls in the proposal package go through the helpers
// declared here (implementations in binary_io.cpp). On-disk format
// (PRPTENS / LFCL / LFSY headers, AoS records) is intentionally
// identical to the prior pure-Python implementation; the io_uring
// developer can swap ``full_write`` for ``io_uring_submit`` inside this
// one file without touching Python or other C++ modules.
//
// Buffer contract: every ``full_write`` call gets a host buffer whose
// base pointer AND length are both 4 KiB (SECTOR_BYTES) aligned. The
// pinned-buffer allocator on the Python side
// (proposal_io.alloc_pinned_uint8_sector_aligned) guarantees the base
// alignment; the helpers below ensure the length is a multiple of
// SECTOR_BYTES (callers compute the padded length, and the helpers
// assert it).
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace proposal { namespace binary_io {

// O_DIRECT sector size used for alignment assertions + io_uring swap.
constexpr int SECTOR_BYTES = 4096;

// Header block sentinel: leading region of every PRPTENS / LFCL file.
// Must be a SECTOR_BYTES multiple so the header + payload form a
// single chunk-aligned write.
constexpr int HEADER_BLOCK_BYTES = 65536;  // 16 sectors

// Memcpy ``header.data()`` into ``pinned_addr[0..header.size())``,
// zero-fill ``pinned_addr[header.size()..HEADER_BLOCK_BYTES)``.
// ``header.size() <= HEADER_BLOCK_BYTES`` required.
void stamp_header_into_pinned(int64_t pinned_addr,
                              const std::string& header);

// Write a PRPTENS file (header stamped at front + payload follows) as
// a single full_write. ``total_padded_bytes`` must be a SECTOR_BYTES
// multiple and the pinned buffer must contain valid data through that
// length (caller zero-pads via alloc_pinned_uint8_sector_aligned).
// ``do_fsync=true`` to fsync after write.
size_t write_prptens(const std::string& final_path,
                     int64_t pinned_addr,
                     const std::string& header,
                     int64_t total_padded_bytes,
                     int mode,
                     bool do_fsync);

// Atomic write to ``<final_path>.tmp`` + fsync + rename(final_path).
size_t write_lossless(const std::string& final_path,
                      int64_t pinned_addr,
                      const std::string& header,
                      int64_t total_padded_bytes,
                      int mode);

// Create (O_APPEND | O_CREAT | O_RDWR | O_TRUNC) + chunk-aligned write
// of a HEADER_BLOCK_BYTES-sized header. ``header_pinned_addr`` is a
// 4 KiB-aligned pinned buffer of size ≥ HEADER_BLOCK_BYTES; the
// helper stamps ``header`` into its front, zero-fills the remainder
// of HEADER_BLOCK_BYTES, then full_write_aligned's the whole block.
// Caller may release the pinned buffer immediately after this returns.
int open_lossy_log_for_append(const std::string& path,
                              int64_t header_pinned_addr,
                              const std::string& header,
                              int mode);

// Append a (count u32, step u32, records...) blob to an fd opened by
// open_lossy_log_for_append. ``total_padded_bytes`` must be a
// SECTOR_BYTES multiple.
size_t append_lossy_log(int fd,
                        int64_t pinned_addr,
                        int64_t total_padded_bytes);

// fsync(fd) / close(fd) wrappers; swallow errors for idempotent teardown.
void fsync_fd(int fd);
void close_fd(int fd);

// Full-write helper exposed so proposal_io.cpp's aos_write (which
// builds the AoS blob in pinned memory itself) can emit its writes
// through this single file. ``total_padded_bytes`` must be a
// SECTOR_BYTES multiple.
size_t full_write_pinned(int fd,
                         int64_t pinned_addr,
                         int64_t total_padded_bytes);

// File deletion. Swallows ENOENT for idempotent caller paths.
void unlink_path(const std::string& path);

// Read helpers. All recovery-side reads (rare events) go through
// these so file-IO syscalls live in one place. ``read_file_range``
// returns ``nbytes`` bytes starting at ``offset`` (uses ::pread so
// no fd-position state is touched).
std::string read_file_range(const std::string& path,
                            int64_t offset,
                            int64_t nbytes);

// ``::stat()``-based file size in bytes, or -1 if the file is missing.
int64_t file_size(const std::string& path);

}}  // namespace proposal::binary_io
