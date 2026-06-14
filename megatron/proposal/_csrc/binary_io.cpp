// binary_io.cpp: single home for all ``::write()`` syscalls in the
// proposal package. See binary_io.h for the contract.
//
// io_uring developer: every ``full_write`` call in this file is a
// drop-in swap site. The caller-provided ``pinned_addr`` is a pointer
// to a host pinned buffer whose base AND length are 4 KiB-aligned
// (see the asserts in full_write_aligned). Replacing the synchronous
// ``::write()`` with an io_uring submit + wait should be transparent
// to all callers (Python, C++ proposal_io.cpp).
#include "binary_io.h"

#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <sys/stat.h>
#include <unistd.h>

namespace proposal { namespace binary_io {

// ---------------------------------------------------------------------------
// Syscall wrappers (anonymous namespace, internal-only).
//
// All raw ``::syscall(...)`` calls in this translation unit MUST go through
// one of these wrappers. Every public API below builds on these — no public
// API touches `::write/::read/::open/::fsync/::close/::unlink/::rename/
// ::pread/::stat` directly. The wrappers are also the io_uring swap site
// (drop-in replacement candidates).
// ---------------------------------------------------------------------------

namespace {

inline void check_aligned(int64_t addr, int64_t n, const char* where) {
    if (addr == 0)
        throw std::invalid_argument(std::string(where) + ": null pinned ptr");
    if ((addr & (SECTOR_BYTES - 1)) != 0)
        throw std::invalid_argument(
            std::string(where) + ": pinned ptr not 4 KiB-aligned");
    if (n < 0 || (n & (SECTOR_BYTES - 1)) != 0)
        throw std::invalid_argument(
            std::string(where) + ": bytes not 4 KiB-aligned");
}

// Synchronous full-write. EINTR retry loop. io_uring swap site.
void syscall_write_full_aligned(int fd, const void* buf, size_t n,
                                const char* where) {
    check_aligned(reinterpret_cast<int64_t>(buf), static_cast<int64_t>(n),
                  where);
    const char* p = static_cast<const char*>(buf);
    while (n > 0) {
        ssize_t w = ::write(fd, p, n);
        if (w < 0) {
            if (errno == EINTR) continue;
            throw std::runtime_error(
                std::string(where) + ": write failed: "
                + std::strerror(errno));
        }
        p += w;
        n -= static_cast<size_t>(w);
    }
}

// Backwards-compat alias used by some headers.
inline void full_write_aligned(int fd, const void* buf, size_t n,
                               const char* where) {
    syscall_write_full_aligned(fd, buf, n, where);
}

// Open for create + read/write + truncate. Caller-provided flags allow
// e.g. O_APPEND. Throws std::runtime_error on failure.
int syscall_open_create(const std::string& path, int extra_flags, int mode,
                        const char* where) {
    const int flags = O_CREAT | O_RDWR | O_TRUNC | extra_flags;
    int fd = ::open(path.c_str(), flags, mode);
    if (fd < 0)
        throw std::runtime_error(
            std::string(where) + ": open " + path + " failed: "
            + std::strerror(errno));
    return fd;
}

int syscall_open_rdonly(const std::string& path, const char* where) {
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0)
        throw std::runtime_error(
            std::string(where) + ": open " + path + " failed: "
            + std::strerror(errno));
    return fd;
}

void syscall_fsync(int fd, const char* where) {
    if (::fsync(fd) < 0)
        throw std::runtime_error(
            std::string(where) + ": fsync failed: " + std::strerror(errno));
}

// Best-effort close: never throws (used in destructors / catch blocks).
void syscall_close_noexcept(int fd) noexcept {
    if (fd < 0) return;
    (void)::close(fd);
}

void syscall_rename(const std::string& from, const std::string& to,
                    const char* where) {
    if (::rename(from.c_str(), to.c_str()) < 0)
        throw std::runtime_error(
            std::string(where) + ": rename " + from + " -> " + to
            + " failed: " + std::strerror(errno));
}

// Returns true on unlink success, false if path was already absent
// (ENOENT swallowed). Other errno → throws.
bool syscall_unlink_allow_enoent(const std::string& path, const char* where) {
    if (::unlink(path.c_str()) < 0) {
        if (errno == ENOENT) return false;
        throw std::runtime_error(
            std::string(where) + ": unlink " + path + " failed: "
            + std::strerror(errno));
    }
    return true;
}

// Loop pread until ``n`` bytes read or EOF. Returns bytes actually read.
size_t syscall_pread_loop(int fd, void* buf, size_t n, int64_t offset,
                          const char* where) {
    size_t read_total = 0;
    char* dst = static_cast<char*>(buf);
    while (read_total < n) {
        ssize_t r = ::pread(fd, dst + read_total, n - read_total,
                            offset + static_cast<int64_t>(read_total));
        if (r < 0) {
            if (errno == EINTR) continue;
            throw std::runtime_error(
                std::string(where) + ": pread failed: " + std::strerror(errno));
        }
        if (r == 0) break;  // EOF before requested n — truncate
        read_total += static_cast<size_t>(r);
    }
    return read_total;
}

// Returns file size in bytes, or -1 if ENOENT. Other errno → throws.
int64_t syscall_stat_size(const std::string& path, const char* where) {
    struct stat st;
    if (::stat(path.c_str(), &st) < 0) {
        if (errno == ENOENT) return -1;
        throw std::runtime_error(
            std::string(where) + ": stat " + path + " failed: "
            + std::strerror(errno));
    }
    return static_cast<int64_t>(st.st_size);
}

}  // anonymous namespace

void stamp_header_into_pinned(int64_t pinned_addr,
                              const std::string& header) {
    if (pinned_addr == 0)
        throw std::invalid_argument("stamp_header: null pinned ptr");
    const ssize_t hlen = static_cast<ssize_t>(header.size());
    if (hlen > HEADER_BLOCK_BYTES)
        throw std::invalid_argument("stamp_header: header > HEADER_BLOCK_BYTES");
    char* dst = reinterpret_cast<char*>(pinned_addr);
    std::memcpy(dst, header.data(), static_cast<size_t>(hlen));
    if (hlen < HEADER_BLOCK_BYTES)
        std::memset(dst + hlen, 0,
                    static_cast<size_t>(HEADER_BLOCK_BYTES - hlen));
}

size_t write_prptens(const std::string& final_path,
                     int64_t pinned_addr,
                     const std::string& header,
                     int64_t total_padded_bytes,
                     int mode,
                     bool do_fsync) {
    stamp_header_into_pinned(pinned_addr, header);
    int fd = syscall_open_create(final_path, /*extra_flags=*/0, mode,
                                 "write_prptens");
    try {
        syscall_write_full_aligned(fd, reinterpret_cast<void*>(pinned_addr),
                                   static_cast<size_t>(total_padded_bytes),
                                   "write_prptens");
        if (do_fsync) syscall_fsync(fd, "write_prptens");
    } catch (...) {
        syscall_close_noexcept(fd);
        throw;
    }
    syscall_close_noexcept(fd);
    return static_cast<size_t>(total_padded_bytes);
}

int write_prptens_keep_open(const std::string& final_path,
                            int64_t pinned_addr,
                            const std::string& header,
                            int64_t total_padded_bytes,
                            int mode) {
    return write_header_payload_keep_open(
        final_path, pinned_addr, header, total_padded_bytes, mode);
}

int write_header_payload_keep_open(const std::string& final_path,
                                   int64_t pinned_addr,
                                   const std::string& header,
                                   int64_t total_padded_bytes,
                                   int mode) {
    stamp_header_into_pinned(pinned_addr, header);
    int fd = syscall_open_create(final_path, /*extra_flags=*/0, mode,
                                 "write_header_payload_keep_open");
    try {
        syscall_write_full_aligned(
            fd,
            reinterpret_cast<void*>(pinned_addr),
            static_cast<size_t>(total_padded_bytes),
            "write_header_payload_keep_open");
    } catch (...) {
        syscall_close_noexcept(fd);
        throw;
    }
    return fd;
}

int open_lossy_log_for_append(const std::string& path,
                              int64_t header_pinned_addr,
                              const std::string& header,
                              int mode) {
    // Stamp the (variable-length) header into the front of a caller-
    // allocated pinned buffer sized to HEADER_BLOCK_BYTES, then write that
    // block aligned. Reader treats [0..HEADER_BLOCK_BYTES) as the header
    // region; tail is zero-filled.
    stamp_header_into_pinned(header_pinned_addr, header);
    int fd = syscall_open_create(path, O_APPEND, mode, "open_lossy_log");
    try {
        syscall_write_full_aligned(
            fd, reinterpret_cast<void*>(header_pinned_addr),
            static_cast<size_t>(HEADER_BLOCK_BYTES),
            "open_lossy_log_header");
    } catch (...) {
        syscall_close_noexcept(fd);
        throw;
    }
    return fd;
}

size_t append_lossy_log(int fd,
                        int64_t pinned_addr,
                        int64_t total_padded_bytes) {
    if (fd < 0) throw std::invalid_argument("append_lossy_log: bad fd");
    if (total_padded_bytes <= 0) return 0;
    syscall_write_full_aligned(fd, reinterpret_cast<void*>(pinned_addr),
                               static_cast<size_t>(total_padded_bytes),
                               "append_lossy_log");
    return static_cast<size_t>(total_padded_bytes);
}

void fsync_fd(int fd) {
    if (fd < 0) return;
    syscall_fsync(fd, "fsync_fd");
}

void close_fd(int fd) {
    syscall_close_noexcept(fd);
}

void unlink_path(const std::string& path) {
    (void)syscall_unlink_allow_enoent(path, "unlink_path");
}

std::string read_file_range(const std::string& path,
                            int64_t offset,
                            int64_t nbytes) {
    if (nbytes < 0)
        throw std::invalid_argument("read_file_range: negative nbytes");
    if (nbytes == 0) return std::string();
    int fd = syscall_open_rdonly(path, "read_file_range");
    std::string out(static_cast<size_t>(nbytes), '\0');
    size_t read_total = 0;
    try {
        read_total = syscall_pread_loop(
            fd, &out[0], static_cast<size_t>(nbytes), offset,
            "read_file_range");
    } catch (...) {
        syscall_close_noexcept(fd);
        throw;
    }
    syscall_close_noexcept(fd);
    if (read_total < static_cast<size_t>(nbytes))
        out.resize(read_total);
    return out;
}

int64_t file_size(const std::string& path) {
    return syscall_stat_size(path, "file_size");
}

}}  // namespace proposal::binary_io
