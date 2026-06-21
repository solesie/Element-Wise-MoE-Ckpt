// proposal_io: C++ pybind bindings for binary_io.cpp helpers.
//
// GPU direct emission produces lossy records on device into pinned AOS.
// This module keeps the CPU-side file IO and recovery helpers in one pybind
// extension so Python does not loop over small pread/scatter calls.
#include "binary_io.h"

#include <torch/extension.h>
#include <pybind11/stl.h>

#include <atomic>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace py = pybind11;

namespace {

void check_cpu_contiguous(const torch::Tensor& t,
                          const char* name,
                          c10::ScalarType dtype) {
    if (!t.is_cpu())
        throw std::invalid_argument(std::string(name) + " must be a CPU tensor");
    if (!t.is_contiguous())
        throw std::invalid_argument(std::string(name) + " must be contiguous");
    if (t.scalar_type() != dtype)
        throw std::invalid_argument(std::string(name) + " has wrong dtype");
}

void scatter_lossy_records_to_dense_by_intervals(
    const torch::Tensor& aos_u8,
    int64_t record_byte_offset,
    int64_t n_records,
    const torch::Tensor& starts_i64,
    const torch::Tensor& ends_i64,
    const std::vector<torch::Tensor>& p_tensors,
    const std::vector<torch::Tensor>& m_tensors,
    const std::vector<torch::Tensor>& v_tensors) {

    check_cpu_contiguous(aos_u8, "aos_u8", c10::ScalarType::Byte);
    check_cpu_contiguous(starts_i64, "starts_i64", c10::ScalarType::Long);
    check_cpu_contiguous(ends_i64, "ends_i64", c10::ScalarType::Long);
    if (record_byte_offset < 0 || n_records < 0)
        throw std::invalid_argument("scatter_lossy_records: negative offset/count");
    const int64_t n_intervals = starts_i64.numel();
    if (ends_i64.numel() != n_intervals ||
        static_cast<int64_t>(p_tensors.size()) != n_intervals ||
        static_cast<int64_t>(m_tensors.size()) != n_intervals ||
        static_cast<int64_t>(v_tensors.size()) != n_intervals) {
        throw std::invalid_argument(
            "scatter_lossy_records: interval/tensor count mismatch");
    }
    if (n_records == 0 || n_intervals == 0) return;

    const int64_t record_bytes = 16;
    const int64_t byte_end = record_byte_offset + n_records * record_bytes;
    if (byte_end > aos_u8.numel())
        throw std::out_of_range("scatter_lossy_records: AOS slice out of range");

    std::vector<float*> p_ptrs;
    std::vector<float*> m_ptrs;
    std::vector<float*> v_ptrs;
    p_ptrs.reserve(static_cast<size_t>(n_intervals));
    m_ptrs.reserve(static_cast<size_t>(n_intervals));
    v_ptrs.reserve(static_cast<size_t>(n_intervals));
    for (int64_t i = 0; i < n_intervals; ++i) {
        check_cpu_contiguous(p_tensors[i], "p_tensor", c10::ScalarType::Float);
        check_cpu_contiguous(m_tensors[i], "m_tensor", c10::ScalarType::Float);
        check_cpu_contiguous(v_tensors[i], "v_tensor", c10::ScalarType::Float);
        p_ptrs.push_back(p_tensors[i].data_ptr<float>());
        m_ptrs.push_back(m_tensors[i].data_ptr<float>());
        v_ptrs.push_back(v_tensors[i].data_ptr<float>());
    }

    const auto* starts = starts_i64.data_ptr<int64_t>();
    const auto* ends = ends_i64.data_ptr<int64_t>();
    const char* base = reinterpret_cast<const char*>(aos_u8.data_ptr<uint8_t>())
        + record_byte_offset;

    std::atomic<int64_t> bad_record{-1};
    std::atomic<int64_t> bad_index{0};

    #pragma omp parallel for schedule(static) if(n_records > 4096)
    for (int64_t r = 0; r < n_records; ++r) {
        if (bad_record.load(std::memory_order_relaxed) >= 0) continue;
        const char* rec = base + r * record_bytes;
        uint32_t idx_u32;
        float p_val, m_val, v_val;
        std::memcpy(&idx_u32, rec + 0, sizeof(uint32_t));
        const int64_t idx = static_cast<int64_t>(idx_u32);
        std::memcpy(&p_val, rec + 4, sizeof(float));
        std::memcpy(&m_val, rec + 8, sizeof(float));
        std::memcpy(&v_val, rec + 12, sizeof(float));

        int64_t lo = 0;
        int64_t hi = n_intervals;
        while (lo < hi) {
            const int64_t mid = (lo + hi) >> 1;
            if (starts[mid] <= idx) lo = mid + 1;
            else hi = mid;
        }
        const int64_t slot = lo - 1;
        if (slot < 0 || idx >= ends[slot]) {
            int64_t expected = -1;
            if (bad_record.compare_exchange_strong(expected, r)) {
                bad_index.store(idx, std::memory_order_relaxed);
            }
            continue;
        }
        const int64_t local = idx - starts[slot];
        p_ptrs[slot][local] = p_val;
        m_ptrs[slot][local] = m_val;
        v_ptrs[slot][local] = v_val;
    }

    const int64_t bad = bad_record.load(std::memory_order_relaxed);
    if (bad >= 0) {
        throw std::runtime_error(
            "scatter_lossy_records: invalid expert-local idx at record "
            + std::to_string(bad) + ": "
            + std::to_string(bad_index.load(std::memory_order_relaxed)));
    }
}

void load_lfcl_payload_interleaved(
    const torch::Tensor& lfcl_u8,
    int64_t payload_offset_bytes,
    const torch::Tensor& numels_i64,
    const torch::Tensor& p_flat,
    const torch::Tensor& m_flat,
    const torch::Tensor& v_flat) {

    check_cpu_contiguous(lfcl_u8, "lfcl_u8", c10::ScalarType::Byte);
    check_cpu_contiguous(numels_i64, "numels_i64", c10::ScalarType::Long);
    check_cpu_contiguous(p_flat, "p_flat", c10::ScalarType::Float);
    check_cpu_contiguous(m_flat, "m_flat", c10::ScalarType::Float);
    check_cpu_contiguous(v_flat, "v_flat", c10::ScalarType::Float);
    if (payload_offset_bytes < 0)
        throw std::invalid_argument("load_lfcl_payload: negative payload offset");
    const int64_t total_numel = p_flat.numel();
    if (m_flat.numel() != total_numel || v_flat.numel() != total_numel)
        throw std::invalid_argument("load_lfcl_payload: flat size mismatch");

    const auto* numels = numels_i64.data_ptr<int64_t>();
    const int64_t n_entries = numels_i64.numel();
    const char* src = reinterpret_cast<const char*>(lfcl_u8.data_ptr<uint8_t>())
        + payload_offset_bytes;
    const int64_t src_bytes = lfcl_u8.numel() - payload_offset_bytes;
    float* p = p_flat.data_ptr<float>();
    float* m = m_flat.data_ptr<float>();
    float* v = v_flat.data_ptr<float>();

    int64_t cursor = 0;
    int64_t byte_cursor = 0;
    for (int64_t i = 0; i < n_entries; ++i) {
        const int64_t n = numels[i];
        if (n < 0 || cursor + n > total_numel)
            throw std::out_of_range("load_lfcl_payload: numel out of range");
        const int64_t bytes = n * static_cast<int64_t>(sizeof(float));
        if (byte_cursor + bytes * 3 > src_bytes)
            throw std::out_of_range("load_lfcl_payload: payload truncated");
        std::memcpy(p + cursor, src + byte_cursor, static_cast<size_t>(bytes));
        byte_cursor += bytes;
        std::memcpy(m + cursor, src + byte_cursor, static_cast<size_t>(bytes));
        byte_cursor += bytes;
        std::memcpy(v + cursor, src + byte_cursor, static_cast<size_t>(bytes));
        byte_cursor += bytes;
        cursor += n;
    }
}

std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>
replay_lfsy_log_into_dense(
    const torch::Tensor& lfsy_u8,
    int64_t total_numel,
    const torch::Tensor& p_flat,
    const torch::Tensor& m_flat,
    const torch::Tensor& v_flat,
    int64_t header_block_bytes,
    int64_t sector_bytes,
    int64_t block_prefix_bytes,
    int64_t record_bytes) {

    check_cpu_contiguous(lfsy_u8, "lfsy_u8", c10::ScalarType::Byte);
    check_cpu_contiguous(p_flat, "p_flat", c10::ScalarType::Float);
    check_cpu_contiguous(m_flat, "m_flat", c10::ScalarType::Float);
    check_cpu_contiguous(v_flat, "v_flat", c10::ScalarType::Float);
    if (total_numel < 0)
        throw std::invalid_argument("replay_lfsy: negative total_numel");
    if (p_flat.numel() != total_numel ||
        m_flat.numel() != total_numel ||
        v_flat.numel() != total_numel) {
        throw std::invalid_argument("replay_lfsy: flat size mismatch");
    }
    if (block_prefix_bytes != 8 || record_bytes != 16)
        throw std::invalid_argument("replay_lfsy: unsupported record format");

    struct Block {
        int64_t data_off;
        int64_t count;
        int64_t step;
    };
    const auto* base = lfsy_u8.data_ptr<uint8_t>();
    const int64_t file_bytes = lfsy_u8.numel();
    std::vector<Block> blocks;
    blocks.reserve(1024);

    auto round_up = [sector_bytes](int64_t x) -> int64_t {
        return ((x + sector_bytes - 1) / sector_bytes) * sector_bytes;
    };

    int64_t max_step = 0;
    int64_t offset = header_block_bytes;
    while (offset + block_prefix_bytes <= file_bytes) {
        uint32_t count_u32 = 0;
        uint32_t step_u32 = 0;
        std::memcpy(&count_u32, base + offset, sizeof(uint32_t));
        std::memcpy(&step_u32, base + offset + 4, sizeof(uint32_t));
        if (count_u32 == 0) {
            offset += sector_bytes;
            continue;
        }
        const int64_t count = static_cast<int64_t>(count_u32);
        const int64_t data_off = offset + block_prefix_bytes;
        const int64_t payload_bytes = count * record_bytes;
        if (data_off + payload_bytes > file_bytes) break;
        const int64_t step = static_cast<int64_t>(step_u32);
        if (step > max_step) max_step = step;
        blocks.push_back(Block{data_off, count, step});
        offset = round_up(data_off + payload_bytes);
    }

    std::vector<uint64_t> seen(
        static_cast<size_t>((total_numel + 63) / 64), 0);
    int64_t remaining = total_numel;
    int64_t applied = 0;
    int64_t records_seen = 0;
    int64_t blocks_replayed = 0;
    float* p = p_flat.data_ptr<float>();
    float* m = m_flat.data_ptr<float>();
    float* v = v_flat.data_ptr<float>();

    for (int64_t b = static_cast<int64_t>(blocks.size()) - 1; b >= 0; --b) {
        const Block& block = blocks[static_cast<size_t>(b)];
        const uint8_t* rec_base = base + block.data_off;
        for (int64_t r = block.count - 1; r >= 0; --r) {
            const uint8_t* rec = rec_base + r * record_bytes;
            uint32_t idx_u32 = 0;
            float p_val = 0.0f, m_val = 0.0f, v_val = 0.0f;
            std::memcpy(&idx_u32, rec, sizeof(uint32_t));
            records_seen += 1;
            const int64_t idx = static_cast<int64_t>(idx_u32);
            if (idx < 0 || idx >= total_numel) continue;
            const uint64_t bit = uint64_t{1} << (idx & 63);
            uint64_t& word = seen[static_cast<size_t>(idx >> 6)];
            if ((word & bit) != 0) continue;
            word |= bit;
            std::memcpy(&p_val, rec + 4, sizeof(float));
            std::memcpy(&m_val, rec + 8, sizeof(float));
            std::memcpy(&v_val, rec + 12, sizeof(float));
            p[idx] = p_val;
            m[idx] = m_val;
            v[idx] = v_val;
            applied += 1;
            remaining -= 1;
            if (remaining <= 0) break;
        }
        blocks_replayed += 1;
        if (remaining <= 0) break;
    }

    return std::make_tuple(
        max_step,
        static_cast<int64_t>(blocks.size()),
        blocks_replayed,
        records_seen,
        applied,
        remaining);
}

}  // namespace


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    namespace bio = proposal::binary_io;
    m.doc() = "proposal CP I/O hot paths (scatter, AoS interleave, "
              "binary_io writes)";

    // binary_io.cpp bindings. All direct ``::write()`` syscalls live in
    // binary_io.cpp; these expose them to Python. io_uring developer
    // swaps inside binary_io.cpp's full_write_aligned.
    m.attr("HEADER_BLOCK_BYTES") = bio::HEADER_BLOCK_BYTES;
    m.attr("SECTOR_BYTES") = bio::SECTOR_BYTES;

    m.def("stamp_header_into_pinned", &bio::stamp_header_into_pinned,
          py::arg("pinned_addr"), py::arg("header"),
          py::call_guard<py::gil_scoped_release>());
    m.def("write_prptens", &bio::write_prptens,
          py::arg("final_path"), py::arg("pinned_addr"), py::arg("header"),
          py::arg("total_padded_bytes"),
          py::arg("mode") = 0644, py::arg("do_fsync") = false,
          py::call_guard<py::gil_scoped_release>());
    m.def("write_prptens_keep_open", &bio::write_prptens_keep_open,
          py::arg("final_path"), py::arg("pinned_addr"), py::arg("header"),
          py::arg("total_padded_bytes"),
          py::arg("mode") = 0644,
          py::call_guard<py::gil_scoped_release>());
    m.def("write_header_payload_keep_open", &bio::write_header_payload_keep_open,
          py::arg("final_path"), py::arg("pinned_addr"), py::arg("header"),
          py::arg("total_padded_bytes"),
          py::arg("mode") = 0644,
          py::call_guard<py::gil_scoped_release>());
    m.def("open_lossy_log_for_append", &bio::open_lossy_log_for_append,
          py::arg("path"), py::arg("header_pinned_addr"), py::arg("header"),
          py::arg("mode") = 0644,
          py::call_guard<py::gil_scoped_release>());
    m.def("append_lossy_log", &bio::append_lossy_log,
          py::arg("fd"), py::arg("pinned_addr"),
          py::arg("total_padded_bytes"),
          py::call_guard<py::gil_scoped_release>());
    m.def("fsync_fd", &bio::fsync_fd, py::arg("fd"),
          py::call_guard<py::gil_scoped_release>());
    m.def("close_fd", &bio::close_fd, py::arg("fd"),
          py::call_guard<py::gil_scoped_release>());
    m.def("unlink_path", &bio::unlink_path, py::arg("path"),
          py::call_guard<py::gil_scoped_release>(),
          "::unlink(path); swallow ENOENT.");
    m.def("read_file_into_tensor",
          [](const std::string& path, const torch::Tensor& dst_u8,
             int64_t offset, int64_t nbytes) {
              check_cpu_contiguous(dst_u8, "dst_u8", c10::ScalarType::Byte);
              if (offset < 0 || nbytes < 0)
                  throw std::invalid_argument("read_file_into_tensor: negative offset/nbytes");
              if (nbytes > dst_u8.numel())
                  throw std::out_of_range("read_file_into_tensor: dst too small");
              size_t got = 0;
              {
                  py::gil_scoped_release rel;
                  got = bio::read_file_into_buffer(
                      path,
                      reinterpret_cast<int64_t>(dst_u8.data_ptr<uint8_t>()),
                      offset,
                      nbytes);
              }
              return static_cast<int64_t>(got);
          },
          py::arg("path"), py::arg("dst_u8"), py::arg("offset"), py::arg("nbytes"),
          "Read file bytes directly into a CPU uint8 tensor.");
    m.def("file_size", &bio::file_size, py::arg("path"),
          py::call_guard<py::gil_scoped_release>(),
          "::stat-based file size in bytes, or -1 if missing.");
    m.def("scatter_lossy_records_to_dense_by_intervals",
          &scatter_lossy_records_to_dense_by_intervals,
          py::arg("aos_u8"), py::arg("record_byte_offset"),
          py::arg("n_records"), py::arg("starts_i64"), py::arg("ends_i64"),
          py::arg("p_tensors"), py::arg("m_tensors"), py::arg("v_tensors"),
          py::call_guard<py::gil_scoped_release>(),
          "Scatter 16-byte lossy AOS records into pinned dense p/m/v tensors.");
    m.def("load_lfcl_payload_interleaved",
          &load_lfcl_payload_interleaved,
          py::arg("lfcl_u8"), py::arg("payload_offset_bytes"),
          py::arg("numels_i64"), py::arg("p_flat"), py::arg("m_flat"),
          py::arg("v_flat"),
          py::call_guard<py::gil_scoped_release>(),
          "Load LFCL tensor-wise p/m/v payload into dense flat CPU tensors.");
    m.def("replay_lfsy_log_into_dense",
          &replay_lfsy_log_into_dense,
          py::arg("lfsy_u8"), py::arg("total_numel"),
          py::arg("p_flat"), py::arg("m_flat"), py::arg("v_flat"),
          py::arg("header_block_bytes"), py::arg("sector_bytes"),
          py::arg("block_prefix_bytes"), py::arg("record_bytes"),
          py::call_guard<py::gil_scoped_release>(),
          "Reverse replay LFSY AOS log into dense flat CPU tensors.");
}
