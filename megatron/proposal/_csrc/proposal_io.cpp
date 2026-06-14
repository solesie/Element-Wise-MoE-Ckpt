// proposal_io: C++ pybind bindings for binary_io.cpp helpers.
//
// Legacy CPU paths (scatter_into_flat, aos_write) removed — the GPU AOS
// pipeline (gpu_aos.py + segment_pipeline.py) now produces records on
// device directly into pinned_aos. Only the file-IO helpers in
// binary_io.cpp remain CPU-side and are re-exported here for Python.
#include "binary_io.h"

#include <torch/extension.h>
#include <pybind11/stl.h>

#include <atomic>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
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
    m.def("read_file_range", [](const std::string& path,
                                int64_t offset, int64_t nbytes) {
              std::string buf;
              {
                  py::gil_scoped_release rel;
                  buf = bio::read_file_range(path, offset, nbytes);
              }
              return py::bytes(buf);
          },
          py::arg("path"), py::arg("offset"), py::arg("nbytes"),
          "::pread bytes [offset, offset+nbytes) from path; returns bytes.");
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
}
