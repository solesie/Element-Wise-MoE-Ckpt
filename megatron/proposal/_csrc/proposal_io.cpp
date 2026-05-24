// proposal_io: C++ pybind bindings for binary_io.cpp helpers.
//
// Legacy CPU paths (scatter_into_flat, aos_write) removed — the GPU AOS
// pipeline (gpu_aos.py + segment_pipeline.py) now produces records on
// device directly into pinned_aos. Only the file-IO helpers in
// binary_io.cpp remain CPU-side and are re-exported here for Python.
#include "binary_io.h"

#include <torch/extension.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace py = pybind11;


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
    m.def("write_lossless", &bio::write_lossless,
          py::arg("final_path"), py::arg("pinned_addr"), py::arg("header"),
          py::arg("total_padded_bytes"), py::arg("mode") = 0644,
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
}
