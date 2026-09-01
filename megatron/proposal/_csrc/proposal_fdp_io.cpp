#include "fdp_store.h"

#include <torch/extension.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace {

py::dict stats_to_dict(const proposal::fdp::FdpStatsSnapshot& s) {
    py::dict d;
    d["backend"] = s.backend;
    d["device_path"] = s.device_path;
    d["nsid"] = s.nsid;
    d["lba_size"] = s.lba_size;
    d["namespace_nlb"] = s.namespace_nlb;
    d["fdp_supported"] = s.fdp_supported;
    d["fdp_pid_enabled"] = s.fdp_pid_enabled;
    d["metadata_start_lba"] = s.metadata_start_lba;
    d["metadata_nlb"] = s.metadata_nlb;
    d["metadata_image_lba"] = s.metadata_image_lba;
    d["metadata_image_nlb"] = s.metadata_image_nlb;
    d["metadata_image_bytes"] = s.metadata_image_bytes;
    d["data_start_lba"] = s.data_start_lba;
    d["data_nlb"] = s.data_nlb;
    d["chunk_bytes"] = s.chunk_bytes;
    d["chunk_lba_count"] = s.chunk_lba_count;
    d["data_chunk_count"] = s.data_chunk_count;
    d["allocated_chunks"] = s.allocated_chunks;
    d["rank"] = s.rank;
    d["world_size"] = s.world_size;
    d["num_moe_layers"] = s.num_moe_layers;
    d["experts_per_layer"] = s.experts_per_layer;
    d["expert_slots"] = s.expert_slots;
    d["dp_rank"] = s.dp_rank;
    d["dp_size"] = s.dp_size;
    d["pp_rank"] = s.pp_rank;
    d["pp_size"] = s.pp_size;
    d["ep_rank"] = s.ep_rank;
    d["ep_size"] = s.ep_size;
    d["nefc_shard_index"] = s.nefc_shard_index;
    d["nefc_shard_count"] = s.nefc_shard_count;
    d["expert_id_base"] = s.expert_id_base;
    d["expert_ids_global"] = s.expert_ids_global;
    d["nefc_qd"] = s.nefc_qd;
    d["expert_qd"] = s.expert_qd;
    d["metadata_qd"] = s.metadata_qd;
    d["max_io_nlb"] = s.max_io_nlb;
    d["max_io_bytes"] = s.max_io_bytes;
    d["write_fua"] = s.write_fua;
    d["metadata_mmap"] = s.metadata_mmap;
    d["metadata_flush_count"] = s.metadata_flush_count;
    d["seq"] = s.seq;
    d["last_flush_iter"] = s.last_flush_iter;
    return d;
}

py::dict summary_to_dict(const proposal::fdp::MetadataSummary& s) {
    py::dict d;
    d["magic"] = s.magic;
    d["version"] = s.version;
    d["seq"] = s.seq;
    d["last_flush_iter"] = s.last_flush_iter;
    d["allocated_chunks"] = s.allocated_chunks;
    d["data_chunk_count"] = s.data_chunk_count;
    d["bitmap_bytes"] = s.bitmap_bytes;
    d["bitmap_offset"] = s.bitmap_offset;
    d["nefc_offset"] = s.nefc_offset;
    d["lfcs_offset"] = s.lfcs_offset;
    d["lfsy_offset"] = s.lfsy_offset;
    d["metadata_image_bytes"] = s.metadata_image_bytes;
    d["nefc_valid"] = s.nefc_valid;
    d["nefc_iter"] = s.nefc_iter;
    d["nefc_logical_bytes"] = s.nefc_logical_bytes;
    d["nefc_chunk_count"] = s.nefc_chunk_count;
    d["nefc_shard_count"] = s.nefc_shard_count;
    d["nefc_valid_count"] = s.nefc_valid_count;
    d["lfcs_valid_count"] = s.lfcs_valid_count;
    d["lfcs_total_chunks"] = s.lfcs_total_chunks;
    d["lfcs_max_chunk_count"] = s.lfcs_max_chunk_count;
    d["lfsy_valid_count"] = s.lfsy_valid_count;
    return d;
}

py::dict batch_stats_to_dict(const proposal::fdp::UcmdRingBatchStats& s) {
    py::dict d;
    d["commands"] = s.commands;
    d["submitted"] = s.submitted;
    d["queue_depth"] = s.queue_depth;
    d["submit_calls"] = s.submit_calls;
    d["registered_file_submits"] = s.registered_file_submits;
    d["submission_policy"] = "sq_space";
    d["completion_thread"] = true;
    d["sqpoll"] = s.sqpoll;
    d["blocking_completion"] = s.blocking_completion;
    d["batch_begin_mono_raw_ns"] = s.batch_begin_mono_raw_ns;
    d["first_submit_mono_raw_ns"] = s.first_submit_mono_raw_ns;
    d["last_submit_mono_raw_ns"] = s.last_submit_mono_raw_ns;
    d["end_wait_begin_mono_raw_ns"] = s.end_wait_begin_mono_raw_ns;
    d["first_completion_mono_raw_ns"] = s.first_completion_mono_raw_ns;
    d["last_completion_mono_raw_ns"] = s.last_completion_mono_raw_ns;
    d["batch_end_mono_raw_ns"] = s.batch_end_mono_raw_ns;
    return d;
}

py::dict ring_runtime_to_dict(const proposal::fdp::UcmdRingRuntimeStats& s) {
    py::dict d;
    d["name"] = s.name;
    d["ring_fd"] = s.ring_fd;
    d["completion_tid"] = s.completion_tid;
    d["sqpoll"] = s.sqpoll;
    d["blocking_completion"] = s.blocking_completion;
    d["completion_waits"] = s.completion_waits;
    d["completed_cqes"] = s.completed_cqes;
    return d;
}

py::dict metadata_perf_to_dict(
    const proposal::fdp::MetadataCommitPerfStats& s) {
    py::dict d;
    d["valid"] = s.valid;
    d["iter"] = s.iter;
    d["owner_rank"] = s.owner_rank;
    d["metadata_image_bytes"] = s.metadata_image_bytes;
    d["nvme_flush_enabled"] = s.nvme_flush_enabled;
    d["start_mono_raw_ns"] = s.start_mono_raw_ns;
    d["recalc_done_mono_raw_ns"] = s.recalc_done_mono_raw_ns;
    d["header_done_mono_raw_ns"] = s.header_done_mono_raw_ns;
    d["image_write_done_mono_raw_ns"] = s.image_write_done_mono_raw_ns;
    d["flush_done_mono_raw_ns"] = s.flush_done_mono_raw_ns;
    d["end_mono_raw_ns"] = s.end_mono_raw_ns;
    return d;
}

py::dict nefc_entry_to_dict(const proposal::fdp::NefcEntrySnapshot& entry) {
    py::dict d;
    d["valid"] = entry.valid;
    d["shard_index"] = entry.shard_index;
    d["rank"] = entry.rank;
    d["dp_rank"] = entry.dp_rank;
    d["pp_rank"] = entry.pp_rank;
    d["ep_rank"] = entry.ep_rank;
    d["iter"] = entry.iter;
    d["logical_bytes"] = entry.logical_bytes;
    d["chunk_count"] = entry.chunk_count;
    return d;
}

py::dict lfcs_entry_to_dict(const proposal::fdp::LfcsEntrySnapshot& entry) {
    py::dict d;
    d["valid"] = entry.valid;
    d["slot"] = entry.slot;
    d["rank"] = entry.rank;
    d["layer_id"] = entry.layer_id;
    d["expert_id"] = entry.expert_id;
    d["dp_rank"] = entry.dp_rank;
    d["pp_rank"] = entry.pp_rank;
    d["ep_rank"] = entry.ep_rank;
    d["generation"] = entry.generation;
    d["iter"] = entry.iter;
    d["logical_bytes"] = entry.logical_bytes;
    d["chunk_count"] = entry.chunk_count;
    d["chunks"] = entry.chunks;
    return d;
}

py::dict lfsy_entry_to_dict(const proposal::fdp::LfsyEntrySnapshot& entry) {
    py::dict d;
    d["valid"] = entry.valid;
    d["slot"] = entry.slot;
    d["rank"] = entry.rank;
    d["layer_id"] = entry.layer_id;
    d["expert_id"] = entry.expert_id;
    d["dp_rank"] = entry.dp_rank;
    d["pp_rank"] = entry.pp_rank;
    d["ep_rank"] = entry.ep_rank;
    d["generation"] = entry.generation;
    d["start_iter"] = entry.start_iter;
    d["logical_tail"] = entry.logical_tail;
    d["chunk_count"] = entry.chunk_count;
    d["chunks"] = entry.chunks;
    return d;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    namespace fdp = proposal::fdp;
    m.doc() = "proposal raw NVMe FDP backend bootstrap";

    m.attr("FDP_FORMAT_VERSION") = fdp::kFormatVersion;
    m.attr("FDP_CHUNK_BYTES_DEFAULT") = fdp::kDefaultChunkBytes;
    m.attr("FDP_MAX_CHUNKS_NEFC") = fdp::kMaxChunksNefc;
    m.attr("FDP_MAX_CHUNKS_LFCS") = fdp::kMaxChunksLfcs;
    m.attr("FDP_MAX_CHUNKS_LFSY") = fdp::kMaxChunksLfsy;

    m.def("open_session_from_env", []() {
        py::gil_scoped_release rel;
        fdp::global_store().open_from_env();
    });
    m.def("close_session", []() {
        py::gil_scoped_release rel;
        fdp::global_store().close();
    });
    m.def("format_or_load", []() {
        py::gil_scoped_release rel;
        fdp::global_store().format_or_load();
    });
    m.def("flush_metadata", [](uint64_t iter) {
        py::gil_scoped_release rel;
        fdp::global_store().flush_metadata(iter);
    }, py::arg("iter") = 0);
    m.def("begin_checkpoint_iter", [](uint64_t iter) {
        py::gil_scoped_release rel;
        fdp::global_store().begin_checkpoint_iter(iter);
    }, py::arg("iter"));
    m.def("mark_writer_done", [](uint64_t iter,
                                 const std::string& writer_kind) {
        py::gil_scoped_release rel;
        return fdp::global_store().mark_writer_done(iter, writer_kind);
    }, py::arg("iter"), py::arg("writer_kind"));
    m.def("begin_data_batch", []() {
        py::gil_scoped_release rel;
        fdp::global_store().begin_data_batch();
    });
    m.def("end_data_batch", []() {
        std::vector<uint32_t> counts;
        {
            py::gil_scoped_release rel;
            counts = fdp::global_store().end_data_batch();
        }
        py::dict d;
        d["nefc_commands"] = counts.size() > 0 ? counts[0] : 0;
        d["expert_commands"] = counts.size() > 1 ? counts[1] : 0;
        d["total_commands"] =
            (counts.size() > 0 ? counts[0] : 0) +
            (counts.size() > 1 ? counts[1] : 0);
        return d;
    });
    m.def("end_data_batch_stats", []() {
        std::vector<fdp::UcmdRingBatchStats> stats;
        {
            py::gil_scoped_release rel;
            stats = fdp::global_store().end_data_batch_stats();
        }
        py::dict nefc = batch_stats_to_dict(
            stats.size() > 0 ? stats[0] : fdp::UcmdRingBatchStats{});
        py::dict expert = batch_stats_to_dict(
            stats.size() > 1 ? stats[1] : fdp::UcmdRingBatchStats{});
        py::dict d;
        d["nefc"] = nefc;
        d["expert"] = expert;
        d["nefc_commands"] = nefc["commands"];
        d["expert_commands"] = expert["commands"];
        d["total_commands"] =
            (stats.size() > 0 ? stats[0].commands : 0) +
            (stats.size() > 1 ? stats[1].commands : 0);
        return d;
    });
    m.def("begin_nefc_batch", []() {
        py::gil_scoped_release rel;
        fdp::global_store().begin_nefc_batch();
    });
    m.def("end_nefc_batch", []() {
        py::gil_scoped_release rel;
        return fdp::global_store().end_nefc_batch();
    });
    m.def("end_nefc_batch_stats", []() {
        fdp::UcmdRingBatchStats stats;
        {
            py::gil_scoped_release rel;
            stats = fdp::global_store().end_nefc_batch_stats();
        }
        return batch_stats_to_dict(stats);
    });
    m.def("begin_expert_batch", []() {
        py::gil_scoped_release rel;
        fdp::global_store().begin_expert_batch();
    });
    m.def("end_expert_batch", []() {
        py::gil_scoped_release rel;
        return fdp::global_store().end_expert_batch();
    });
    m.def("end_expert_batch_stats", []() {
        fdp::UcmdRingBatchStats stats;
        {
            py::gil_scoped_release rel;
            stats = fdp::global_store().end_expert_batch_stats();
        }
        return batch_stats_to_dict(stats);
    });
    m.def("begin_expert_read_batch", []() {
        py::gil_scoped_release rel;
        fdp::global_store().begin_expert_read_batch();
    });
    m.def("end_expert_read_batch", []() {
        py::gil_scoped_release rel;
        return fdp::global_store().end_expert_read_batch();
    });
    m.def("end_expert_read_batch_stats", []() {
        fdp::UcmdRingBatchStats stats;
        {
            py::gil_scoped_release rel;
            stats = fdp::global_store().end_expert_read_batch_stats();
        }
        return batch_stats_to_dict(stats);
    });
    m.def("write_nefc", [](const std::string& object_name,
                           int64_t pinned_addr,
                           const std::string& header,
                           uint64_t payload_bytes,
                           uint64_t total_padded_bytes,
                           uint64_t iter) {
        py::gil_scoped_release rel;
        return fdp::global_store().write_nefc(
            object_name, pinned_addr, header, payload_bytes,
            total_padded_bytes, iter);
    }, py::arg("object_name"), py::arg("pinned_addr"), py::arg("header"),
       py::arg("payload_bytes"), py::arg("total_padded_bytes"),
       py::arg("iter"));
    m.def("write_lfcs", [](const std::string& object_name,
                           int64_t pinned_addr,
                           const std::string& header,
                           uint64_t payload_bytes,
                           uint64_t total_padded_bytes,
                           uint32_t layer_id,
                           uint32_t expert_id,
                           uint64_t iter,
                           uint64_t generation) {
        py::gil_scoped_release rel;
        return fdp::global_store().write_lfcs(
            object_name, pinned_addr, header, payload_bytes,
            total_padded_bytes, layer_id, expert_id, iter, generation);
    }, py::arg("object_name"), py::arg("pinned_addr"), py::arg("header"),
       py::arg("payload_bytes"), py::arg("total_padded_bytes"),
       py::arg("layer_id"), py::arg("expert_id"), py::arg("iter"),
       py::arg("generation"));
    m.def("open_lfsy", [](const std::string& object_name,
                          const std::string& header,
                          uint32_t layer_id,
                          uint32_t expert_id,
                          int64_t start_iter,
                          uint64_t generation) {
        py::gil_scoped_release rel;
        return fdp::global_store().open_lfsy(
            object_name, header, layer_id, expert_id, start_iter, generation);
    }, py::arg("object_name"), py::arg("header"), py::arg("layer_id"),
       py::arg("expert_id"), py::arg("start_iter"), py::arg("generation"));
    m.def("append_lfsy", [](int64_t handle,
                            int64_t pinned_addr,
                            uint64_t total_bytes,
                            uint64_t total_padded_bytes,
                            uint32_t layer_id,
                            uint32_t expert_id,
                            uint64_t iter,
                            uint64_t generation) {
        py::gil_scoped_release rel;
        return fdp::global_store().append_lfsy(
            handle, pinned_addr, total_bytes, total_padded_bytes,
            layer_id, expert_id, iter, generation);
    }, py::arg("handle"), py::arg("pinned_addr"), py::arg("total_bytes"),
       py::arg("total_padded_bytes"), py::arg("layer_id"),
       py::arg("expert_id"), py::arg("iter"), py::arg("generation"));
    m.def("fsync_handle", [](int64_t handle) {
        py::gil_scoped_release rel;
        fdp::global_store().fsync_handle(handle);
    }, py::arg("handle"));
    m.def("close_lfsy", [](int64_t handle) {
        py::gil_scoped_release rel;
        fdp::global_store().close_lfsy(handle);
    }, py::arg("handle"));
    m.def("stats", []() {
        return stats_to_dict(fdp::global_store().stats());
    });
    m.def("perf_runtime_stats", []() {
        py::dict d;
        d["nefc_ring"] = ring_runtime_to_dict(
            fdp::global_store().nefc_ring_runtime_stats());
        d["expert_ring"] = ring_runtime_to_dict(
            fdp::global_store().expert_ring_runtime_stats());
        d["last_metadata_commit"] = metadata_perf_to_dict(
            fdp::global_store().last_metadata_commit_perf());
        return d;
    });
    m.def("debug_read_metadata_summary", []() {
        fdp::MetadataSummary s;
        {
            py::gil_scoped_release rel;
            s = fdp::global_store().debug_read_metadata_summary();
        }
        return summary_to_dict(s);
    });
    m.def("debug_read_lfsy_entries", []() {
        std::vector<fdp::LfsyEntrySnapshot> entries;
        {
            py::gil_scoped_release rel;
            entries = fdp::global_store().debug_read_lfsy_entries();
        }
        py::list out;
        for (const auto& entry : entries) {
            out.append(lfsy_entry_to_dict(entry));
        }
        return out;
    });
    m.def("recovery_entries", []() {
        fdp::NefcEntrySnapshot nefc;
        std::vector<fdp::NefcEntrySnapshot> nefc_entries;
        std::vector<fdp::LfcsEntrySnapshot> lfcs;
        std::vector<fdp::LfsyEntrySnapshot> lfsy;
        {
            py::gil_scoped_release rel;
            nefc = fdp::global_store().read_nefc_entry();
            nefc_entries = fdp::global_store().read_nefc_entries();
            lfcs = fdp::global_store().read_lfcs_entries();
            lfsy = fdp::global_store().read_lfsy_entries();
        }
        py::list nefc_out;
        for (const auto& entry : nefc_entries) {
            nefc_out.append(nefc_entry_to_dict(entry));
        }
        py::list lfcs_out;
        for (const auto& entry : lfcs) {
            lfcs_out.append(lfcs_entry_to_dict(entry));
        }
        py::list lfsy_out;
        for (const auto& entry : lfsy) {
            lfsy_out.append(lfsy_entry_to_dict(entry));
        }
        py::dict d;
        d["nefc"] = nefc_entry_to_dict(nefc);
        d["nefc_shards"] = nefc_out;
        d["lfcs"] = lfcs_out;
        d["lfsy"] = lfsy_out;
        return d;
    });
    m.def("read_nefc_into", [](int64_t dst_addr, uint64_t dst_bytes) {
        py::gil_scoped_release rel;
        return fdp::global_store().read_nefc_into(dst_addr, dst_bytes);
    }, py::arg("dst_addr"), py::arg("dst_bytes"));
    m.def("read_nefc_shard_into", [](uint32_t shard_index,
                                     int64_t dst_addr,
                                     uint64_t dst_bytes) {
        py::gil_scoped_release rel;
        return fdp::global_store().read_nefc_shard_into(
            shard_index, dst_addr, dst_bytes);
    }, py::arg("shard_index"), py::arg("dst_addr"), py::arg("dst_bytes"));
    m.def("read_lfcs_into", [](uint32_t layer_id, uint32_t expert_id,
                               int64_t dst_addr, uint64_t dst_bytes) {
        py::gil_scoped_release rel;
        return fdp::global_store().read_lfcs_into(
            layer_id, expert_id, dst_addr, dst_bytes);
    }, py::arg("layer_id"), py::arg("expert_id"),
       py::arg("dst_addr"), py::arg("dst_bytes"));
    m.def("read_lfsy_into", [](uint32_t layer_id, uint32_t expert_id,
                               int64_t dst_addr, uint64_t dst_bytes) {
        py::gil_scoped_release rel;
        return fdp::global_store().read_lfsy_into(
            layer_id, expert_id, dst_addr, dst_bytes);
    }, py::arg("layer_id"), py::arg("expert_id"),
       py::arg("dst_addr"), py::arg("dst_bytes"));
    m.def("debug_raw_write_read_smoke", []() {
        py::gil_scoped_release rel;
        return fdp::global_store().debug_raw_write_read_smoke();
    });
    m.def("debug_read_nefc", [](uint64_t offset, uint64_t nbytes) {
        std::string out;
        {
            py::gil_scoped_release rel;
            out = fdp::global_store().debug_read_nefc(offset, nbytes);
        }
        return py::bytes(out);
    }, py::arg("offset"), py::arg("nbytes"));
    m.def("debug_read_lfcs", [](uint32_t layer_id, uint32_t expert_id,
                                uint64_t offset, uint64_t nbytes) {
        std::string out;
        {
            py::gil_scoped_release rel;
            out = fdp::global_store().debug_read_lfcs(
                layer_id, expert_id, offset, nbytes);
        }
        return py::bytes(out);
    }, py::arg("layer_id"), py::arg("expert_id"),
       py::arg("offset"), py::arg("nbytes"));
    m.def("debug_read_lfsy", [](uint32_t layer_id, uint32_t expert_id,
                                uint64_t offset, uint64_t nbytes) {
        std::string out;
        {
            py::gil_scoped_release rel;
            out = fdp::global_store().debug_read_lfsy(
                layer_id, expert_id, offset, nbytes);
        }
        return py::bytes(out);
    }, py::arg("layer_id"), py::arg("expert_id"),
       py::arg("offset"), py::arg("nbytes"));
    m.def("debug_alloc_chunk", []() {
        return fdp::global_store().debug_alloc_chunk();
    });
    m.def("debug_free_chunk", [](uint32_t chunk_id) {
        fdp::global_store().debug_free_chunk(chunk_id);
    }, py::arg("chunk_id"));
}
