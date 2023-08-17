#include "gstsavantframemeta.h"
#include "savantrsprobes.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(pygstsavantframemeta, m) {
    m.doc() = "GstSavantFrameMeta library";

    py::class_<GstSavantFrameMeta>(m, "GstSavantFrameMeta",
                                   "Contains frame related metadata.")
        .def_readwrite("idx", &GstSavantFrameMeta::idx, "Frame IDX.");

    m.def(
        "gst_buffer_get_savant_frame_meta",
        [](size_t gst_buffer) {
            auto *buffer = reinterpret_cast<GstBuffer *>(gst_buffer);
            return gst_buffer_get_savant_frame_meta(buffer);
        },
        py::return_value_policy::reference);

    m.def(
        "gst_buffer_add_savant_frame_meta",
        [](size_t gst_buffer, int idx) {
            auto *buffer = reinterpret_cast<GstBuffer *>(gst_buffer);
            return gst_buffer_add_savant_frame_meta(buffer, idx);
        },
        py::return_value_policy::reference);

    m.def(
        "gst_buffer_get_nvds_savant_frame_meta",
        [](size_t gst_buffer) {
            auto *buffer = reinterpret_cast<GstBuffer *>(gst_buffer);
            return gst_buffer_get_nvds_savant_frame_meta(buffer);
        },
        py::return_value_policy::reference);

    m.def(
        "gst_buffer_add_nvds_savant_frame_meta",
        [](size_t gst_buffer, int idx) {
            auto *buffer = reinterpret_cast<GstBuffer *>(gst_buffer);
            return gst_buffer_add_nvds_savant_frame_meta(buffer, idx);
        },
        py::return_value_policy::reference);

    m.def(
        "nvds_frame_meta_get_nvds_savant_frame_meta",
        [](NvDsFrameMeta *frame_meta) {
            return nvds_frame_meta_get_nvds_savant_frame_meta(frame_meta);
        },
        py::return_value_policy::reference);

    m.def("add_convert_savant_frame_meta_pad_probe", [](size_t gst_pad,
                                                        bool to_nvds) {
        auto *pad = reinterpret_cast<GstPad *>(gst_pad);
        gpointer conv_func;
        if (to_nvds) {
            conv_func = (gpointer)gst_savant_frame_meta_to_nvds;
        } else {
            conv_func = (gpointer)nvds_savant_frame_meta_to_gst;
        }
        gst_pad_add_probe(pad, GST_PAD_PROBE_TYPE_BUFFER,
                          convert_savant_frame_meta_pad_probe, conv_func, NULL);
    });

    m.def("move_frame_as_is_pad_probe", [](size_t gst_pad,
                                           size_t probe_info,
                                           uintptr_t video_pipeline,
                                           const char *stage) {
        auto *pad = reinterpret_cast<GstPad *>(gst_pad);
        auto *info = reinterpret_cast<GstPadProbeInfo *>(probe_info);
        SavantRsPadProbeData *data = create_savant_rs_pad_probe_data(
            video_pipeline, stage);
        move_frame_as_is_pad_probe(pad, info, (gpointer)data);
        release_savant_rs_pad_probe_data(data);
    });

    m.def("move_and_pack_frames_pad_probe", [](size_t gst_pad,
                                               size_t probe_info,
                                               uintptr_t video_pipeline,
                                               const char *stage) {
        auto *pad = reinterpret_cast<GstPad *>(gst_pad);
        auto *info = reinterpret_cast<GstPadProbeInfo *>(probe_info);
        SavantRsPadProbeData *data = create_savant_rs_pad_probe_data(
            video_pipeline, stage);
        move_and_pack_frames_pad_probe(pad, info, (gpointer)data);
        release_savant_rs_pad_probe_data(data);
    });

    m.def("move_batch_as_is_pad_probe", [](size_t gst_pad,
                                           size_t probe_info,
                                           uintptr_t video_pipeline,
                                           const char *stage) {
        auto *pad = reinterpret_cast<GstPad *>(gst_pad);
        auto *info = reinterpret_cast<GstPadProbeInfo *>(probe_info);
        SavantRsPadProbeData *data = create_savant_rs_pad_probe_data(
            video_pipeline, stage);
        move_batch_as_is_pad_probe(pad, info, (gpointer)data);
        release_savant_rs_pad_probe_data(data);
    });

    m.def("move_and_unpack_batch_pad_probe", [](size_t gst_pad,
                                                size_t probe_info,
                                                uintptr_t video_pipeline,
                                                const char *stage) {
        auto *pad = reinterpret_cast<GstPad *>(gst_pad);
        auto *info = reinterpret_cast<GstPadProbeInfo *>(probe_info);
        SavantRsPadProbeData *data = create_savant_rs_pad_probe_data(
            video_pipeline, stage);
        move_and_unpack_batch_pad_probe(pad, info, (gpointer)data);
        release_savant_rs_pad_probe_data(data);
    });
}
