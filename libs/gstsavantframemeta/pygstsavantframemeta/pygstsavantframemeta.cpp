#include "gstsavantframemeta.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(pygstsavantframemeta, m) {
    m.doc() = "GstSavantFrameMeta library";

    py::class_<GstSavantFrameMeta>(m, "GstSavantFrameMeta")
        .def_readwrite("idx", &GstSavantFrameMeta::idx);

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
}
