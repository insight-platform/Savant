#include "nvsurfaceptr.h"
#include <gst/gst.h>
#include <iostream>


namespace pysavantboost {
    PyDSCudaMemory::PyDSCudaMemory(uint64_t gst_buffer, guint batch_id) {
        GstMapInfo map_info;

        auto *buffer = (GstBuffer *) (gst_buffer);
        gst_buffer_map(buffer, &map_info, GST_MAP_READ);
        auto *nv_buf_surface = (NvBufSurface *) (map_info.data);
        gst_buffer_unmap(buffer, &map_info);
        ds_cuda_memory = new DSCudaMemory(nv_buf_surface, batch_id);
    }

    uint64_t PyDSCudaMemory::GetMapCudaPtr() {
        return (uint64_t) ds_cuda_memory->GetMapCudaPtr();
    }

    void PyDSCudaMemory::UnMapCudaPtr() {
        ds_cuda_memory->UnMapCudaPtr();
    }

    guint PyDSCudaMemory::width() {
        return ds_cuda_memory->width();
    }

    guint PyDSCudaMemory::height() {
        return ds_cuda_memory->height();
    }

    guint PyDSCudaMemory::size() {
        return ds_cuda_memory->size();
    }

    guint PyDSCudaMemory::pitch() {
        return ds_cuda_memory->pitch();
    }

    void bindnvsurfaceptr(py::module &m) {
        py::class_<PyDSCudaMemory>(m, "PyDSCudaMemory")
                .def(py::init<uint64_t, guint>())
                .def("GetMapCudaPtr", &PyDSCudaMemory::GetMapCudaPtr)
                .def("UnMapCudaPtr", &PyDSCudaMemory::UnMapCudaPtr)
                .def_property_readonly("width", &PyDSCudaMemory::width)
                .def_property_readonly("height", &PyDSCudaMemory::height)
                .def_property_readonly("size", &PyDSCudaMemory::size)
                .def_property_readonly("pitch", &PyDSCudaMemory::pitch);
    }
}
