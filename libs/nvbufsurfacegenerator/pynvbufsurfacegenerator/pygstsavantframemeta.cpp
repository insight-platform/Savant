#include "nvbufsurface_generator.h"
#include <pybind11/pybind11.h>


namespace py = pybind11;

PYBIND11_MODULE(pynvbufsurfacegenerator, m) {
    m.doc() = "NvBufSurfaceGenerator library";

    py::class_<NvBufSurfaceGenerator>(m, "NvBufSurfaceGenerator")
        .def(py::init([](
            size_t gst_caps,
            uint32_t gpuId,
            uint32_t memType
        ) {
            auto *caps = reinterpret_cast<GstCaps *>(gst_caps);
            return new NvBufSurfaceGenerator(caps, gpuId, memType);
        }))
        .def("create_surface", [](
            NvBufSurfaceGenerator &self,
            size_t gst_buffer_dest
        ) {
            auto *buffer_dest = reinterpret_cast<GstBuffer *>(gst_buffer_dest);
            py::gil_scoped_release release;
            self.create_surface(buffer_dest);
        });

}
