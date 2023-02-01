#pragma  once

#include "pysavantboost.h"
#include "deepstream/nvsurfaceptr.h"
#include <pybind11/functional.h>


namespace pysavantboost {
    void bindnvsurfaceptr(py::module &m);

    class PyDSCudaMemory {
    private:
        DSCudaMemory *ds_cuda_memory;
    public:
        PyDSCudaMemory(uint64_t gst_buffer, guint batch_id);
        uint64_t GetMapCudaPtr();
        void UnMapCudaPtr();
        guint width();
        guint height();
        guint size();
    };
}
