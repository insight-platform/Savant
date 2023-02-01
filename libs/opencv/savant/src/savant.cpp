#include "opencv2/savant.hpp"
#include "opencv2/core.hpp"
#include "opencv2/core/cuda.hpp"

namespace cv {
    namespace savant {
        cuda::GpuMat createGpuMat(int rows, int cols, int type, uint64 dataAddr, size_t step) {
            cuda::GpuMat gpuMat(rows, cols, type, (void *) dataAddr, step);
            return gpuMat;
        }
    }
}
