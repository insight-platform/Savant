#ifndef OPENCV_SAVANT_HPP
#define OPENCV_SAVANT_HPP

#include "opencv2/core.hpp"
#include "opencv2/core/cuda.hpp"

namespace cv {
    namespace savant {
        /** Constructs GpuMat headers pointing to user-allocated data */
        CV_EXPORTS_W cuda::GpuMat createGpuMat(int rows, int cols, int type, uint64 dataAddr, size_t step = Mat::AUTO_STEP);
    }
}

#endif //OPENCV_SAVANT_HPP
