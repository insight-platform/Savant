#include "cudaCrop.h"
#include "cuda_utils.h"
#include "npp.h"


inline void __getLastCudaError(const char *errorMessage, const char *file,
                               const int line) {
    cudaError_t err = cudaGetLastError();

    if (cudaSuccess != err) {
        fprintf(stderr,
                "%s(%i) : getLastCudaError() CUDA error :"
                " %s : (%d) %s.\n",
                file, line, errorMessage, static_cast<int>(err),
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

inline void __printLastCudaError(const char *errorMessage, const char *file,
                                 const int line) {
    cudaError_t err = cudaGetLastError();

    if (cudaSuccess != err) {
        fprintf(stderr,
                "%s(%i) : getLastCudaError() CUDA error :"
                " %s : (%d) %s.\n",
                file, line, errorMessage, static_cast<int>(err),
                cudaGetErrorString(err));
    }
}


__global__ void crop_8u_C4R(
        const Npp32u* input,
        const NppiSize inputSize,
        const unsigned int src_pitch,
        const NppiRect crop_rect,
        const unsigned int dst_pitch,
        Npp32u* output,
        int shiftX,
        int shiftY
        )
{
    unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    if( x >= crop_rect.width || y >= crop_rect.height )
        return;
    output[(y + shiftY) * dst_pitch + x + shiftX] = input[(y + crop_rect.y) * src_pitch + x+crop_rect.x];

}

cudaError_t cudaCrop(
        const Npp8u *ref_frame,
        const NppiSize inputSize,
        const NppiRect crop_rect,
        Npp8u* output_image,
        const NppiSize outputSize,
        int shiftX,
        int shiftY)
{
    if( !ref_frame || !output_image )
        return cudaErrorInvalidDevicePointer;
    if( inputSize.width == 0 || inputSize.height == 0 || crop_rect.width == 0 || crop_rect.height == 0 ||
    crop_rect.x < 0 || crop_rect.y < 0 || crop_rect.x > inputSize.width || crop_rect.y > inputSize.height ||
    crop_rect.x + crop_rect.width - 1 >= inputSize.width || crop_rect.y + crop_rect.height - 1>= inputSize.height)
        return cudaErrorInvalidValue;

    // launch kernel
    dim3 block(8, 8);
    dim3 grid(iDivUp(crop_rect.width, block.x), iDivUp(crop_rect.height,block.y));

    unsigned int src_pitch = inputSize.width;
    unsigned int dst_pitch = outputSize.width;

    crop_8u_C4R<<<grid, block>>>(
            (Npp32u *) ref_frame,
            inputSize,
            src_pitch,
            crop_rect,
            dst_pitch,
            (Npp32u *) output_image,
            shiftX,
            shiftY);

    return CUDA(cudaGetLastError());
}
