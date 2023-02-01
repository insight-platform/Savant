#include "nvsurfaceptr.h"
#include <cuda_egl_interop.h>
#include <gst/gstinfo.h>
#include <iostream>

GST_DEBUG_CATEGORY_STATIC (gst_dsclcprepro_debug);
#define GST_CAT_DEFAULT gst_dsclcprepro_debug

#define cudaCheckError() { \
    cudaError_t e=cudaGetLastError(); \
    if (e!=cudaSuccess) { \
        GST_ERROR("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e)); \
        exit(0); \
    } \
}

DSCudaMemory::DSCudaMemory(NvBufSurface *surface, guint batch_id) {
    GST_DEBUG_CATEGORY_INIT (
            gst_dsclcprepro_debug,
            "dsclcprepro",
            150,
            "dsclcprepro plugin");
    if (!surface) {
        throw std::invalid_argument("Invalid pointer to NvBufSurface");
    }
    if (batch_id >= surface->batchSize) {
        throw std::invalid_argument("batch_id is out of bound");
    }
    _surface = surface;
    _batch_id = batch_id;
}

Npp8u *DSCudaMemory::GetMapCudaPtr() {
    Npp8u *frame_ptr = nullptr;
    cudaEglFrame cudaEgl;
    cudaError_t err;
    NvBufSurfaceParams &surface = _surface->surfaceList[_batch_id];

    switch (_surface->memType) {

        case NVBUF_MEM_SURFACE_ARRAY:
            if (_egl_frame_ptr != nullptr) {
                return _egl_frame_ptr;
            }
            if (!surface.mappedAddr.eglImage) {
                if (NvBufSurfaceMapEglImage(_surface, _batch_id) != 0) {
                    GST_ERROR("Error NvBufSurfaceMapEglImage");
                    goto error;
                }
            }
            if (_pResource == nullptr) {
                err = cudaGraphicsEGLRegisterImage(
                        &_pResource,
                        surface.mappedAddr.eglImage,
                        cudaGraphicsRegisterFlagsNone
                );
                if (err != cudaSuccess) {
                    GST_ERROR("Error cudaGraphicsEGLRegisterImage");
                    goto error;
                }
            }

            err = cudaGraphicsResourceGetMappedEglFrame(&cudaEgl, _pResource, 0, 0);
            if (err != cudaSuccess) {
                GST_ERROR("Error cudaGraphicsResourceGetMappedEglFrame");
                goto error;
            }

            size_t inputSize;
            cudaGraphicsResourceGetMappedPointer((void **) &frame_ptr, &inputSize, _pResource);
            _egl_frame_ptr = frame_ptr;
            break;
        case NVBUF_MEM_CUDA_DEVICE: case NVBUF_MEM_CUDA_UNIFIED:
            frame_ptr = (Npp8u *) surface.dataPtr;
            break;
        default:
            GST_ERROR("Not supported memory type");
            break;
    }
    return frame_ptr;

    error:
        return nullptr;
}

void DSCudaMemory::UnMapCudaPtr() {
    if (_surface->memType == NVBUF_MEM_SURFACE_ARRAY) {
        if (_pResource != nullptr) {
            if (cudaGraphicsUnregisterResource(_pResource) != cudaSuccess) {
                GST_ERROR("Error cudaGraphicsUnregisterResource");
                return;
            }
            _pResource = nullptr;
            _egl_frame_ptr = nullptr;
        }
        if (_surface->surfaceList[_batch_id].mappedAddr.eglImage) {
            if (NvBufSurfaceUnMapEglImage(_surface, _batch_id) != 0) {
                GST_ERROR("Error NvBufSurfaceUnMapEglImage");
                return;
            }
        }
    }
}

guint DSCudaMemory::width() {
    return _surface->surfaceList[_batch_id].width;
}

guint DSCudaMemory::height() {
    return _surface->surfaceList[_batch_id].height;
}

guint DSCudaMemory::size() {
    return _surface->surfaceList[_batch_id].dataSize;
}
