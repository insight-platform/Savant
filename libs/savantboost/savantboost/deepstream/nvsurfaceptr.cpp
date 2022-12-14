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

DSCudaMemory::DSCudaMemory(NvBufSurface * surface, bool EglImageMapBuffer){
    GST_DEBUG_CATEGORY_INIT (
            gst_dsclcprepro_debug,
            "dsclcprepro",
            150,
            "dsclcprepro plugin");
    _EglImageMapBuffer = EglImageMapBuffer;
    _surface = surface;
}

Npp8u* DSCudaMemory::GetMapCudaPtr(guint batch_id){
    Npp8u* frame_ptr = nullptr;
    Npp8u* _gpu_frame = nullptr;
    Npp8u* _gpu_frame_1 = nullptr;
    cudaEglFrame cudaEgl;
    cudaError_t err;
    unsigned long int buffer_size;
    int status;

    switch (_surface->memType) {

        case NVBUF_MEM_SURFACE_ARRAY:
            if (_EglImageMapBuffer){
                GST_WARNING("Interaction with the buffers directly on the GPU of Jetson NX using NvBufSurfaceMapEglImage "
                            "unfortunately does not work correctly on long videos and long processing time. "
                            "This can lead to errors and crashes of the pipeline.");
                if (NvBufSurfaceMapEglImage (_surface, 0) !=0 ) {
                    GST_ERROR("Error NvBufSurfaceMapEglImage");
                    goto error;
                }
                err = cudaGraphicsEGLRegisterImage(
                    &_pResource,
                    _surface->surfaceList[batch_id].mappedAddr.eglImage,
                    cudaGraphicsRegisterFlagsNone
                );
                if (err != cudaSuccess) {
                    GST_ERROR("Error cudaGraphicsEGLRegisterImage");
                    goto error;
                }

                err = cudaGraphicsResourceGetMappedEglFrame(&cudaEgl, _pResource, 0, 0);
                if (err != cudaSuccess) {
                    GST_ERROR("Error cudaGraphicsResourceGetMappedEglFrame");
                    goto error;
                }

                size_t inputSize;
                cudaGraphicsResourceGetMappedPointer((void **) &frame_ptr, &inputSize, _pResource);
            }
            else {
                if (this->mapped_surface.find(batch_id) == this->mapped_surface.end()) {
                    status = NvBufSurfaceMap(_surface, -1, -1, NVBUF_MAP_READ_WRITE);
                    if (status != 0) GST_ERROR("Mapping error NvBufSurfaceMap");
                    NvBufSurfaceSyncForCpu(_surface, -1, -1);
                    cudaCheckError()
                    if (status != 0) GST_ERROR("Mapping error NvBufSurfaceSyncForCpu");
                    buffer_size = _surface->surfaceList[batch_id].width * _surface->surfaceList[batch_id].height * sizeof(Npp32u);
                    cudaMalloc((void **) &_gpu_frame, buffer_size);
                    cudaCheckError()
                    cudaMemcpy(_gpu_frame, _surface->surfaceList[batch_id].mappedAddr.addr[0], buffer_size, cudaMemcpyHostToDevice);
                    cudaCheckError()
                    this->mapped_surface[batch_id] = _gpu_frame;
                    frame_ptr = _gpu_frame;
                }
                else
                {
                    frame_ptr = this->mapped_surface[batch_id];
                }

            }
            break;
        case NVBUF_MEM_CUDA_DEVICE: case NVBUF_MEM_CUDA_UNIFIED:
            frame_ptr = (Npp8u*) _surface->surfaceList[batch_id].dataPtr;
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
    cudaError_t err;
    Npp8u* _gpu_frame = nullptr;
    int status;
    if (_surface != nullptr)
        if (_surface->memType == NVBUF_MEM_SURFACE_ARRAY){
            if (_EglImageMapBuffer){
                if (NvBufSurfaceUnMapEglImage(_surface, 0) !=0) GST_ERROR("Error NvBufSurfaceUnMapEglImage");
            }
            else {
                guint batch_id;
                for (std::pair<guint, Npp8u *> element : this->mapped_surface)
                {
                    batch_id = element.first;
                    unsigned long int buffer_size = _surface->surfaceList[batch_id].width * _surface->surfaceList[batch_id].height * sizeof(Npp32u);
                    _gpu_frame = (Npp8u*) element.second;
                    err = cudaMemcpy(
                      (void*) _surface->surfaceList[batch_id].mappedAddr.addr[0], 
                        _gpu_frame, 
                        buffer_size,
                        cudaMemcpyDeviceToHost
                    );
                    if (err != cudaSuccess) GST_ERROR("Error cudaMemcpy");
                    cudaFree(_gpu_frame);  
                }
                this->mapped_surface.erase(this->mapped_surface.begin(), this->mapped_surface.end());
                NvBufSurfaceSyncForDevice(_surface, -1, -1);
                status = NvBufSurfaceUnMap(_surface, -1, -1);
                if (status!=0) GST_ERROR("UnMapping error NvBufSurfaceUnMap");
            }
        }
}

