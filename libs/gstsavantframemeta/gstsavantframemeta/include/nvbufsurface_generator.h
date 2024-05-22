#ifndef _NV_BUF_SURFACE_GENERATOR_
#define _NV_BUF_SURFACE_GENERATOR_

#include <cstdint>
#include <gst/gst.h>


class NvBufSurfaceGenerator {
public:
    NvBufSurfaceGenerator(
        GstCaps *caps,
        uint32_t gpu_id,
        uint32_t mem_type
    );

    ~NvBufSurfaceGenerator();

    void create_surface(GstBuffer *buffer_dest);

private:
    GstBufferPool *buffer_pool;
};


#endif //_NV_BUF_SURFACE_GENERATOR_
