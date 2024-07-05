#ifndef _NV_BUF_SURFACE_GENERATOR_
#define _NV_BUF_SURFACE_GENERATOR_

#include <cstdint>
#include <gst/gst.h>


/**
 * Generates GStreamer buffers with NvBufSurface memory allocated.
 */
class NvBufSurfaceGenerator {
public:

    /**
     * Create a new NvBufSurfaceGenerator.
     *
     * @param caps Caps for the generated buffers.
     * @param gpu_id ID of the GPU to allocate NvBufSurface.
     * @param mem_type Memory type for the NvBufSurface.
     */
    NvBufSurfaceGenerator(
        GstCaps *caps,
        uint32_t gpu_id,
        uint32_t mem_type
    );

    ~NvBufSurfaceGenerator();

    /**
     * Create a new NvBufSurface and attach it to the given buffer.
     *
     * @param buffer_dest Buffer to attach the NvBufSurface to.
     */
    void create_surface(GstBuffer *buffer_dest);

private:
    GstBufferPool *buffer_pool;
};


#endif //_NV_BUF_SURFACE_GENERATOR_
