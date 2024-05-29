#include "nvbufsurface_generator.h"

#include <gst/gst.h>
#include <stdexcept>
#include "gstnvdsbufferpool.h"
#ifdef PLATFORM_X86_64
#include "nvbufsurface.h"
#endif


NvBufSurfaceGenerator::NvBufSurfaceGenerator(
    GstCaps *caps,
    uint32_t gpu_id,
    uint32_t mem_type
) {
#ifdef PLATFORM_X86_64

    GST_DEBUG("Creating NvBufSurfaceGenerator");
    GstStructure *config;

    buffer_pool = gst_nvds_buffer_pool_new();
    GST_DEBUG("Buffer pool %p created", buffer_pool);

    config = gst_buffer_pool_get_config(buffer_pool);
    gst_buffer_pool_config_set_params(
        config, caps, sizeof(NvBufSurface), 4, 4 + 4);
    gst_structure_set(
        config,
        "memtype", G_TYPE_UINT, mem_type,
        "gpu-id", G_TYPE_UINT, gpu_id,
        "batch-size", G_TYPE_UINT, 1,
        NULL);

    if (!gst_buffer_pool_set_config(buffer_pool, config)) {
        GST_ERROR("Failed to configure buffer pool %p", buffer_pool);
        throw std::runtime_error("Failed to configure buffer pool");
    }
    GST_DEBUG("Buffer pool %p configured", buffer_pool);

    gst_buffer_pool_set_active(buffer_pool, TRUE);

    GST_DEBUG("NvBufSurfaceGenerator created");

#else

    throw std::runtime_error("NvBufSurfaceGenerator is not supported on this platform");

#endif
}


NvBufSurfaceGenerator::~NvBufSurfaceGenerator() {
#ifdef PLATFORM_X86_64

    GST_DEBUG("Destroying NvBufSurfaceGenerator");
    gst_object_unref(buffer_pool);
    GST_DEBUG("Buffer pool %p destroyed", buffer_pool);

#else

    throw std::runtime_error("NvBufSurfaceGenerator is not supported on this platform");

#endif
}


void NvBufSurfaceGenerator::create_surface(GstBuffer *buffer_dest) {
    #ifdef PLATFORM_X86_64

    GST_DEBUG("Creating NvBufSurface at buffer %p", buffer_dest);
    GstBuffer *buffer;
    GstFlowReturn result;

    GST_DEBUG("Acquiring buffer from pool %p", buffer_pool);
    result = gst_buffer_pool_acquire_buffer(
        buffer_pool, &buffer, nullptr);

    if (result != GST_FLOW_OK) {
        GST_ERROR(
            "Failed to acquire buffer from pool %p: %d",
            buffer_pool, result);
        throw std::runtime_error("Failed to acquire buffer from pool");
    }

    gst_buffer_copy_into(buffer_dest, buffer, GST_BUFFER_COPY_ALL, 0, -1);
    GST_DEBUG("Buffer %p copied into buffer %p", buffer, buffer_dest);
    gst_buffer_unref(buffer);
    gst_buffer_pool_release_buffer(buffer_pool, buffer);

#else

    throw std::runtime_error("NvBufSurfaceGenerator is not supported on this platform");

#endif
}
