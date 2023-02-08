#ifndef DEEPSTREAM_PENCILS_NVSURFACEPTR_H
#define DEEPSTREAM_PENCILS_NVSURFACEPTR_H

#include <glib.h>
#include "npp.h"
#include "nvbufsurface.h"
#include <unordered_map>

class DSCudaMemory {
private:
    struct cudaGraphicsResource *_pResource = nullptr;
    Npp8u *_egl_frame_ptr = nullptr;
    guint _batch_id;
    NvBufSurface *_surface;
public:
    DSCudaMemory(NvBufSurface *surface, guint batch_id);
    Npp8u *GetMapCudaPtr();
    void UnMapCudaPtr();
    guint width();
    guint height();
    guint size();
    guint pitch();
};


#endif //DEEPSTREAM_PENCILS_NVSURFACEPTR_H
