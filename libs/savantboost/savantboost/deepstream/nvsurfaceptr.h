#ifndef DEEPSTREAM_PENCILS_NVSURFACEPTR_H
#define DEEPSTREAM_PENCILS_NVSURFACEPTR_H

#include <glib.h>
#include "npp.h"
#include "nvbufsurface.h"
#include <unordered_map>

class DSCudaMemory{
private:
    struct cudaGraphicsResource* _pResource = nullptr;
    guint _batch_id;
    bool _EglImageMapBuffer;
    NvBufSurface *_surface;
    std::unordered_map<guint, Npp8u *> mapped_surface;
public:
    DSCudaMemory(NvBufSurface * surface, bool EglImageMapBuffer);
    Npp8u* GetMapCudaPtr(guint batch_id);
    void UnMapCudaPtr();
};


#endif //DEEPSTREAM_PENCILS_NVSURFACEPTR_H
