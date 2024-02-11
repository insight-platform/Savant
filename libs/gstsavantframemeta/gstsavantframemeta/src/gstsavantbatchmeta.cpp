#include "gstsavantbatchmeta.h"

#include <glib.h>
#include <gst/gst.h>

#include "gstnvdsmeta.h"
#include "nvdsmeta.h"

/* GstMeta functions */

gboolean gst_savant_batch_meta_init(GstMeta *meta, gpointer params,
                                    GstBuffer *buffer) {
    GST_LOG("Initialize GstSavantBatchMeta for buffer %p", buffer);
    GstSavantBatchMeta *emeta = (GstSavantBatchMeta *)meta;
    emeta->idx = 0;
    return TRUE;
}

gboolean gst_savant_batch_meta_transform(GstBuffer *dest, GstMeta *meta,
                                         GstBuffer *buffer, GQuark type,
                                         gpointer data) {
    GstSavantBatchMeta *dmeta, *smeta;
    smeta = (GstSavantBatchMeta *)meta;
    GST_DEBUG("Transform GstSavantBatchMeta with IDX %d from buffer %p to buffer %p",
              smeta->idx, buffer, dest);
    dmeta = (GstSavantBatchMeta *)gst_buffer_add_savant_batch_meta(dest,
                                                                   smeta->idx);
    if (!dmeta)
        return FALSE;
    return TRUE;
}

GType gst_savant_batch_meta_api_get_type(void) {
    GST_TRACE("Get GstSavantBatchMeta API type");
    static GType type = 0;
    static const gchar *tags[] = {NULL};

    if (g_once_init_enter(&type)) {
        GType _type = gst_meta_api_type_register("GstSavantBatchMetaAPI", tags);
        g_once_init_leave(&type, _type);
    }
    return type;
}

const GstMetaInfo *gst_savant_batch_meta_get_info(void) {
    GST_TRACE("Get GstSavantBatchMeta info");
    static const GstMetaInfo *savant_batch_meta_info = NULL;

    if (g_once_init_enter((GstMetaInfo **)&savant_batch_meta_info)) {
        const GstMetaInfo *meta = gst_meta_register(
            GST_SAVANT_BATCH_META_API_TYPE, "GstSavantBatchMeta",
            sizeof(GstSavantBatchMeta),
            (GstMetaInitFunction)gst_savant_batch_meta_init,
            (GstMetaFreeFunction)NULL, gst_savant_batch_meta_transform);
        g_once_init_leave((GstMetaInfo **)&savant_batch_meta_info,
                          (GstMetaInfo *)meta);
    }
    return savant_batch_meta_info;
}

GstSavantBatchMeta *gst_buffer_get_savant_batch_meta(GstBuffer *buffer) {
    GST_DEBUG("Get GstSavantBatchMeta for buffer %p", buffer);
    GstSavantBatchMeta *meta = (GstSavantBatchMeta *)gst_buffer_get_meta(
        buffer, GST_SAVANT_BATCH_META_API_TYPE);
    if (meta) {
        GST_DEBUG("GstSavantBatchMeta with IDX %d found for buffer %p",
                  meta->idx, buffer);
    } else {
        GST_DEBUG("GstSavantBatchMeta not found for buffer %p", buffer);
    }

    return meta;
}

GstSavantBatchMeta *gst_buffer_add_savant_batch_meta(GstBuffer *buffer,
                                                     guint32 idx) {
    GST_DEBUG("Adding GstSavantBatchMeta with IDX %d to buffer %p", idx, buffer);
    GstSavantBatchMeta *meta;
    if (!gst_buffer_is_writable(buffer)) {
        GST_WARNING("Failed to add GstSavantBatchMeta with IDX %d to buffer %p: "
                    "buffer is not writable",
                    idx, buffer);
        return NULL;
    }
    meta = (GstSavantBatchMeta *)gst_buffer_add_meta(
        buffer, gst_savant_batch_meta_get_info(), NULL);
    if (!meta) {
        GST_WARNING("Failed to add GstSavantBatchMeta with IDX %d to buffer %p",
                    idx, buffer);
        return NULL;
    }
    meta->idx = idx;
    GST_INFO("Added GstSavantBatchMeta with IDX %d to buffer %p", idx, buffer);

    return meta;
}
