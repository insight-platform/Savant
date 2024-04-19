#include "gstsavantframemeta.h"

#include <glib.h>
#include <gst/gst.h>

/* GstMeta functions */

gboolean gst_savant_frame_meta_init(GstMeta *meta, gpointer params,
                                    GstBuffer *buffer) {
    GST_LOG("Initialize GstSavantFrameMeta for buffer %p", buffer);
    GstSavantFrameMeta *emeta = (GstSavantFrameMeta *)meta;
    emeta->idx = 0;
    return TRUE;
}

gboolean gst_savant_frame_meta_transform(GstBuffer *dest, GstMeta *meta,
                                         GstBuffer *buffer, GQuark type,
                                         gpointer data) {
    GstSavantFrameMeta *dmeta, *smeta;
    smeta = (GstSavantFrameMeta *)meta;
    GST_DEBUG("Transform GstSavantFrameMeta with IDX %d from buffer %p to buffer %p",
              smeta->idx, buffer, dest);
    dmeta = (GstSavantFrameMeta *)gst_buffer_add_savant_frame_meta(dest, smeta->idx);
    if (!dmeta)
        return FALSE;
    return TRUE;
}

GType gst_savant_frame_meta_api_get_type(void) {
    GST_TRACE("Get GstSavantFrameMeta API type");
    static GType type = 0;
    static const gchar *tags[] = {NULL};

    if (g_once_init_enter(&type)) {
        GType _type = gst_meta_api_type_register("GstSavantFrameMetaAPI", tags);
        g_once_init_leave(&type, _type);
    }
    return type;
}

const GstMetaInfo *gst_savant_frame_meta_get_info(void) {
    GST_TRACE("Get GstSavantFrameMeta info");
    static const GstMetaInfo *savant_frame_meta_info = NULL;

    if (g_once_init_enter((GstMetaInfo **)&savant_frame_meta_info)) {
        const GstMetaInfo *meta = gst_meta_register(
            GST_SAVANT_FRAME_META_API_TYPE, "GstSavantFrameMeta",
            sizeof(GstSavantFrameMeta),
            (GstMetaInitFunction)gst_savant_frame_meta_init,
            (GstMetaFreeFunction)NULL, gst_savant_frame_meta_transform);
        g_once_init_leave((GstMetaInfo **)&savant_frame_meta_info,
                          (GstMetaInfo *)meta);
    }
    return savant_frame_meta_info;
}

GstSavantFrameMeta *gst_buffer_get_savant_frame_meta(GstBuffer *buffer) {
    GST_DEBUG("Get GstSavantFrameMeta for buffer %p", buffer);
    GstSavantFrameMeta *meta = (GstSavantFrameMeta *)gst_buffer_get_meta(
        buffer, GST_SAVANT_FRAME_META_API_TYPE);
    if (meta) {
        GST_DEBUG("GstSavantFrameMeta with IDX %d found for buffer %p",
                  meta->idx, buffer);
    } else {
        GST_DEBUG("GstSavantFrameMeta not found for buffer %p", buffer);
    }

    return meta;
}

GstSavantFrameMeta *gst_buffer_add_savant_frame_meta(GstBuffer *buffer,
                                                     guint32 idx) {
    GST_DEBUG("Adding GstSavantFrameMeta with IDX %d to buffer %p", idx, buffer);
    GstSavantFrameMeta *meta;
    if (!gst_buffer_is_writable(buffer)) {
        GST_WARNING("Failed to add GstSavantFrameMeta with IDX %d to buffer %p: "
                    "buffer is not writable",
                    idx, buffer);
        return NULL;
    }
    meta = (GstSavantFrameMeta *)gst_buffer_add_meta(
        buffer, gst_savant_frame_meta_get_info(), NULL);
    if (!meta) {
        GST_WARNING("Failed to add GstSavantFrameMeta with IDX %d to buffer %p",
                    idx, buffer);
        return NULL;
    }
    meta->idx = idx;
    GST_INFO("Added GstSavantFrameMeta with IDX %d to buffer %p", idx, buffer);

    return meta;
}


static GstSavantFrameMeta *new_savant_frame_meta(guint32 frame_idx) {
    GstSavantFrameMeta *savant_frame_meta =
        (GstSavantFrameMeta *)g_malloc0(sizeof(GstSavantFrameMeta));
    if (savant_frame_meta != NULL) {
        savant_frame_meta->idx = frame_idx;
    }
    return savant_frame_meta;
}
