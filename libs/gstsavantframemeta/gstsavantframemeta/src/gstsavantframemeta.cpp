#include "gstsavantframemeta.h"

#include <glib.h>
#include <gst/gst.h>

#include "gstnvdsmeta.h"
#include "nvdsmeta.h"

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

/* NvDsMeta functions */

gpointer nvds_savant_frame_meta_copy_func(gpointer data, gpointer user_data) {
    GstSavantFrameMeta *src_meta = (GstSavantFrameMeta *)data;
    GST_LOG("Copying GstSavantFrameMeta with IDX %d", src_meta->idx);
    GstSavantFrameMeta *dst_meta =
        (GstSavantFrameMeta *)g_malloc0(sizeof(GstSavantFrameMeta));
    memcpy(dst_meta, src_meta, sizeof(GstSavantFrameMeta));
    return (gpointer)dst_meta;
}

void nvds_savant_frame_meta_release_func(gpointer data, gpointer user_data) {
    GstSavantFrameMeta *meta = (GstSavantFrameMeta *)data;
    if (meta) {
        GST_LOG("Releasing GstSavantFrameMeta with IDX %d", meta->idx);
        g_free(meta);
        meta = NULL;
    }
}

gpointer nvds_savant_frame_meta_transform_func(gpointer data,
                                               gpointer user_data) {
    NvDsUserMeta *user_meta = (NvDsUserMeta *)data;
    GstSavantFrameMeta *src_meta =
        (GstSavantFrameMeta *)user_meta->user_meta_data;
    GST_DEBUG("Transforming GstSavantFrameMeta with IDX %d", src_meta->idx);
    GstSavantFrameMeta *dst_meta =
        (GstSavantFrameMeta *)nvds_savant_frame_meta_copy_func(src_meta, NULL);
    return (gpointer)dst_meta;
}

void nvds_user_meta_savant_frame_meta_release_func(gpointer data,
                                                   gpointer user_data) {
    NvDsUserMeta *user_meta = (NvDsUserMeta *)data;
    GstSavantFrameMeta *meta = (GstSavantFrameMeta *)user_meta->user_meta_data;
    GST_LOG("Releasing GstSavantFrameMeta with IDX %d from NvDsUserMeta",
            meta->idx);
    // nvds_savant_frame_meta_release_func(meta, NULL);
}

static GstSavantFrameMeta *new_savant_frame_meta(guint32 frame_idx) {
    GstSavantFrameMeta *savant_frame_meta =
        (GstSavantFrameMeta *)g_malloc0(sizeof(GstSavantFrameMeta));
    if (savant_frame_meta != NULL) {
        savant_frame_meta->idx = frame_idx;
    }
    return savant_frame_meta;
}

GstSavantFrameMeta *gst_buffer_add_nvds_savant_frame_meta(GstBuffer *buffer,
                                                          guint32 idx) {
    GST_DEBUG("Adding GstSavantFrameMeta with IDX %d to buffer %p", idx, buffer);
    NvDsMeta *meta = NULL;
    GstSavantFrameMeta *savant_frame_meta =
        (GstSavantFrameMeta *)g_malloc0(sizeof(GstSavantFrameMeta));
    if (!savant_frame_meta) {
        GST_ERROR("Failed to allocate GstSavantFrameMeta for buffer %p", buffer);
        // TODO: stop pipeline?
        return NULL;
    }
    savant_frame_meta->idx = idx;

    meta = gst_buffer_add_nvds_meta(buffer, savant_frame_meta, NULL,
                                    nvds_savant_frame_meta_copy_func,
                                    nvds_savant_frame_meta_release_func);
    if (!meta) {
        GST_ERROR("Failed to add NvDsMeta with GstSavantFrameMeta with IDX %d "
                  "to buffer %p",
                  idx, buffer);
        // TODO: stop pipeline?
        return NULL;
    }

    meta->meta_type = (GstNvDsMetaType)GST_NVDS_SAVANT_FRAME_META;

    meta->gst_to_nvds_meta_transform_func =
        nvds_savant_frame_meta_transform_func;

    meta->gst_to_nvds_meta_release_func =
        nvds_user_meta_savant_frame_meta_release_func;

    return savant_frame_meta;
}

GstSavantFrameMeta *gst_buffer_get_nvds_savant_frame_meta(GstBuffer *buffer) {
    NvDsMeta *meta = gst_buffer_get_nvds_meta(buffer);
    if (meta->meta_type == GST_NVDS_SAVANT_FRAME_META) {
        GstSavantFrameMeta *savant_frame_meta =
            (GstSavantFrameMeta *)meta->meta_data;
        return savant_frame_meta;
    }
    return NULL;
}

GstSavantFrameMeta *
nvds_frame_meta_get_nvds_savant_frame_meta(NvDsFrameMeta *frame_meta) {
    GST_DEBUG("Get GstSavantFrameMeta from NvDsFrameMeta %p", frame_meta);
    NvDsMetaList *l_user_meta = NULL;
    NvDsUserMeta *user_meta = NULL;
    GstSavantFrameMeta *savant_frame_meta = NULL;
    for (l_user_meta = frame_meta->frame_user_meta_list; l_user_meta != NULL;
         l_user_meta = l_user_meta->next) {
        user_meta = (NvDsUserMeta *)(l_user_meta->data);
        if (user_meta->base_meta.meta_type == GST_NVDS_SAVANT_FRAME_META) {
            savant_frame_meta = (GstSavantFrameMeta *)user_meta->user_meta_data;
            GST_DEBUG("Got GstSavantFrameMeta %p from NvDsFrameMeta %p",
                      frame_meta, savant_frame_meta);
            return savant_frame_meta;
        }
    }
    GST_INFO("GstSavantFrameMeta not found in NvDsFrameMeta %p", frame_meta);
    return NULL;
}

GstSavantFrameMeta *gst_savant_frame_meta_to_nvds(GstBuffer *buffer) {
    GST_DEBUG("Convert SavantFrameMeta from Gst to NvDs for buffer %p", buffer);
    GstSavantFrameMeta *s_meta, *d_meta;
    s_meta = gst_buffer_get_savant_frame_meta(buffer);
    if (!s_meta) {
        GST_INFO("Buffer %p has no SavantFrameMeta", buffer);
        return NULL;
    }
    d_meta = gst_buffer_add_nvds_savant_frame_meta(buffer, s_meta->idx);
    return d_meta;
}

GstSavantFrameMeta *nvds_savant_frame_meta_to_gst(GstBuffer *buffer) {
    GST_DEBUG("Convert SavantFrameMeta from NvDs to Gst for buffer %p", buffer);
    NvDsMetaList *l_frame;
    NvDsFrameMeta *frame_meta;
    GstSavantFrameMeta *s_meta, *d_meta;
    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buffer);
    if (!batch_meta) {
        GST_INFO("Buffer %p has no NvDsBatchMeta", buffer);
        return NULL;
    }
    GST_DEBUG("Buffer %p with NvDsBatchMeta %p has %d frames",
              buffer, batch_meta, batch_meta->num_frames_in_batch);
    if (batch_meta->num_frames_in_batch != 1) {
        GST_WARNING("Buffer %p with NvDsBatchMeta has %d frames",
                    buffer, batch_meta->num_frames_in_batch);
        return NULL;
    }
    l_frame = batch_meta->frame_meta_list;
    frame_meta = (NvDsFrameMeta *)l_frame->data;
    s_meta = nvds_frame_meta_get_nvds_savant_frame_meta(frame_meta);
    if (!s_meta) {
        GST_INFO("Buffer %p has no SavantFrameMeta in NvDsBatchMeta", buffer);
        return NULL;
    }
    // Copying s_meta since it will be released when NvDsBatchMeta is released
    d_meta = gst_buffer_add_savant_frame_meta(buffer, s_meta->idx);
    return d_meta;
}

typedef GstSavantFrameMeta *(*GstSavantMetaConvertFunction)(GstBuffer *buffer);

GstPadProbeReturn convert_savant_frame_meta_pad_probe(GstPad *pad,
                                                      GstPadProbeInfo *info,
                                                      gpointer user_data) {
    GST_DEBUG_OBJECT(pad, "Convert SavantFrameMeta");
    GstSavantMetaConvertFunction conv_func =
        (GstSavantMetaConvertFunction)user_data;
    GstBuffer *buffer = GST_PAD_PROBE_INFO_BUFFER(info);
    if (!buffer) {
        GST_WARNING_OBJECT(pad, "Buffer is NULL");
        return GST_PAD_PROBE_OK;
    }
    if (!gst_buffer_is_writable(buffer)) {
        GST_INFO_OBJECT(pad, "Buffer is not writable");
        buffer = gst_buffer_make_writable(buffer);
    }
    conv_func(buffer);
    GST_PAD_PROBE_INFO_DATA(info) = buffer;

    return GST_PAD_PROBE_OK;
}
