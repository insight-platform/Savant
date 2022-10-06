#ifndef _GST_SAVANT_FRAME_META_
#define _GST_SAVANT_FRAME_META_

#include <glib.h>
#include <gst/gst.h>

#include "nvdsmeta.h"

#define GST_SAVANT_FRAME_META_API_TYPE (gst_savant_frame_meta_api_get_type())
#define GST_NVDS_SAVANT_FRAME_META                                             \
    (nvds_get_user_meta_type((gchar *)"SAVANT.FRAME"))

/**
 * Contains frame related metadata.
 */
typedef struct _GstSavantFrameMeta_ {
    GstMeta meta; /** Allocation for base GstMeta structure. */
    guint32 idx;  /** Frame IDX. */
} GstSavantFrameMeta;

/* GstMeta functions */

/**
 * Get savant frame metadata from GStreamer buffer.
 *
 * @param buffer GStreamer buffer.
 * @return Pointer to metadata, or NULL if the buffer doesn't have meta.
 */
GstSavantFrameMeta *gst_buffer_get_savant_frame_meta(GstBuffer *buffer);

/**
 * Add savant frame metadata to GStreamer buffer as GstMeta.
 *
 * @param buffer GStreamer buffer.
 * @param idx Frame IDX.
 * @return Pointer to added metadata, or NULL if failed to add meta.
 */
GstSavantFrameMeta *gst_buffer_add_savant_frame_meta(GstBuffer *buffer,
                                                     guint32 idx);

/* NvDsMeta functions */

/**
 * Add savant frame metadata to GStreamer buffer as NvDsMeta.
 * Needed to pass metadata through deepstream elements.
 *
 * See:
 * https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_metadata.html#adding-metadata-to-the-plugin-before-gst-nvstreammux
 *
 * @param buffer GStreamer buffer.
 * @param idx Frame IDX.
 * @return Pointer to added metadata, or NULL if failed to add meta.
 */
GstSavantFrameMeta *gst_buffer_add_nvds_savant_frame_meta(GstBuffer *buffer,
                                                          guint32 idx);

/**
 * Get savant frame metadata from NvDsMeta of GStreamer buffer.
 *
 * @param buffer GStreamer buffer.
 * @return Pointer to metadata, or NULL if the buffer doesn't have meta.
 */
GstSavantFrameMeta *gst_buffer_get_nvds_savant_frame_meta(GstBuffer *buffer);

/**
 * Get savant frame metadata from NvDs frame metadata.
 *
 * @param frame_meta NvDs frame metadata.
 * @return Pointer to metadata, or NULL if the buffer doesn't have meta.
 */
GstSavantFrameMeta *
nvds_frame_meta_get_nvds_savant_frame_meta(NvDsFrameMeta *frame_meta);

/* Gst-NvDs conversion */

/**
 * Convert savant frame metadata from GstMeta to NvDsMeta.
 *
 * @param buffer GStreamer buffer.
 * @return Pointer to metadata, or NULL if the buffer doesn't have meta.
 */
GstSavantFrameMeta *gst_savant_frame_meta_to_nvds(GstBuffer *buffer);

/**
 * Convert savant frame metadata from NvDsMeta to GstMeta.
 *
 * @param buffer GStreamer buffer.
 * @return Pointer to metadata, or NULL if the buffer doesn't have meta.
 */
GstSavantFrameMeta *nvds_savant_frame_meta_to_gst(GstBuffer *buffer);

/**
 * Pad probe to convert savant frame metadata from NvDsMeta to GstMeta
 * or vice versa.
 *
 * @param pad GStreamer pad.
 * @param info Pad probe info.
 * @param user_data Conversion method, either gst_savant_frame_meta_to_nvds
 *                  or nvds_savant_frame_meta_to_gst
 */
GstPadProbeReturn convert_savant_frame_meta_pad_probe(GstPad *pad,
                                                      GstPadProbeInfo *info,
                                                      gpointer user_data);

#endif /* _GST_SAVANT_FRAME_META_ */
