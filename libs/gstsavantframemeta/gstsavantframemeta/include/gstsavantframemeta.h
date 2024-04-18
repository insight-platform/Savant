#ifndef _GST_SAVANT_FRAME_META_
#define _GST_SAVANT_FRAME_META_

#include <glib.h>
#include <gst/gst.h>

#define GST_SAVANT_FRAME_META_API_TYPE (gst_savant_frame_meta_api_get_type())

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

#endif /* _GST_SAVANT_FRAME_META_ */
