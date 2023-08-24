#ifndef _GST_SAVANT_BATCH_META_
#define _GST_SAVANT_BATCH_META_

#include <glib.h>
#include <gst/gst.h>

#define GST_SAVANT_BATCH_META_API_TYPE (gst_savant_batch_meta_api_get_type())

/**
 * Contains batch related metadata.
 */
typedef struct _GstSavantBatchMeta_ {
    GstMeta meta; /** Allocation for base GstMeta structure. */
    guint32 idx;  /** Batch IDX. */
} GstSavantBatchMeta;

/* GstMeta functions */

/**
 * Get savant batch metadata from GStreamer buffer.
 *
 * @param buffer GStreamer buffer.
 * @return Pointer to metadata, or NULL if the buffer doesn't have meta.
 */
GstSavantBatchMeta *gst_buffer_get_savant_batch_meta(GstBuffer *buffer);

/**
 * Add savant batch metadata to GStreamer buffer as GstMeta.
 *
 * @param buffer GStreamer buffer.
 * @param idx Batch IDX.
 * @return Pointer to added metadata, or NULL if failed to add meta.
 */
GstSavantBatchMeta *gst_buffer_add_savant_batch_meta(GstBuffer *buffer,
                                                     guint32 idx);


#endif /* _GST_SAVANT_BATCH_META_ */
