#include "savantrsprobes.h"
#include "gstsavantbatchmeta.h"
#include "gstsavantframemeta.h"
#include "nvdssavantframemeta.h"

#include <glib.h>
#include <gst/gst.h>

#include "gstnvdsmeta.h"
#include "nvdsmeta.h"


typedef struct Arc_Vec_BorrowedVideoObject {
} Arc_Vec_BorrowedVideoObject;

extern "C" {
#include "savant_rs.h"
}

// SavantRsPadProbeData

SavantRsPadProbeData *
create_savant_rs_pad_probe_data(uintptr_t handle, const char *stage) {
    auto *data = (SavantRsPadProbeData *)malloc(sizeof(SavantRsPadProbeData));
    data->handle = handle;
    data->stage = (char *)malloc(strlen(stage) + 1);
    strcpy(data->stage, stage);

    GST_LOG("Pipeline stage %s. Created SavantRsPadProbeData", data->stage);

    return data;
}

void release_savant_rs_pad_probe_data(SavantRsPadProbeData *data) {
    GST_LOG("Pipeline stage %s. Releasing SavantRsPadProbeData", data->stage);
    free(data->stage);
    free(data);
}

// Gst Pad Probes

GstPadProbeReturn move_frame_as_is_pad_probe(GstPad *pad,
                                             GstPadProbeInfo *info,
                                             gpointer user_data) {
    GstSavantFrameMeta *savant_frame_meta;
    int64_t *idx_list;
    auto *data = (SavantRsPadProbeData *)user_data;
    GstBuffer *buffer = GST_PAD_PROBE_INFO_BUFFER(info);
    if (!buffer) {
        GST_INFO_OBJECT(pad, "Buffer is NULL");
        return GST_PAD_PROBE_PASS;
    }

    GST_INFO_OBJECT(
        pad,
        "Pipeline stage %s. Moving frame at buffer %ld.",
        data->stage, buffer->pts);
    savant_frame_meta = gst_buffer_get_savant_frame_meta(buffer);
    if (!savant_frame_meta) {
        GST_WARNING_OBJECT(
            pad,
            "Pipeline stage %s. Failed to move frame at buffer %ld. "
            "Frame has no GstSavantFrameMeta.",
            data->stage, buffer->pts);
        return GST_PAD_PROBE_PASS;
    }

    idx_list = (int64_t *)malloc(sizeof(int64_t));
    idx_list[0] = savant_frame_meta->idx;
    pipeline2_move_as_is(data->handle, data->stage, idx_list, 1);
    GST_INFO_OBJECT(
        pad,
        "Pipeline stage %s. Frame %d at buffer %ld moved.",
        data->stage, savant_frame_meta->idx, buffer->pts);
    free(idx_list);

    return GST_PAD_PROBE_OK;
}

GstPadProbeReturn move_and_pack_frames_pad_probe(GstPad *pad,
                                                 GstPadProbeInfo *info,
                                                 gpointer user_data) {
    GstSavantFrameMeta *savant_frame_meta;
    int64_t *idx_list;
    int64_t batch_id;
    NvDsBatchMeta *batch_meta;
    NvDsMetaList *l_frame;
    NvDsFrameMeta *frame_meta;
    auto *data = (SavantRsPadProbeData *)user_data;
    GstBuffer *buffer = GST_PAD_PROBE_INFO_BUFFER(info);
    if (!buffer) {
        GST_INFO_OBJECT(pad, "Buffer is NULL");
        return GST_PAD_PROBE_PASS;
    }

    GST_INFO_OBJECT(
        pad,
        "Pipeline stage %s. Packing frames at buffer %ld to a batch.",
        data->stage, buffer->pts);
    if (!gst_buffer_is_writable(buffer)) {
        GST_INFO_OBJECT(
            pad,
            "Pipeline stage %s. Buffer %ld is not writable. Making it writable.",
            data->stage, buffer->pts);
        buffer = gst_buffer_make_writable(buffer);
    }

    batch_meta = gst_buffer_get_nvds_batch_meta(buffer);
    if (batch_meta->num_frames_in_batch == 0) {
        GST_INFO_OBJECT(
            pad,
            "Pipeline stage %s. Skipping buffer %ld: batch is empty.",
            data->stage, buffer->pts);
        return GST_PAD_PROBE_PASS;
    }

    idx_list = (int64_t *)malloc(
        sizeof(int64_t) * batch_meta->num_frames_in_batch);
    l_frame = batch_meta->frame_meta_list;
    for (int i = 0; i < batch_meta->num_frames_in_batch; i++) {
        frame_meta = (NvDsFrameMeta *)l_frame->data;
        savant_frame_meta = nvds_frame_meta_get_nvds_savant_frame_meta(
            frame_meta);
        if (!savant_frame_meta) {
            GST_WARNING_OBJECT(
                pad,
                "Pipeline stage %s. Failed to pack frames at buffer %ld. "
                "Frame %ld has no GstSavantFrameMeta.",
                data->stage, buffer->pts, frame_meta->buf_pts);
            return GST_PAD_PROBE_PASS;
        }
        idx_list[i] = savant_frame_meta->idx;
        l_frame = l_frame->next;
    }

    batch_id = pipeline2_move_and_pack_frames(
        data->handle, data->stage, idx_list, batch_meta->num_frames_in_batch);

    gst_buffer_add_savant_batch_meta(buffer, batch_id);
    GST_INFO_OBJECT(
        pad,
        "Pipeline stage %s. %d frames at buffer %ld are packed to batch %ld.",
        data->stage, batch_meta->num_frames_in_batch, buffer->pts, batch_id);
    free(idx_list);

    return GST_PAD_PROBE_OK;
}

GstPadProbeReturn move_batch_as_is_pad_probe(GstPad *pad,
                                             GstPadProbeInfo *info,
                                             gpointer user_data) {
    GstSavantBatchMeta *savant_batch_meta;
    int64_t *idx_list;
    NvDsBatchMeta *batch_meta;
    auto *data = (SavantRsPadProbeData *)user_data;
    GstBuffer *buffer = GST_PAD_PROBE_INFO_BUFFER(info);
    if (!buffer) {
        GST_INFO_OBJECT(pad, "Buffer is NULL");
        return GST_PAD_PROBE_PASS;
    }

    GST_INFO_OBJECT(
        pad,
        "Pipeline stage %s. Moving batch at buffer %ld.",
        data->stage, buffer->pts);

    batch_meta = gst_buffer_get_nvds_batch_meta(buffer);
    if (batch_meta->num_frames_in_batch == 0) {
        GST_INFO_OBJECT(
            pad,
            "Pipeline stage %s. Skipping buffer %ld: batch is empty.",
            data->stage, buffer->pts);
        return GST_PAD_PROBE_PASS;
    }

    savant_batch_meta = gst_buffer_get_savant_batch_meta(buffer);
    if (!savant_batch_meta) {
        GST_WARNING_OBJECT(
            pad,
            "Pipeline stage %s. Failed to move batch at buffer %ld. "
            "Batch has no GstSavantFrameMeta.",
            data->stage, buffer->pts);
        return GST_PAD_PROBE_PASS;
    }

    idx_list = (int64_t *)malloc(sizeof(int64_t));
    idx_list[0] = savant_batch_meta->idx;
    pipeline2_move_as_is(data->handle, data->stage, idx_list, 1);
    GST_INFO_OBJECT(
        pad,
        "Pipeline stage %s. Batch %d at buffer %ld moved.",
        data->stage, savant_batch_meta->idx, buffer->pts);
    free(idx_list);

    return GST_PAD_PROBE_OK;
}

GstPadProbeReturn move_and_unpack_batch_pad_probe(GstPad *pad,
                                                  GstPadProbeInfo *info,
                                                  gpointer user_data) {
    GstSavantBatchMeta *savant_batch_meta;
    int64_t *resulting_ids;
    NvDsBatchMeta *batch_meta;
    auto *data = (SavantRsPadProbeData *)user_data;
    GstBuffer *buffer = GST_PAD_PROBE_INFO_BUFFER(info);
    if (!buffer) {
        GST_INFO_OBJECT(pad, "Buffer is NULL");
        return GST_PAD_PROBE_PASS;
    }

    GST_INFO_OBJECT(
        pad,
        "Pipeline stage %s. Unpacking batch at buffer %ld to frames.",
        data->stage, buffer->pts);
    if (!gst_buffer_is_writable(buffer)) {
        GST_INFO_OBJECT(
            pad,
            "Pipeline stage %s. Buffer %ld is not writable. Making it writable.",
            data->stage, buffer->pts);
        buffer = gst_buffer_make_writable(buffer);
    }

    batch_meta = gst_buffer_get_nvds_batch_meta(buffer);
    if (batch_meta->num_frames_in_batch == 0) {
        GST_INFO_OBJECT(
            pad,
            "Pipeline stage %s. Skipping buffer %ld: batch is empty.",
            data->stage, buffer->pts);
        return GST_PAD_PROBE_PASS;
    }

    savant_batch_meta = gst_buffer_get_savant_batch_meta(buffer);
    if (!savant_batch_meta) {
        GST_WARNING_OBJECT(
            pad,
            "Pipeline stage %s. Failed to unpack batch at buffer %ld. "
            "Batch has no GstSavantFrameMeta.",
            data->stage, buffer->pts);
        return GST_PAD_PROBE_PASS;
    }

    resulting_ids = (int64_t *)malloc(
        sizeof(int64_t) * batch_meta->num_frames_in_batch);
    pipeline2_move_and_unpack_batch(
        data->handle,
        data->stage,
        savant_batch_meta->idx,
        resulting_ids,
        batch_meta->num_frames_in_batch);

    GST_INFO_OBJECT(
        pad,
        "Pipeline stage %s. Batch %d at buffer %ld unpacked %d frames.",
        data->stage, savant_batch_meta->idx, buffer->pts,
        batch_meta->num_frames_in_batch);
    free(resulting_ids);

    return GST_PAD_PROBE_OK;
}
