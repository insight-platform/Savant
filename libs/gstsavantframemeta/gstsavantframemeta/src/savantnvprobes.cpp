#include "savantnvprobes.h"

#include <glib.h>
#include <gst/gst.h>
#include <vector>

#include "gstnvdsmeta.h"
#include "nvdsmeta.h"

GstPadProbeReturn remove_tracker_objs_pad_probe(GstPad *pad,
                                                GstPadProbeInfo *info,
                                                gpointer user_data) {
    GstBuffer *buffer = GST_PAD_PROBE_INFO_BUFFER(info);
    if (!buffer) {
        GST_INFO_OBJECT(pad, "Buffer is NULL");
        return GST_PAD_PROBE_PASS;
    }

    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buffer);
    if (batch_meta->num_frames_in_batch == 0) {
        GST_INFO_OBJECT(
            pad,
            "Tracker obj remover. Skipping buffer %ld: batch is empty.",
            buffer->pts);
        return GST_PAD_PROBE_PASS;
    }

    NvDsFrameMetaList *l_frame = NULL;
    for (l_frame = batch_meta->frame_meta_list; l_frame != NULL; l_frame = l_frame->next) {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) (l_frame->data);
        NvDsObjectMetaList *l_obj = NULL;
        std::vector<NvDsObjectMeta*> removalList;
        for (l_obj = frame_meta->obj_meta_list; l_obj != NULL; l_obj = l_obj->next) {
            NvDsObjectMeta *obj_meta = (NvDsObjectMeta *) (l_obj->data);
            float det_l = obj_meta->detector_bbox_info.org_bbox_coords.left;
            float det_t = obj_meta->detector_bbox_info.org_bbox_coords.top;
            float det_w = obj_meta->detector_bbox_info.org_bbox_coords.width;
            float det_h = obj_meta->detector_bbox_info.org_bbox_coords.height;
            float det_c = (float)obj_meta->confidence;

            if (det_l == 0.0f && det_t == 0.0f && det_w == 0.0f && det_h == 0.0f && det_c == -0.1f) {
                GST_WARNING("Found obj, detector box [%f %f %f %f]/%f, tracker [%f %f %f %f]/%f", det_l, det_t, det_w, det_h, det_c, tr_l, tr_t, tr_w, tr_h, tr_c);
                removalList.push_back(obj_meta);
            }
        }

        for (uint32_t i = 0; i < removalList.size(); i++) {
            nvds_remove_obj_meta_from_frame(frame_meta, removalList[i]);
        }
    }

    return GST_PAD_PROBE_OK;
}
