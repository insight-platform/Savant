#include "savantnvprobes.h"
#include <vector>
#include "gstnvdsmeta.h"


GstPadProbeReturn remove_tracker_objs_pad_probe(GstPad *pad,
                                                GstPadProbeInfo *info,
                                                gpointer user_data) {
    GstBuffer *buffer = GST_PAD_PROBE_INFO_BUFFER(info);
    if (!buffer) {
        GST_INFO_OBJECT(pad, "Tracker obj remover: skipping NULL buffer.");
        return GST_PAD_PROBE_PASS;
    }

    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buffer);
    if (batch_meta->num_frames_in_batch == 0) {
        GST_INFO_OBJECT(
            pad,
            "Tracker obj remover: skipping buffer %ld, batch is empty.",
            buffer->pts);
        return GST_PAD_PROBE_PASS;
    }

    NvDsFrameMetaList *l_frame = NULL;
    for (l_frame = batch_meta->frame_meta_list; l_frame != NULL; l_frame = l_frame->next) {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) (l_frame->data);
        std::vector<NvDsObjectMeta*> removal_list;
        NvDsObjectMetaList *l_obj = NULL;
        for (l_obj = frame_meta->obj_meta_list; l_obj != NULL; l_obj = l_obj->next) {
            NvDsObjectMeta *obj_meta = (NvDsObjectMeta *) (l_obj->data);

            if (obj_is_tracker_created(obj_meta)) {
                GST_WARNING_OBJECT(pad, "Tracker obj remover: found obj created by the tracker, marking for removal.");
                removal_list.push_back(obj_meta);
            }
        }

        for (auto obj : removal_list) {
            nvds_remove_obj_meta_from_frame(frame_meta, obj);
        }
    }

    return GST_PAD_PROBE_OK;
}


bool obj_is_tracker_created(NvDsObjectMeta *obj_meta) {
    float left = obj_meta->detector_bbox_info.org_bbox_coords.left;
    float top = obj_meta->detector_bbox_info.org_bbox_coords.top;
    float width = obj_meta->detector_bbox_info.org_bbox_coords.width;
    float height = obj_meta->detector_bbox_info.org_bbox_coords.height;
    float conf = (float)obj_meta->confidence;

    return left == 0.0f && top == 0.0f && width == 0.0f && height == 0.0f && conf == -0.1f;
}
