#include <vector>
#include <cmath>
#include "cuda/nms_iou.h"
#include "types.h"
#include "nvdsparserapid.h"

int nms(
        void *raw_input_bboxes,
        float *raw_output_bboxes,
        int total_bboxes,
        float nms_thresh,
        float conf_thresh,
        int topk)
{
    int boxes_point = 6;
    auto input_boxes  = static_cast<float6_rapid_bbox *>((void *)raw_input_bboxes);
    auto output_bboxes  = static_cast<float6_rapid_bbox *>((void *)raw_output_bboxes);
    auto *bboxes = new float[boxes_point*total_bboxes];
    auto *scores = new float[total_bboxes];
    auto *classes = new float[total_bboxes];

    for (int i = 0; i < total_bboxes; i++)
    {
        float width=input_boxes[i].width;
        float height=input_boxes[i].height;
        bboxes[i*boxes_point] = input_boxes[i].xc - width / 2;
        bboxes[i*boxes_point+1] = input_boxes[i].yc - height / 2;
        bboxes[i*boxes_point+2] = input_boxes[i].xc + width / 2;
        bboxes[i*boxes_point+3] = input_boxes[i].yc + height / 2;
        bboxes[i*boxes_point+4] = (float) sin(input_boxes[i].angle / 180 * 3.14);
        bboxes[i*boxes_point+5] = (float) cos(input_boxes[i].angle / 180 * 3.14);
        scores[i] = input_boxes[i].conf;
        classes[i] = 0.;
    }


    // Perform NMS
    int num_detections_end = odtk::cuda::nms_rotate(scores, bboxes, classes, input_boxes, output_bboxes, total_bboxes,
        topk, nms_thresh, conf_thresh);
    delete[] bboxes;
    delete[] scores;
    delete[] classes;
    return num_detections_end;
}


