#ifndef savantboost_NVDSPARSERAPID_H
#define savantboost_NVDSPARSERAPID_H
int nms(void *raw_input_bboxes, float *output_bboxes,
        int total_bboxes, float nms_thresh, float conf_thresh, int topk);
#endif //savantboost_NVDSPARSERAPID_H
