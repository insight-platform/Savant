#pragma once

#include "video_frame.h"


class FrameBuffer {
public:
    FrameBuffer();

    void put_frame(VideoFrame *frame, bool keyframe);

    void drop_frames(uint64_t fragment_ts);

    std::vector<VideoFrame *> get_all_frames();

    void clear();

private:
    std::map<uint64_t, std::vector<VideoFrame *>> frames;
    uint64_t last_fragment_ts;
};
