#pragma once

#include "KinesisVideoStream.h"


namespace video = com::amazonaws::kinesis::video;


class FrameBuffer {
public:
    FrameBuffer();

    void put_frame(video::KinesisVideoFrame *frame, bool keyframe);

    void drop_frames(uint64_t fragment_ts);

    std::vector<video::KinesisVideoFrame *> get_all_frames();

    void clear();

private:
    std::map<uint64_t, std::vector<video::KinesisVideoFrame *>> frames;
    uint64_t last_fragment_ts;
};
