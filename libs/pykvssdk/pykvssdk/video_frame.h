#pragma once

#include <map>
#include <string>
#include "KinesisVideoStream.h"


namespace video = com::amazonaws::kinesis::video;

typedef struct _VideoFrame {
    video::KinesisVideoFrame *frame;
    std::string &uuid;

    ~_VideoFrame() {
        delete[] frame->frameData;
        delete frame;
    }
} VideoFrame;
