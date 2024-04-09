#include "KinesisVideoStream.h"
#include "consts.h"
#include "logger.h"
#include "frame_buffer.h"


namespace video = com::amazonaws::kinesis::video;


FrameBuffer::FrameBuffer() {
    frames = std::map<uint64_t, std::vector<video::KinesisVideoFrame *>>();
    last_fragment_ts = 0;
}


void FrameBuffer::put_frame(video::KinesisVideoFrame *frame, bool keyframe) {
    uint64_t fragment_ts;
    if (keyframe) {
        // 100ns -> ms
        fragment_ts = frame->presentationTs / TIME_100NS_TO_1MS;
        last_fragment_ts = fragment_ts;
    } else {
        fragment_ts = last_fragment_ts;
    }
    LOG_PYBIND_DEBUG(
        "Put frame %lu to buffer (fragment %lu).",
        frame->presentationTs, fragment_ts);
    if (frames.find(fragment_ts) == frames.end()) {
        frames[fragment_ts] = std::vector<video::KinesisVideoFrame *>();
    }
    frames[fragment_ts].push_back(frame);
}


void FrameBuffer::drop_frames(uint64_t fragment_ts) {
    LOG_PYBIND_DEBUG("Dropping frames of fragment %lu.", fragment_ts);
    if (frames.find(fragment_ts) != frames.end()) {
        for (auto frame: frames[fragment_ts]) {
            LOG_PYBIND_TRACE(
                "Drop frame %lu from buffer (fragment %lu).",
                frame->presentationTs, fragment_ts);
            delete[] frame->frameData;
            delete frame;
        }
        frames.erase(fragment_ts);
    }
}


std::vector<video::KinesisVideoFrame *> FrameBuffer::get_all_frames() {
    std::vector<video::KinesisVideoFrame *> all_frames;
    for (auto &kv: frames) {
        for (auto frame: kv.second) {
            all_frames.push_back(frame);
        }
    }
    return all_frames;
}


void FrameBuffer::clear() {
    LOG_PYBIND_DEBUG("Clearing frame buffer.");
    for (auto &kv: frames) {
        for (auto frame: kv.second) {
            delete[] frame->frameData;
            delete frame;
        }
    }
    frames.clear();
    LOG_PYBIND_DEBUG("Frame buffer cleared.");
}
