#include "consts.h"
#include "logger.h"
#include "video_frame.h"
#include "frame_buffer.h"


FrameBuffer::FrameBuffer() {
    frames = std::map<uint64_t, std::vector<VideoFrame *>>();
    last_fragment_ts = 0;
}


void FrameBuffer::put_frame(VideoFrame *frame, bool keyframe) {
    uint64_t fragment_ts;
    if (keyframe) {
        // 100ns -> ms
        fragment_ts = frame->frame->presentationTs / TIME_100NS_TO_1MS;
        last_fragment_ts = fragment_ts;
    } else {
        fragment_ts = last_fragment_ts;
    }
    LOG_PYBIND_DEBUG(
        "Put frame %lu to buffer (fragment %lu).",
        frame->frame->presentationTs, fragment_ts);
    if (frames.find(fragment_ts) == frames.end()) {
        frames[fragment_ts] = std::vector<VideoFrame *>();
    }
    frames[fragment_ts].push_back(frame);
}


void FrameBuffer::drop_frames(uint64_t fragment_ts) {
    LOG_PYBIND_DEBUG("Dropping frames of fragment %lu.", fragment_ts);
    if (frames.find(fragment_ts) != frames.end()) {
        for (auto frame: frames[fragment_ts]) {
            LOG_PYBIND_TRACE(
                "Drop frame %lu from buffer (fragment %lu).",
                frame->frame->presentationTs, fragment_ts);
            delete frame;
        }
        frames.erase(fragment_ts);
    }
}


std::vector<VideoFrame *> FrameBuffer::get_all_frames() {
    std::vector<VideoFrame *> all_frames;
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
            delete[] frame->frame->frameData;
            delete frame;
        }
    }
    frames.clear();
    LOG_PYBIND_DEBUG("Frame buffer cleared.");
}
