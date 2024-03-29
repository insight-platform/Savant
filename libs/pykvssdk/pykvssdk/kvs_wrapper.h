#pragma once

#include "KinesisVideoProducer.h"
#include "KinesisVideoStream.h"
#include "producer_state.h"
#include "client_callback_provider.h"
#include "stream_callback_provider.h"
#include "frame_buffer.h"


namespace video = com::amazonaws::kinesis::video;


class KvsWrapper {

public:
    KvsWrapper(
        const std::string &region,
        const std::string &access_key,
        const std::string &secret_key,
        const std::string &stream_name,
        const std::string &content_type,
        bool allow_stream_creation,
        uint32_t framerate,
        uint32_t low_threshold,
        uint32_t high_threshold
    );

    bool start(const char *codec_private_data, size_t codec_private_data_size);

    void put_frame(
        char *frame_data,
        uint32_t data_size,
        uint32_t idx,
        uint64_t pts,
        uint64_t dts,
        uint64_t duration,
        bool keyframe
    );

    bool stop_sync();

private:
    std::unique_ptr<video::KinesisVideoProducer> producer;
    std::shared_ptr<video::KinesisVideoStream> stream;
    ProducerState *state;
    FrameBuffer *frame_buffer;
    uint32_t low_threshold;
    uint64_t low_threshold_100ns;
    uint32_t high_threshold;
    uint64_t high_threshold_100ns;
    uint64_t max_fragment_duration;
    uint64_t current_fragment_start_ts = 0;

    bool push_frame(video::KinesisVideoFrame *frame);

    bool check_connection();
};
