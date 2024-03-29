#include "KinesisVideoProducer.h"
#include "KinesisVideoStream.h"
#include "client_callback_provider.h"
#include "stream_callback_provider.h"
#include "frame_buffer.h"
#include "consts.h"
#include "logger.h"
#include "kvs_wrapper.h"


namespace video = com::amazonaws::kinesis::video;


KvsWrapper::KvsWrapper(
    const std::string &region,
    const std::string &access_key,
    const std::string &secret_key,
    const std::string &stream_name,
    const std::string &content_type,
    bool allow_stream_creation,
    uint32_t framerate,
    uint32_t low_threshold,
    uint32_t high_threshold
) {
    state = new ProducerState();
    frame_buffer = new FrameBuffer();
    this->low_threshold = low_threshold;
    this->low_threshold_100ns = ((uint64_t)low_threshold) * TIME_100NS_TO_1S;
    this->high_threshold = high_threshold;
    this->high_threshold_100ns = ((uint64_t)high_threshold) * TIME_100NS_TO_1S;
    max_fragment_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(
        DEFAULT_FRAGMENT_DURATION).count();

    std::unique_ptr<video::DeviceInfoProvider> device_info_provider(
        new video::DefaultDeviceInfoProvider());
    std::unique_ptr<video::ClientCallbackProvider> client_callback_provider(
        new PyBindClientCallbackProvider());
    std::unique_ptr<video::StreamCallbackProvider> stream_callback_provider(
        new PyBindStreamCallbackProvider(state, frame_buffer));
    video::Credentials credentials = video::Credentials(
        access_key,
        secret_key
    );
    std::unique_ptr<video::CredentialProvider> credential_provider(
        new video::StaticCredentialProvider(credentials));
    producer = video::KinesisVideoProducer::createSync(
        std::move(device_info_provider),
        std::move(client_callback_provider),
        std::move(stream_callback_provider),
        std::move(credential_provider),
//        API_CALL_CACHE_TYPE_ALL,
        API_CALL_CACHE_TYPE_NONE,
        region
    );

    std::unique_ptr<video::StreamDefinition> stream_definition(
        new video::StreamDefinition(
            stream_name,
            DEFAULT_RETENTION_PERIOD,
            nullptr, // tags
            "", // kms_key_id
            STREAMING_TYPE_OFFLINE,
            content_type,
            DEFAULT_MAX_LATENCY,
            DEFAULT_FRAGMENT_DURATION,
            DEFAULT_TIMECODE_SCALE,
            true, // key_frame_fragmentation
            true, // frame_timecodes
            true, // absolute_fragment_times
            true, // fragment_acks
            true, // restart_on_error
            true, // recalculate_metrics
            allow_stream_creation,
            0, // nal_adaptation_flags
            framerate,
            DEFAULT_AVG_BANDWIDTH_BPS,
            DEFAULT_BUFFER_DURATION,
            DEFAULT_REPLAY_DURATION,
            DEFAULT_CONNECTION_STALENESS
        )
    );
    stream = producer->createStreamSync(std::move(stream_definition));
}


bool KvsWrapper::start(
    const char *codec_private_data,
    size_t codec_private_data_size
) {
    LOG_PYBIND_INFO("Starting KVS stream.");
    bool ret = stream->start(
        (unsigned char *)codec_private_data,
        codec_private_data_size,
        DEFAULT_VIDEO_TRACK_ID
    );
    if (ret) {
        LOG_PYBIND_INFO("Stream started successfully.");
    } else {
        LOG_PYBIND_ERROR("Failed to start stream.");
    }

    return ret;
}


void KvsWrapper::put_frame(
    char *frame_data,
    uint32_t data_size,
    uint32_t idx,
    uint64_t pts,
    uint64_t dts,
    uint64_t duration,
    bool keyframe
) {
    if (keyframe) {
        current_fragment_start_ts = pts;
        LOG_PYBIND_DEBUG(
            "Keyframe detected. Starting new fragment at %lu.",
            current_fragment_start_ts);
    } else {
        uint64_t new_fragment_duration =
            pts - current_fragment_start_ts + duration;
        LOG_PYBIND_DEBUG(
            "Fragment start timestamp: %lu, frame timestamp: %lu, "
            "frame duration: %lu, new fragment duration: %lu",
            current_fragment_start_ts, pts,
            duration, new_fragment_duration);
        if (new_fragment_duration > max_fragment_duration) {
            LOG_PYBIND_WARN(
                "Frame %lu would exceed max fragment duration. Dropping frame.",
                pts);
            return;
        }
    }
    auto *frame = new video::KinesisVideoFrame();
    frame->size = data_size;
    frame->frameData = new uint8_t[frame->size];
    memcpy(frame->frameData, frame_data, frame->size);
    frame->index = idx;
    // Convert to 100ns units
    frame->presentationTs = pts / TIME_1NS_TO_100NS;
    frame->decodingTs = dts / TIME_1NS_TO_100NS;
    frame->duration = duration / TIME_1NS_TO_100NS;
    frame->flags = keyframe ? FRAME_FLAG_KEY_FRAME : FRAME_FLAG_NONE;
    frame->trackId = DEFAULT_VIDEO_TRACK_ID;
    frame_buffer->put_frame(frame, keyframe);
    std::vector<video::KinesisVideoFrame *> pending_frames = std::vector<video::KinesisVideoFrame *>();
    pending_frames.push_back(frame);

    while (!pending_frames.empty()) {
        if (!check_connection()) {
            LOG_PYBIND_WARN(
                "Stream was reset. Resending all buffered frames.");
            pending_frames = frame_buffer->get_all_frames();
        } else if (push_frame(pending_frames.front())) {
            LOG_PYBIND_DEBUG("Frame pushed successfully.");
            pending_frames.erase(pending_frames.begin());
        } else {
            LOG_PYBIND_ERROR("Failed to push frame. Retrying.");
        }
    }
}


bool KvsWrapper::stop_sync() {
    LOG_PYBIND_INFO("Stopping KVS stream.");
    bool ret = stream->stopSync();
    if (ret) {
        LOG_PYBIND_INFO("Stream stopped successfully.");
    } else {
        LOG_PYBIND_ERROR("Failed to stop stream.");
    }
    frame_buffer->clear();

    return ret;
}


bool KvsWrapper::push_frame(video::KinesisVideoFrame *frame) {
    uint64_t duration_available = state->get_duration_available();
    if (duration_available > high_threshold_100ns) {
        LOG_PYBIND_WARN(
            "Buffer in KVS SDK has frames for more than %d seconds (high threshold). "
            "Pushing frames at a slower rate to avoid overflow.",
            high_threshold);
        std::this_thread::sleep_for(
            DEFAULT_OVERFLOW_PREVENTION_HIGH_THR_INTERVAL);
    } else if (duration_available > low_threshold_100ns) {
        LOG_PYBIND_WARN(
            "Buffer in KVS SDK has frames for more than %d seconds (low threshold). "
            "Pushing frames at a slower rate to avoid overflow.",
            low_threshold);
        std::this_thread::sleep_for(
            DEFAULT_OVERFLOW_PREVENTION_LOW_THR_INTERVAL);
    }
    LOG_PYBIND_DEBUG(
        "Pushing frame %lu to KVS SDK.", frame->presentationTs);

    return stream->putFrame(*frame);
}


bool KvsWrapper::check_connection() {
    bool stream_was_restarted = false;
    while (!state->is_connection_ready()) {
        LOG_PYBIND_ERROR("Connection is not ready. Resetting stream.");
        stream_was_restarted = true;
        while (!stream->resetStream()) {
            LOG_PYBIND_ERROR("Failed to reset stream. Retrying.");
            std::this_thread::sleep_for(DEFAULT_RETRY_INTERVAL);
        }
        std::this_thread::sleep_for(DEFAULT_RETRY_INTERVAL);
    }
    return !stream_was_restarted;
}
