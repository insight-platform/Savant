#include "producer_state.h"
#include "StreamCallbackProvider.h"
#include "logger.h"
#include "stream_callback_provider.h"


PyBindStreamCallbackProvider::PyBindStreamCallbackProvider(
    ProducerState *state,
    FrameBuffer *frame_buffer
) {
    this->state = state;
    this->frame_buffer = frame_buffer;
    this->custom_data_ = (UINT64)this;
}


STATUS PyBindStreamCallbackProvider::streamErrorReportHandler(
    UINT64 handle,
    STREAM_HANDLE stream_handle,
    UPLOAD_HANDLE upload_handle,
    UINT64 errored_timecode,
    UINT32 status_code
) {
    LOG_PYBIND_ERROR(
        "Got stream error. Status code: 0x%08x, errored timecode: %lu",
        status_code, errored_timecode);
    get_state(handle)->set_connection_ready(false);

    return STATUS_SUCCESS;
}


STATUS PyBindStreamCallbackProvider::streamReadyHandler(
    UINT64 handle,
    STREAM_HANDLE stream_handle
) {
    LOG_PYBIND_INFO("Stream is ready.");
    get_state(handle)->set_connection_ready(true);

    return STATUS_SUCCESS;
}


STATUS PyBindStreamCallbackProvider::streamClosedHandler(
    UINT64 handle,
    STREAM_HANDLE stream_handle,
    UPLOAD_HANDLE upload_handle
) {
    LOG_PYBIND_INFO("Stream is closed.");
    get_state(handle)->set_connection_ready(false);

    return STATUS_SUCCESS;
}


STATUS PyBindStreamCallbackProvider::streamDataAvailableHandler(
    UINT64 handle,
    STREAM_HANDLE stream_handle,
    PCHAR stream_name,
    UPLOAD_HANDLE upload_handle,
    UINT64 duration_available,
    UINT64 size_available
) {
    LOG_PYBIND_DEBUG(
        "KVS SDK buffer has %lu bytes of data available for stream %s "
        "(duration: %lu of 100ns units).",
        size_available, stream_name, duration_available);
    get_state(handle)->set_duration_available(duration_available);

    return STATUS_SUCCESS;
}


STATUS PyBindStreamCallbackProvider::fragmentAckReceivedHandler(
    UINT64 handle,
    STREAM_HANDLE stream_handle,
    UPLOAD_HANDLE upload_handle,
    PFragmentAck pFragmentAck
) {
    LOG_PYBIND_DEBUG(
        "Got fragment ack. Fragment timestamp: %lu, ack type: %d",
        pFragmentAck->timestamp, pFragmentAck->ackType);
    if (pFragmentAck->ackType == FRAGMENT_ACK_TYPE_RECEIVED) {
        get_buffer(handle)->drop_frames(pFragmentAck->timestamp);
    }

    return STATUS_SUCCESS;
}
