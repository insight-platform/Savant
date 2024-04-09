#pragma once

#include "StreamCallbackProvider.h"
#include "producer_state.h"
#include "frame_buffer.h"


namespace video = com::amazonaws::kinesis::video;


class PyBindStreamCallbackProvider : public video::StreamCallbackProvider {
    UINT64 custom_data_;
public:
    PyBindStreamCallbackProvider(
        ProducerState *state,
        FrameBuffer *frame_buffer
    );


    UINT64 getCallbackCustomData() override {
        return custom_data_;
    }


    StreamErrorReportFunc getStreamErrorReportCallback() override {
        return streamErrorReportHandler;
    };


    StreamReadyFunc getStreamReadyCallback() override {
        return streamReadyHandler;
    };


    StreamClosedFunc getStreamClosedCallback() override {
        return streamClosedHandler;
    };


    StreamDataAvailableFunc getStreamDataAvailableCallback() override {
        return streamDataAvailableHandler;
    };


    FragmentAckReceivedFunc getFragmentAckReceivedCallback() override {
        return fragmentAckReceivedHandler;
    };


private:

    ProducerState *state;
    FrameBuffer *frame_buffer;

    static STATUS streamErrorReportHandler(
        UINT64 handle,
        STREAM_HANDLE stream_handle,
        UPLOAD_HANDLE upload_handle,
        UINT64 errored_timecode,
        STATUS status_code
    );

    static STATUS streamReadyHandler(
        UINT64 handle,
        STREAM_HANDLE stream_handle
    );

    static STATUS streamClosedHandler(
        UINT64 handle,
        STREAM_HANDLE stream_handle,
        UPLOAD_HANDLE upload_handle
    );

    static STATUS streamDataAvailableHandler(
        UINT64 handle,
        STREAM_HANDLE stream_handle,
        PCHAR stream_name,
        UPLOAD_HANDLE upload_handle,
        UINT64 duration_available,
        UINT64 size_available
    );

    static STATUS fragmentAckReceivedHandler(
        UINT64 handle,
        STREAM_HANDLE stream_handle,
        UPLOAD_HANDLE upload_handle,
        PFragmentAck pFragmentAck
    );


    inline static FrameBuffer *get_buffer(UINT64 provider_handle) {
        return ((PyBindStreamCallbackProvider *)provider_handle)->frame_buffer;
    }


    inline static ProducerState *get_state(UINT64 provider_handle) {
        return ((PyBindStreamCallbackProvider *)provider_handle)->state;
    }
};
