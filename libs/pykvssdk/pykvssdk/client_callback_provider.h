#pragma once

#include "ClientCallbackProvider.h"


namespace video = com::amazonaws::kinesis::video;


class PyBindClientCallbackProvider : public video::ClientCallbackProvider {
public:

    UINT64 getCallbackCustomData() override {
        return reinterpret_cast<UINT64> (this);
    }

};
