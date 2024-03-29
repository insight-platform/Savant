#pragma once

#include <cstdint>


class ProducerState {
public:
    inline uint64_t get_duration_available() const {
        return duration_available;
    }


    inline void set_duration_available(uint64_t value) {
        this->duration_available = value;
    }


    inline bool is_connection_ready() const {
        return connection_ready;
    }


    inline void set_connection_ready(bool value) {
        this->connection_ready = value;
    }


private:
    uint64_t duration_available = 0;
    bool connection_ready = false;
};
