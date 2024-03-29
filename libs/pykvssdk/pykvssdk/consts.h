#pragma once

#include <chrono>


#define TIME_1NS_TO_100NS 100
#define TIME_100NS_TO_1MS 10000
#define TIME_100NS_TO_1S (TIME_100NS_TO_1MS * 1000)

// Defaults
#define DEFAULT_RETENTION_PERIOD std::chrono::hours(24)
#define DEFAULT_MAX_LATENCY std::chrono::seconds(10)
#define DEFAULT_FRAGMENT_DURATION std::chrono::seconds(20)
#define DEFAULT_TIMECODE_SCALE std::chrono::milliseconds(1)
#define DEFAULT_AVG_BANDWIDTH_BPS (4 * 1024 * 1024)
#define DEFAULT_BUFFER_DURATION std::chrono::seconds(120)
#define DEFAULT_REPLAY_DURATION std::chrono::seconds(120)
#define DEFAULT_CONNECTION_STALENESS std::chrono::seconds(20)
#define DEFAULT_RETRY_INTERVAL std::chrono::seconds(1)
#define DEFAULT_OVERFLOW_PREVENTION_LOW_THR_INTERVAL std::chrono::milliseconds(100)
#define DEFAULT_OVERFLOW_PREVENTION_HIGH_THR_INTERVAL std::chrono::seconds(1)
