#pragma once

#include <log4cplus/loggingmacros.h>
#include <log4cplus/logger.h>


#define LOG_PYBIND_TRACE(...) LOG4CPLUS_TRACE_FMT(log4cplus::Logger::getInstance("PyKvsBinding"), __VA_ARGS__)
#define LOG_PYBIND_DEBUG(...) LOG4CPLUS_DEBUG_FMT(log4cplus::Logger::getInstance("PyKvsBinding"), __VA_ARGS__)
#define LOG_PYBIND_INFO(...)  LOG4CPLUS_INFO_FMT(log4cplus::Logger::getInstance("PyKvsBinding"), __VA_ARGS__)
#define LOG_PYBIND_WARN(...)  LOG4CPLUS_WARN_FMT(log4cplus::Logger::getInstance("PyKvsBinding"), __VA_ARGS__)
#define LOG_PYBIND_ERROR(...) LOG4CPLUS_ERROR_FMT(log4cplus::Logger::getInstance("PyKvsBinding"), __VA_ARGS__)
