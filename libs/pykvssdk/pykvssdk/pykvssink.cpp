#include <pybind11/pybind11.h>
#include <log4cplus/config.hxx>
#include "kvs_wrapper.h"


namespace py = pybind11;


PYBIND11_MODULE(pykvssdk, m) {
    m.doc() = "Python bindings for KVS SDK";

    py::class_<KvsWrapper>(
        m, "KvsWrapper"
    )
        .def(
            py::init<
                const std::string &,
                const std::string &,
                const std::string &,
                const std::string &,
                const std::string &,
                bool,
                uint32_t,
                uint32_t,
                uint32_t,
                uint32_t
            >(),
            py::arg("region"),
            py::arg("access_key"),
            py::arg("secret_key"),
            py::arg("stream_name"),
            py::arg("codec"),
            py::arg("allow_stream_creation"),
            py::arg("framerate"),
            py::arg("low_threshold"),
            py::arg("high_threshold"),
            py::arg("retention_period_hours") = 24
        )
        .def("start", &KvsWrapper::start)
        .def("stop_sync", &KvsWrapper::stop_sync)
        .def("put_frame", &KvsWrapper::put_frame);

    m.def("configure_logging", [](
        const std::string &log_config_path
    ) {
        log4cplus::initialize();
        log4cplus::PropertyConfigurator::doConfigure(log_config_path);
    });

}
