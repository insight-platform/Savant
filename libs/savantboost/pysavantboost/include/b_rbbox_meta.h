#pragma  once

#include "pysavantboost.h"

namespace py = pybind11;

namespace pysavantboost {
    void bindrbboxsmeta(py::module &m);
}