#pragma  once

#include "pysavantboost.h"
#include "deepstream/preprocessing.h"
#include "deepstream/image.h"
#include <pybind11/functional.h>


namespace pysavantboost {
    void bindpreprocessing(py::module &m);

    class PyImage{
        private:
            uint8_t *tmp_pointer=nullptr;
        public:
            savantboost::Image* image = nullptr;
            PyImage(py::array_t<uint8_t, py::array::c_style> numpy_image);
            PyImage(savantboost::Image *image);
            ~PyImage();
            py::array_t<uint8_t, py::array::c_style> to_numpy();

    };
}