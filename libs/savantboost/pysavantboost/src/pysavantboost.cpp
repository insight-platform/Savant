#include "pysavantboost.h"
#include "nms.h"
#include "b_rbbox_meta.h"
#include "preprocessing.h"
#include "nvsurfaceptr.h"


using namespace std;
namespace py = pybind11;

namespace pysavantboost {

    PYBIND11_MODULE(pysavantboost, m) {
        m.doc() = "pybind11 bindings for savantboost library functions"; /* this will be the doc string*/

        bindrbboxsmeta(m);
        bindnms(m);
        bindpreprocessing(m);
        bindnvsurfaceptr(m);
        py::add_ostream_redirect(m, "ostream_redirect");

    }   // end PYBIND11_MODULE(pyds, m)

}
