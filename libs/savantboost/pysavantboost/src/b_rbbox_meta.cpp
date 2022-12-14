#include "b_rbbox_meta.h"
#include "deepstream/dsrbbox_meta.h"
#include "pysavantboost.h"


namespace py = pybind11;

namespace pysavantboost {

    void bindrbboxsmeta(py::module &m) {
        py::class_<NvRBboxCoords>(m, "NvRBboxCoords", pydsdoc::usermeta::RBboxCoords::descr)
            .def(py::init<>())
            .def_readwrite("x_center", &NvRBboxCoords::x_center)
            .def_readwrite("y_center", &NvRBboxCoords::y_center)
            .def_readwrite("width", &NvRBboxCoords::width)
            .def_readwrite("height", &NvRBboxCoords::height)
            .def_readwrite("angle", &NvRBboxCoords::angle)
            .def("cast",
                     [](void *data) {
                         return (NvRBboxCoords *) data;
                     },
                     py::return_value_policy::reference,
                     pydsdoc::usermeta::RBboxCoords::cast)
            
            .def("cast",
                     [](size_t data) {
                         return (NvRBboxCoords *) data;
                     },
                     py::return_value_policy::reference,
                     pydsdoc::usermeta::RBboxCoords::cast);
        
        m.def("acquire_rbbox",
              &acquire_rbbox,
              py::return_value_policy::reference,
              pydsdoc::functionDoc::acquire_rbbox);
        m.def("add_rbbox_to_object_meta",
            &add_rbbox_to_object_meta
        );
        m.def(
            "iter_over_rbbox", 
            [](NvDsUserMetaList* user_meta_list) {return py::make_iterator(RBBoxIterator(user_meta_list), RBBoxIterator(nullptr));}, 
            py::keep_alive<0, 1>()
        );
        m.def(
            "get_rbbox",
            [](NvDsObjectMeta* nvdsobjmeta) 
            {
                NvRBboxCoords* return_value = new NvRBboxCoords;
                std::memcpy (return_value, get_rbbox(nvdsobjmeta), sizeof(NvRBboxCoords));
                return return_value;
            },
            py::return_value_policy::take_ownership
        );
    }
}