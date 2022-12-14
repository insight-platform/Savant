#include "preprocessing.h"
#include <string>
#include <pybind11/embed.h>
#include <stdexcept>
#include <dlfcn.h>


namespace pysavantboost {

     bool hasEnding (std::string const &fullString, std::string const &ending) {
          if (fullString.length() >= ending.length()) {
               return (0 == fullString.compare (fullString.length() - ending.length(), ending.length(), ending));
          } else {
               return false;
          }
     }

     PyImage::PyImage(py::array_t<uint8_t, py::array::c_style> numpy_image){
          // TODO: Add dimension validity checks
          
          image = new savantboost::Image((Npp32u *) numpy_image.data(), numpy_image.shape(1), numpy_image.shape(0), true);
     }
     
     PyImage::PyImage(savantboost::Image *_image){
          image = _image;
     }

     PyImage::~PyImage(){
          if (this->tmp_pointer) free(this->tmp_pointer);
     }

     py::array_t<uint8_t, py::array::c_style> PyImage::to_numpy(){
          ssize_t              ndim    = 3;
          std::vector<ssize_t> shape   = { image->getHeight(), image->getWidth(), 4};
          std::vector<ssize_t> strides = {
            (ssize_t) (sizeof(Npp8u) * image->getWidth() * 4),
            (ssize_t) (sizeof(Npp8u)*4),
            sizeof(Npp8u)
            };
          auto dtype = py::dtype(py::format_descriptor<unsigned char>::format());
          return py::array(
                    dtype,
                    shape,
                    strides,
                    (const unsigned char *) image->getCPUDataPtr(),
                    py::cast(image->getCPUDataPtr())
          );

          // return py::array(py::buffer_info(
          //           (Npp8u*) image->getCPUDataPtr(),          /* data as contiguous array  */
          //           sizeof(Npp8u),                            /* size of one scalar        */
          //           py::format_descriptor<uint8_t>::format(), /* data type                 */
          //           ndim,                                          /* number of dimensions      */
          //           shape,                                 /* shape of the matrix       */
          //           strides                                /* strides for each axis     */
          //      )
          // );
     }


     void bindpreprocessing(py::module &m) {

          py::class_<PyImage>(m, "Image", pydsdoc::preprocessing::Image::class_desc)
               .def(py::init<py::array_t<uint8_t, py::array::c_style>>(), py::arg("image"))
               .def("to_numpy", &PyImage::to_numpy, py::return_value_policy::reference);

          py::class_<ObjectsPreprocessing>(m, "ObjectsPreprocessing", pydsdoc::preprocessing::ObjectsPreprocessing::class_desc)
               .def(py::init<>())
               .def(
                    "add_preprocessing_function",
                    [](
                          ObjectsPreprocessing &self,
                          std::string element_name,
                          std::string custom_preprocessing
                    )
                    {
                         
                         std::function<savantboost::Image *(savantboost::Image* )> custom_preprocess_func;
                         int i = custom_preprocessing.find(":");
                         std::string lib_path = custom_preprocessing.substr(0, i);
                         std::string func_name = custom_preprocessing.substr(i+1, custom_preprocessing.length() - i);
                         /* open the needed object */
                         if (hasEnding (lib_path, std::string(".so")))
                         {
                              self.add_preprocessing_function(element_name, lib_path, func_name);
                         }
                         else
                         {
                              py::object pymodule = py::module::import(lib_path.c_str());
                              py::object custom_func = pymodule.attr(func_name.c_str());
                              custom_preprocess_func = [=](savantboost::Image *image) {
                                   return custom_func(PyImage(image)).cast<PyImage>().image;
                              };
                         }
                         self.add_preprocessing_function(element_name, custom_preprocess_func);
                    }
               )
               .def("restore_frame", [] (
                    ObjectsPreprocessing &self,
                    size_t inbuf
               ){
                    auto *buffer = reinterpret_cast<GstBuffer *>(inbuf);
                    self.restore_frame(buffer);
               }, 
               pydsdoc::preprocessing::ObjectsPreprocessing::restore_frame)
               .def(
                    "preprocessing",
                    [](
                         ObjectsPreprocessing &self,
                         std::string element_name,
                         size_t inbuf, 
                         gint model_uid, 
                         gint class_id, 
                         gint padding_width,
                         gint padding_height
                    ){
                         // std::cout << "Test21sdfsdfsdfds" << std::endl;
                         auto *buffer = reinterpret_cast<GstBuffer *>(inbuf);
                         int status = (int) self.preprocessing(
                              element_name, buffer, model_uid, class_id, padding_width, padding_height
                         );
                         
                    },
                    py::arg("element_name"),
                    py::arg("inbuf"),
                    py::arg("model_uid"),
                    py::arg("class_id"),
                    py::arg("padding_width") = 0,
                    py::arg("padding_height") = 0
               );
     }
}
