cmake_minimum_required(VERSION 3.22)
project(pysavantboost VERSION "1.0.2" LANGUAGES C CXX CUDA)

set(python_module_name pysavantboost)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--no-undefined" )
set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} -Wl,--no-undefined" )


set(CMAKE_CUDA_COMPILER_TOOLKIT_ROOT "/usr/local/cuda")
set(CMAKE_CUDA_COMPILER_LIBRARY_ROOT "/usr/local/cuda")
set(CMAKE_BUILD_TYPE Debug)

find_package(PkgConfig REQUIRED)
find_package(CUDA REQUIRED)

find_package(CUDAToolkit REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development)
find_package(PythonExtensions)
find_package(pybind11 CONFIG REQUIRED)


if (DEFINED $ENV{ENABLE_DEBUG})
    message("Try found OpenCV package")
    find_package( OpenCV REQUIRED )
endif(DEFINED $ENV{ENABLE_DEBUG})

pkg_check_modules(GLIB REQUIRED glib-2.0)
pkg_check_modules(GSTREAMER gstreamer-1.0)
pkg_check_modules(GSTREAMER_PLUGIN gstreamer-plugins-base-1.0)

if(NOT DEFINED DeepStream_DIR)
    set(DeepStream_DIR /opt/nvidia/deepstream/deepstream)
endif()

include_directories(
        "${DeepStream_DIR}/sources/includes"
        ${GSTREAMER_INCLUDE_DIRS}
        ${GLIB_INCLUDE_DIRS}
        ${CUDA_INCLUDE_DIRS}
        ${GSTREAMER_PLUGIN_INCLUDE_DIRS}
        pysavantboost/include/
        savantboost/
)
link_directories(
        ${GSTREAMER_LIBRARY_DIRS}
        ${GLIB_LIBRARY_DIRS}
        ${CUDA_LIBRARY_DIRS}
        "${DeepStream_DIR}/lib"
)
if (DEFINE $ENV{ENABLE_DEBUG})
    link_directories(
        ${OPENCV_LIBRARY_DIRS}   
    )
    include_directories(
        ${OpenCV_INCLUDE_DIRS}
    )
endif(DEFINE $ENV{ENABLE_DEBUG})

add_subdirectory(savantboost ../build/savantboost)

add_library(${python_module_name} MODULE pysavantboost/src/nms.cpp pysavantboost/src/b_rbbox_meta.cpp pysavantboost/src/pysavantboost.cpp 
pysavantboost/src/preprocessing.cpp pysavantboost/src/nvsurfaceptr.cpp)

if ($ENV{ENABLE_DEBUG})
    target_compile_definitions(${python_module_name} PUBLIC ENABLE_DEBUG)
endif($ENV{ENABLE_DEBUG})



if (DEFINE $ENV{ENABLE_DEBUG})
    target_link_libraries(${python_module_name}
                ${PYTHON_LIBRARIES}
                ${CUDA_LIBRARIES}
                ${OPENCV_LIBRARY_DIRS}
                ${OpenCV_LIBS} 
                pybind11::module
                pybind11::lto
                savantboost
            )
else()
    target_link_libraries(${python_module_name}
    ${PYTHON_LIBRARIES}
    ${CUDA_LIBRARIES}
    pybind11::module
    pybind11::lto
    savantboost
)
endif(DEFINE $ENV{ENABLE_DEBUG})


pybind11_extension(${python_module_name})
pybind11_strip(${python_module_name})
install(TARGETS ${python_module_name} DESTINATION .)
# Quiet a warning, since this project is only valid with SKBUILD
set(ignoreMe "${SKBUILD}")
