cmake_minimum_required(VERSION 3.22...3.27)

set(python_module_name pysavantboost)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--no-undefined" )
set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} -Wl,--no-undefined" )
set(PYBIND11_FINDPYTHON ON)

if(NOT DEFINED DeepStream_DIR)
    set(DeepStream_DIR /opt/nvidia/deepstream/deepstream)
endif()

project(pysavantboost LANGUAGES C CXX CUDA)

find_package(PkgConfig REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development)
find_package(CUDAToolkit REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

if(CMAKE_BUILD_TYPE STREQUAL Debug)
    message("Added OpenCV package for debuging")
    find_package( OpenCV REQUIRED )
endif()

pkg_check_modules(GLIB REQUIRED glib-2.0)
pkg_check_modules(GSTREAMER gstreamer-1.0)
pkg_check_modules(GSTREAMER_PLUGIN gstreamer-plugins-base-1.0)


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
if(CMAKE_BUILD_TYPE STREQUAL Debug)
    link_directories(
        ${OPENCV_LIBRARY_DIRS}   
    )
    include_directories(
        ${OpenCV_INCLUDE_DIRS}
    )
endif()

add_subdirectory(savantboost ../build/savantboost)

file(GLOB SRC_FILES pysavantboost/src/*.cpp)

pybind11_add_module(${python_module_name} MODULE ${SRC_FILES})

set(CUDA_LINK_LIBRARIES_KEYWORD PUBLIC)

target_link_libraries(${python_module_name} PRIVATE
    ${Python3_LIBRARIES}
    pybind11::module
    pybind11::lto
    CUDA::cudart
    CUDA::nppc
    CUDA::nppig
    CUDA::npps
    savantboost
)

if(CMAKE_BUILD_TYPE STREQUAL Debug)
    target_link_libraries(${python_module_name} PRIVATE
                ${OPENCV_LIBRARY_DIRS}
                ${OpenCV_LIBS}
            )
endif()

pybind11_extension(${python_module_name})
pybind11_strip(${python_module_name})
install(TARGETS ${python_module_name} DESTINATION .)

