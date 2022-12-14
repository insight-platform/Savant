cmake_minimum_required(VERSION 3.14...3.18)

include(FindCUDA)

project(savantboost VERSION "1.0.2" LANGUAGES C CXX CUDA)

set(python_module_name pysavantboost)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--no-undefined" )
set(CMAKE_BUILD_TYPE Release)
# Compile option for debug
#set(CMAKE_BUILD_TYPE Debug)
#set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}" "-G -Xcompiler -rdynamic" )


find_package(PkgConfig REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development)
find_package(PythonExtensions)
find_package(pybind11 CONFIG REQUIRED)
if (DEFINE $ENV{ENABLE_DEBUG})
        find_package(OpenCV REQUIRED)
endif(DEFINE $ENV{ENABLE_DEBUG})
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
endif(DEFINE $ENV{ENABLE_DEBUG})

if(CUDAToolkit_VERSION_MAJOR STREQUAL "11")
        message("Building for CUDA 11")
        set(ARCH "
        -gencode=arch=compute_53,code=sm_53
        -gencode=arch=compute_72,code=sm_72
        -gencode=arch=compute_75,code=sm_75
        -gencode=arch=compute_80,code=sm_80
        -gencode=arch=compute_80,code=compute_80
        ")
else()
        message("Building for CUDA 10")
        set(ARCH "
        -gencode=arch=compute_53,code=sm_53
        -gencode=arch=compute_72,code=sm_72
        -gencode=arch=compute_75,code=sm_75
        -gencode=arch=compute_75,code=compute_75
        ")
endif()

cuda_add_library(savantboost SHARED
        cuda/nms_iou.h
        cuda/nms_iou.cu
        cuda/cudaCrop.h
        cuda/cudaCrop.cu
        bbox/rotatebbox.h
        bbox/rotatebbox.cpp
        deepstream/nvsurfaceptr.h
        deepstream/nvsurfaceptr.cpp
        deepstream/dsrbbox_meta.h
        deepstream/dsrbbox_meta.cpp
        deepstream/preprocessing.h
        deepstream/preprocessing.cpp
        deepstream/user_meta_type.h
        deepstream/image.h
        deepstream/image.cpp
        nvdsparserapid.cpp
        nvdsparserapid.h

        OPTIONS ${ARCH} -std=c++14 --expt-extended-lambda
        )

set(CUDA_LINK_LIBRARIES_KEYWORD PUBLIC)

if (DEFINE $ENV{ENABLE_DEBUG})
        target_link_libraries(
                savantboost
                ${CUDA_LIBRARIES}
                ${GLIB_LIBRARIES}
                ${GSTREAMER_LIBRARIES}
                ${GSTREAMER_PLUGIN_LIBRARIES}
                ${OpenCV_LIBS}
                gstbase-1.0
                gstvideo-1.0
                nvdsgst_helper
                nvdsgst_meta
                nvds_meta
                nvbufsurface
                nvbufsurftransform
                cuda
                CUDA::nppc
                CUDA::nppig
                CUDA::npps
        )
else()
        target_link_libraries(
                savantboost
                ${CUDA_LIBRARIES}
                ${GLIB_LIBRARIES}
                ${GSTREAMER_LIBRARIES}
                ${GSTREAMER_PLUGIN_LIBRARIES}
                gstbase-1.0
                gstvideo-1.0
                nvdsgst_helper
                nvdsgst_meta
                nvds_meta
                nvbufsurface
                nvbufsurftransform
                cuda
                CUDA::nppc
                CUDA::nppig
                CUDA::npps
        )
endif(DEFINE $ENV{ENABLE_DEBUG})

#
set_target_properties(savantboost PROPERTIES PUBLIC_HEADER "cuda/cudaCrop.h;bbox/rotatebbox.h;deepstream/nvsurfaceptr.h;deepstream/image.h")
#

install(TARGETS savantboost LIBRARY DESTINATION ../lib
        PUBLIC_HEADER DESTINATION ../include)

# Quiet a warning, since this project is only valid with SKBUILD
set(ignoreMe "${SKBUILD}")
