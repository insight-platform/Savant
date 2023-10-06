cmake_minimum_required(VERSION 3.22...3.27)

set(library_name savantboost)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--no-undefined" )

project(${library_name} LANGUAGES C CXX CUDA)

find_package(PkgConfig REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED)

IF(CMAKE_BUILD_TYPE MATCHES Debug)
    message("Added OpenCV package for debuging")
    find_package( OpenCV REQUIRED )
ENDIF()

pkg_check_modules(GLIB REQUIRED glib-2.0)
pkg_check_modules(GSTREAMER gstreamer-1.0)
pkg_check_modules(GSTREAMER_PLUGIN gstreamer-plugins-base-1.0)

IF(NOT DEFINED DeepStream_DIR)
    set(DeepStream_DIR /opt/nvidia/deepstream/deepstream)
ENDIF()

include_directories(
        "${DeepStream_DIR}/sources/includes"
        ${GSTREAMER_INCLUDE_DIRS}
        ${GLIB_INCLUDE_DIRS}
        ${GSTREAMER_PLUGIN_INCLUDE_DIRS}
)

link_directories(
        ${GSTREAMER_LIBRARY_DIRS}
        ${GLIB_LIBRARY_DIRS}
        "${DeepStream_DIR}/lib"
)

IF(CMAKE_BUILD_TYPE MATCHES Debug)
        link_directories(
                ${OPENCV_LIBRARY_DIRS}
        )
ENDIF()

message("CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")

file(GLOB SOURCE_FILES *.cpp *.h */*.cpp */*.cu */*.h )

add_library(${library_name} SHARED    ${SOURCE_FILES})

target_compile_features(${library_name} PRIVATE cuda_std_14 )

set(CUDA_LINK_LIBRARIES_KEYWORD PUBLIC)

target_link_libraries(
        ${library_name}
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
        CUDA::nppig
        CUDA::npps
)

IF(CMAKE_BUILD_TYPE MATCHES Debug)
        target_link_libraries(
                ${library_name}
                ${OpenCV_LIBS}
        )
ENDIF()

#
install(TARGETS ${library_name} LIBRARY DESTINATION ../lib
        PUBLIC_HEADER DESTINATION ../include)