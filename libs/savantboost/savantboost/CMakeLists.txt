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

if(CMAKE_BUILD_TYPE EQUAL Debug)
    message("Added OpenCV package for debuging")
    find_package( OpenCV REQUIRED )
endif()

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
        ${GSTREAMER_PLUGIN_INCLUDE_DIRS}
)

link_directories(
        ${GSTREAMER_LIBRARY_DIRS}
        ${GLIB_LIBRARY_DIRS}
        "${DeepStream_DIR}/lib"
)

if(CMAKE_BUILD_TYPE MATCHES Debug)
        link_directories(
                ${OPENCV_LIBRARY_DIRS}
        )
endif()

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

if(CMAKE_BUILD_TYPE EQUAL Debug)
        target_link_libraries(
                ${library_name}
                ${OpenCV_LIBS}
        )
endif()

if(CMAKE_SYSTEM_PROCESSOR MATCHES arm)
    message("Building for ${CMAKE_SYSTEM_PROCESSOR}")
    set(ARCH "72\;87")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES x86_64)
    message("Building for ${CMAKE_SYSTEM_PROCESSOR}")
    set(ARCH "61\;70\;75\;80\;86\;89\;90")
else()
    message(FATAL_ERROR "Unsupported arhitecture: ${CMAKE_SYSTEM_PROCESSOR}")
endif()

file(GLOB HEADER_FILES *.h  */*.h )
set_target_properties(savantboost PROPERTIES CUDA_ARCHITECTURES ${ARCH} PUBLIC_HEADER ${HEADER_FILES})

install(TARGETS ${library_name} LIBRARY DESTINATION ../lib
        PUBLIC_HEADER DESTINATION ../include)