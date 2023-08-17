cmake_minimum_required(VERSION 3.12)

project(pygstsavantframemeta VERSION "0.0.1" LANGUAGES CXX CUDA)
set(python_module_name pygstsavantframemeta)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--no-undefined" )
set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} -Wl,--no-undefined" )

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-O3")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

find_package(PkgConfig REQUIRED)
find_package(CUDA REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

if(NOT DEFINED DeepStream_DIR)
    set(DeepStream_DIR /opt/nvidia/deepstream/deepstream)
endif()

if(NOT DEFINED SavantRsCapi_DIR)
    set(SavantRsCapi_DIR /opt/savant_capi)
endif()

pkg_check_modules(GLIB REQUIRED glib-2.0)
pkg_check_modules(GSTREAMER REQUIRED gstreamer-1.0)

include_directories(
        ${Python3_INCLUDE_DIRS}
        ${GLIB_INCLUDE_DIRS}
        ${GSTREAMER_INCLUDE_DIRS}
        ${CUDA_INCLUDE_DIRS}
        ${DeepStream_DIR}/sources/includes
        ${SavantRsCapi_DIR}/include
        ${CMAKE_SOURCE_DIR}/gstsavantframemeta/include
        ${CMAKE_SOURCE_DIR}/pygstsavantframemeta
 )

link_directories(
        ${Python3_LIBRARY_DIRS}
        ${GLIB_LIBRARY_DIRS}
        ${GSTREAMER_LIBRARY_DIRS}
        ${CUDA_LIBRARY_DIRS}
        ${DeepStream_DIR}/lib
        ${SavantRsCapi_DIR}/lib
 )

add_subdirectory(gstsavantframemeta ../build/gstsavantframemeta)

file (GLOB SOURCE_FILES "gstsavantframemeta/src/*.cpp")
file (GLOB HEADER_FILES "gstsavantframemeta/include/*.h")
file (GLOB PYTHON_FILES "pygstsavantframemeta/*.cpp" "pygstsavantframemeta/*.h")

source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCE_FILES} ${HEADER_FILES} ${PYTHON_FILES} )

pybind11_add_module(${python_module_name} SHARED
    ${SOURCE_FILES}
    ${HEADER_FILES}
    ${PYTHON_FILES}
 )

target_link_libraries(${python_module_name} PRIVATE
    ${Python3_LIBRARIES}
    ${GLIB_LIBRARIES}
    ${GSTREAMER_LIBRARIES}
    ${CUDA_LIBRARIES}
    nvds_meta
    nvdsgst_meta
    savant_capi
    pybind11::module
    pybind11::lto
    gstsavantframemeta
 )

pybind11_extension(${python_module_name})
pybind11_strip(${python_module_name})

install(TARGETS ${python_module_name} DESTINATION .)
