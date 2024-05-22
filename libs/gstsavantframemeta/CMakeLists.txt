cmake_minimum_required(VERSION 3.12)
include(CheckLanguage)

project(pygstsavantframemeta VERSION "0.0.3" LANGUAGES CXX)
check_language(CUDA)
if (CMAKE_CUDA_COMPILER)
    set(SAVANT_NVDS_ENABLED ON)
    enable_language(CUDA)
else()
    set(SAVANT_NVDS_ENABLED OFF)
endif()

set(python_module_name pygstsavantframemeta)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--no-undefined" )
set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} -Wl,--no-undefined" )

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-O3")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

if (SAVANT_NVDS_ENABLED)
    add_definitions(-DSAVANT_NVDS_ENABLED=${SAVANT_NVDS_ENABLED})
endif()

find_package(PkgConfig REQUIRED)
if (SAVANT_NVDS_ENABLED)
    find_package(CUDA REQUIRED)
endif()
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# Find savant-rs package
execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import savant_rs; print(savant_rs.__path__[0], end='')"
        OUTPUT_VARIABLE SAVANT_RS_LIB_DIR
        RESULT_VARIABLE EXIT_CODE
)
if (NOT ${EXIT_CODE} EQUAL 0)
    message(
            FATAL_ERROR
            "The \"savant_rs\" Python3 package is not installed. Please install it using the following command: \"pip3 install savant_rs\"."
    )
endif()
set(SAVANT_RS_CORE_LIB_DIR "${SAVANT_RS_LIB_DIR}.libs")

file(GLOB SAVANT_RS_CORE_LIB_FILE "${SAVANT_RS_CORE_LIB_DIR}/libsavant_core-*.so")
message(STATUS "Found core library for savant-rs: ${SAVANT_RS_CORE_LIB_FILE}")
file(GLOB SAVANT_RS_CORE_PY_LIB_FILE "${SAVANT_RS_CORE_LIB_DIR}/libsavant_core_py-*.so")
message(STATUS "Found core-py library for savant-rs: ${SAVANT_RS_CORE_PY_LIB_FILE}")
file(GLOB SAVANT_RS_LIB_FILE "${SAVANT_RS_LIB_DIR}/savant_rs.cpython*.so")
message(STATUS "Found library for savant-rs: ${SAVANT_RS_LIB_FILE}")

if(SAVANT_NVDS_ENABLED AND NOT DEFINED DeepStream_DIR)
    set(DeepStream_DIR /opt/nvidia/deepstream/deepstream)
endif()

pkg_check_modules(GLIB REQUIRED glib-2.0)
pkg_check_modules(GSTREAMER REQUIRED gstreamer-1.0)

include_directories(
        ${Python3_INCLUDE_DIRS}
        ${GLIB_INCLUDE_DIRS}
        ${GSTREAMER_INCLUDE_DIRS}
        ${SAVANT_RS_LIB_DIR}/include
        ${CMAKE_SOURCE_DIR}/gstsavantframemeta/include
        ${CMAKE_SOURCE_DIR}/pygstsavantframemeta
)
if (SAVANT_NVDS_ENABLED)
    include_directories(
            ${CUDA_INCLUDE_DIRS}
            ${DeepStream_DIR}/sources/includes
    )
endif()

link_directories(
        ${Python3_LIBRARY_DIRS}
        ${GLIB_LIBRARY_DIRS}
        ${GSTREAMER_LIBRARY_DIRS}
        ${SAVANT_RS_CORE_LIB_DIR}
        ${SAVANT_RS_LIB_DIR}
)
if (SAVANT_NVDS_ENABLED)
    link_directories(
            ${CUDA_LIBRARY_DIRS}
            ${DeepStream_DIR}/lib
    )
endif()

add_subdirectory(gstsavantframemeta ../build/gstsavantframemeta)

set(SOURCE_FILES
    "gstsavantframemeta/src/gstsavantframemeta.cpp"
)
if (SAVANT_NVDS_ENABLED)
    set(SOURCE_FILES
        ${SOURCE_FILES}
        "gstsavantframemeta/src/gstsavantbatchmeta.cpp"
        "gstsavantframemeta/src/nvbufsurface_generator.cpp"
        "gstsavantframemeta/src/nvdssavantframemeta.cpp"
        "gstsavantframemeta/src/savantrsprobes.cpp"
        "gstsavantframemeta/src/savantnvprobes.cpp"
    )
endif()

set(HEADER_FILES
    "gstsavantframemeta/include/gstsavantframemeta.h"
)
if (SAVANT_NVDS_ENABLED)
    set(HEADER_FILES
        ${HEADER_FILES}
        "gstsavantframemeta/include/gstsavantbatchmeta.h"
        "gstsavantframemeta/include/nvbufsurface_generator.h"
        "gstsavantframemeta/include/nvdssavantframemeta.h"
        "gstsavantframemeta/include/savantrsprobes.h"
        "gstsavantframemeta/include/savantnvprobes.h"
    )
endif()

file (GLOB PYTHON_FILES "pygstsavantframemeta/*.cpp" "pygstsavantframemeta/*.h")

source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCE_FILES} ${HEADER_FILES} ${PYTHON_FILES} )

add_library(savant_core UNKNOWN IMPORTED)
set_property(TARGET savant_core PROPERTY IMPORTED_LOCATION ${SAVANT_RS_CORE_LIB_FILE})
add_library(savant_core_py UNKNOWN IMPORTED)
set_property(TARGET savant_core_py PROPERTY IMPORTED_LOCATION ${SAVANT_RS_CORE_PY_LIB_FILE})
add_library(savant_rs UNKNOWN IMPORTED)
set_property(TARGET savant_rs PROPERTY IMPORTED_LOCATION ${SAVANT_RS_LIB_FILE})

pybind11_add_module(${python_module_name} SHARED
    ${SOURCE_FILES}
    ${HEADER_FILES}
    ${PYTHON_FILES}
)

target_link_libraries(${python_module_name} PRIVATE
    ${Python3_LIBRARIES}
    ${GLIB_LIBRARIES}
    ${GSTREAMER_LIBRARIES}
    savant_rs
    pybind11::module
    pybind11::lto
    gstsavantframemeta
)
if (SAVANT_NVDS_ENABLED)
    target_link_libraries(${python_module_name} PRIVATE
            ${CUDA_LIBRARIES}
            nvds_meta
            nvdsgst_meta
            nvdsgst_bufferpool
            nvdsbufferpool
            nvbufsurface
            nvdsgst_helper
            nvbufsurftransform
    )
endif()

pybind11_extension(${python_module_name})
pybind11_strip(${python_module_name})

install(TARGETS ${python_module_name} DESTINATION .)
