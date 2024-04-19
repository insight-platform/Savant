cmake_minimum_required(VERSION 3.12)
include(CheckLanguage)

project(gstsavantframemeta VERSION "0.0.3" LANGUAGES CXX)
check_language(CUDA)
if (CMAKE_CUDA_COMPILER)
    set(SAVANT_NVDS_ENABLED ON)
    enable_language(CUDA)
else()
    set(SAVANT_NVDS_ENABLED OFF)
endif()

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--no-undefined" )
set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} -Wl,--no-undefined" )

find_package(PkgConfig REQUIRED)
if (SAVANT_NVDS_ENABLED)
    find_package(CUDA REQUIRED)
endif()
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

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

if (SAVANT_NVDS_ENABLED AND NOT DEFINED DeepStream_DIR)
    set(DeepStream_DIR /opt/nvidia/deepstream/deepstream)
endif()

pkg_check_modules(GLIB REQUIRED glib-2.0)
pkg_check_modules(GSTREAMER REQUIRED gstreamer-1.0)

include_directories(
        ${GLIB_INCLUDE_DIRS}
        ${GSTREAMER_INCLUDE_DIRS}
        ${SAVANT_RS_LIB_DIR}/include
)
if (SAVANT_NVDS_ENABLED)
    include_directories(
            ${CUDA_INCLUDE_DIRS}
            ${DeepStream_DIR}/sources/includes
    )
endif()

link_directories(
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

AUX_SOURCE_DIRECTORY(src SRC_FILES)

set(PROJECT_SOURCE_DIR "src")
set(PROJECT_INCLUDE_DIR "include")

set(SOURCE_FILES
    ${PROJECT_INCLUDE_DIR}/gstsavantframemeta.h
    ${PROJECT_SOURCE_DIR}/gstsavantframemeta.cpp
)
if (SAVANT_NVDS_ENABLED)
    set(SOURCE_FILES
        ${SOURCE_FILES}
        ${PROJECT_INCLUDE_DIR}/gstsavantbatchmeta.h
        ${PROJECT_SOURCE_DIR}/gstsavantbatchmeta.cpp
        ${PROJECT_INCLUDE_DIR}/nvdssavantframemeta.h
        ${PROJECT_SOURCE_DIR}/nvdssavantframemeta.cpp
        ${PROJECT_INCLUDE_DIR}/savantrsprobes.h
        ${PROJECT_SOURCE_DIR}/savantrsprobes.cpp
        ${PROJECT_INCLUDE_DIR}/savantnvprobes.h
        ${PROJECT_SOURCE_DIR}/savantnvprobes.cpp
    )
endif()

source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCE_FILES})

add_library(savant_core UNKNOWN IMPORTED)
set_property(TARGET savant_core PROPERTY IMPORTED_LOCATION ${SAVANT_RS_CORE_LIB_FILE})
add_library(savant_core_py UNKNOWN IMPORTED)
set_property(TARGET savant_core_py PROPERTY IMPORTED_LOCATION ${SAVANT_RS_CORE_PY_LIB_FILE})
add_library(savant_rs UNKNOWN IMPORTED)
set_property(TARGET savant_rs PROPERTY IMPORTED_LOCATION ${SAVANT_RS_LIB_FILE})

add_library(gstsavantframemeta SHARED ${SOURCE_FILES})
target_link_libraries(
        gstsavantframemeta
        ${GLIB_LIBRARIES}
        ${GSTREAMER_LIBRARIES}
        savant_core
        savant_core_py
        savant_rs
)
if (SAVANT_NVDS_ENABLED)
    target_link_libraries(
            gstsavantframemeta
            ${CUDA_LIBRARIES}
            nvds_meta
            nvdsgst_meta
    )
endif()

target_include_directories(gstsavantframemeta PUBLIC include)

install(TARGETS gstsavantframemeta LIBRARY PUBLIC_HEADER)
