cmake_minimum_required(VERSION 3.12)

project(gstsavantframemeta VERSION "0.0.2" LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--no-undefined" )
set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} -Wl,--no-undefined" )

find_package(PkgConfig REQUIRED)
find_package(CUDA REQUIRED)
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
file(GLOB SAVANT_RS_LIB_FILES "${SAVANT_RS_LIB_DIR}/savant_rs.cpython*.so")
message(STATUS "Found library for savant-rs: ${SAVANT_RS_LIB_FILES}")
if(NOT DEFINED SAVANT_RS_CAPI_INCLUDE_DIR)
    set(SAVANT_RS_CAPI_INCLUDE_DIR /opt/savant_capi/include)
endif()

if(NOT DEFINED DeepStream_DIR)
    set(DeepStream_DIR /opt/nvidia/deepstream/deepstream)
endif()

pkg_check_modules(GLIB REQUIRED glib-2.0)
pkg_check_modules(GSTREAMER REQUIRED gstreamer-1.0)

include_directories(
        ${GLIB_INCLUDE_DIRS}
        ${GSTREAMER_INCLUDE_DIRS}
        ${CUDA_INCLUDE_DIRS}
        ${DeepStream_DIR}/sources/includes
        ${SAVANT_RS_CAPI_INCLUDE_DIR}
)

link_directories(
        ${GLIB_LIBRARY_DIRS}
        ${GSTREAMER_LIBRARY_DIRS}
        ${CUDA_LIBRARY_DIRS}
        ${DeepStream_DIR}/lib
        ${SAVANT_RS_LIB_DIR}
)

AUX_SOURCE_DIRECTORY(src SRC_FILES)

set(PROJECT_SOURCE_DIR "src")
set(PROJECT_INCLUDE_DIR "include")

set(SOURCE_FILES
    ${PROJECT_INCLUDE_DIR}/gstsavantbatchmeta.h
    ${PROJECT_SOURCE_DIR}/gstsavantbatchmeta.cpp
    ${PROJECT_INCLUDE_DIR}/gstsavantframemeta.h
    ${PROJECT_SOURCE_DIR}/gstsavantframemeta.cpp
    ${PROJECT_INCLUDE_DIR}/savantrsprobes.h
    ${PROJECT_SOURCE_DIR}/savantrsprobes.cpp
)

source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCE_FILES})

add_library(savant_rs UNKNOWN IMPORTED)
set_property(TARGET savant_rs PROPERTY IMPORTED_LOCATION ${SAVANT_RS_LIB_FILES})

add_library(gstsavantframemeta SHARED ${SOURCE_FILES})
target_link_libraries(
        gstsavantframemeta
        ${GLIB_LIBRARIES}
        ${GSTREAMER_LIBRARIES}
        ${CUDA_LIBRARIES}
        nvds_meta
        nvdsgst_meta
        savant_rs
)

target_include_directories(gstsavantframemeta PUBLIC include)

install(TARGETS gstsavantframemeta LIBRARY PUBLIC_HEADER)
