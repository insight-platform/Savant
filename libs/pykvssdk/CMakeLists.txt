cmake_minimum_required(VERSION 3.12)

project(pykvssdk VERSION "0.0.2" LANGUAGES CXX)
set(python_module_name pykvssdk)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--no-undefined")
set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} -Wl,--no-undefined")

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()

set(CMAKE_CXX_FLAGS "-O3")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

if (NOT DEFINED KVS_SDK_BASE_DIR)
    message(FATAL_ERROR "KVS_SDK_BASE_DIR is not defined")
endif ()

set(KVS_SDK_INCLUDE_DIR "${KVS_SDK_BASE_DIR}/dist/src")
set(KVS_SDK_BIN_DIR "${KVS_SDK_BASE_DIR}/dist/lib")

find_package(PkgConfig REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
#find_package(Threads REQUIRED)

include_directories(
        ${Python3_INCLUDE_DIRS}
        ${KVS_SDK_BASE_DIR}/src
        ${KVS_SDK_BASE_DIR}/src/JNI/include
        ${KVS_SDK_BASE_DIR}/dependency/libkvscproducer/kvscproducer-src/dependency/libkvspic/kvspic-src/src/client/include
        ${KVS_SDK_BASE_DIR}/dependency/libkvscproducer/kvscproducer-src/dependency/libkvspic/kvspic-src/src/common/include
        ${KVS_SDK_BASE_DIR}/dependency/libkvscproducer/kvscproducer-src/dependency/libkvspic/kvspic-src/src/utils/include
        ${KVS_SDK_BASE_DIR}/dependency/libkvscproducer/kvscproducer-src/dependency/libkvspic/kvspic-src/src/mkvgen/include
        ${KVS_SDK_BASE_DIR}/dependency/libkvscproducer/kvscproducer-src/dependency/libkvspic/kvspic-src/src/view/include
        ${KVS_SDK_BASE_DIR}/dependency/libkvscproducer/kvscproducer-src/dependency/libkvspic/kvspic-src/src/heap/include
        ${KVS_SDK_BASE_DIR}/dependency/libkvscproducer/kvscproducer-src/dependency/libkvspic/kvspic-src/src/state/include
        ${KVS_SDK_BASE_DIR}/dependency/libkvscproducer/kvscproducer-src/src/include
        ${CMAKE_SOURCE_DIR}/pykvssdk
)

link_directories(
        ${Python3_LIBRARY_DIRS}
        ${KVS_SDK_BIN_DIR}
)

file(GLOB PYTHON_FILES "pykvssdk/*.cpp" "pykvssdk/*.h")

source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCE_FILES} ${HEADER_FILES} ${PYTHON_FILES})

pybind11_add_module(${python_module_name} SHARED
        ${SOURCE_FILES}
        ${HEADER_FILES}
        ${PYTHON_FILES}
)

target_link_libraries(${python_module_name} PRIVATE
        ${Python3_LIBRARIES}
        pybind11::module
        pybind11::lto
        cproducer
        KinesisVideoProducer
        log4cplus
)

pybind11_extension(${python_module_name})
pybind11_strip(${python_module_name})

install(TARGETS ${python_module_name} DESTINATION .)
