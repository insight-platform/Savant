cmake_minimum_required(VERSION 3.12)

project(nvbufsurfacegenerator VERSION "0.0.1" LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--no-undefined" )
set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} -Wl,--no-undefined" )

if(CMAKE_SYSTEM_PROCESSOR STREQUAL aarch64)
    message("Building for ${CMAKE_SYSTEM_PROCESSOR}")
    add_definitions(-DPLATFORM_AARCH64)
elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL x86_64)
    message("Building for ${CMAKE_SYSTEM_PROCESSOR}")
    add_definitions(-DPLATFORM_X86_64)
else()
    message(FATAL_ERROR "Unsupported arhitecture: ${CMAKE_SYSTEM_PROCESSOR}")
endif()

find_package(PkgConfig REQUIRED)
find_package(CUDA REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

if (NOT DEFINED DeepStream_DIR)
    set(DeepStream_DIR /opt/nvidia/deepstream/deepstream)
endif()

pkg_check_modules(GLIB REQUIRED glib-2.0)
pkg_check_modules(GSTREAMER REQUIRED gstreamer-1.0)

include_directories(
        ${GLIB_INCLUDE_DIRS}
        ${GSTREAMER_INCLUDE_DIRS}
        ${CUDA_INCLUDE_DIRS}
        ${DeepStream_DIR}/sources/includes
)

link_directories(
        ${GLIB_LIBRARY_DIRS}
        ${GSTREAMER_LIBRARY_DIRS}
        ${CUDA_LIBRARY_DIRS}
        ${DeepStream_DIR}/lib
)

AUX_SOURCE_DIRECTORY(src SRC_FILES)

set(PROJECT_SOURCE_DIR "src")
set(PROJECT_INCLUDE_DIR "include")

set(SOURCE_FILES
    ${PROJECT_INCLUDE_DIR}/nvbufsurface_generator.h
    ${PROJECT_SOURCE_DIR}/nvbufsurface_generator.cpp
)

source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCE_FILES})

add_library(nvbufsurfacegenerator SHARED ${SOURCE_FILES})
target_link_libraries(
        nvbufsurfacegenerator
        ${GLIB_LIBRARIES}
        ${GSTREAMER_LIBRARIES}
        ${CUDA_LIBRARIES}
)
if (CMAKE_SYSTEM_PROCESSOR STREQUAL x86_64)
    target_link_libraries(
            nvbufsurfacegenerator
            nvdsbufferpool
    )
endif()

target_include_directories(nvbufsurfacegenerator PUBLIC include)

install(TARGETS nvbufsurfacegenerator LIBRARY PUBLIC_HEADER)
