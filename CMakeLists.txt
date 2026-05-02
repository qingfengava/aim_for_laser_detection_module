cmake_minimum_required(VERSION 3.20)
project(aim_for_laser_detection_module LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(LASER_ENABLE_NATIVE_OPT "Enable -march=native in release" ON)
option(LASER_ENABLE_LTO "Enable LTO/IPO in release" ON)
option(LASER_ENABLE_FAST_MATH "Enable -ffast-math in release" OFF)
option(LASER_WARNINGS_AS_ERRORS "Treat warnings as errors" OFF)

if(NOT DEFINED ENV{MVCAM_SDK_PATH} OR "$ENV{MVCAM_SDK_PATH}" STREQUAL "")
    set(ENV{MVCAM_SDK_PATH} "/opt/MVS")
endif()

list(APPEND CMAKE_MODULE_PATH
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake
    /home/robo/wust_vl/cmake
)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(fmt REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(CUDAToolkit QUIET)
find_package(TensorRT QUIET)
find_package(wust_vl REQUIRED)

add_library(laser_build_options INTERFACE)

target_compile_options(laser_build_options INTERFACE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall>
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wextra>
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wpedantic>
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wshadow>
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wconversion>
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wnon-virtual-dtor>
    $<$<CXX_COMPILER_ID:GNU,Clang>:$<$<CONFIG:Release>:-O3>>
    $<$<CXX_COMPILER_ID:GNU,Clang>:$<$<CONFIG:Release>:-DNDEBUG>>
    $<$<CXX_COMPILER_ID:GNU,Clang>:$<$<CONFIG:Release>:-fno-omit-frame-pointer>>
)

if(LASER_ENABLE_NATIVE_OPT)
    target_compile_options(laser_build_options INTERFACE
        $<$<CXX_COMPILER_ID:GNU,Clang>:$<$<CONFIG:Release>:-march=native>>
    )
endif()

if(LASER_ENABLE_FAST_MATH)
    target_compile_options(laser_build_options INTERFACE
        $<$<CXX_COMPILER_ID:GNU,Clang>:$<$<CONFIG:Release>:-ffast-math>>
    )
endif()

if(LASER_WARNINGS_AS_ERRORS)
    target_compile_options(laser_build_options INTERFACE
        $<$<CXX_COMPILER_ID:GNU,Clang>:-Werror>
    )
endif()

if(LASER_ENABLE_LTO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT ipo_supported OUTPUT ipo_error)
    if(ipo_supported)
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
        message(STATUS "IPO/LTO enabled for Release")
    else()
        message(WARNING "IPO/LTO not supported: ${ipo_error}")
    endif()
endif()

set(LASER_SOURCES
    src/modules/config/parameter_hub.cpp
    src/modules/perception/laser_pipeline.cpp
    src/modules/tracking/laser_track_filter.cpp
    src/app/laser_vision_node.cpp
)

add_library(laser_core STATIC ${LASER_SOURCES})

target_include_directories(laser_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/laser_aim/utils
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(laser_core PUBLIC
    laser_build_options
    wust_vl::wust_vl
    yaml-cpp
    fmt::fmt
    Eigen3::Eigen
    ${OpenCV_LIBS}
)

add_executable(laser_aim src/app/main.cpp)
target_link_libraries(laser_aim PRIVATE laser_core)

set_target_properties(laser_aim PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin
)

file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
if(EXISTS ${CMAKE_SOURCE_DIR}/bin/config)
    file(REMOVE_RECURSE ${CMAKE_SOURCE_DIR}/bin/config)
endif()
file(CREATE_LINK ${CMAKE_SOURCE_DIR}/config ${CMAKE_SOURCE_DIR}/bin/config SYMBOLIC)
