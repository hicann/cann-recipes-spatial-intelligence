cmake_minimum_required(VERSION 3.19.0)
project(meta_gauss_render)

include(cmake/config.cmake)
include(cmake/func.cmake)
include(cmake/intf.cmake)

add_subdirectory(${PROJECT_SOURCE_DIR}/kernels)
add_subdirectory(${PROJECT_SOURCE_DIR}/meta_gauss_render/csrc)