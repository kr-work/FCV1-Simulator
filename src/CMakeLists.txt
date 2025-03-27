cmake_minimum_required(VERSION 3.12)
project(simulator)
list(APPEND CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../extern/box2d/build/lib/cmake/box2d")
set(CMAKE_CXX_STANDARD 17)
set(box2d_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../extern/box2d/build/lib/cmake/box2d")

find_package(pybind11 REQUIRED)
find_package(box2D REQUIRED)

link_directories("/workspaces/FCV1-Simulator/extern/box2d/build/bin")

add_library(simulator MODULE fcv1_simulator.cpp)
target_include_directories(simulator PRIVATE "${CMAKE_SOURCE_DIR}/../extern/box2D/include")
target_link_libraries(simulator PRIVATE pybind11::module)
target_link_libraries(simulator PRIVATE box2d)
set_target_properties(simulator PROPERTIES PREFIX "")