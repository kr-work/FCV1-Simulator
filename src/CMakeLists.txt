cmake_minimum_required(VERSION 3.12)
project(simulator)
list(APPEND CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../extern/box2d/build/lib/cmake/box2d")
set(CMAKE_CXX_STANDARD 17)

find_package(pybind11 REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Box2D REQUIRED)

link_directories("/workspaces/FCV1-Simulator/extern/box2d/build/bin")

add_library(simulator MODULE fcv1_simulator.cpp)
target_include_directories(simulator PRIVATE "${CMAKE_SOURCE_DIR}/../extern/Box2D/include")
target_link_libraries(simulator PRIVATE pybind11::module)
target_link_libraries(simulator PRIVATE box2d)
target_link_libraries(simulator PRIVATE OpenMP::OpenMP_CXX)  # OpenMPをリンクする
set_target_properties(simulator PROPERTIES PREFIX "")