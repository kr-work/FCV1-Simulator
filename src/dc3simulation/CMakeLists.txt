cmake_minimum_required(VERSION 3.12)
project(simulator)
list(APPEND CMAKE_PREFIX_PATH "/home/vscode/.local/lib/python3.11/site-packages/pybind11/share/cmake/pybind11")
set(CMAKE_CXX_STANDARD 17)

find_package(pybind11 REQUIRED)
find_package(OpenMP REQUIRED)

add_library(simulator MODULE simulator.cpp)
target_link_libraries(simulator PRIVATE pybind11::module)
target_link_libraries(simulator PRIVATE box2d)
target_link_libraries(simulator PRIVATE OpenMP::OpenMP_CXX)  # OpenMPをリンクする
set_target_properties(simulator PROPERTIES PREFIX "")