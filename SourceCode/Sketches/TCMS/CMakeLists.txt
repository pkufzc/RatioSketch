
cmake_minimum_required(VERSION 3.14)
project(tcms_cpp)
set(CMAKE_CXX_STANDARD 14)
find_package(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)
# 强制使用 pip 安装的 pybind11
set(pybind11_DIR "$ENV{HOME}/.local/lib/python3.10/site-packages/pybind11/share/cmake/pybind11")
find_package(pybind11 REQUIRED)
find_package(OpenMP)

pybind11_add_module(tcms_cpp tcms_cpp.cpp)
target_include_directories(tcms_cpp PRIVATE ${Python3_INCLUDE_DIRS})
target_link_libraries(tcms_cpp PRIVATE pybind11::module ${Python3_LIBRARIES})
if(OpenMP_CXX_FOUND)
    target_link_libraries(tcms_cpp PRIVATE OpenMP::OpenMP_CXX)
endif()
