cmake_minimum_required(VERSION 3.23)

cmake_policy(SET CMP0126 NEW)
cmake_policy(SET CMP0077 NEW)
cmake_policy(SET CMP0102 NEW)

project(neuralnet)

set(EXPORT_COMPILE_COMMANDS true)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED true)

add_library(neuralnet neuralnet/neuralnet.cpp)

add_executable(main main.cpp)
target_include_directories(main PUBLIC mnist-master/mnist-master/include)
target_link_libraries(main neuralnet)
