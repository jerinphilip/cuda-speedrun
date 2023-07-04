cmake_minimum_required(VERSION 3.5.1)
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

project(fcl LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_language("CUDA")
find_package(CUDA 8.0 REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

add_subdirectory(3rd-party)
include_directories(3rd-party)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory(fcl)
