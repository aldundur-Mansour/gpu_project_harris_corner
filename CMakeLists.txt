cmake_minimum_required(VERSION 3.26)
project(CPROJECT)

set(CMAKE_CXX_STANDARD 17)

#find_library(OpenCV)
find_package(OpenCV)

include_directories(${OpenCV_INCLUDE_DIRS})
set(CMAKE_CXX_STANDARD 11)


add_executable(CPROJECT main.cpp)

target_link_libraries(CPROJECT ${OpenCV_LIBS})
