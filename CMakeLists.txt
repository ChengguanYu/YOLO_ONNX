cmake_minimum_required(VERSION 3.25)
project(yolo)

set(CMAKE_CXX_STANDARD 14)
find_package(OpenCV 4 REQUIRED)
add_executable(yolo main.cpp yolo.cpp yolo.h)
target_link_libraries(yolo ${OpenCV_LIBS})