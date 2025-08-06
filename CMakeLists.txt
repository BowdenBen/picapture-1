cmake_minimum_required(VERSION 3.12)
project(PiCapture)
set(CMAKE_CXX_STANDARD 20)
add_compile_options(-O2 -Wall)
find_package(OpenCV REQUIRED)
add_executable(PiCapture src/motion_detector.cpp)

target_include_directories(PiCapture
  PRIVATE
    ${CMAKE_SOURCE_DIR}
)

target_link_libraries(PiCapture PRIVATE ${OpenCV_LIBS})
