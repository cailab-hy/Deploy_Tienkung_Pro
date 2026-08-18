cmake_minimum_required(VERSION 3.8)
project(hdmi_inference)

add_compile_options(-std=c++17)

# ROS2
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(eigen3_cmake_module REQUIRED)
find_package(Eigen3 REQUIRED)

# yaml-cpp
find_package(PkgConfig REQUIRED)
pkg_check_modules(YAML_CPP REQUIRED yaml-cpp)

# spdlog / fmt
find_package(spdlog REQUIRED)
find_package(fmt REQUIRED)

# ONNX Runtime
set(ONNXRUNTIME_ROOT_DIR "/usr/local" CACHE PATH "ONNX Runtime install prefix")
find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
  HINTS ${ONNXRUNTIME_ROOT_DIR}/include/onnxruntime
        ${ONNXRUNTIME_ROOT_DIR}/include
        /usr/include/onnxruntime
)
find_library(ONNXRUNTIME_LIB onnxruntime
  HINTS ${ONNXRUNTIME_ROOT_DIR}/lib
        /usr/lib
        /usr/lib/aarch64-linux-gnu
)
if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIB)
  message(FATAL_ERROR "ONNX Runtime not found. Set ONNXRUNTIME_ROOT_DIR.")
endif()
message(STATUS "ONNX Runtime include: ${ONNXRUNTIME_INCLUDE_DIR}")
message(STATUS "ONNX Runtime lib: ${ONNXRUNTIME_LIB}")

# OpenCV (for depth image resize)
find_package(OpenCV REQUIRED)

# zlib (for cnpy)
find_package(ZLIB REQUIRED)

# Include directories
include_directories(
  include
  third_party
  ${EIGEN3_INCLUDE_DIR}
  ${ONNXRUNTIME_INCLUDE_DIR}
  ${OpenCV_INCLUDE_DIRS}
)

# Executable
add_executable(hdmi_inference_node
  src/hdmi_inference_node.cpp
  src/HdmiPolicyRunner.cpp
  src/OnnxInference.cpp
  src/ObservationBuilder.cpp
  src/MotionData.cpp
  src/Joystick.cpp
  third_party/cnpy.cpp
)

target_link_libraries(hdmi_inference_node
  spdlog::spdlog
  fmt::fmt
  ${YAML_CPP_LIBRARIES}
  ${ONNXRUNTIME_LIB}
  ZLIB::ZLIB
  ${OpenCV_LIBS}
)

ament_target_dependencies(hdmi_inference_node
  rclcpp
  std_msgs
  sensor_msgs
  Eigen3
)

# Install
install(TARGETS hdmi_inference_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
  FILES_MATCHING PATTERN "*.launch.py"
)

install(DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
)

ament_package()
