cmake_minimum_required(VERSION 3.8)
project(rl_control_new)

# Set CMAKE policies for compatibility
if(POLICY CMP0057)
  cmake_policy(SET CMP0057 NEW)
endif()

## Compile as C++17, supported in ROS2
add_compile_options(-std=c++17)

## Find ament packages  
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(bodyctrl_msgs REQUIRED)
find_package(pluginlib REQUIRED)
find_package(eigen3_cmake_module REQUIRED)
find_package(Eigen3 REQUIRED)


# Find yaml-cpp
find_package(PkgConfig REQUIRED)
pkg_check_modules(YAML_CPP REQUIRED yaml-cpp)

# Find ament_index_cpp for ROS2 package path resolution
find_package(ament_index_cpp REQUIRED)

# Find spdlog and fmt
find_package(spdlog REQUIRED)
find_package(fmt REQUIRED)

# Platform specific settings
if(UNIX AND NOT APPLE)
  set(OS "linux")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ")
  set(OS_LIBS pthread rt)
elseif(WIN32)
  set(OS "windows")
  add_definitions(-DWIN32_LEAN_AND_MEAN)
elseif(APPLE)
  set(OS "apple")
endif()

# Include directories
include_directories(
  include
  src/plugins
  src/plugins/rl_control_new/include
  src/plugins/p2s/include
  src/plugins/JoyStick/include
  src/plugins/x_humanoid_rl_sdk/include
  src/plugins/x_humanoid_rl_sdk/include/robot_interface
  src/plugins/x_humanoid_rl_sdk/include/robot_FSM
  ${EIGEN3_INCLUDE_DIR}
)

link_directories("/usr/local/lib")
include_directories("/usr/local/include")
link_directories("/usr/lib")
include_directories("/usr/include")

find_package(Eigen3 3.0.0 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(src/common/)

# Link directories
link_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/src/plugins/x_humanoid_rl_sdk/lib
)
link_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/src/plugins/p2s/lib
)
# Set install directory (for backward compatibility)
set(INSTALL_DIR ${PROJECT_SOURCE_DIR}/../../install/lib)

# Create the plugin library
add_library(rlctrlnew_plugin SHARED
  src/plugins/rl_control_new/src/RLControlNewPlugin.cpp
  src/plugins/JoyStick/src/Joystick.cpp
)

# Link libraries
target_link_libraries(rlctrlnew_plugin
  spdlog::spdlog
  fmt::fmt
  ${YAML_CPP_LIBRARIES}
  x_humanoid_rl_sdk
  funcSPTrans

)

# Check if fast_ros library exists and link if available
if(EXISTS "${INSTALL_DIR}/libfast_ros.so")
  target_link_libraries(rlctrlnew_plugin ${INSTALL_DIR}/libfast_ros.so)
endif()

# Link ament_index_cpp if found
if(ament_index_cpp_FOUND)
  target_link_libraries(rlctrlnew_plugin ament_index_cpp::ament_index_cpp)
endif()

# ROS2 dependencies
ament_target_dependencies(rlctrlnew_plugin
  rclcpp
  rclcpp_components
  std_msgs
  sensor_msgs
  bodyctrl_msgs
  pluginlib
  Eigen3
)

# Install the plugin library
install(TARGETS rlctrlnew_plugin
ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

# Install header files
install(DIRECTORY include/
  DESTINATION include
  FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)

# Install launch files
install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
  FILES_MATCHING PATTERN "*.launch" PATTERN "*.launch.py"
)

# Install config files
install(DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
  # FILES_MATCHING PATTERN "*.yaml" PATTERN "*.yml" PATTERN "*.json"
)

# Install plugin description file
install(FILES rlctrlnew_plugin.xml
  DESTINATION share/${PROJECT_NAME}
)

# Collect and install x_humanoid_rl_sdk library files
file(GLOB XHS_LIBS ${CMAKE_CURRENT_SOURCE_DIR}/src/plugins/x_humanoid_rl_sdk/lib/*)
if(XHS_LIBS)
  install(FILES ${XHS_LIBS}
    DESTINATION lib
  )
endif()

# Collect and install p2s library files
file(GLOB P2S_LIBS ${CMAKE_CURRENT_SOURCE_DIR}/src/plugins/p2s/lib/*)
if(P2S_LIBS)
  install(FILES ${P2S_LIBS}
    DESTINATION lib
  )
endif()

# Export plugin
pluginlib_export_plugin_description_file(${PROJECT_NAME} rlctrlnew_plugin.xml)

# Register component with rclcpp_components
rclcpp_components_register_nodes(rlctrlnew_plugin "rl_control_new::RLControlNewPlugin")
set(node_plugins "${node_plugins}rl_control_new::RLControlNewPlugin;$<TARGET_FILE:rlctrlnew_plugin>\n")

# Testing
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

# Export dependencies
ament_export_dependencies(
  rclcpp
  rclcpp_components
  std_msgs
  bodyctrl_msgs
  pluginlib
  Eigen3
)

ament_package()
