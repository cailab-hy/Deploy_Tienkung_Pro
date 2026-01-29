cmake_minimum_required(VERSION 3.0.2)
project(x_humanoid_rl_sdk)

# Set policy for IN_LIST operator
cmake_policy(SET CMP0057 NEW)

add_compile_options(-std=c++17)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(pluginlib REQUIRED)
find_package(bodyctrl_msgs REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(sensor_msgs REQUIRED)
# 查找OpenVINO包
find_package(OpenVINO REQUIRED)
find_package (Eigen3 3.0.0 REQUIRED)
find_package (pinocchio REQUIRED)



include_directories(
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
  "${CMAKE_CURRENT_SOURCE_DIR}/../rl_control_new/src/plugins/JoyStick/include"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/robot_interface"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/robot_FSM"

)

include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${OpenVINO_INCLUDE_DIRS})


link_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/lib
     /usr/lib  # 添加系统库目录
)

file(GLOB_RECURSE SRC_FILES
    src/*.cpp
    robot_FSM/*.cpp
    robot_interface/*.cpp
)

add_library(${PROJECT_NAME} SHARED ${SRC_FILES})

target_link_libraries(${PROJECT_NAME}
    yaml-cpp
    openvino
    openvino_c
    pinocchio::pinocchio
)
# ROS2 dependencies
ament_target_dependencies(${PROJECT_NAME}
  rclcpp
  pluginlib
  bodyctrl_msgs
  Eigen3
  sensor_msgs
  pinocchio
)
# 设置目标目录
set(PLUGIN_TARGET_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../rl_control_new/src/plugins/${PROJECT_NAME})

# 收集需要复制的文件
file(GLOB XHS_LIBS ${CMAKE_CURRENT_SOURCE_DIR}/lib/*)

# 安装库文件到目标目录
install(FILES ${XHS_LIBS}
    DESTINATION ${PLUGIN_TARGET_DIR}/lib
)

# 安装生成的共享库
install(TARGETS ${PROJECT_NAME}
    LIBRARY DESTINATION ${PLUGIN_TARGET_DIR}/lib
)

# 安装头文件 - robot_FSM中的指定文件
install(
    FILES 
        include/robot_FSM/FSMState.h
        include/robot_FSM/RobotFSM.h
    DESTINATION ${PLUGIN_TARGET_DIR}/include/robot_FSM
)

# 安装头文件 - robot_interface中的指定文件
install(
    FILES include/robot_interface/RobotInterface.h
    DESTINATION ${PLUGIN_TARGET_DIR}/include/robot_interface
)

# 如果有配置文件，也可以安装
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/config)
    install(
        DIRECTORY config/
        DESTINATION ${PLUGIN_TARGET_DIR}/config
    )
endif()

# Standard ament package
ament_package()
