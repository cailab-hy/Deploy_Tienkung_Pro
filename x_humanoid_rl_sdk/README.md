# Tienkung Humanoid Robot Reinforcement Learning Control SDK

This is a C++ SDK for humanoid robot reinforcement learning control, including finite state machine implementation, robot interface, and control algorithms.

## Directory Structure

```
x_humanoid_rl_sdk/
├── include/
│   ├── robot_FSM/          # Robot finite state machine implementation
│   └── robot_interface/    # Robot interface definition
├── src/
│   ├── robot_FSM/          # State machine implementation source code
│   └── robot_interface/    # Robot interface implementation source code
├── lib/                    # Third-party library files
├── CMakeLists.txt          # CMake compilation configuration
└── package.xml             # ROS2 package description file
```

## Dependencies

### System Dependencies
- Ubuntu 22.04 LTS
- ROS2 Humble
- C++17 or higher
- CMake 3.0.2 or higher
- Eigen3
- yaml-cpp
- sensor_msgs
- bodyctrl_msgs
- OpenVINO (default, for neural network inference acceleration)

### Third-party Libraries
- OpenVINO (default)

## Compilation Instructions

### 1. Ensure Workspace Structure is Correct
```bash
# Assuming your workspace structure is as follows:
# tklab_ws/src/
# ├── x_humanoid_rl_sdk/
# ├── rl_control_new/
# └── ... other packages
```

### 2. Install Dependencies
```bash
# Ubuntu/Debian systems
sudo apt update
sudo apt install -y cmake build-essential libeigen3-dev libyaml-cpp-dev

# Install ROS 2 dependencies
sudo apt install -y ros-${ROS_DISTRO}-rclcpp ros-${ROS_DISTRO}-pluginlib
sudo apt install -y ros-${ROS_DISTRO}-sensor-msgs

# Install bodyctrl_msgs (if not already installed)
# Need to obtain from corresponding repository

# Install OpenVINO (optional, for neural network inference acceleration)
# Please refer to Intel's official documentation to install OpenVINO
wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor | sudo tee /usr/share/keyrings/intel-sw-products.gpg > /dev/null

echo "deb [signed-by=/usr/share/keyrings/intel-sw-products.gpg] https://apt.repos.intel.com/openvino/2023 ubuntu20 main" | sudo tee /etc/apt/sources.list.d/intel-openvino-2023.list

sudo apt update
sudo apt install openvino
```

### 3. Compile the Project
```bash
cd ~/tklab_ws
colcon build --packages-select x_humanoid_rl_sdk
```

### 4. Set Environment Variables
```bash
source install/setup.bash
```

## Usage Instructions

### Header File Inclusion
```cpp
#include "robot_FSM/RobotFSM.h"
#include "robot_interface/RobotInterface.h"
```

### Main Components

#### 1. Robot State Machine (FSM)
- `FSMState`: Base state class
- `RobotFSM`: Robot state machine manager
- Supports multiple states: STOP (stop), ZERO (zeroing), MLP (reinforcement learning control)

#### 2. Robot Interface
- `RobotInterface`: Robot hardware interface abstraction
- Provides joint control, state reading and other functions
- `RobotData`: Robot data structure, including joint positions, velocities, torques and other information

#### 3. Neural Network Inference
- OpenVINO (default)

### API Reference

#### RobotFSM Class
```cpp
// Get robot state machine instance
RobotFSM* robot_fsm = get_robot_FSM(robot_data);

// Run state machine
robot_fsm->RunFSM(flag);

// Get current state
FSMStateName current_state = robot_fsm->getCurrentState();
```

#### RobotInterface Class
```cpp
// Get robot interface instance
RobotInterface* robot_interface = get_robot_interface();

// Initialize interface
robot_interface->Init();

// Get state
robot_interface->GetState(time, robot_data);

// Set control command
robot_interface->SetCommand(robot_data);

// Disable all joints
robot_interface->DisableAllJoints();
```

### RobotData Structure
```cpp
struct RobotData {
  Eigen::VectorXd q_a_;       // Actual joint positions (26 dimensions: 6 DOF floating base + 20 joints)
  Eigen::VectorXd q_dot_a_;   // Actual joint velocities (26 dimensions)
  Eigen::VectorXd tau_a_;     // Actual joint torques (26 dimensions)
  
  Eigen::VectorXd q_d_;       // Desired joint positions (26 dimensions)
  Eigen::VectorXd q_dot_d_;   // Desired joint velocities (26 dimensions)
  Eigen::VectorXd tau_d_;     // Desired joint torques (26 dimensions)
  
  Eigen::VectorXd joint_kp_p_; // Joint position control P gains (20 dimensions)
  Eigen::VectorXd joint_kd_p_; // Joint position control D gains (20 dimensions)
  
  Eigen::VectorXd imu_data_;  // IMU data (9 dimensions: 3 Euler angles + 3 angular velocities + 3 accelerations)
  double gait_a;              // Gait parameter
  double time_now_;           // Current time
  std::string config_file_;   // Configuration file path
  bool pos_mode_;             // Position control mode flag
};
```

## Notes

1. Ensure compilation and execution in ROS 2 environment
2. YAML configuration file path needs to be configured correctly
3. Policy network model files need to be consistent with the path specified in the configuration file
4. IMU and motor data need to be properly subscribed and processed

## Troubleshooting

### Compilation Errors
- Check if Eigen3 and YAML-CPP are properly installed
- Confirm CMake version meets requirements
- Verify ROS 2 environment variables are set correctly
- Ensure bodyctrl_msgs package is properly installed

### Runtime Errors
- Check if configuration file path is correct
- Confirm model files exist and are in correct format
- Verify hardware connections and data subscription

## Contact

If you have any questions, please contact the project maintenance team.
