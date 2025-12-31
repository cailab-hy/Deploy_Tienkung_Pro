# Tienkung Humanoid Robot Reinforcement Learning Control Library (rl_control_new)

This is a ROS2-based reinforcement learning control library for humanoid robots, designed to control the Tienkung series of humanoid robots. This library uses reinforcement learning algorithms to implement robot motion control, supporting both simulation and real robot environments.

## Directory Structure

```
rl_control_new/
├── CMakeLists.txt              # CMake compilation configuration file
├── package.xml                 # ROS2 package description file
├── rlctrlnew_plugin.xml        # Plugin description file
├── README.md                   # This documentation file
├── config/                     # Configuration files directory
│   ├── policy/                 # Policy network model files
│   │   └── policy.xml          # Policy network model
│   └── tg22_config.yaml        # Robot configuration file
├── include/                    # Header files directory
│   └── broccoli/               # Broccoli library header files
├── launch/                     # ROS2 launch files
│   └── rl.launch.py            # Launch script
├── src/                        # Source code directory
│   ├── common/                 # Common utility code
│   │   └── util/               # Utility classes
│   └── plugins/                # Plugins directory
│       ├── JoyStick/           # Joystick control plugin
│       ├── p2s/                # Serial-parallel conversion library
│       ├── rl_control_new/     # Main control plugin
│       │   ├── include/        # Header files directory
│       │   │   ├── RLControlNewPlugin.h  # Main control logic interface
│       │   │   └── bodyIdMap.h # Robot joint ID mapping
│       │   └── src/            # Source files directory
│       │       └── RLControlNewPlugin.cpp  # Main control logic implementation
│       └── x_humanoid_rl_sdk/  # Robot SDK library
└── README.md                   # Documentation
```

## Dependencies

### System Dependencies
- Ubuntu 22.04 LTS
- ROS2 Humble
- C++17 compiler
- CMake 3.8 or higher

### ROS2 Package Dependencies
- rclcpp
- rclpy
- std_msgs
- sensor_msgs
- bodyctrl_msgs
- pluginlib
- eigen3_cmake_module
- Eigen3
- yaml-cpp
- ament_index_cpp
- joy (for joystick control)

### Third-party Library Dependencies
- spdlog (logging library)
- fmt (formatting library)
- x_humanoid_rl_sdk (robot reinforcement learning SDK)
- OpenVINO (for neural network inference, default)

## Compilation Instructions

### 1. Environment Setup

Ensure ROS2 and basic dependencies are installed:

```bash
# Update system
sudo apt update

# Install basic dependencies
sudo apt install -y cmake build-essential git

# Install ROS2 dependencies
sudo apt install -y ros-${ROS_DISTRO}-rclcpp ros-${ROS_DISTRO}-rclpy
sudo apt install -y ros-${ROS_DISTRO}-std-msgs ros-${ROS_DISTRO}-sensor-msgs
sudo apt install -y ros-${ROS_DISTRO}-pluginlib ros-${ROS_DISTRO}-eigen3-cmake-module
sudo apt install -y ros-${ROS_DISTRO}-ament-cmake ros-${ROS_DISTRO}-ament-index-cpp

# Install third-party library dependencies
sudo apt install -y libyaml-cpp-dev libspdlog-dev libfmt-dev

# Install joystick control dependencies
sudo apt install -y ros-${ROS_DISTRO}-joy
```

### 2. Compile the Project

```bash
# Enter colcon workspace
cd ~/tklab_ws

# Compile rl_control_new package and its dependencies
colcon build --packages-select rl_control_new x_humanoid_rl_sdk

# Or compile all packages
colcon build
```

### 3. Set Environment Variables

```bash
# Source workspace
source install/setup.bash
```

## Usage Instructions

### Start Control Node

Launch the reinforcement learning control node using the launch file:

```bash
# Basic launch command
ros2 launch rl_control_new rl.launch.py
```

### Configuration File Description

The configuration file [tg22_config.yaml](./config/tg22_config.yaml) contains the following main parameters:

- `motor_num`: Number of motors
- `actions_size`: Action space dimension
- `dt`: Control period
- `ct_scale`: Control parameter scaling factor
- `joint_kp_p`: Joint position control P gain
- `joint_kd_p`: Joint position control D gain
- `simulation`: Whether it is a simulation environment
- `mlp.path`: Policy network model path

### Node Information

The main control node `rl_control_new::RLControlNewPlugin` provides the following functions:

1. Read robot status (joint positions, velocities, IMU data, etc.)
2. Run reinforcement learning policy network
3. Calculate and publish joint control commands
4. Support state machine management (STOP, ZERO, MLP states)
5. Support joystick control input

### Main Components

#### RLControlNewPlugin Class
This is the main control class, inheriting from rclcpp::Node, responsible for:
- ROS2 node management
- Robot state subscription and control command publishing
- Reinforcement learning policy execution
- Communication with robot hardware interface

#### x_humanoid_rl_sdk Plugin
Provides core robot control functionality:
- Finite state machine implementation (FSM)
- Robot interface abstraction
- OpenVINO model inference support

## Joystick Control Logic

The system supports Xbox controller wired mode and the robot's built-in controller (Yunzhuo): The two controllers cannot be used simultaneously!!!

### Mode 1: Xbox Control Mode (Wired Connection)
Startup method:
```bash
ros2 run joy joy_node --ros-args --remap joy:=sbus_data # Ensure it's in the same domain_id as the robot
```
Controller button function allocation:

| Button | Function |
|--------|----------|
| A key | Switch to MLP (Machine Learning Policy) control mode |
| X key | Switch to ZERO (Zeroing) control mode |
| Y key | Switch to STOP (Stop) control mode |
| Left joystick | Control robot forward/backward and left/right movement |
| Right joystick | Control robot turning (left/right) |

Control logic description:
1. The robot's initial state is STOP mode. After startup, press ZERO mode to ensure all joints of the robot return to the set zero position
2. Press A key to switch to MLP mode, the robot starts walking
3. Press X key to return to ZERO mode, the robot returns to the initial posture
4. Press Y key to enter STOP mode, maintaining the current posture
5. State switching flow: STOP -> ZERO -> MLP -> STOP

### Mode 2: Yunzhuo Control Mode
Controller button function allocation:

| Button | Function |
|--------|----------|
| A key + G key (switch to middle zero position) | Switch to MLP (Machine Learning Policy) control mode |
| D key | Switch to ZERO (Zeroing) control mode |
| C key | Switch to STOP (Stop) control mode |
| Left joystick | Control robot forward/backward and left/right movement |
| Right joystick | Control robot turning (left/right) |

Control logic description:
1. The robot's initial state is STOP mode. After startup, press ZERO mode to ensure all joints of the robot return to the set zero position
2. Press A key + G key (switch to middle zero position) to switch to MLP mode, the robot starts walking
3. Press D key to return to ZERO mode, the robot returns to the initial posture
4. Press C key to enter STOP mode, maintaining the current posture
5. State switching flow: STOP -> ZERO -> MLP -> STOP

## Troubleshooting

### Compilation Issues

1. Ensure all dependency packages are properly installed
2. Check if Eigen3 is properly installed:
   ```bash
   sudo apt install libeigen3-dev
   ```
3. Ensure yaml-cpp library is properly installed:
   ```bash
   sudo apt install libyaml-cpp-dev
   ```

### Runtime Issues

1. Ensure configuration file path is correct
2. Check if policy network model file exists
3. Verify ROS2 environment variables are properly set
4. Confirm robot hardware connection is normal

### Common Errors

1. If you encounter an error that the [bodyctrl_msgs] package cannot be found, please ensure the package is installed:
   ```bash
   # Need to obtain and compile the bodyctrl_msgs package from the corresponding repository
   ```

## Contact

If you have any questions, please contact the project maintenance team.
