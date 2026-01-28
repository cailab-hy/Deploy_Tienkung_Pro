# Tienkung Humanoid Robot Reinforcement Learning Control Project Collection

This repository contains the main components of the Tienkung series humanoid robot reinforcement learning control system.

## Project Structure

```
Deploy_Tienkung/
├── rl_control_new/         # Reinforcement learning control library
└── x_humanoid_rl_sdk/      # Robot control SDK
```

## Component Introduction

### rl_control_new

A ROS2-based reinforcement learning control library for Tienkung series humanoid robots. This library uses reinforcement learning algorithms to achieve robot motion control, supporting both simulation and real robot environments.

Key Features:
- Robot control strategy based on reinforcement learning
- Support for simulation and real robot deployment
- ROS2 integration for easy system expansion
- Plugin-based architecture design

### x_humanoid_rl_sdk

Tienkunghumanoid robot reinforcement learning control SDK, including finite state machine implementation, robot interface, and control algorithms.

Key Features:
- Finite state machine implementation
- Standardized robot interface definition
- Control algorithm encapsulation
- Easy-to-integrate C++ library

## Usage Instructions

Please refer to the README documents of each sub-project for detailed installation and usage instructions:

- [rl_control_new README](./rl_control_new/README_en.md)
- [x_humanoid_rl_sdk README](./x_humanoid_rl_sdk/README_en.md)

## System Requirements

- Ubuntu 22.04 LTS
- ROS2 Humble
- C++17 compiler
- CMake 3.8+
- Eigen3
- Other dependencies as detailed in each sub-project documentation

## Contact

If you have any questions, please contact the project maintenance team.
