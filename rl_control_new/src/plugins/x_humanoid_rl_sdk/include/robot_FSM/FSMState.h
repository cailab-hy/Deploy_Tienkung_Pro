#ifndef FSM_STATE_H_
#define FSM_STATE_H_

#include "RobotInterface.h"
#include "Joystick.h"

// #define MWR

enum FSMStateName
{
  STOP = 0, // Stopped, remote control C
  ZERO,    // Zero position, remote control D
  MLP,    // humanoidgym walking policy, remote control undefined
};

class FSMState
{
public:
  explicit FSMState(RobotData *robot_data) { robot_data_ = robot_data; }
  virtual ~FSMState() = default;

  // Behavior to be carried out when entering a state
  virtual void OnEnter() = 0;
  // Run the normal behavior for the state
  virtual void Run(xbox_flag &flag) = 0;
  // Manages state specific transitions
  virtual FSMStateName CheckTransition(const xbox_flag &flag) = 0;
  // Behavior to be carried out when exiting a state
  virtual void OnExit() = 0;

  FSMStateName current_state_name_;

protected:
  double dt_ = 0.0025;    // 1/0.0025 = 400Hz
  int freq_ratio_ = 8;    // ??? 400Hz/50Hz = 8  
  // int joint_num_ = 20; // Tienkung Lite 
  int joint_num_ = 30;    // Tienkung Pro
  
  RobotData *robot_data_;
};

#endif // FSM_STATE_H_
