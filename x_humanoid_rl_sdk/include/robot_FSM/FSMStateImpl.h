
#ifndef FSM_STATE_IMPL_H_
#define FSM_STATE_IMPL_H_

#include <time.h>
#include "FSMState.h"
#include "BasicFunction.h"
#include "Joystick.h"
#include <openvino/runtime/core.hpp>
#include <fstream>

void FSMInit(const std::string& config_file);

class StateZero : public FSMState {
 public:
  explicit StateZero(RobotData *robot_data);

  void OnEnter() override;

  void Run(xbox_flag& flag) override;

  FSMStateName CheckTransition(const xbox_flag& flag) override;

  void OnExit() override;

 private:
  double timer = 0.;
  double total_time = 2.0;

  bool zero_finish_flag = false;
  bool first_run = false;
  Eigen::VectorXd init_joint_pos = Eigen::VectorXd::Zero(joint_num_);
};

class StateMLP : public FSMState {
 public:
  explicit StateMLP(RobotData *robot_data);

  void OnEnter() override;

  void Run(xbox_flag& flag) override;

  FSMStateName CheckTransition(const xbox_flag& flag) override;

  void OnExit() override;

 private:
  double timer = 0.;

  Eigen::VectorXd input_data_mlp;
  // double action_num = 20;  // Tienkung Lite
  double action_num = 30;     // Tienkung Pro
  Eigen::VectorXd output_data_mlp = Eigen::VectorXd::Zero(action_num);
  Eigen::VectorXd last_action_d = Eigen::VectorXd::Zero(action_num);
  Eigen::VectorXd last_action_dot_d = Eigen::VectorXd::Zero(action_num);

  Eigen::Vector3d command = Eigen::Vector3d::Zero();
  Eigen::Vector3d joystick_command = Eigen::Vector3d::Zero();
  Eigen::Vector3d command_scales = Eigen::Vector3d::Zero();
  Eigen::VectorXd default_dof_pos = Eigen::VectorXd::Zero(action_num);
  Eigen::VectorXd action_last = Eigen::VectorXd::Zero(action_num);
  Eigen::VectorXd joint_pos_last_gait = Eigen::VectorXd::Zero(action_num);
  Eigen::VectorXd joint_vel_last_gait = Eigen::VectorXd::Zero(action_num);

  double obs_scales_lin_vel = 1.0;
  double obs_scales_ang_vel = 1.0;
  double obs_scales_dof_pos = 1.0;
  double obs_scales_dof_vel = 1.0;
  double height_scales = 1.0;
  double action_scales = 0.25;

  bool first_Run = true;
  Eigen::VectorXd baseLinVel_Est = Eigen::VectorXd::Zero(3);
  Eigen::VectorXd baseLinVel_Est_last = Eigen::VectorXd::Zero(3);

  // walk
  double gait_cycle = 0.85; // run 0.5
  double left_phase_ratio = 0.38; // run 0.6
  double right_phase_ratio = 0.38; // run 0.6
  double left_theta_offset = 0.38;  // run 0.6
  double right_theta_offset = 0.88; // run 0.1

  // run
  // double gait_cycle = 0.5; // run 0.5
  // double left_phase_ratio = 0.6; // run 0.6
  // double right_phase_ratio = 0.6; // run 0.6
  // double left_theta_offset = 0.6;  // run 0.6
  // double right_theta_offset = 0.1; // run 0.1

  LowPassFilter *omega_filter;
  double timer_gait = 0.0;
  double gait_d = 0.0;
  float gait_a = 0.0;
  double trans_time = 0.3;

  double trans_time_lower = 0.3; 
  double trans_time_upper = 0.6; 

  double left_phase;
  double right_phase;
  double x_vel_command_offset = 0.0;
  double y_vel_command_offset = 0.0;
  bool run2walk = false;
};

class StateStop : public FSMState {
 public:
  explicit StateStop(RobotData *robot_data);

  void OnEnter() override;

  void Run(xbox_flag& flag) override;

  FSMStateName CheckTransition(const xbox_flag& flag) override;

  void OnExit() override;

 private:
  double timer = 0.;
  bool first_run = false;
  Eigen::VectorXd init_joint_pos = Eigen::VectorXd::Zero(joint_num_);
};

#endif //FSM_STATE_IMPL_H_
