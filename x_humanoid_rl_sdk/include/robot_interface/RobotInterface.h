/**
 * @file RobotInterface.h
 * @brief
 * @version 1.0
 * @date 2024-03-06
 *
 * @copyright Copyright (c) 2024 x-humanoid_
 *
 */

#ifndef ROBOT_INTERFACE_H_
#define ROBOT_INTERFACE_H_

#include <Eigen/Dense>

struct RobotData {
  // (Tienkung Lite) 26 = floating base xyz rpy (6) + joints (20)
  // (Tienkung Pro) 36 = floating base xyz rpy (6) + joints (30)
  
  // Eigen::VectorXd q_a_ = Eigen::VectorXd::Zero(26);
  // Eigen::VectorXd q_dot_a_ = Eigen::VectorXd::Zero(26);
  // Eigen::VectorXd tau_a_ = Eigen::VectorXd::Zero(26);

  Eigen::VectorXd q_a_ = Eigen::VectorXd::Zero(36);
  Eigen::VectorXd q_dot_a_ = Eigen::VectorXd::Zero(36);
  Eigen::VectorXd tau_a_ = Eigen::VectorXd::Zero(36);

  // Eigen::VectorXd q_d_ = Eigen::VectorXd::Zero(26);
  // Eigen::VectorXd q_dot_d_ = Eigen::VectorXd::Zero(26);
  // Eigen::VectorXd tau_d_ = Eigen::VectorXd::Zero(26);

  Eigen::VectorXd q_d_ = Eigen::VectorXd::Zero(36);
  Eigen::VectorXd q_dot_d_ = Eigen::VectorXd::Zero(36);
  Eigen::VectorXd tau_d_ = Eigen::VectorXd::Zero(36);

  // Eigen::VectorXd joint_kp_p_ = Eigen::VectorXd::Zero(20);
  // Eigen::VectorXd joint_kd_p_ = Eigen::VectorXd::Zero(20);

  Eigen::VectorXd joint_kp_p_ = Eigen::VectorXd::Zero(30);
  Eigen::VectorXd joint_kd_p_ = Eigen::VectorXd::Zero(30);

  Eigen::VectorXd imu_data_ = Eigen::VectorXd::Zero(9); 
  double gait_a = 0.0;
  double time_now_ = 0.0;  // time in seconds
  std::string config_file_ = "";

  bool pos_mode_ = true;

};


class RobotInterface {
 public:
  RobotInterface() = default;
  virtual ~RobotInterface() = default;
  virtual void Init() = 0;
  virtual void GetState(double t, RobotData &robot_data) = 0;
  virtual void SetCommand(RobotData &robot_data) = 0;
  virtual void DisableAllJoints() = 0;

  bool error_state_ = false;
};

RobotInterface *get_robot_interface();

#endif //ROBOT_INTERFACE_H_
