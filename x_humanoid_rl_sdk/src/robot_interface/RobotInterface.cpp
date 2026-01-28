#include "RobotInterface.h"

// Singleton implementation
class RobotInterfaceImpl : public RobotInterface {
public:
  void Init() override {
    // ou can establish device connections and subscribe to ROS topics here.
  }

  void GetState(double t, RobotData &robot_data) override {
    // robot_data is updated in the main loop, you can leave it empty or add logging
  }

  void SetCommand(RobotData &robot_data) override {
    // Commands are published in the main loop, you can leave it empty here
  }

  void DisableAllJoints() override {
    error_state_ = true;
    robot_data_.q_d_.setZero();
    robot_data_.q_dot_d_.setZero();
    robot_data_.tau_d_.setZero();
    robot_data_.joint_kp_p_.setZero();
    robot_data_.joint_kd_p_.setZero();
  }

private:
  RobotData robot_data_;
};

static RobotInterfaceImpl *instance = nullptr;

RobotInterface *get_robot_interface() {
  if (!instance) instance = new RobotInterfaceImpl();
  return instance;
}
