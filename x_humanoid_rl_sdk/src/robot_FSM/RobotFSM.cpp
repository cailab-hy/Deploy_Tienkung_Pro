#include "RobotFSM.h"
#include "RobotInterface.h"
#include <iostream>
#include "FSMStateImpl.h"

enum class FSMMode
{
  NORMAL,
  CHANGE
};

struct FSMStateList
{
  FSMState *stop = nullptr;
  FSMState *zero = nullptr;
  FSMState *mlp = nullptr;

};

class RobotFSMImpl : public RobotFSM
{
public:
  explicit RobotFSMImpl(RobotData &robot_data)
      : RobotFSM(robot_data), robot_data_(robot_data)
  {
    InitStates(); 
    FSMInit(robot_data_.config_file_);
    current_state_ = state_list_.stop; // 初始是stop (Initially, it is in a stopped state.)
    current_state_name_ = FSMStateName::STOP;
    current_state_->OnEnter();
    mode_ = FSMMode::NORMAL;
  }

  ~RobotFSMImpl() override
  {
    delete state_list_.stop;
    delete state_list_.zero;
    delete state_list_.mlp;

  }

  void RunFSM(xbox_flag &flag) override
  {
    // std::cout << flag.fsm_state_command << std::endl;
    // 急停逻辑 (Emergency stop logic)
    if (flag.is_disable)
    {
      disable_joints_ = true;
      std::cout << "[FSM] Emergency Stop Triggered." << std::endl;
      return;
    }
    else
    {
      disable_joints_ = false;
    }

    // 安全检测（例如速度异常、NaN (Safety check (e.g., abnormal speed, NaN))
    if (!CheckSafety(flag))
    {
      std::cerr << "[FSM] Safety Check Failed, disabling control!" << std::endl;
      return;
    }


    // 正常运行或状态切换 (Normal operation or state switching)
    switch (mode_)
    {
    case FSMMode::NORMAL:
      current_state_->Run(flag);                                // 运行FSM (Run the FSM current state)
      next_state_name_ = current_state_->CheckTransition(flag); // 检测是否发生变化 (Check for state transitions)
      if (next_state_name_ != current_state_name_)
      {
        std::cout << "current_state_name_1: " << current_state_name_ << std::endl;
        std::cout << "next_state_name_1: " << next_state_name_ << std::endl;
        mode_ = FSMMode::CHANGE;
        next_state_ = GetState(next_state_name_);
      }
      break;

    case FSMMode::CHANGE:
      current_state_->OnExit();
      current_state_ = next_state_;
      current_state_name_ = next_state_name_;
      current_state_->OnEnter();
      mode_ = FSMMode::NORMAL;
      break;
    }
  }

  FSMStateName getCurrentState() override
  {
    return current_state_name_;
  }

private:
  RobotData &robot_data_;
  FSMMode mode_;
  FSMState *current_state_ = nullptr;
  FSMState *next_state_ = nullptr;
  FSMStateName current_state_name_;
  FSMStateName next_state_name_;
  FSMStateList state_list_;

  void InitStates()
  {
    state_list_.stop = new StateStop(&robot_data_);
    state_list_.zero = new StateZero(&robot_data_);
    state_list_.mlp = new StateMLP(&robot_data_);

  }

  FSMState *GetState(FSMStateName state_name)
  {
    switch (state_name)
    {
    case FSMStateName::STOP:
      return state_list_.stop;
    case FSMStateName::ZERO:
      return state_list_.zero;
    case FSMStateName::MLP:
      return state_list_.mlp;
    default:
      return state_list_.zero;
    }
  }

  bool CheckSafety(const xbox_flag &flag)
  {
    if (robot_data_.q_a_.hasNaN() || robot_data_.q_dot_a_.hasNaN())
      return false;
    if (std::abs(flag.x_speed_command) > 5.0 ||
        std::abs(flag.y_speed_command) > 5.0 ||
        std::abs(flag.yaw_speed_command) > 5.0)
      return false;
    return true;
  }
};

RobotFSM *get_robot_FSM(RobotData &robot_data)
{
  return new RobotFSMImpl(robot_data);
}
