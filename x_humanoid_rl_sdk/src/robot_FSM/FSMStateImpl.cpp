#include "FSMStateImpl.h"
#include <yaml-cpp/yaml.h>
#include <ament_index_cpp/get_package_share_directory.hpp>
#ifndef PI
#define PI 3.141592654
#endif

// const int obs_num = 750; // Tienkung Lite
const int obs_num = 1050;   // Tienkung Pro

ov::Core core;
std::shared_ptr<ov::Model> model;
ov::CompiledModel compiled_model;
ov::InferRequest infer_request;
std::vector<float> input_vec(obs_num, 0.);
ov::Tensor ov_in_tensor0(ov::element::f32, ov::Shape{obs_num}, input_vec.data());

// Eigen::VectorXd zero_pos = Eigen::VectorXd::Zero(20); // Tienkung Lite
Eigen::VectorXd zero_pos = Eigen::VectorXd::Zero(30);    // Tienkung Pro

void FSMInit(const std::string& config_file) {
  
  YAML::Node config = YAML::LoadFile(config_file);
  auto relative_path = config["mlp"]["path"].as<std::string>();

  std::string package_path = ament_index_cpp::get_package_share_directory("rl_control_new");
  std::string mlp_path = package_path + relative_path;

  model = core.read_model(mlp_path + ".xml",  mlp_path + ".bin");
  compiled_model = core.compile_model(model, "CPU");
  infer_request = compiled_model.create_infer_request();

  core.set_property("CPU", ov::inference_num_threads(1)); 
  float gait_a = 0.0f;
  infer_request.set_input_tensor(0, ov_in_tensor0);

  for (int i = 0; i < 3; i++) {
    std::cout << "MLP forward: " << i << std::endl;
    infer_request.infer();
  }
  std::cout << "MLP forward succeed!" << std::endl;

  """
  // zero_pos for Tienkung Lite
  zero_pos << 0.0, -0.5, 0.0, 1.0, -0.5, 0.0,
      0.0, -0.5, 0.0, 1.0, -0.5, 0.0,
      0.0, 0.1, 0.0, -0.3,
      0.0, -0.1, -0.0, -0.3;
  """

  // zero_pos for Tienkung Pro
  zero_pos << 0.0, -0.5, 0.0, 1.0, -0.5, -0.5,    // left leg:  "l_hip_roll", "l_hip_pitch", "l_hip_yaw", "l_knee", "l_ankle_pitch", "l_ankle_roll",
    0.0, -0.5, 0.0, 1.0, -0.5, -0.5,              // right leg: "r_hip_roll", "r_hip_pitch", "r_hip_yaw", "r_knee", "r_ankle_pitch", "r_ankle_roll"
    0.0, 0.1, 0.0, -0.3, 0.0, 0.0, 0.0,           // left arm:  "l_shoulder_pitch", "l_shoulder_roll", "l_shoulder_yaw", "l_elbow", "l_wrist_yaw", "l_wrist_pitch", "l_wrist_roll",
    0.0, -0.1, -0.0, -0.3, 0.0, 0.0, 0.0,         // right arm: "r_shoulder_pitch", "r_shoulder_roll", "r_shoulder_yaw", "r_elbow", "r_wrist_yaw", "r_wrist_pitch", "r_wrist_roll"
    0.0, 0.0, 0.0, 0.0;                           // head and waist: "head_roll", "head_pitch", "head_yaw", "waist_yaw"
}

StateZero::StateZero(RobotData *robot_data) : FSMState(robot_data) {
  robot_data_ = robot_data;
  current_state_name_ = FSMStateName::ZERO;
}

void StateZero::OnEnter() {
  timer = 0.;
  init_joint_pos = robot_data_->q_a_.tail(joint_num_);
  std::cout << "enter zero" << std::endl;
}

void StateZero::Run(xbox_flag &flag) {
  if (!first_run){
    init_joint_pos = robot_data_->q_a_.tail(joint_num_);
    std::cout << "init_joint_pos for enter zero: " << init_joint_pos.transpose() << std::endl;
    first_run = true;    
  }
  
  // keep warm
  if ((int) (timer / dt_) % freq_ratio_ == 0) {
    float gait_a = 0.0f;
    infer_request.set_input_tensor(0, ov_in_tensor0);
    infer_request.infer();
  }
  // zero
  Eigen::VectorXd zero = Eigen::VectorXd::Zero(joint_num_);
  Eigen::VectorXd joint_pos = Eigen::VectorXd::Zero(joint_num_);
  Eigen::VectorXd joint_vel = Eigen::VectorXd::Zero(joint_num_);
  Eigen::VectorXd joint_acc = Eigen::VectorXd::Zero(joint_num_);

  if (timer < total_time) {
    FifthPoly(init_joint_pos, zero, zero,
              zero_pos, zero, zero,
              total_time, timer,
              joint_pos, joint_vel, joint_acc);
  } else {
    zero_finish_flag = true;
    joint_pos = zero_pos;
    joint_vel.setZero();
  }
  robot_data_->q_d_.tail(joint_num_) = joint_pos;
  robot_data_->q_dot_d_.tail(joint_num_) = joint_vel;
  robot_data_->tau_d_.setZero();
  robot_data_->pos_mode_ = true;
  timer += dt_;

}

FSMStateName StateZero::CheckTransition(const xbox_flag &flag) {

  if (flag.fsm_state_command == "gotoMLP" && zero_finish_flag) {
    std::cout << "Zero2MLP" << std::endl;
    return FSMStateName::MLP;
  } else if (flag.fsm_state_command == "gotoStop") {
    std::cout << "Zero2Stop" << std::endl;
    return FSMStateName::STOP;
  } else {
    return FSMStateName::ZERO;
  }
}

void StateZero::OnExit() {
  zero_finish_flag = false;
}

StateMLP::StateMLP(RobotData *robot_data) : FSMState(robot_data) {
  robot_data_ = robot_data;
  current_state_name_ = FSMStateName::MLP;

  input_data_mlp = Eigen::VectorXd::Zero(obs_num);
  command_scales[0] = obs_scales_lin_vel;
  command_scales[1] = obs_scales_lin_vel;
  command_scales[2] = obs_scales_ang_vel;
  omega_filter = new LowPassFilter(30, 0.707, 0.0025, 3);
}

void StateMLP::OnEnter() {
  std::cout << "MLPenter" << std::endl;
  timer = 0.;
  timer_gait = 0.;
  gait_d = 0.0;
  gait_a = 0.0;
  trans_time = 0.5;
  left_phase = left_theta_offset;
  right_phase = right_theta_offset;
  """
  // default_dof_pos for tienkung lite
  default_dof_pos << 0.0, -0.5, 0.0, 1.0, -0.5, 0.0,
      0.0, -0.5, -0.0, 1.0, -0.5, 0.0,
      0.0, 0.1, 0.0, -0.3,
      0.0, -0.1, -0.0, -0.3;
  """
  // default_dof_pos for tienkung pro
  default_dof_pos << 0.0, -0.5, 0.0, 1.0, -0.5, -0.5,   // left leg:  "l_hip_roll", "l_hip_pitch", "l_hip_yaw", "l_knee", "l_ankle_pitch", "l_ankle_roll",
      0.0, -0.5, 0.0, 1.0, -0.5, -0.5,                  // right leg: "r_hip_roll", "r_hip_pitch", "r_hip_yaw", "r_knee", "r_ankle_pitch", "r_ankle_roll"
      0.0, 0.1, 0.0, -0.3, 0.0, 0.0, 0.0,               // left arm:  "l_shoulder_pitch", "l_shoulder_roll", "l_shoulder_yaw", "l_elbow", "l_wrist_yaw", "l_wrist_pitch", "l_wrist_roll",
      0.0, -0.1, -0.0, -0.3, 0.0, 0.0, 0.0,             // right arm: "r_shoulder_pitch", "r_shoulder_roll", "r_shoulder_yaw", "r_elbow", "r_wrist_yaw", "r_wrist_pitch", "r_wrist_roll"
      0.0, 0.0, 0.0, 0.0;                               // head and waist: "head_roll", "head_pitch", "head_yaw", "waist_yaw"
  action_last.setZero();
  joint_pos_last_gait = zero_pos;
  joint_vel_last_gait.setZero();
  last_action_d.setZero();
  last_action_dot_d.setZero();
  input_data_mlp.setZero();
  x_vel_command_offset = 0.0;
  y_vel_command_offset = 0.0;
  run2walk = false;

}

void StateMLP::Run(xbox_flag &flag) {
  // command
  joystick_command(0) = flag.x_speed_command;
  joystick_command(1) = flag.y_speed_command;
  joystick_command(2) = flag.yaw_speed_command;


  if (gait_a > 0.5) {
    x_vel_command_offset += flag.x_speed_offset;
    y_vel_command_offset += flag.y_speed_offset;

    joystick_command(0) += x_vel_command_offset;
    joystick_command(1) += y_vel_command_offset;
  }

  double x_command_slope = 0.002;

  if ((fabs(joystick_command(0)) > 0.1)) {
    if ((fabs(joystick_command(0) - command(0)) > x_command_slope)) {
      command(0) += x_command_slope * (joystick_command(0) - command(0)) / fabs(joystick_command(0) - command(0));
    } else {
      command(0) = joystick_command(0);
    }
  } else {
    if ((fabs(joystick_command(0) - command(0)) > 0.0015)) {
      command(0) += 0.0015 * (joystick_command(0) - command(0)) / fabs(joystick_command(0) - command(0));
    } else {
      command(0) = joystick_command(0);
    }
  }
  command(1) = joystick_command(1);
  if ((fabs(joystick_command(2) - command(2)) > 0.001) && (fabs(joystick_command(2)) > 0.1)) {
    command(2) += 0.001 * (joystick_command(2) - command(2)) / fabs(joystick_command(2) - command(2));
  } else {
    command(2) = joystick_command(2);
  }

  // MLP obs

  // imu data trans to world coordinate
  Eigen::Vector3d ypr_a = robot_data_->imu_data_.head(3);
  Eigen::Matrix3d NED_R_YPR_a = Eigen::Matrix3d::Zero();
  Eigen::Vector3d rpy = Eigen::Vector3d::Zero();
  ypr_a(0) = 0.0;

  EulerZYXToMatrix(NED_R_YPR_a, ypr_a);
  MatrixToEulerXYZ(NED_R_YPR_a, rpy);

  // rpy(1) -= 0.01;
  robot_data_->q_a_.segment(3, 3) = rpy;

  Eigen::Matrix3d R_xyz_omega = Eigen::Matrix3d::Identity();
  R_xyz_omega.row(1) = RotX(rpy(0)).row(1);
  R_xyz_omega.row(2) = (RotX(rpy(0)) * RotY(rpy(1))).row(2);
  robot_data_->q_dot_a_.segment(3, 3) = R_xyz_omega.transpose() * NED_R_YPR_a * robot_data_->imu_data_.segment(3, 3);


  // Std vector obs input
  Eigen::Matrix3d Rb_w = Eigen::Matrix3d::Identity();
  Euler_XYZToMatrix(Rb_w, robot_data_->q_a_.segment(3, 3));
  Eigen::Vector3d rpy_a = robot_data_->q_a_.segment(3, 3);
  Eigen::Matrix3d R_xyz_omega_a = Eigen::Matrix3d::Identity();
  R_xyz_omega_a.row(1) = RotX(rpy_a(0)).row(1);
  R_xyz_omega_a.row(2) = (RotX(rpy_a(0)) * RotY(rpy_a(1))).row(2);


  Eigen::Vector3d ang_vel = Rb_w.transpose() * R_xyz_omega_a * robot_data_->q_dot_a_.segment(3, 3) * obs_scales_ang_vel;
  ang_vel = omega_filter->mFilter(ang_vel);

  // 把 Eigen 里的数据写到 input_vec (Write the data from Eigen into input_vec)
  input_vec[0] = ang_vel(0);
  input_vec[1] = ang_vel(1);
  input_vec[2] = ang_vel(2);
  

  Eigen::Vector3d base_z = -Rb_w.transpose().col(2);
  input_vec[3] = base_z(0);
  input_vec[4] = base_z(1);
  input_vec[5] = base_z(2);


  // command
  input_vec[6] = command[0] * command_scales[0];
  input_vec[7] = command[1] * command_scales[1];
  input_vec[8] = command[2] * command_scales[2];

  """
  // Tienkung Lite
  std::vector<int> mujoco_to_isaac_idx = {
    0,  6,  12, 16, 1,  7,  13, 17, 2,  8,
    14, 18, 3,  9,  15, 19, 4,  10, 5,  11
  };
  """
  // Tienkung Pro (https://www.notion.so/ID-2dad117f088d8090811ed17e424d4b7c)
  std::vector<int> mujoco_to_isaac_idx = {
    1,  6,  11, 16, 20, 24, 2,  7,  12, 17, 21, 25, 0,  3,  8,
    13, 4,  9,  14, 18, 22, 26, 28, 5,  10, 15, 19, 23, 27, 29
  };

  // Dof pos
  for (int i = 0; i < joint_num_; i++) {
    int idx = mujoco_to_isaac_idx[i];
    input_vec[9 + i] = (robot_data_->q_a_(6 + idx) - default_dof_pos(idx)) * obs_scales_dof_pos;
  }

  // Dof vel
 for (int i = 0; i < joint_num_; i++) {
    int idx = mujoco_to_isaac_idx[i];
    input_vec[29 + i] = robot_data_->q_dot_a_(6 + idx) * obs_scales_dof_vel;
  }

  // actions
  clip(action_last, -100.0, 100.0);
  for (int i = 0; i < action_num; i++) {
      input_vec[49 + i] = action_last(i);
  }

  // gait phase
  Eigen::VectorXd gait = gait_phase(timer_gait, gait_cycle,
                                    left_theta_offset,
                                    right_theta_offset,
                                    left_phase_ratio,
                                    right_phase_ratio);
  for (int i = 0; i < 6; i++) {
      input_vec[69 + i] = gait(i);
  }

  if ((int)(timer / dt_) % freq_ratio_ == 0) {
      if (first_Run) {
          first_Run = false;
      } else {
          std::vector<float> new_obs(input_vec.begin(), input_vec.begin() + 75);
          std::move(input_vec.begin() + 75, input_vec.end(), input_vec.begin());
          std::copy(new_obs.begin(), new_obs.end(), input_vec.end() - 75);
      }
  }

  """
  // Tienkung Lite
  std::vector<int> isaac_to_mujoco_idx = {
    0, 4, 8, 12, 16, 18, 1, 5, 9, 13, 
    17, 19, 2, 6, 10, 14, 3, 7, 11, 15
  };
  """

  // Tienkung Pro (https://www.notion.so/ID-2dad117f088d8090811ed17e424d4b7c)
  std::vector<int> isaac_to_mujoco_idx = {
    12, 0, 6, 13, 16, 23, 1, 7, 14, 17, 24, 2, 8, 15, 18,
    25, 3, 9, 19, 26, 4, 10, 20, 27, 5, 11, 21, 28, 22, 29
  };

  if ((int) (timer / dt_) % freq_ratio_ == 0) {
    for (size_t i = 0; i < ov_in_tensor0.get_size(); ++i) {
      ov_in_tensor0.data<float>()[i] = input_vec[i];
    }

    infer_request.set_input_tensor(0, ov_in_tensor0);
    infer_request.infer();

    ov::Tensor ov_out_tensor = infer_request.get_output_tensor();
    const float *ov_out_data = ov_out_tensor.data<float>();
    for (int i = 0; i < ov_out_tensor.get_size(); i++) {
      output_data_mlp(i) = ov_out_data[i];
    }
  }
  robot_data_->gait_a = gait_a;

  Eigen::VectorXd mlp_out = output_data_mlp.head(action_num);
  Eigen::VectorXd mlp_out_reordered = Eigen::VectorXd::Zero(action_num);
  for (int i = 0; i < action_num; ++i) {
    mlp_out_reordered(i) = mlp_out(isaac_to_mujoco_idx[i]);
  }
  Eigen::VectorXd mlp_out_dot = Eigen::VectorXd::Zero(action_num);


  Eigen::VectorXd mlp_out_scaled = mlp_out_reordered * action_scales + default_dof_pos;
  Eigen::VectorXd mlp_out_dot_scaled = mlp_out_dot * action_scales;

  robot_data_->q_d_.tail(joint_num_) = mlp_out_scaled;
  robot_data_->q_dot_d_.setZero();
  robot_data_->tau_d_.setZero();
  robot_data_->pos_mode_ = false;

  last_action_d = mlp_out;
  last_action_dot_d = mlp_out_dot;
  action_last = output_data_mlp.head(action_num);
  timer += dt_;
  timer_gait += dt_;

  robot_data_->q_d_.segment(0, 3) = Rb_w.transpose() * robot_data_->q_d_.segment(0, 3);

}

FSMStateName StateMLP::CheckTransition(const xbox_flag &flag) {
  bool outPoslimit = false;
  if (robot_data_->q_a_[7] >= 1.0 || robot_data_->q_a_[13] >= 1.0) {
    outPoslimit = true;
    std::cout << "out of limit!!!" << std::endl;
    std::cout << "left 2: " << robot_data_->q_a_[7] << "  right 2: " << robot_data_->q_a_[13] << std::endl;
  }
  if (flag.fsm_state_command == "gotoStop" || outPoslimit) {
    std::cout << "MLP2Stop" << std::endl;
    return FSMStateName::STOP;
  } else {
    return FSMStateName::MLP;
  }
}

void StateMLP::OnExit() {
}

StateStop::StateStop(RobotData *robot_data) : FSMState(robot_data) {
  robot_data_ = robot_data;
  current_state_name_ = FSMStateName::STOP;
}

void StateStop::OnEnter() {
  std::cout << "enter into stop" << std::endl;
  timer = 0.;
  init_joint_pos = robot_data_->q_a_.tail(joint_num_);

}

void StateStop::Run(xbox_flag &flag) {
    if (!first_run){
    init_joint_pos = robot_data_->q_a_.tail(joint_num_);
    std::cout << "init_joint_pos for enter stop: " << init_joint_pos.transpose() << std::endl;
    first_run = true;    
  }
  // keep warm
  if ((int) (timer / dt_) % freq_ratio_ == 0) {
    float gait_a = 0.0f;
    infer_request.set_input_tensor(0, ov_in_tensor0);
    infer_request.infer();
  }
  robot_data_->q_d_.tail(joint_num_) = init_joint_pos;
  robot_data_->q_dot_d_.setZero();
  robot_data_->tau_d_.setZero();
  robot_data_->pos_mode_ = true;

  timer += dt_;

}

FSMStateName StateStop::CheckTransition(const xbox_flag &flag) {
  if (flag.fsm_state_command == "gotoZero") {
    std::cout << "Stop2Zero" << std::endl;
    return FSMStateName::ZERO;
  } else {
    return FSMStateName::STOP;
  }
}

void StateStop::OnExit() {
}
