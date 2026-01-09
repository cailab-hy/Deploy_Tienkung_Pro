#include "FSMStateImpl.h"
#include <yaml-cpp/yaml.h>
#include <ament_index_cpp/get_package_share_directory.hpp>

// -----------------------------------------------------------------------------
// This file implements FSM (Finite State Machine) states for humanoid deployment.
// The FSM consists of three main states:
//   1) ZERO : Move robot to a safe nominal posture
//   2) MLP  : Run learned policy (neural network) for control
//   3) STOP : Emergency / safe stop state
//
// The policy inference is done using OpenVINO.
// -----------------------------------------------------------------------------

#ifndef PI
#define PI 3.141592654
#endif

// Total observation dimension expected by the policy network.
// This includes stacked observation history.
const int obs_num = 1050;
const int tienkung_pro_dof = 30;

// -----------------------------------------------------------------------------
// Details for OBS (Tienkung Pro)

// 1) Per-frame observation structure
// Index    : Feature                 (Dim)
// ----------------------------------------
// 0-2      : [1] Angular velocity        (3)
// 3-5      : [2] Gravity direction       (3)
// 6-8      : [3] Commands (vx, vy, yaw)  (3)
// 9-38     : [4] Joint positions         (30)
// 39-68    : [5] Joint velocities        (30)
// 69-99    : [6] Previous actions        (30)
// 100-105  : [7] Phase info              (6)   // e.g., sin/cos/phase or gait signals
// Total per-frame = 105

// 2) History stacking
// History length = 10
// Total obs_num = Per-frame obs * History length = 1050
// -----------------------------------------------------------------------------

// OpenVINO objects are declared globally so they are reused
// across FSM states and control loops.
ov::Core core;
std::shared_ptr<ov::Model> model;
ov::CompiledModel compiled_model;
ov::InferRequest infer_request;

// Input buffer for the policy network.
// The memory is directly bound to OpenVINO tensor.
std::vector<float> input_vec(obs_num, 0.);
ov::Tensor ov_in_tensor0(
    ov::element::f32,
    ov::Shape{1, static_cast<size_t>(obs_num)},
    input_vec.data()
);

// Nominal joint configuration used in ZERO state.
// This posture is used to safely initialize the humanoid before enabling learned control.
Eigen::VectorXd zero_pos = Eigen::VectorXd::Zero(tienkung_pro_dof);


// -----------------------------------------------------------------------------
// FSM initialization
// Loads the MLP policy and prepares OpenVINO inference.
// -----------------------------------------------------------------------------
void FSMInit(const std::string& config_file) {
  // Load YAML configuration file
  YAML::Node config = YAML::LoadFile(config_file);

  // Relative path to the policy model (without extension)
  auto relative_path = config["mlp"]["path"].as<std::string>();

  // Resolve absolute path using ROS2 package index
  std::string package_path = ament_index_cpp::get_package_share_directory("rl_control_new");
  std::string mlp_path = package_path + relative_path;

  // Load OpenVINO IR model (.xml + .bin)
  model = core.read_model(mlp_path + ".xml",  mlp_path + ".bin");

  // Compile model for CPU execution
  compiled_model = core.compile_model(model, "CPU");
  infer_request = compiled_model.create_infer_request();

  // Restrict CPU threads to reduce timing jitter during deployment
  core.set_property("CPU", ov::inference_num_threads(1)); 

  float gait_a = 0.0f;

  // Bind preallocated input tensor
  infer_request.set_input_tensor(0, ov_in_tensor0);

  // Warm-up inference to avoid first-step latency spikes
  for (int i = 0; i < 3; i++) {
    std::cout << "MLP forward: " << i << std::endl;
    // Run a single forward pass of the policy network.
    infer_request.infer();
  }
  std::cout << "MLP forward succeed!" << std::endl;



  // Hard-coded safe default posture for Tienkung Pro
  zero_pos << 0.0, -0.5, 0.0, 1.0, -0.5, 0.0,           // left leg:  "l_hip_roll", "l_hip_pitch", "l_hip_yaw", "l_knee", "l_ankle_pitch", "l_ankle_roll",
              0.0, -0.5, 0.0, 1.0, -0.5, 0.0,           // right leg: "r_hip_roll", "r_hip_pitch", "r_hip_yaw", "r_knee", "r_ankle_pitch", "r_ankle_roll"
              0.0, 0.1, 0.0, -0.3, 0.0, 0.0, 0.0,       // left arm:  "l_shoulder_pitch", "l_shoulder_roll", "l_shoulder_yaw", "l_elbow", "l_wrist_yaw", "l_wrist_pitch", "l_wrist_roll",
              0.0, -0.1, -0.0, -0.3, 0.0, 0.0, 0.0,     // right arm: "r_shoulder_pitch", "r_shoulder_roll", "r_shoulder_yaw", "r_elbow", "r_wrist_yaw", "r_wrist_pitch", "r_wrist_roll"
              0.0, 0.0, 0.0, 0.0;                       // head and waist: "head_roll", "head_pitch", "head_yaw", "waist_yaw"
}

// -----------------------------------------------------------------------------
// ZERO STATE
// Moves robot smoothly to a predefined safe posture.
// -----------------------------------------------------------------------------
StateZero::StateZero(RobotData *robot_data) : FSMState(robot_data) {
  robot_data_ = robot_data;
  current_state_name_ = FSMStateName::ZERO;
}

void StateZero::OnEnter() {
  timer = 0.;
  // Cache current joint positions at state entry
  init_joint_pos = robot_data_->q_a_.tail(joint_num_);
  std::cout << "enter zero" << std::endl;
}

void StateZero::Run(xbox_flag &flag) {
  // Cache initial joint state only once
  if (!first_run){
    init_joint_pos = robot_data_->q_a_.tail(joint_num_);
    std::cout << "init_joint_pos for enter zero: " << init_joint_pos.transpose() << std::endl;
    first_run = true;    
  }
  
  // Periodically run dummy inference to keep the model "warm" even when policy control is not active.
  if ((int) (timer / dt_) % freq_ratio_ == 0) {
    float gait_a = 0.0f;
    infer_request.set_input_tensor(0, ov_in_tensor0);
    infer_request.infer();
  }

  Eigen::VectorXd zero = Eigen::VectorXd::Zero(joint_num_);
  Eigen::VectorXd joint_pos = Eigen::VectorXd::Zero(joint_num_);
  Eigen::VectorXd joint_vel = Eigen::VectorXd::Zero(joint_num_);
  Eigen::VectorXd joint_acc = Eigen::VectorXd::Zero(joint_num_);

  // Generate smooth fifth-order polynomial trajectory from current posture to zero_pos
  if (timer < total_time) {
    FifthPoly(init_joint_pos, zero, zero,
              zero_pos, zero, zero,
              total_time, timer,
              joint_pos, joint_vel, joint_acc);
  } else {
    // Hold final posture after transition is complete
    zero_finish_flag = true;
    joint_pos = zero_pos;
    joint_vel.setZero();
  }

  // Position control mode during ZERO state
  robot_data_->q_d_.tail(joint_num_) = joint_pos;
  robot_data_->q_dot_d_.tail(joint_num_) = joint_vel;
  robot_data_->tau_d_.setZero();
  robot_data_->pos_mode_ = true;

  timer += dt_;
}

FSMStateName StateZero::CheckTransition(const xbox_flag &flag) {
  // Transition to MLP state only after reaching zero posture
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
  // Reset entry flag for next time
  zero_finish_flag = false;
}

// -----------------------------------------------------------------------------
// MLP STATE
// Runs the learned neural network policy for humanoid control.
// -----------------------------------------------------------------------------
StateMLP::StateMLP(RobotData *robot_data) : FSMState(robot_data) {
  robot_data_ = robot_data;
  current_state_name_ = FSMStateName::MLP;

  // Full stacked observation buffer
  input_data_mlp = Eigen::VectorXd::Zero(obs_num);

  // Scaling factors for commanded velocities
  command_scales[0] = obs_scales_lin_vel;
  command_scales[1] = obs_scales_lin_vel;
  command_scales[2] = obs_scales_ang_vel;

  // Low-pass filter to reduce IMU noise in angular velocity
  omega_filter = new LowPassFilter(tienkung_pro_dof, 0.707, 0.0025, 3);
}

void StateMLP::OnEnter() {
  std::cout << "MLPenter" << std::endl;

  timer = 0.;
  timer_gait = 0.;

  gait_d = 0.0;
  gait_a = 0.0;
  trans_time = 0.5;

  // Initialize gait phases
  left_phase = left_theta_offset;
  right_phase = right_theta_offset;

  // default_dof_pos for tienkung pro
  default_dof_pos << 0.0, -0.5, 0.0, 1.0, -0.5, 0.0,   // left leg:  "l_hip_roll", "l_hip_pitch", "l_hip_yaw", "l_knee", "l_ankle_pitch", "l_ankle_roll",
      0.0, -0.5, 0.0, 1.0, -0.5, 0.0,                  // right leg: "r_hip_roll", "r_hip_pitch", "r_hip_yaw", "r_knee", "r_ankle_pitch", "r_ankle_roll"
      0.0, 0.1, 0.0, -0.3, 0.0, 0.0, 0.0,              // left arm:  "l_shoulder_pitch", "l_shoulder_roll", "l_shoulder_yaw", "l_elbow", "l_wrist_yaw", "l_wrist_pitch", "l_wrist_roll",
      0.0, -0.1, -0.0, -0.3, 0.0, 0.0, 0.0,            // right arm: "r_shoulder_pitch", "r_shoulder_roll", "r_shoulder_yaw", "r_elbow", "r_wrist_yaw", "r_wrist_pitch", "r_wrist_roll"
      0.0, 0.0, 0.0, 0.0;                              // head and waist: "head_roll", "head_pitch", "head_yaw", "waist_yaw"

  action_last.setZero();
  joint_pos_last_gait = zero_pos;
  joint_vel_last_gait.setZero();

  // Reset previous action history
  last_action_d.setZero();
  last_action_dot_d.setZero();
  input_data_mlp.setZero();
  x_vel_command_offset = 0.0;
  y_vel_command_offset = 0.0;
  run2walk = false;
}

void StateMLP::Run(xbox_flag &flag) {
  // ---------------------------------------------------------------------------
  // Main control tick for the learned-policy (MLP) state.
  // This function:
  //  (1) updates user commands (velocity/yaw) with a simple rate limiter
  //  (2) builds the per-step observation frame and writes it into input_vec
  //  (3) maintains a 10-step observation history (1050-dim) by shifting input_vec
  //  (4) runs OpenVINO inference at freq_ratio_ and applies scaled joint targets
  // ---------------------------------------------------------------------------

  // [Command] Read user/requested velocity commands from the UI/controller.
  joystick_command(0) = flag.x_speed_command;
  joystick_command(1) = flag.y_speed_command;
  joystick_command(2) = flag.yaw_speed_command;


  // [Command Offset] Optionally accumulate command offsets after gait is active
  if (gait_a > 0.5) {
    x_vel_command_offset += flag.x_speed_offset;
    y_vel_command_offset += flag.y_speed_offset;

    joystick_command(0) += x_vel_command_offset;
    joystick_command(1) += y_vel_command_offset;
  }

  // [Rate Limiter] Smooth/limit changes of command(0)/command(2) to reduce jerks.
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

  // [Observation] Build the per-step observation (105 dims) for the policy.
  // The policy expects body-frame angular velocity, gravity direction, commands,
  // normalized joint pos/vel, previous action, and gait phase signals.

  // IMU processing: convert raw IMU orientation to the representation used downstream.
  // ypr_a(0)=0.0 forces yaw to zero (common trick to avoid yaw drift affecting policy).
  // imu data trans to world coordinate
  Eigen::Vector3d ypr_a = robot_data_->imu_data_.head(3);
  Eigen::Matrix3d NED_R_YPR_a = Eigen::Matrix3d::Zero();
  Eigen::Vector3d rpy = Eigen::Vector3d::Zero();

  // Ignore yaw for policy input if desired (common for locomotion policies).
  ypr_a(0) = 0.0;

  EulerZYXToMatrix(NED_R_YPR_a, ypr_a);
  MatrixToEulerXYZ(NED_R_YPR_a, rpy);

  // Save processed orientation to robot state buffer.
  robot_data_->q_a_.segment(3, 3) = rpy;

  Eigen::Matrix3d R_xyz_omega = Eigen::Matrix3d::Identity();
  R_xyz_omega.row(1) = RotX(rpy(0)).row(1);
  R_xyz_omega.row(2) = (RotX(rpy(0)) * RotY(rpy(1))).row(2);

  // Convert IMU angular velocity into the same convention/frame used by the controller.
  robot_data_->q_dot_a_.segment(3, 3) = R_xyz_omega.transpose() * NED_R_YPR_a * robot_data_->imu_data_.segment(3, 3);


  // Compute rotation matrix from current base RPY and use it to express quantities in body frame.
  // Std vector obs input
  Eigen::Matrix3d Rb_w = Eigen::Matrix3d::Identity();
  Euler_XYZToMatrix(Rb_w, robot_data_->q_a_.segment(3, 3));
  Eigen::Vector3d rpy_a = robot_data_->q_a_.segment(3, 3);
  Eigen::Matrix3d R_xyz_omega_a = Eigen::Matrix3d::Identity();
  R_xyz_omega_a.row(1) = RotX(rpy_a(0)).row(1);
  R_xyz_omega_a.row(2) = (RotX(rpy_a(0)) * RotY(rpy_a(1))).row(2);

  Eigen::Vector3d ang_vel = Rb_w.transpose() * R_xyz_omega_a * robot_data_->q_dot_a_.segment(3, 3) * obs_scales_ang_vel;
  
  // Low-pass filter to reduce high-frequency IMU noise before feeding the policy.
  ang_vel = omega_filter->mFilter(ang_vel);

  // Write [0:3] angular velocity into the OpenVINO input buffer (input_vec).
  input_vec[0] = ang_vel(0);
  input_vec[1] = ang_vel(1);
  input_vec[2] = ang_vel(2);
  
  // Write [3:6] projected gravity direction (negative body Z axis).
  Eigen::Vector3d base_z = -Rb_w.transpose().col(2);
  input_vec[3] = base_z(0);
  input_vec[4] = base_z(1);
  input_vec[5] = base_z(2);

  // Write [6:9] scaled user commands into observation.
  input_vec[6] = command[0] * command_scales[0];
  input_vec[7] = command[1] * command_scales[1];
  input_vec[8] = command[2] * command_scales[2];


  // Joint-index mapping: controller/MuJoCo joint order -> Isaac/policy joint order.
  std::vector<int> mujoco_to_isaac_idx = {
    12,  0, 6, 13, 16, 23, 1, 7, 14, 17,
    24, 2, 8, 15, 18, 25, 3, 9, 19, 26,
    4, 10, 20, 27, 5, 11, 21, 28, 22, 29};

  // Write [9:39] normalized joint position (relative to default_dof_pos).
  for (int i = 0; i < joint_num_; i++) {
    int idx = mujoco_to_isaac_idx[i];
    input_vec[9 + i] = (robot_data_->q_a_(6 + idx) - default_dof_pos(idx)) * obs_scales_dof_pos;
  }

  // Write [39:69] normalized joint velocity.
  for (int i = 0; i < joint_num_; i++) {
    int idx = mujoco_to_isaac_idx[i];
    input_vec[39 + i] = robot_data_->q_dot_a_(6 + idx) * obs_scales_dof_vel;
  }

  // Write [69:99] previous action (clipped) so the policy can use action history.
  clip(action_last, -100.0, 100.0);
  for (int i = 0; i < action_num; i++) {
      input_vec[69 + i] = action_last(i);
  }

  // Write [99:105] gait/phase features (e.g., sin/cos or phase flags).
  Eigen::VectorXd gait = gait_phase(timer_gait, gait_cycle,
                                    left_theta_offset,
                                    right_theta_offset,
                                    left_phase_ratio,
                                    right_phase_ratio);
  for (int i = 0; i < 6; i++) {
      input_vec[99 + i] = gait(i);
  }

  // History stacking: shift the 10-frame buffer left by 1 frame (105 dims),
  // and append the newest 105-dim frame to the tail.
  if ((int)(timer / dt_) % freq_ratio_ == 0) {
      if (first_Run) {
          first_Run = false;
      } else {
          std::vector<float> new_obs(input_vec.begin(), input_vec.begin() + 105);
          std::move(input_vec.begin() + 105, input_vec.end(), input_vec.begin());
          std::copy(new_obs.begin(), new_obs.end(), input_vec.end() - 105);
      }
  }

  // Action-index mapping: Isaac/policy output order -> controller/MuJoCo joint order.
  std::vector<int> isaac_to_mujoco_idx = {
    1, 6, 11, 16, 20, 24, 2, 7, 12, 17,
    21, 25, 0, 3, 8, 13, 4, 9, 14, 18, 
    22, 26, 28, 5, 10, 15, 19, 23, 27, 29};

  if ((int) (timer / dt_) % freq_ratio_ == 0) {
    // Copy the full stacked observation history (1050 floats) into the OpenVINO tensor.
    for (size_t i = 0; i < ov_in_tensor0.get_size(); ++i) {
      ov_in_tensor0.data<float>()[i] = input_vec[i];
    }

    infer_request.set_input_tensor(0, ov_in_tensor0);

    // Run policy inference (OpenVINO). Output is typically normalized joint targets.
    infer_request.infer();

    // Read policy output tensor into output_data_mlp (Eigen vector).
    ov::Tensor ov_out_tensor = infer_request.get_output_tensor();
    const float *ov_out_data = ov_out_tensor.data<float>();
    for (int i = 0; i < ov_out_tensor.get_size(); i++) {
      output_data_mlp(i) = ov_out_data[i];
    }
  }

  // Expose gait amplitude/state to other modules (e.g., transitions/safety).
  robot_data_->gait_a = gait_a;

  Eigen::VectorXd mlp_out = output_data_mlp.head(action_num);
  Eigen::VectorXd mlp_out_reordered = Eigen::VectorXd::Zero(action_num);
  for (int i = 0; i < action_num; ++i) {
    mlp_out_reordered(i) = mlp_out(isaac_to_mujoco_idx[i]);
  }
  Eigen::VectorXd mlp_out_dot = Eigen::VectorXd::Zero(action_num);

  // Post-process actions: scale and add default_dof_pos offset to get joint targets.
  Eigen::VectorXd mlp_out_scaled = mlp_out_reordered * action_scales + default_dof_pos;
  Eigen::VectorXd mlp_out_dot_scaled = mlp_out_dot * action_scales;


  robot_data_->q_d_.tail(joint_num_) = mlp_out_scaled;
  robot_data_->q_dot_d_.setZero();
  robot_data_->tau_d_.setZero();

  // Switch to non-position mode if the downstream controller expects hybrid/torque control.
  robot_data_->pos_mode_ = false;

  // Cache current action for next-step observation (previous action feature).
  last_action_d = mlp_out;
  last_action_dot_d = mlp_out_dot;
  action_last = output_data_mlp.head(action_num);
  timer += dt_;
  timer_gait += dt_;

  // Transform desired base commands into the frame expected by the low-level controller.
  robot_data_->q_d_.segment(0, 3) = Rb_w.transpose() * robot_data_->q_d_.segment(0, 3);
}

FSMStateName StateMLP::CheckTransition(const xbox_flag &flag) {
  // Flag indicating whether any joint exceeds a predefined safety limit.
  bool outPoslimit = false;

  // Safety check: monitor specific joints (hard-coded indices)
  // If joint positions exceed the threshold, trigger emergency transition.
  if (robot_data_->q_a_[7] >= 1.0 || robot_data_->q_a_[13] >= 1.0) {
    outPoslimit = true;
    std::cout << "out of limit!!!" << std::endl;

    // Print joint values for debugging/logging purposes.
    std::cout << "left 2: " << robot_data_->q_a_[7] << "  right 2: " << robot_data_->q_a_[13] << std::endl;
  }

  // Transition to STOP state either by user command or safety violation.
  if (flag.fsm_state_command == "gotoStop" || outPoslimit) {
    std::cout << "MLP2Stop" << std::endl;
    return FSMStateName::STOP;
  } else {
    // Remain in MLP state if no stop condition is triggered.
    return FSMStateName::MLP;
  }
}

void StateMLP::OnExit() {
  // No special cleanup is required when exiting the MLP state.
  // Timers and internal variables are handled elsewhere.
}

StateStop::StateStop(RobotData *robot_data) : FSMState(robot_data) {
  // Store robot data pointer and set current FSM state.
  robot_data_ = robot_data;
  current_state_name_ = FSMStateName::STOP;
}

void StateStop::OnEnter() {
  // Entering STOP state: used for emergency or safe halt.
  std::cout << "enter into stop" << std::endl;
  
  // Reset local timer.
  timer = 0.;

  // Cache the current joint positions to hold them during STOP.
  init_joint_pos = robot_data_->q_a_.tail(joint_num_);

}

void StateStop::Run(xbox_flag &flag) {
  // Cache initial joint positions only once when STOP state starts.
  if (!first_run){
    init_joint_pos = robot_data_->q_a_.tail(joint_num_);
    std::cout << "init_joint_pos for enter stop: " << init_joint_pos.transpose() << std::endl;
    first_run = true;    
  }

  // Keep the neural network inference "warm" even in STOP state.
  // This avoids latency spikes if transitioning back to MLP.
  if ((int) (timer / dt_) % freq_ratio_ == 0) {
    float gait_a = 0.0f;
    infer_request.set_input_tensor(0, ov_in_tensor0);
    infer_request.infer();
  }

  // Hold the robot at the joint configuration captured on STOP entry.
  robot_data_->q_d_.tail(joint_num_) = init_joint_pos;

  // Zero desired joint velocities and torques for safety.
  robot_data_->q_dot_d_.setZero();
  robot_data_->tau_d_.setZero();

  // Enable position control mode to passively hold posture.
  robot_data_->pos_mode_ = true;

  timer += dt_;
}

FSMStateName StateStop::CheckTransition(const xbox_flag &flag) {
  // Allow transition back to ZERO state upon explicit user command.
  if (flag.fsm_state_command == "gotoZero") {
    std::cout << "Stop2Zero" << std::endl;
    return FSMStateName::ZERO;
  } else {
    // Otherwise, remain in STOP state.
    return FSMStateName::STOP;
  }
}

void StateStop::OnExit() {
  // No special exit handling for STOP state.
}


