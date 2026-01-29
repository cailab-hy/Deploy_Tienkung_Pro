#include "FSMStateImpl.h"
#include <yaml-cpp/yaml.h>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <iostream>
#include <algorithm>
#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/model.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
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
const int obs_num = 159;
const int tienkung_pro_dof = 30;

// -----------------------------------------------------------------------------
// Details for OBS (Tienkung Pro)

// 1) Per-frame observation structure
// Index    : Feature                 (Dim)
// ----------------------------------------
// 0-59      : [1] motion_command        (60)
// 60-65     : [2] motion_ref_ori_b      (6)
// 66-68     : [3] base_ang_vel          (3)
// 69-98     : [4] Joint positions       (30)
// 99-128    : [5] Joint velocities      (30)
// 129-158   : [6] Previous actions      (30)
// Total per-frame = 159

// 2) History stacking
// History length = 1
// Total obs_num = Per-frame obs * History length = 159
// -----------------------------------------------------------------------------

// OpenVINO objects are declared globally so they are reused
// across FSM states and control loops.
ov::Core core;
std::shared_ptr<ov::Model> model;
ov::CompiledModel compiled_model;
ov::InferRequest infer_request;

ReferenceMotion g_ref_motion;

bool has_time_input = false;
std::vector<float> time_input_vec;
ov::Tensor ov_in_tensor1;

namespace {
size_t ShapeSize(const ov::Shape& shape) {
  size_t size = 1;
  for (auto dim : shape) {
    size *= dim;
  }
  return size;
}

void LogObsBlockMinMax(const char *label,
                       const std::vector<float> &vec,
                       size_t start,
                       size_t count) {
  if (count == 0 || start >= vec.size()) {
    std::cout << "[Obs] " << label << " empty" << std::endl;
    return;
  }
  size_t end = start + count;
  if (end > vec.size()) {
    end = vec.size();
  }
  auto begin_it = vec.begin() + static_cast<std::ptrdiff_t>(start);
  auto end_it = vec.begin() + static_cast<std::ptrdiff_t>(end);
  auto minmax = std::minmax_element(begin_it, end_it);
  std::cout << "[Obs] " << label << " [" << start << ":" << (end - 1)
            << "] min=" << *minmax.first << " max=" << *minmax.second << std::endl;
}

ReferenceMotion::Output ref_motion_cache;
bool ref_motion_ready = false;
int motion_timestep = 0;
bool pinocchio_ready = false;
int ref_body_frame_id = -1;
} // namespace

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
bool ReferenceMotion::Load(const std::string& config_file) {
  YAML::Node root = YAML::LoadFile(config_file);

  YAML::Node ref_joint_pos = root["joint_pos"];
  YAML::Node ref_joint_vel = root["joint_vel"];
  YAML::Node ref_body_quat_w = root["body_quat_w"];
  if (!ref_body_quat_w || !ref_body_quat_w.IsSequence()) {
    ref_body_quat_w = root["body_pos_w"];
  }

  // 기본 체크
  if (!ref_joint_pos || !ref_joint_pos.IsSequence() || ref_joint_pos.size() == 0) return false;
  if (!ref_joint_vel || !ref_joint_vel.IsSequence() || ref_joint_vel.size() != ref_joint_pos.size()) return false;
  if (!ref_body_quat_w || !ref_body_quat_w.IsSequence() || ref_body_quat_w.size() != ref_joint_pos.size()) return false;

  seq_size_ = (int)ref_joint_pos.size();

  // 멤버 행렬로 resize (Load 끝나도 유지됨)
  ref_joint_pos_matrix_.resize(seq_size_, tienkung_pro_dof);
  ref_joint_vel_matrix_.resize(seq_size_, tienkung_pro_dof);

  const int body_dim = 4;
  ref_body_pos_w_matrix_.resize(seq_size_, body_dim);

  for (int k = 0; k < seq_size_; ++k) {
    YAML::Node row_pos  = ref_joint_pos[k];
    YAML::Node row_vel  = ref_joint_vel[k];
    YAML::Node row_body = ref_body_quat_w[k];

    if (!row_pos.IsSequence() || (int)row_pos.size() != tienkung_pro_dof) return false;
    if (!row_vel.IsSequence() || (int)row_vel.size() != tienkung_pro_dof) return false;
    if (!row_body.IsSequence() || (int)row_body.size() != body_dim) return false;

    for (int j = 0; j < tienkung_pro_dof; ++j) {
      ref_joint_pos_matrix_(k, j) = row_pos[j].as<double>();
      ref_joint_vel_matrix_(k, j) = row_vel[j].as<double>();
    }
    for (int j = 0; j < body_dim; ++j) {
      ref_body_pos_w_matrix_(k, j) = row_body[j].as<double>();
    }
  }
  loaded_ = true;
  return true;
}

void ReferenceMotion::Reset() {
  current_frame_ = 0;
}

ReferenceMotion::Output ReferenceMotion::Run() {
  Output out;
  if (!loaded_ || seq_size_ <= 0) return out;

  // 범위 클램프 (또는 loop 처리)
  out.joint_pos  = ref_joint_pos_matrix_.row(current_frame_).transpose();
  out.joint_vel  = ref_joint_vel_matrix_.row(current_frame_).transpose();
  out.body_pos_w = ref_body_pos_w_matrix_.row(current_frame_).transpose();

  if (current_frame_ < seq_size_ - 1) {
    current_frame_++;
  }
  return out;
}
ReferenceMotion::Output ReferenceMotion::SpecificRun(int motion_time) const {
  Output out;
  if (!loaded_ || seq_size_ <= 0) return out;
  out.joint_pos  = ref_joint_pos_matrix_.row(motion_time).transpose();
  out.joint_vel  = ref_joint_vel_matrix_.row(motion_time).transpose();
  out.body_pos_w = ref_body_pos_w_matrix_.row(motion_time).transpose();
  return out;
}

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

  const auto inputs = compiled_model.inputs();
  if (inputs.empty()) {
    std::cerr << "[MLP] No input tensors found in model." << std::endl;
    has_time_input = false;
    time_input_vec.clear();
  } else {
    const auto obs_shape = inputs[0].get_shape();
    if (!obs_shape.empty()) {
      size_t obs_dim = obs_shape.back();
      if (obs_dim != static_cast<size_t>(obs_num)) {
        std::cerr << "[MLP] obs dim mismatch: model expects " << obs_dim
                  << ", code uses " << obs_num << std::endl;
      }
    }

    // Bind preallocated input tensor
    infer_request.set_input_tensor(0, ov_in_tensor0);

    has_time_input = inputs.size() > 1;
    if (has_time_input) {
      const auto time_shape = inputs[1].get_shape();
      time_input_vec.assign(ShapeSize(time_shape), 0.0f);
      ov_in_tensor1 = ov::Tensor(ov::element::f32, time_shape, time_input_vec.data());
      infer_request.set_input_tensor(1, ov_in_tensor1);
    } else {
      time_input_vec.clear();
    }
  }

  // Warm-up inference to avoid first-step latency spikes
  for (int i = 0; i < 3; i++) {
    std::cout << "MLP forward: " << i << std::endl;
    if (has_time_input && !time_input_vec.empty()) {
      time_input_vec[0] = 0.0f;
    }
    // Run a single forward pass of the policy network.
    infer_request.infer();
  }
  std::cout << "MLP forward succeed!" << std::endl;

  // Hard-coded safe default posture for Tienkung Pro


  zero_pos << 0.0, -0.5, 0.0, 1.0, -0.5, 0.0,           // left leg:  "l_hip_roll", "l_hip_pitch", "l_hip_yaw", "l_knee", "l_ankle_pitch", "l_ankle_roll",
              0.0, -0.5, 0.0, 1.0, -0.5, 0.0,           // right leg: "r_hip_roll", "r_hip_pitch", "r_hip_yaw", "r_knee", "r_ankle_pitch", "r_ankle_roll"
              0.0, 0.0, 0.0, 0.0,                       // waist and head: "waist_yaw", "head_roll", "head_pitch", "head_yaw"
              0.0, 0.1, 0.0, -0.3, 0.0, 0.0, 0.0,       // left arm:  "l_shoulder_pitch", "l_shoulder_roll", "l_shoulder_yaw", "l_elbow", "l_wrist_yaw", "l_wrist_pitch", "l_wrist_roll",
              0.0, -0.1, -0.0, -0.3, 0.0, 0.0, 0.0;     // right arm: "r_shoulder_pitch", "r_shoulder_roll", "r_shoulder_yaw", "r_elbow", "r_wrist_yaw", "r_wrist_pitch", "r_wrist_roll"
  std::string ref_path = config["reference_motion"]["motion_path"].as<std::string>();
  std::string ref_full_path = package_path + ref_path;

  if (!g_ref_motion.Load(ref_full_path)) {
    std::cout << "[ReferenceMotion] Load failed: " << ref_full_path << std::endl;
  } else {
    g_ref_motion.Reset(); // 로드 후 시작 프레임 0으로
    std::cout << "[ReferenceMotion] Load OK" << std::endl;
  }


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

  const std::string filename = "/home/cai/Deploy_Tienkung_Pro/rl_control_new/config/urdf_folder/urdf/tiangong2.0_pro_urdf.urdf";
  std::ifstream file;
  file.open(filename.c_str());
  std::string filestr((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
  pinocchio::Model pinocchio_model = pinocchio::urdf::buildModelFromXML(filestr,pinocchio::JointModelFreeFlyer());
  pinocchio::Data pinocchio_robot_data(pinocchio_model);

}

void StateZero::Run(xbox_flag &flag) {
  // Cache initial joint state only once
  if (!first_run){
    init_joint_pos = robot_data_->q_a_.tail(joint_num_);
    std::cout << "init_joint_pos for enter zero: " << init_joint_pos.transpose() << std::endl;
    first_run = true;    
  }
  // Periodically run dummy inference to keep the model "warm" even when policy control is not active.
  

  if ((int)(timer / dt_) % freq_ratio_ == 0) {
    if (has_time_input && !time_input_vec.empty()) {
      time_input_vec[0] = 0.0f;
      infer_request.set_input_tensor(1, ov_in_tensor1);
    }
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

  // Low-pass filter to reduce IMU noise in angular velocity
  omega_filter = new LowPassFilter(tienkung_pro_dof, 0.707, 0.0025, 3);
}

void StateMLP::OnEnter() {
  std::cout << "MLPenter" << std::endl;

  timer = 0.;
  trans_time = 0.5;

  // default_dof_pos for tienkung pro
  default_dof_pos << 0.0, -0.5, 0.0, 1.0, -0.5, 0.0,    // left leg:  "l_hip_roll", "l_hip_pitch", "l_hip_yaw", "l_knee", "l_ankle_pitch", "l_ankle_roll",
              0.0, -0.5, 0.0, 1.0, -0.5, 0.0,           // right leg: "r_hip_roll", "r_hip_pitch", "r_hip_yaw", "r_knee", "r_ankle_pitch", "r_ankle_roll"
              0.0, 0.0, 0.0, 0.0,                       // waist and head: "waist_yaw", "head_roll", "head_pitch", "head_yaw"
              0.0, 0.1, 0.0, -0.3, 0.0, 0.0, 0.0,       // left arm:  "l_shoulder_pitch", "l_shoulder_roll", "l_shoulder_yaw", "l_elbow", "l_wrist_yaw", "l_wrist_pitch", "l_wrist_roll",
              0.0, -0.1, -0.0, -0.3, 0.0, 0.0, 0.0;     // right arm: "r_shoulder_pitch", "r_shoulder_roll", "r_shoulder_yaw", "r_elbow", "r_wrist_yaw", "r_wrist_pitch", "r_wrist_roll"

  ref_motion_ready = false;
  motion_timestep = 0;
  g_ref_motion.Reset();

  if (!pinocchio_ready) {
    std::string package_path = ament_index_cpp::get_package_share_directory("rl_control_new");
    std::string urdf_path = package_path + "/config/urdf_folder/urdf/tiangong2.0_pro_urdf.urdf";
    std::ifstream urdf_file(urdf_path);
    if (!urdf_file.is_open()) {
      std::cerr << "[Pinocchio] URDF not found: " << urdf_path << std::endl;
    } else {
      pinocchio_model = pinocchio::urdf::buildModel(urdf_path, pinocchio::JointModelFreeFlyer());
      pinocchio_robot_data = pinocchio::Data(pinocchio_model);
      ref_body_frame_id = pinocchio_model.getFrameId("body_yaw_link");
      pinocchio_ready = true;
    }
  }

  action_last.setZero();
  // Reset previous action history
  last_action_d.setZero();
  last_action_dot_d.setZero();
  input_data_mlp.setZero();
  run2walk = false;
}


void StateMLP::Run(xbox_flag &flag) {
  const bool do_infer = ((int)(timer / dt_) % freq_ratio_ == 0);

  if (do_infer) {
    ref_motion_cache = g_ref_motion.Run();
    ref_motion_ready = (ref_motion_cache.joint_pos.size() == joint_num_);
  }
  if (!ref_motion_ready) {
    ref_motion_cache = g_ref_motion.SpecificRun(0);
    ref_motion_ready = (ref_motion_cache.joint_pos.size() == joint_num_);
  }

  Eigen::VectorXd ref_joint_pos = Eigen::VectorXd::Zero(joint_num_);
  Eigen::VectorXd ref_joint_vel = Eigen::VectorXd::Zero(joint_num_);
  if (ref_motion_cache.joint_pos.size() == joint_num_) {
    ref_joint_pos = ref_motion_cache.joint_pos;
  }
  if (ref_motion_cache.joint_vel.size() == joint_num_) {
    ref_joint_vel = ref_motion_cache.joint_vel;
  }

  for (int i = 0; i < joint_num_; i++) {
    input_vec[i] = static_cast<float>(ref_joint_pos(i));
  }
  for (int i = 0; i < joint_num_; i++) {
    input_vec[30 + i] = static_cast<float>(ref_joint_vel(i));
  }

  Eigen::Vector4d motion_ref_ori_1(1.0, 0.0, 0.0, 0.0);
  if (ref_motion_cache.body_pos_w.size() == 4) {
    motion_ref_ori_1 = ref_motion_cache.body_pos_w;
  }
  double motion_yaw_offset = quat_yaw(motion_ref_ori_1);
  Eigen::Vector4d motion_ref_ori_2 = remove_yaw_offset(motion_ref_ori_1, motion_yaw_offset);

  Eigen::Vector3d root_pos = Eigen::Vector3d::Zero();
  Eigen::Vector4d root_ori_xyzw = robot_data_->imu_data_.tail(4); // xyzw

  Eigen::VectorXd c(root_pos.size() + root_ori_xyzw.size());
  c.head(root_pos.size()) = root_pos;
  c.tail(root_ori_xyzw.size()) = root_ori_xyzw;

  const std::vector<int> mujoco_to_isaac_idx = {
    12, 0, 6, 13, 16, 23, 1, 7, 14, 17,
    24, 2, 8, 15, 18, 25, 3, 9, 19, 26,
    4, 10, 20, 27, 5, 11, 21, 28, 22, 29
  };

  Eigen::VectorXd dof_pos_in_pinocchio = Eigen::VectorXd::Zero(joint_num_);
  for (int i = 0; i < joint_num_; i++) {
    int idx = mujoco_to_isaac_idx[i];
    dof_pos_in_pinocchio[i] = robot_data_->q_a_(6 + idx);
  }

  Eigen::VectorXd configuration(c.size() + dof_pos_in_pinocchio.size());
  configuration.head(c.size()) = c;
  configuration.tail(dof_pos_in_pinocchio.size()) = dof_pos_in_pinocchio;

  Eigen::Vector4d robot_ref_ori_2(1.0, 0.0, 0.0, 0.0);
  if (pinocchio_ready && ref_body_frame_id >= 0) {
    pinocchio::framesForwardKinematics(pinocchio_model, pinocchio_robot_data, configuration);
    const auto &ref_body_pose_in_world = pinocchio_robot_data.oMf[ref_body_frame_id];
    auto quaternion = pinocchio::Quaternion(ref_body_pose_in_world.rotation());
    auto ref_ori_xyzw = quaternion.coeffs();
    Eigen::Vector4d robot_ref_ori_1(ref_ori_xyzw(3), ref_ori_xyzw(0), ref_ori_xyzw(1), ref_ori_xyzw(2));
    double robot_yaw_offset = quat_yaw(robot_ref_ori_1);
    robot_ref_ori_2 = remove_yaw_offset(robot_ref_ori_1, robot_yaw_offset);
  }

  Eigen::Vector4d transformed = subtract_frame_transforms(robot_ref_ori_2, motion_ref_ori_2);
  Eigen::Matrix3d motion_ref_ori_b = matrix_from_quat(transformed);

  input_vec[60] = motion_ref_ori_b(0,0);
  input_vec[61] = motion_ref_ori_b(0,1);
  input_vec[62] = motion_ref_ori_b(1,0);
  input_vec[63] = motion_ref_ori_b(1,1);
  input_vec[64] = motion_ref_ori_b(2,0);
  input_vec[65] = motion_ref_ori_b(2,1);

  Eigen::Vector3d ypr_a = robot_data_->imu_data_.head(3);
  Eigen::Matrix3d NED_R_YPR_a = Eigen::Matrix3d::Zero();
  Eigen::Vector3d rpy = Eigen::Vector3d::Zero();
  ypr_a(0) = 0.0;

  EulerZYXToMatrix(NED_R_YPR_a, ypr_a);
  MatrixToEulerXYZ(NED_R_YPR_a, rpy);
  robot_data_->q_a_.segment(3, 3) = rpy;
  Eigen::Matrix3d R_xyz_omega = Eigen::Matrix3d::Identity();
  R_xyz_omega.row(1) = RotX(rpy(0)).row(1);
  R_xyz_omega.row(2) = (RotX(rpy(0)) * RotY(rpy(1))).row(2);
  robot_data_->q_dot_a_.segment(3, 3) = R_xyz_omega.transpose() * NED_R_YPR_a * robot_data_->imu_data_.segment(3, 3);
  Eigen::Matrix3d Rb_w = Eigen::Matrix3d::Identity();
  Euler_XYZToMatrix(Rb_w, robot_data_->q_a_.segment(3, 3));
  Eigen::Vector3d rpy_a = robot_data_->q_a_.segment(3, 3);
  Eigen::Matrix3d R_xyz_omega_a = Eigen::Matrix3d::Identity();
  R_xyz_omega_a.row(1) = RotX(rpy_a(0)).row(1);
  R_xyz_omega_a.row(2) = (RotX(rpy_a(0)) * RotY(rpy_a(1))).row(2);
  Eigen::Vector3d ang_vel = Rb_w.transpose() * R_xyz_omega_a * robot_data_->q_dot_a_.segment(3, 3);
  // Low-pass filter to reduce high-frequency IMU noise before feeding the policy.
  ang_vel = omega_filter->mFilter(ang_vel);

  // Write [0:3] angular velocity into the OpenVINO input buffer (input_vec).
  input_vec[66] = ang_vel(0);
  input_vec[67] = ang_vel(1);
  input_vec[68] = ang_vel(2);

  // dof_pos
  for (int i = 0; i < joint_num_; i++) {
    int idx = mujoco_to_isaac_idx[i];
    input_vec[69 + i] = (robot_data_->q_a_(6 + idx) - default_dof_pos(idx));
  }
  // dof_vel
  for (int i = 0; i < joint_num_; i++) {
    int idx = mujoco_to_isaac_idx[i];
    input_vec[99 + i] = robot_data_->q_dot_a_(6 + idx);
  }
  clip(action_last, -100.0, 100.0);
  for (int i = 0; i < action_num; i++) {
    input_vec[129 + i] = action_last(i);
  }

  const std::vector<int> isaac_to_mujoco_idx = {
    1, 6, 11, 16, 20, 24, 2, 7, 12, 17,
    21, 25, 0, 3, 8, 13, 4, 9, 14, 18,
    22, 26, 28, 5, 10, 15, 19, 23, 27, 29
  };

  if (do_infer) {
    static int debug_log_count = 0;
    const int debug_log_stride = 10;
    if ((debug_log_count++ % debug_log_stride) == 0) {
      std::cout << "[Obs] step=" << motion_timestep << std::endl;
      LogObsBlockMinMax("motion_command", input_vec, 0, 60);
      LogObsBlockMinMax("motion_ref_ori_b", input_vec, 60, 6);
      LogObsBlockMinMax("base_ang_vel", input_vec, 66, 3);
      LogObsBlockMinMax("dof_pos", input_vec, 69, 30);
      LogObsBlockMinMax("dof_vel", input_vec, 99, 30);
      LogObsBlockMinMax("prev_actions", input_vec, 129, 30);
    }
    // Copy the 159-dim observation into the OpenVINO tensor.
    for (size_t i = 0; i < ov_in_tensor0.get_size(); ++i) {
      ov_in_tensor0.data<float>()[i] = input_vec[i];
    }
    if (has_time_input && !time_input_vec.empty()) {
      time_input_vec[0] = static_cast<float>(motion_timestep);
      infer_request.set_input_tensor(1, ov_in_tensor1);
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
    motion_timestep++;
  }
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
  // Transform desired base commands into the frame expected by the low-level controller.
  robot_data_->q_d_.segment(0, 3) = Rb_w.transpose() * robot_data_->q_d_.segment(0, 3);
}
