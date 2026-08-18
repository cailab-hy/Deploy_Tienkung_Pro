#include "HdmiPolicyRunner.h"
#include <yaml-cpp/yaml.h>
#include <iostream>
#include <regex>
#include <cassert>

// ============================================================================
// Joint mapping: hw (bodyIdMap) ↔ isaac (policy) order
// ============================================================================
// hw order (bodyIdMap.h):
//   [0-5]:   l_hip_roll, l_hip_pitch, l_hip_yaw, l_knee, l_ankle_pitch, l_ankle_roll
//   [6-11]:  r_hip_roll, r_hip_pitch, r_hip_yaw, r_knee, r_ankle_pitch, r_ankle_roll
//   [12]:    waist_yaw
//   [13-15]: head_roll, head_pitch, head_yaw
//   [16-22]: l_shoulder_pitch, l_shoulder_roll, l_shoulder_yaw, l_elbow, l_wrist_yaw, l_wrist_pitch, l_wrist_roll
//   [23-29]: r_shoulder_pitch, r_shoulder_roll, r_shoulder_yaw, r_elbow, r_wrist_yaw, r_wrist_pitch, r_wrist_roll
//
// isaac order (policy YAML isaac_joint_names):
//   [0]:  body_yaw_joint        → hw[12]
//   [1]:  hip_roll_l_joint      → hw[0]
//   [2]:  hip_roll_r_joint      → hw[6]
//   [3]:  head_yaw_joint        → hw[15]
//   [4]:  shoulder_pitch_l_joint→ hw[16]
//   [5]:  shoulder_pitch_r_joint→ hw[23]
//   [6]:  hip_pitch_l_joint     → hw[1]
//   [7]:  hip_pitch_r_joint     → hw[7]
//   [8]:  head_pitch_joint      → hw[14]
//   [9]:  shoulder_roll_l_joint → hw[17]
//   [10]: shoulder_roll_r_joint → hw[24]
//   [11]: hip_yaw_l_joint       → hw[2]
//   [12]: hip_yaw_r_joint       → hw[8]
//   [13]: head_roll_joint       → hw[13]
//   [14]: shoulder_yaw_l_joint  → hw[18]
//   [15]: shoulder_yaw_r_joint  → hw[25]
//   [16]: knee_pitch_l_joint    → hw[3]
//   [17]: knee_pitch_r_joint    → hw[9]
//   [18]: elbow_pitch_l_joint   → hw[19]
//   [19]: elbow_pitch_r_joint   → hw[26]
//   [20]: ankle_pitch_l_joint   → hw[4]  (S/P)
//   [21]: ankle_pitch_r_joint   → hw[10] (S/P)
//   [22]: elbow_yaw_l_joint     → hw[20]
//   [23]: elbow_yaw_r_joint     → hw[27]
//   [24]: ankle_roll_l_joint    → hw[5]  (S/P)
//   [25]: ankle_roll_r_joint    → hw[11] (S/P)
//   [26]: wrist_pitch_l_joint   → hw[21]
//   [27]: wrist_pitch_r_joint   → hw[28]
//   [28]: wrist_roll_l_joint    → hw[22]
//   [29]: wrist_roll_r_joint    → hw[29]

const int HdmiPolicyRunner::ISAAC_TO_HW[30] = {
    12, 0, 6, 15, 16, 23, 1, 7, 14, 17,
    24, 2, 8, 13, 18, 25, 3, 9, 19, 26,
    4, 10, 20, 27, 5, 11, 21, 28, 22, 29
};

const int HdmiPolicyRunner::HW_TO_ISAAC[30] = {
    1, 6, 11, 16, 20, 24, 2, 7, 12, 17,
    21, 25, 0, 13, 8, 3, 4, 9, 14, 18,
    22, 26, 28, 5, 10, 15, 19, 23, 27, 29
};

// ============================================================================
// Initialization
// ============================================================================

void HdmiPolicyRunner::init(const std::string& policy_dir, const std::string& policy_name) {
    std::string yaml_path = policy_dir + "/" + policy_name + ".yaml";
    std::string onnx_path = policy_dir + "/" + policy_name + ".onnx";
    std::string json_path = policy_dir + "/" + policy_name + ".json";

    // Load policy configuration from YAML
    loadPolicyConfig(yaml_path);

    // Load ONNX model
    onnx_.loadModel(onnx_path, json_path);

    // Load motion data
    motion_.load(policy_dir, isaac_joint_names_, policy_body_names_,
                 policy_joint_names_, future_steps_, motion_duration_sec_);

    // Initialize state vectors
    q_isaac_ = Eigen::VectorXd::Zero(NUM_JOINTS);
    imu_data_ = Eigen::VectorXd::Zero(9);
    depth_.resize(ObservationBuilder::DEPTH_SIZE, 0.0f);

    q_d_isaac_ = Eigen::VectorXd::Zero(NUM_JOINTS);
    qdot_d_isaac_ = Eigen::VectorXd::Zero(NUM_JOINTS);
    tor_d_isaac_ = Eigen::VectorXd::Zero(NUM_JOINTS);

    start_pos_ = Eigen::VectorXd::Zero(NUM_JOINTS);
    last_action_.resize(NUM_JOINTS, 0.0f);

    obs_builder_.reset();
    motion_.reset();

    state_ = STOP;
    step_count_ = 0;
    zero_count_ = 0;
    disable_joints_ = false;

    std::cout << "[HdmiPolicyRunner] Initialized successfully" << std::endl;
}

void HdmiPolicyRunner::loadPolicyConfig(const std::string& yaml_path) {
    YAML::Node config = YAML::LoadFile(yaml_path);

    // isaac_joint_names (30 joints)
    isaac_joint_names_.clear();
    for (auto& name : config["isaac_joint_names"]) {
        isaac_joint_names_.push_back(name.as<std::string>());
    }

    // policy body names (17 bodies from command.ref_body_pos_future_local.body_names)
    policy_body_names_.clear();
    for (auto& name : config["observation"]["command"]["ref_body_pos_future_local"]["body_names"]) {
        policy_body_names_.push_back(name.as<std::string>());
    }

    // policy joint names (24 joints from command.ref_body_pos_future_local.joint_names)
    policy_joint_names_.clear();
    for (auto& name : config["observation"]["command"]["ref_body_pos_future_local"]["joint_names"]) {
        policy_joint_names_.push_back(name.as<std::string>());
    }

    // future steps
    future_steps_.clear();
    for (auto& s : config["observation"]["command"]["ref_body_pos_future_local"]["future_steps"]) {
        future_steps_.push_back(s.as<int>());
    }

    // motion duration
    motion_duration_sec_ = config["observation"]["command"]["ref_body_pos_future_local"]["motion_duration_second"].as<double>();

    // action scale
    action_scale_ = config["action_scale"][".*"].as<double>();

    // default joint positions (isaac order)
    default_joint_pos_ = Eigen::VectorXd::Zero(NUM_JOINTS);
    auto default_pos_node = config["default_joint_pos"];
    for (int i = 0; i < NUM_JOINTS; i++) {
        const std::string& jname = isaac_joint_names_[i];
        if (default_pos_node[jname]) {
            default_joint_pos_(i) = default_pos_node[jname].as<double>();
        }
    }

    // PD gains (regex patterns → isaac order)
    kp_isaac_ = Eigen::VectorXd::Zero(NUM_JOINTS);
    kd_isaac_ = Eigen::VectorXd::Zero(NUM_JOINTS);

    auto kp_node = config["joint_kp"];
    auto kd_node = config["joint_kd"];

    for (int i = 0; i < NUM_JOINTS; i++) {
        const std::string& jname = isaac_joint_names_[i];
        // Match against regex patterns in the YAML
        for (auto it = kp_node.begin(); it != kp_node.end(); ++it) {
            std::string pattern = it->first.as<std::string>();
            try {
                std::regex re(pattern);
                if (std::regex_match(jname, re)) {
                    kp_isaac_(i) = it->second.as<double>();
                    break;
                }
            } catch (...) {
                // Exact match fallback
                if (pattern == jname) {
                    kp_isaac_(i) = it->second.as<double>();
                    break;
                }
            }
        }
        for (auto it = kd_node.begin(); it != kd_node.end(); ++it) {
            std::string pattern = it->first.as<std::string>();
            try {
                std::regex re(pattern);
                if (std::regex_match(jname, re)) {
                    kd_isaac_(i) = it->second.as<double>();
                    break;
                }
            } catch (...) {
                if (pattern == jname) {
                    kd_isaac_(i) = it->second.as<double>();
                    break;
                }
            }
        }
    }

    std::cout << "[HdmiPolicyRunner] Config loaded: "
              << isaac_joint_names_.size() << " isaac joints, "
              << policy_body_names_.size() << " policy bodies, "
              << policy_joint_names_.size() << " policy joints, "
              << "action_scale=" << action_scale_ << std::endl;
    std::cout << "[HdmiPolicyRunner] Default pos head: ";
    for (int i = 0; i < 5; i++) std::cout << default_joint_pos_(i) << " ";
    std::cout << "..." << std::endl;
}

// ============================================================================
// Runtime
// ============================================================================

void HdmiPolicyRunner::updateState(const Eigen::VectorXd& q_isaac,
                                    const Eigen::VectorXd& imu_data,
                                    const std::vector<float>& depth_frame) {
    q_isaac_ = q_isaac;
    imu_data_ = imu_data;
    if (!depth_frame.empty()) {
        depth_ = depth_frame;
    }
}

void HdmiPolicyRunner::runFSM(const xbox_flag& flag) {
    // State transitions based on joystick
    if (flag.fsm_state_command == "gotoSTOP") {
        if (state_ != STOP) {
            state_ = STOP;
            std::cout << "[HdmiPolicyRunner] → STOP" << std::endl;
        }
    } else if (flag.fsm_state_command == "gotoZero") {
        if (state_ == STOP) {
            state_ = ZERO;
            zero_count_ = 0;
            start_pos_ = q_isaac_;
            motion_.reset();
            obs_builder_.reset();
            std::cout << "[HdmiPolicyRunner] → ZERO" << std::endl;
        }
    } else if (flag.fsm_state_command == "gotoMlp") {
        if (state_ == ZERO) {
            state_ = MLP;
            step_count_ = 0;
            std::cout << "[HdmiPolicyRunner] → MLP" << std::endl;
        }
    }

    if (flag.is_disable) {
        disable_joints_ = true;
        return;
    }

    // Execute current state
    switch (state_) {
        case STOP:
            // Hold current position
            q_d_isaac_ = q_isaac_;
            qdot_d_isaac_.setZero();
            tor_d_isaac_.setZero();
            break;

        case ZERO: {
            // Interpolate from start_pos to default_joint_pos
            zero_count_++;
            double ratio = std::min(1.0, static_cast<double>(zero_count_) / ZERO_DURATION);
            q_d_isaac_ = start_pos_ + ratio * (default_joint_pos_ - start_pos_);
            qdot_d_isaac_.setZero();
            tor_d_isaac_.setZero();
            break;
        }

        case MLP: {
            // Run policy at 50Hz (every freq_ratio_ steps)
            if (step_count_ % freq_ratio_ == 0) {
                runPolicy();
            }
            step_count_++;

            // Apply last action: q_d = default_pos + action * action_scale
            for (int i = 0; i < NUM_JOINTS; i++) {
                q_d_isaac_(i) = default_joint_pos_(i) + last_action_[i] * action_scale_;
            }
            qdot_d_isaac_.setZero();
            tor_d_isaac_.setZero();
            break;
        }
    }
}

void HdmiPolicyRunner::runPolicy() {
    // 1. Update observation builder with current sensor data
    Eigen::Vector3d ang_vel(imu_data_(3), imu_data_(4), imu_data_(5));
    Eigen::Vector3d euler(imu_data_(0), imu_data_(1), imu_data_(2));
    obs_builder_.update(q_isaac_, ang_vel, euler);

    // 2. Get command observation from motion data (advances time internally)
    std::vector<float> command_obs = motion_.step();

    // 3. Build policy observation
    std::vector<float> policy_obs = obs_builder_.buildPolicyObs();

    // 4. Run ONNX inference
    last_action_ = onnx_.run(command_obs, policy_obs, depth_);

    // 4.5. Clip actions (safety guard, matches Python reference)
    for (auto& a : last_action_) {
        a = std::max(-100.0f, std::min(100.0f, a));
    }

    // 5. Record action for prev_actions history
    obs_builder_.recordAction(last_action_);
}

Eigen::VectorXd HdmiPolicyRunner::getDesiredPos() const {
    return q_d_isaac_;
}

Eigen::VectorXd HdmiPolicyRunner::getDesiredVel() const {
    return qdot_d_isaac_;
}

Eigen::VectorXd HdmiPolicyRunner::getDesiredTor() const {
    return tor_d_isaac_;
}
