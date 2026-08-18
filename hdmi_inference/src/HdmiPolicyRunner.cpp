#include "HdmiPolicyRunner.h"
#include <yaml-cpp/yaml.h>
#include <iostream>
#include <regex>
#include <cassert>
#include <algorithm>

// Joint mapping arrays (same as rl_control_new version)
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

void HdmiPolicyRunner::init(const std::string& stand1_dir, const std::string& stand1_name,
                             const std::string& pickandplace_dir, const std::string& pickandplace_name,
                             const std::string& stand2_dir, const std::string& stand2_name,
                             const std::string& return_dir, const std::string& return_name) {
    // Load shared config from stand1 policy YAML (all policies share the same observation/action space)
    loadPolicyConfig(stand1_dir + "/" + stand1_name + ".yaml");

    struct PolicyInit {
        PolicyId id;
        const std::string& dir;
        const std::string& name;
        const char* label;
    };
    const PolicyInit init_list[NUM_POLICIES] = {
        {POLICY_STAND1,        stand1_dir,        stand1_name,        "stand1"},
        {POLICY_PICKANDPLACE,  pickandplace_dir,  pickandplace_name,  "pickandplace"},
        {POLICY_STAND2,        stand2_dir,        stand2_name,        "stand2"},
        {POLICY_RETURN,        return_dir,        return_name,        "return"},
    };

    for (const auto& p : init_list) {
        policies_[p.id].name = p.label;
        policies_[p.id].onnx.loadModel(p.dir + "/" + p.name + ".onnx",
                                       p.dir + "/" + p.name + ".json");
        YAML::Node cfg = YAML::LoadFile(p.dir + "/" + p.name + ".yaml");
        policies_[p.id].motion_duration_sec =
            cfg["observation"]["command"]["ref_body_pos_future_local"]["motion_duration_second"].as<double>();
        policies_[p.id].motion.load(p.dir, isaac_joint_names_, policy_body_names_,
                                    policy_joint_names_, future_steps_,
                                    policies_[p.id].motion_duration_sec);
    }

    // Initialize ObservationBuilder with correct action dimension
    obs_builder_.init(num_action_joints_);

    q_isaac_ = Eigen::VectorXd::Zero(NUM_JOINTS);
    imu_data_ = Eigen::VectorXd::Zero(9);
    depth_.resize(ObservationBuilder::DEPTH_SIZE, 0.0f);

    q_d_isaac_ = Eigen::VectorXd::Zero(NUM_JOINTS);
    qdot_d_isaac_ = Eigen::VectorXd::Zero(NUM_JOINTS);
    tor_d_isaac_ = Eigen::VectorXd::Zero(NUM_JOINTS);

    start_pos_ = Eigen::VectorXd::Zero(NUM_JOINTS);
    last_action_.resize(num_action_joints_, 0.0f);

    for (int i = 0; i < NUM_POLICIES; i++) {
        policies_[i].motion.reset();
    }

    active_policy_ = POLICY_STAND1;
    cycle_step_count_ = 0;
    b_was_pressed_ = false;
    state_ = STOP;
    zero_count_ = 0;
    disable_joints_ = false;

    std::cout << "[HdmiPolicyRunner] Initialized with " << NUM_POLICIES << " policies: "
              << "stand1 ("       << policies_[POLICY_STAND1].motion_duration_sec       << "s), "
              << "pickandplace (" << policies_[POLICY_PICKANDPLACE].motion_duration_sec << "s), "
              << "stand2 ("       << policies_[POLICY_STAND2].motion_duration_sec       << "s), "
              << "return ("       << policies_[POLICY_RETURN].motion_duration_sec       << "s)" << std::endl;
}

void HdmiPolicyRunner::loadPolicyConfig(const std::string& yaml_path) {
    YAML::Node config = YAML::LoadFile(yaml_path);

    // Read isaac_joint_names (30 hardware joints)
    isaac_joint_names_.clear();
    for (const auto& name : config["isaac_joint_names"]) {
        isaac_joint_names_.push_back(name.as<std::string>());
    }

    // Read command observation body/joint names
    policy_body_names_.clear();
    for (const auto& name : config["observation"]["command"]["ref_body_pos_future_local"]["body_names"]) {
        policy_body_names_.push_back(name.as<std::string>());
    }

    policy_joint_names_.clear();
    for (const auto& name : config["observation"]["command"]["ref_body_pos_future_local"]["joint_names"]) {
        policy_joint_names_.push_back(name.as<std::string>());
    }

    future_steps_.clear();
    for (const auto& s : config["observation"]["command"]["ref_body_pos_future_local"]["future_steps"]) {
        future_steps_.push_back(s.as<int>());
    }

    // Read action joint names (top-level policy_joint_names)
    // This determines the ONNX output dimension and action-to-isaac mapping
    action_joint_names_.clear();
    for (const auto& name : config["policy_joint_names"]) {
        action_joint_names_.push_back(name.as<std::string>());
    }
    num_action_joints_ = static_cast<int>(action_joint_names_.size());

    // Build action → isaac index mapping
    action_to_isaac_.clear();
    for (const auto& aname : action_joint_names_) {
        auto it = std::find(isaac_joint_names_.begin(), isaac_joint_names_.end(), aname);
        if (it == isaac_joint_names_.end()) {
            throw std::runtime_error("Action joint '" + aname + "' not found in isaac_joint_names");
        }
        action_to_isaac_.push_back(static_cast<int>(std::distance(isaac_joint_names_.begin(), it)));
    }

    // Parse action_scale: try ".*" first, then per-joint regex matching
    action_scales_.resize(num_action_joints_, 0.25);  // default 0.25
    auto action_scale_node = config["action_scale"];
    if (action_scale_node[".*"]) {
        double global_scale = action_scale_node[".*"].as<double>();
        std::fill(action_scales_.begin(), action_scales_.end(), global_scale);
    } else {
        // Per-pattern matching (like kp/kd parsing)
        for (int i = 0; i < num_action_joints_; i++) {
            const std::string& jname = action_joint_names_[i];
            for (auto it = action_scale_node.begin(); it != action_scale_node.end(); ++it) {
                std::string pattern = it->first.as<std::string>();
                try {
                    std::regex re(pattern);
                    if (std::regex_match(jname, re)) {
                        action_scales_[i] = it->second.as<double>();
                        break;
                    }
                } catch (...) {
                    if (pattern == jname) {
                        action_scales_[i] = it->second.as<double>();
                        break;
                    }
                }
            }
        }
    }

    // Default joint positions (all 30 isaac joints)
    default_joint_pos_ = Eigen::VectorXd::Zero(NUM_JOINTS);
    auto default_pos_node = config["default_joint_pos"];
    for (int i = 0; i < NUM_JOINTS; i++) {
        const std::string& jname = isaac_joint_names_[i];
        if (default_pos_node[jname]) {
            default_joint_pos_(i) = default_pos_node[jname].as<double>();
        }
    }

    // PD gains (all 30 isaac joints)
    kp_isaac_ = Eigen::VectorXd::Zero(NUM_JOINTS);
    kd_isaac_ = Eigen::VectorXd::Zero(NUM_JOINTS);

    auto kp_node = config["joint_kp"];
    auto kd_node = config["joint_kd"];

    for (int i = 0; i < NUM_JOINTS; i++) {
        const std::string& jname = isaac_joint_names_[i];
        for (auto it = kp_node.begin(); it != kp_node.end(); ++it) {
            std::string pattern = it->first.as<std::string>();
            try {
                std::regex re(pattern);
                if (std::regex_match(jname, re)) {
                    kp_isaac_(i) = it->second.as<double>();
                    break;
                }
            } catch (...) {
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
              << num_action_joints_ << " action joints, "
              << policy_body_names_.size() << " policy bodies, "
              << policy_joint_names_.size() << " command joints" << std::endl;

    // Log which joints are fixed (not in action_joint_names)
    for (int i = 0; i < NUM_JOINTS; i++) {
        if (std::find(action_to_isaac_.begin(), action_to_isaac_.end(), i) == action_to_isaac_.end()) {
            std::cout << "[HdmiPolicyRunner] Fixed joint: " << isaac_joint_names_[i]
                      << " (isaac idx=" << i << ", held at " << default_joint_pos_(i) << ")" << std::endl;
        }
    }
}

void HdmiPolicyRunner::switchPolicy(PolicyId id) {
    if (id == active_policy_) return;
    active_policy_ = id;
    policies_[id].motion.reset();
    cycle_step_count_ = 0;
    // Do NOT reset obs_builder_ or last_action_ here.
    // All policies share the same observation space, so the accumulated
    // history (joint_pos, ang_vel, prev_actions) from the previous policy
    // is valid and provides continuity. Resetting to zeros causes spasms
    // because the policy receives unrealistic observations.
    std::cout << "[HdmiPolicyRunner] Switched to policy: " << policies_[id].name << std::endl;
}

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
    // State transitions
    if (flag.fsm_state_command == "gotoStop") {
        if (state_ != STOP) {
            state_ = STOP;
            std::cout << "[HdmiPolicyRunner] -> STOP" << std::endl;
        }
    } else if (flag.fsm_state_command == "gotoZero") {
        if (state_ == STOP) {
            state_ = ZERO;
            zero_count_ = 0;
            start_pos_ = q_isaac_;
            for (int i = 0; i < NUM_POLICIES; i++) {
                policies_[i].motion.reset();
            }
            obs_builder_.init(num_action_joints_);
            std::cout << "[HdmiPolicyRunner] -> ZERO" << std::endl;
        }
    } else if (flag.fsm_state_command == "gotoMLP") {
        if (state_ == ZERO) {
            state_ = MLP;
            active_policy_ = POLICY_STAND1;
            cycle_step_count_ = 0;
            b_was_pressed_ = false;
            for (int i = 0; i < NUM_POLICIES; i++) {
                policies_[i].motion.reset();
            }
            obs_builder_.init(num_action_joints_);
            std::cout << "[HdmiPolicyRunner] -> MLP (starting with stand1)" << std::endl;
        }
    }

    if (flag.is_disable) {
        disable_joints_ = true;
        return;
    }

    switch (state_) {
        case STOP:
            q_d_isaac_ = q_isaac_;
            qdot_d_isaac_.setZero();
            tor_d_isaac_.setZero();
            break;

        case ZERO: {
            zero_count_++;
            double ratio = std::min(1.0, static_cast<double>(zero_count_) / ZERO_DURATION);
            q_d_isaac_ = start_pos_ + ratio * (default_joint_pos_ - start_pos_);
            qdot_d_isaac_.setZero();
            tor_d_isaac_.setZero();
            break;
        }

        case MLP: {
            // B-button rising-edge detection:
            //   STAND1 -> PICKANDPLACE,  STAND2 -> RETURN
            bool b_pressed_now = (flag.policy_switch_request == 1);
            bool b_rising_edge = b_pressed_now && !b_was_pressed_;
            b_was_pressed_ = b_pressed_now;

            if (b_rising_edge) {
                if (active_policy_ == POLICY_STAND1) {
                    switchPolicy(POLICY_PICKANDPLACE);
                } else if (active_policy_ == POLICY_STAND2) {
                    switchPolicy(POLICY_RETURN);
                }
            }

            runPolicy();

            // Auto-transition after one full motion cycle:
            //   PICKANDPLACE -> STAND2,  RETURN -> STAND1
            int cycle_len = policies_[active_policy_].motion.numSteps();
            if (cycle_step_count_ >= cycle_len && cycle_len > 0) {
                if (active_policy_ == POLICY_PICKANDPLACE) {
                    switchPolicy(POLICY_STAND2);
                } else if (active_policy_ == POLICY_RETURN) {
                    switchPolicy(POLICY_STAND1);
                }
            }

            // Start from default positions for ALL joints
            q_d_isaac_ = default_joint_pos_;

            // Apply actions only to action joints (non-action joints stay at default)
            for (int i = 0; i < num_action_joints_; i++) {
                int isaac_idx = action_to_isaac_[i];
                q_d_isaac_(isaac_idx) = default_joint_pos_(isaac_idx) + last_action_[i] * action_scales_[i];
            }
            qdot_d_isaac_.setZero();
            tor_d_isaac_.setZero();
            break;
        }
    }
}

void HdmiPolicyRunner::runPolicy() {
    Eigen::Vector3d ang_vel(imu_data_(3), imu_data_(4), imu_data_(5));
    Eigen::Vector3d euler(imu_data_(0), imu_data_(1), imu_data_(2));
    obs_builder_.update(q_isaac_, ang_vel, euler);

    std::vector<float> command_obs = policies_[active_policy_].motion.step();
    cycle_step_count_++;
    std::vector<float> policy_obs = obs_builder_.buildPolicyObs();
    last_action_ = policies_[active_policy_].onnx.run(command_obs, policy_obs, depth_);

    for (auto& a : last_action_) {
        a = std::max(-100.0f, std::min(100.0f, a));
    }

    obs_builder_.recordAction(last_action_);
}

Eigen::VectorXd HdmiPolicyRunner::getDesiredPos() const { return q_d_isaac_; }
Eigen::VectorXd HdmiPolicyRunner::getDesiredVel() const { return qdot_d_isaac_; }
Eigen::VectorXd HdmiPolicyRunner::getDesiredTor() const { return tor_d_isaac_; }
