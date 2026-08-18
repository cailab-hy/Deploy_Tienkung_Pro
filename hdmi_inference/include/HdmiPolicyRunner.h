#ifndef HDMI_POLICY_RUNNER_H_
#define HDMI_POLICY_RUNNER_H_

#include <string>
#include <vector>
#include <Eigen/Dense>
#include "OnnxInference.h"
#include "MotionData.h"
#include "ObservationBuilder.h"
#include "Joystick.h"

enum PolicyId {
    POLICY_STAND1 = 0,
    POLICY_PICKANDPLACE = 1,
    POLICY_STAND2 = 2,
    POLICY_RETURN = 3,
    NUM_POLICIES = 4
};

struct PolicySlot {
    OnnxInference onnx;
    MotionData motion;
    double motion_duration_sec = 0.0;
    std::string name;
};

/**
 * @brief HDMI policy runner with FSM and multi-policy transition support.
 *
 * Runs at 50Hz on Jetson Orin. Transition sequence:
 *   STAND1 --(B button)--> PICKANDPLACE --(auto, 1 cycle)--> STAND2
 *        --(B button)--> RETURN --(auto, 1 cycle)--> STAND1 (repeat)
 *
 * FSM states: STOP -> ZERO -> MLP
 * - STOP: hold current position
 * - ZERO: interpolate to default joint positions over 63 steps (1.26s at 50Hz)
 * - MLP:  run ONNX policy every step (50Hz)
 */
class HdmiPolicyRunner {
public:
    enum State { STOP = 0, ZERO = 1, MLP = 2 };

    HdmiPolicyRunner() = default;

    void init(const std::string& stand1_dir, const std::string& stand1_name,
              const std::string& pickandplace_dir, const std::string& pickandplace_name,
              const std::string& stand2_dir, const std::string& stand2_name,
              const std::string& return_dir, const std::string& return_name);

    void updateState(const Eigen::VectorXd& q_isaac,
                     const Eigen::VectorXd& imu_data,
                     const std::vector<float>& depth_frame);

    void runFSM(const xbox_flag& flag);

    Eigen::VectorXd getDesiredPos() const;
    Eigen::VectorXd getDesiredVel() const;
    Eigen::VectorXd getDesiredTor() const;

    State getCurrentState() const { return state_; }
    PolicyId getActivePolicy() const { return active_policy_; }
    const Eigen::VectorXd& getKp() const { return kp_isaac_; }
    const Eigen::VectorXd& getKd() const { return kd_isaac_; }
    bool disableJoints() const { return disable_joints_; }

    static constexpr int NUM_JOINTS = 30;
    static const int HW_TO_ISAAC[NUM_JOINTS];
    static const int ISAAC_TO_HW[NUM_JOINTS];

private:
    PolicySlot policies_[NUM_POLICIES];
    PolicyId active_policy_ = POLICY_STAND1;
    int cycle_step_count_ = 0;     // motion steps since last policy switch
    bool b_was_pressed_ = false;   // for B-button rising-edge detection

    ObservationBuilder obs_builder_;

    State state_ = STOP;
    bool disable_joints_ = false;

    // ZERO interpolation: 63 steps at 50Hz = 1.26s
    int zero_count_ = 0;
    static constexpr int ZERO_DURATION = 63;
    Eigen::VectorXd start_pos_;

    Eigen::VectorXd default_joint_pos_;

    Eigen::VectorXd q_isaac_;
    Eigen::VectorXd imu_data_;
    std::vector<float> depth_;

    Eigen::VectorXd q_d_isaac_;
    Eigen::VectorXd qdot_d_isaac_;
    Eigen::VectorXd tor_d_isaac_;

    // Action mapping: policy action index → isaac joint index
    int num_action_joints_ = NUM_JOINTS;
    std::vector<int> action_to_isaac_;        // [num_action_joints_] maps action[i] → isaac index
    std::vector<double> action_scales_;       // [num_action_joints_] per-action-joint scale
    std::vector<float> last_action_;          // [num_action_joints_]

    Eigen::VectorXd kp_isaac_;
    Eigen::VectorXd kd_isaac_;

    std::vector<std::string> isaac_joint_names_;
    std::vector<std::string> policy_body_names_;
    std::vector<std::string> policy_joint_names_;  // command obs joints
    std::vector<std::string> action_joint_names_;  // action output joints
    std::vector<int> future_steps_;

    void loadPolicyConfig(const std::string& yaml_path);
    void runPolicy();
    void switchPolicy(PolicyId id);
};

#endif // HDMI_POLICY_RUNNER_H_
