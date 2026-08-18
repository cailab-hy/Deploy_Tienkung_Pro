#ifndef HDMI_POLICY_RUNNER_H_
#define HDMI_POLICY_RUNNER_H_

#include <string>
#include <vector>
#include <Eigen/Dense>
#include "OnnxInference.h"
#include "MotionData.h"
#include "ObservationBuilder.h"
#include "Joystick.h"

/**
 * @brief HDMI policy runner with FSM.
 * Replaces x_humanoid_rl_sdk's RobotInterface + RobotFSM for HDMI-trained policies.
 *
 * FSM states: STOP → ZERO → MLP
 * - STOP: hold current position
 * - ZERO: interpolate to default joint positions over 500 steps (1.25s at 400Hz)
 * - MLP:  run ONNX policy at 50Hz (every 8th step of the 400Hz loop)
 */
class HdmiPolicyRunner {
public:
    enum State { STOP = 0, ZERO = 1, MLP = 2 };

    HdmiPolicyRunner() = default;

    /**
     * @brief Initialize the runner
     * @param policy_dir Directory containing .onnx, .json, .yaml, motion.npz, meta.json
     * @param policy_name Base name of policy files (e.g. "policy-fmtq3g5b-final")
     */
    void init(const std::string& policy_dir, const std::string& policy_name);

    /**
     * @brief Update sensor state (call every 400Hz step)
     * @param q_isaac Joint positions in isaac order [30], after S/P conversion
     * @param imu_data IMU data [yaw, pitch, roll, gx, gy, gz, ax, ay, az]
     * @param depth_frame Depth image [120*160] float32
     */
    void updateState(const Eigen::VectorXd& q_isaac,
                     const Eigen::VectorXd& imu_data,
                     const std::vector<float>& depth_frame);

    /**
     * @brief Run FSM (call every 400Hz step, after updateState)
     * @param flag Joystick flags
     */
    void runFSM(const xbox_flag& flag);

    /**
     * @brief Get desired joint positions in isaac order [30]
     */
    Eigen::VectorXd getDesiredPos() const;

    /**
     * @brief Get desired joint velocities in isaac order [30]
     */
    Eigen::VectorXd getDesiredVel() const;

    /**
     * @brief Get desired joint torques in isaac order [30]
     */
    Eigen::VectorXd getDesiredTor() const;

    /** @brief Get current FSM state */
    State getCurrentState() const { return state_; }

    /** @brief Get PD gains in isaac order (to be remapped to hw order by caller) */
    const Eigen::VectorXd& getKp() const { return kp_isaac_; }
    const Eigen::VectorXd& getKd() const { return kd_isaac_; }

    /** @brief Whether joints should be disabled (emergency stop) */
    bool disableJoints() const { return disable_joints_; }

    // Joint mapping arrays (hw ↔ isaac)
    static constexpr int NUM_JOINTS = 30;
    static const int HW_TO_ISAAC[NUM_JOINTS];
    static const int ISAAC_TO_HW[NUM_JOINTS];

private:
    // Components
    OnnxInference onnx_;
    MotionData motion_;
    ObservationBuilder obs_builder_;

    // FSM
    State state_ = STOP;
    bool disable_joints_ = false;

    // Timing
    int freq_ratio_ = 8;     // 400Hz / 50Hz
    int step_count_ = 0;

    // ZERO state interpolation
    int zero_count_ = 0;
    static constexpr int ZERO_DURATION = 500;  // 500 steps at 400Hz = 1.25s
    Eigen::VectorXd start_pos_;  // position when entering ZERO state

    // Default joint positions (from policy YAML)
    Eigen::VectorXd default_joint_pos_;  // [30] in isaac order

    // Action scale
    double action_scale_ = 0.25;

    // Current state
    Eigen::VectorXd q_isaac_;       // current joint positions [30]
    Eigen::VectorXd imu_data_;      // [9]
    std::vector<float> depth_;      // [120*160]

    // Desired outputs (isaac order)
    Eigen::VectorXd q_d_isaac_;     // desired position [30]
    Eigen::VectorXd qdot_d_isaac_;  // desired velocity [30]
    Eigen::VectorXd tor_d_isaac_;   // desired torque [30]

    // Last policy action (held between 50Hz steps)
    std::vector<float> last_action_;

    // PD gains in isaac order
    Eigen::VectorXd kp_isaac_;
    Eigen::VectorXd kd_isaac_;

    // Policy configuration
    std::vector<std::string> isaac_joint_names_;
    std::vector<std::string> policy_body_names_;
    std::vector<std::string> policy_joint_names_;
    std::vector<int> future_steps_;
    double motion_duration_sec_ = 9.74;

    /** @brief Parse policy YAML for configuration */
    void loadPolicyConfig(const std::string& yaml_path);

    /** @brief Run one policy inference step */
    void runPolicy();
};

#endif // HDMI_POLICY_RUNNER_H_
