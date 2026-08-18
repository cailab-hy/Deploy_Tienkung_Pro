#ifndef OBSERVATION_BUILDER_H_
#define OBSERVATION_BUILDER_H_

#include <vector>
#include <cmath>
#include <Eigen/Dense>

/**
 * @brief Builds policy observation [276] and manages observation history buffers.
 *
 * Policy observation layout:
 *   root_ang_vel_history (step=0):          3
 *   projected_gravity_history (step=0):     3
 *   joint_pos_history (steps=[0,1,2,3,4,8]): 30*6 = 180
 *   prev_actions (3 steps):                 30*3 = 90
 *   Total:                                  276
 */
class ObservationBuilder {
public:
    static constexpr int NUM_JOINTS = 30;
    static constexpr int POLICY_OBS_DIM = 276;
    static constexpr int DEPTH_H = 120;
    static constexpr int DEPTH_W = 160;
    static constexpr int DEPTH_SIZE = DEPTH_H * DEPTH_W;

    ObservationBuilder();

    /**
     * @brief Update sensor data and shift history buffers.
     *        Call this once per policy step (50Hz).
     * @param joint_pos_isaac Joint positions in isaac order [30]
     * @param ang_vel_body Angular velocity in body frame [3] (from IMU)
     * @param euler Euler angles [yaw, pitch, roll] (from IMU)
     */
    void update(const Eigen::VectorXd& joint_pos_isaac,
                const Eigen::Vector3d& ang_vel_body,
                const Eigen::Vector3d& euler);

    /**
     * @brief Record the action taken this step (for prev_actions history)
     * @param action Action vector in isaac order [30]
     */
    void recordAction(const std::vector<float>& action);

    /**
     * @brief Build the policy observation vector [276]
     */
    std::vector<float> buildPolicyObs() const;

    /** @brief Reset all history buffers */
    void reset();

private:
    // History buffers
    // joint_pos_history: circular buffer, max history step = 8, so buffer_size = 9
    static constexpr int JOINT_HIST_BUF_SIZE = 9;
    static constexpr int JOINT_HIST_STEPS[6] = {0, 1, 2, 3, 4, 8};
    double joint_pos_history_[JOINT_HIST_BUF_SIZE][NUM_JOINTS] = {};

    // Angular velocity history (only step=0 used, buffer_size=1)
    double ang_vel_history_[3] = {};

    // Projected gravity history (only step=0 used, buffer_size=1)
    double projected_gravity_[3] = {};

    // Previous actions: [NUM_JOINTS, 3] (layout: action_i_step0, action_i_step1, action_i_step2, ...)
    static constexpr int PREV_ACTION_STEPS = 3;
    float prev_actions_[NUM_JOINTS][PREV_ACTION_STEPS] = {};

    // --- Helper functions ---

    /**
     * @brief Convert euler (yaw, pitch, roll) to quaternion (w, x, y, z)
     */
    static void eulerToQuat(double yaw, double pitch, double roll, double* quat);

    /**
     * @brief Compute projected gravity: quat_rotate_inverse(quat, [0,0,-1])
     */
    static void computeProjectedGravity(const double* quat, double* gravity);
};

#endif // OBSERVATION_BUILDER_H_
