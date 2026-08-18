#ifndef MOTION_DATA_H_
#define MOTION_DATA_H_

#include <string>
#include <vector>
#include <cstddef>

/**
 * @brief Motion reference data loader and command observation builder.
 *
 * Loads motion.npz + meta.json, remaps joints from meta order to isaac order,
 * and computes command observations (ref_body_pos_future_local, ref_joint_pos_future, ref_motion_phase).
 */
class MotionData {
public:
    MotionData() = default;

    /**
     * @brief Load motion data from directory containing motion.npz and meta.json
     * @param motion_dir Path to directory (e.g. .../policy_kbc)
     * @param motion_subpath Relative path to motion data within config (e.g. "data/motion/tienkung_pro/carry_and_place_bread_box")
     * @param isaac_joint_names Joint names in policy/isaac order (30 joints)
     * @param policy_body_names Body names used by policy (17 bodies)
     * @param policy_joint_names Joint names used by policy for command obs (24 joints)
     * @param future_steps Future timesteps for command obs (e.g. [1,2,8,16,32])
     * @param motion_duration_sec Duration in seconds (e.g. 9.74)
     */
    void load(const std::string& motion_dir,
              const std::vector<std::string>& isaac_joint_names,
              const std::vector<std::string>& policy_body_names,
              const std::vector<std::string>& policy_joint_names,
              const std::vector<int>& future_steps,
              double motion_duration_sec);

    /**
     * @brief Advance time by one step and compute command observation
     * @param root_quat Current robot root quaternion [w,x,y,z] (from motion data, not IMU)
     * @return Command observation vector [376]
     */
    std::vector<float> step();

    /** @brief Reset motion time to 0 */
    void reset();

    /** @brief Get current motion timestep */
    int currentStep() const { return t_; }

    /** @brief Get total number of motion steps */
    int numSteps() const { return num_steps_; }

    bool isLoaded() const { return loaded_; }

private:
    bool loaded_ = false;

    // Motion data arrays (after joint remapping to isaac order)
    // body data stays in meta.json order
    int num_steps_ = 0;
    int num_bodies_meta_ = 0;  // 34 (meta.json)
    int num_joints_isaac_ = 0; // 30 (isaac order)

    // Raw motion data [num_steps x ...]
    std::vector<double> body_pos_w_;    // [num_steps, num_bodies_meta, 3]
    std::vector<double> body_quat_w_;   // [num_steps, num_bodies_meta, 4]
    std::vector<double> joint_pos_;     // [num_steps, num_joints_isaac] (remapped to isaac order)

    // Index mappings
    std::vector<int> policy_body_indices_;   // indices into meta body_names for the 17 policy bodies
    std::vector<int> policy_joint_indices_;  // indices into isaac joint_names for the 24 policy joints
    int root_body_idx_ = 0;                  // index of "pelvis" in meta body_names

    // Time management
    int t_ = 0;
    int motion_steps_ = 0;  // int(duration * 50)
    std::vector<int> future_steps_;
    int n_future_steps_ = 0;
    int n_policy_bodies_ = 0;
    int n_policy_joints_ = 0;

    // --- Quaternion math helpers ---
    static void yaw_quat(const double* q_in, double* q_out);
    static void quat_rotate_inverse(const double* q, const double* v, double* out);
};

#endif // MOTION_DATA_H_
