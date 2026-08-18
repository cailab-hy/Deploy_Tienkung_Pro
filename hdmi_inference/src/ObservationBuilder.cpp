#include "ObservationBuilder.h"
#include <cstring>
#include <cassert>

constexpr int ObservationBuilder::JOINT_HIST_STEPS[6];

ObservationBuilder::ObservationBuilder() {
    init(NUM_JOINTS);
}

void ObservationBuilder::init(int num_actions) {
    num_actions_ = num_actions;
    // 3 (ang_vel) + 3 (gravity) + 30*6 (joint_pos_hist) + num_actions*3 (prev_actions)
    policy_obs_dim_ = 3 + 3 + NUM_JOINTS * 6 + num_actions_ * PREV_ACTION_STEPS;

    prev_actions_.assign(num_actions_, std::vector<float>(PREV_ACTION_STEPS, 0.0f));
    reset();
}

void ObservationBuilder::reset() {
    std::memset(joint_pos_history_, 0, sizeof(joint_pos_history_));
    std::memset(ang_vel_history_, 0, sizeof(ang_vel_history_));
    std::memset(projected_gravity_, 0, sizeof(projected_gravity_));
    for (auto& acts : prev_actions_) {
        std::fill(acts.begin(), acts.end(), 0.0f);
    }
    // Default projected gravity (standing upright)
    projected_gravity_[2] = -1.0;
}

void ObservationBuilder::update(const Eigen::VectorXd& joint_pos_isaac,
                                 const Eigen::Vector3d& ang_vel_body,
                                 const Eigen::Vector3d& euler) {
    assert(joint_pos_isaac.size() == NUM_JOINTS);

    // Shift joint position history (roll forward)
    for (int i = JOINT_HIST_BUF_SIZE - 1; i > 0; i--) {
        std::memcpy(joint_pos_history_[i], joint_pos_history_[i - 1], sizeof(double) * NUM_JOINTS);
    }
    // Write current joint positions to step 0
    for (int j = 0; j < NUM_JOINTS; j++) {
        joint_pos_history_[0][j] = joint_pos_isaac(j);
    }

    // Update angular velocity (step 0 only)
    ang_vel_history_[0] = ang_vel_body(0);
    ang_vel_history_[1] = ang_vel_body(1);
    ang_vel_history_[2] = ang_vel_body(2);

    // Compute projected gravity from euler angles
    double quat[4];
    eulerToQuat(euler(0), euler(1), euler(2), quat);
    computeProjectedGravity(quat, projected_gravity_);
}

void ObservationBuilder::recordAction(const std::vector<float>& action) {
    assert(static_cast<int>(action.size()) == num_actions_);

    // Shift previous actions (roll: step[2] = step[1], step[1] = step[0], step[0] = new)
    for (int j = 0; j < num_actions_; j++) {
        for (int s = PREV_ACTION_STEPS - 1; s > 0; s--) {
            prev_actions_[j][s] = prev_actions_[j][s - 1];
        }
        prev_actions_[j][0] = action[j];
    }
}

std::vector<float> ObservationBuilder::buildPolicyObs() const {
    std::vector<float> obs;
    obs.reserve(policy_obs_dim_);

    // 1. root_ang_vel_history (step=0): [3]
    obs.push_back(static_cast<float>(ang_vel_history_[0]));
    obs.push_back(static_cast<float>(ang_vel_history_[1]));
    obs.push_back(static_cast<float>(ang_vel_history_[2]));

    // 2. projected_gravity_history (step=0): [3]
    obs.push_back(static_cast<float>(projected_gravity_[0]));
    obs.push_back(static_cast<float>(projected_gravity_[1]));
    obs.push_back(static_cast<float>(projected_gravity_[2]));

    // 3. joint_pos_history (steps=[0,1,2,3,4,8]): [30*6 = 180]
    for (int si = 0; si < 6; si++) {
        int step = JOINT_HIST_STEPS[si];
        for (int j = 0; j < NUM_JOINTS; j++) {
            obs.push_back(static_cast<float>(joint_pos_history_[step][j]));
        }
    }

    // 4. prev_actions (3 steps): [num_actions*3]
    // Layout: action0_t0, action0_t1, action0_t2, action1_t0, action1_t1, action1_t2, ...
    for (int j = 0; j < num_actions_; j++) {
        for (int s = 0; s < PREV_ACTION_STEPS; s++) {
            obs.push_back(prev_actions_[j][s]);
        }
    }

    assert(static_cast<int>(obs.size()) == policy_obs_dim_);
    return obs;
}

void ObservationBuilder::eulerToQuat(double yaw, double pitch, double roll, double* quat) {
    // ZYX convention: yaw (Z) → pitch (Y) → roll (X)
    double cy = std::cos(yaw * 0.5);
    double sy = std::sin(yaw * 0.5);
    double cp = std::cos(pitch * 0.5);
    double sp = std::sin(pitch * 0.5);
    double cr = std::cos(roll * 0.5);
    double sr = std::sin(roll * 0.5);

    // Quaternion (w, x, y, z)
    quat[0] = cr * cp * cy + sr * sp * sy;  // w
    quat[1] = sr * cp * cy - cr * sp * sy;  // x
    quat[2] = cr * sp * cy + sr * cp * sy;  // y
    quat[3] = cr * cp * sy - sr * sp * cy;  // z
}

void ObservationBuilder::computeProjectedGravity(const double* quat, double* gravity) {
    // quat_rotate_inverse(quat, [0, 0, -1])
    double qw = quat[0], qx = quat[1], qy = quat[2], qz = quat[3];

    // v = [0, 0, -1]
    double vz = -1.0;

    // a = v * (2*qw^2 - 1)
    double s = 2.0 * qw * qw - 1.0;
    // only vz is non-zero
    double az = vz * s;

    // b = cross(q_vec, v) * qw * 2
    // cross([qx,qy,qz], [0,0,-1]) = [-qy, qx, 0]
    double bx = -qy * 2.0 * qw;
    double by = qx * 2.0 * qw;
    double bz = 0.0;

    // c = q_vec * dot(q_vec, v) * 2
    // dot([qx,qy,qz], [0,0,-1]) = -qz
    double dot = -qz;
    double cx = qx * dot * 2.0;
    double cy_val = qy * dot * 2.0;
    double cz = qz * dot * 2.0;

    // result = a - b + c
    gravity[0] = 0.0 - bx + cx;      // ax=0
    gravity[1] = 0.0 - by + cy_val;  // ay=0
    gravity[2] = az - bz + cz;
}
