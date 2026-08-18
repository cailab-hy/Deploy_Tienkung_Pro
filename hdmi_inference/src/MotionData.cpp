#include "MotionData.h"
#include "cnpy.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <cassert>

using json = nlohmann::json;

void MotionData::load(const std::string& motion_dir,
                      const std::vector<std::string>& isaac_joint_names,
                      const std::vector<std::string>& policy_body_names,
                      const std::vector<std::string>& policy_joint_names,
                      const std::vector<int>& future_steps,
                      double motion_duration_sec) {
    // Load meta.json
    std::string meta_path = motion_dir + "/meta.json";
    std::ifstream meta_file(meta_path);
    if (!meta_file.is_open()) {
        throw std::runtime_error("Failed to open meta.json: " + meta_path);
    }
    json meta;
    meta_file >> meta;

    std::vector<std::string> meta_body_names = meta["body_names"].get<std::vector<std::string>>();
    std::vector<std::string> meta_joint_names = meta["joint_names"].get<std::vector<std::string>>();
    double fps = meta["fps"].get<double>();

    num_bodies_meta_ = static_cast<int>(meta_body_names.size());
    num_joints_isaac_ = static_cast<int>(isaac_joint_names.size());

    // Load motion.npz
    std::string npz_path = motion_dir + "/motion.npz";
    auto npz = cnpy::npz_load(npz_path);

    auto& body_pos_arr = npz["body_pos_w"];     // [T, num_bodies_meta, 3]
    auto& body_quat_arr = npz["body_quat_w"];   // [T, num_bodies_meta, 4]
    auto& joint_pos_arr = npz["joint_pos"];      // [T, num_joints_meta]

    num_steps_ = static_cast<int>(body_pos_arr.shape[0]);
    int num_joints_meta = static_cast<int>(joint_pos_arr.shape[1]);

    std::cout << "[MotionData] Loaded: " << num_steps_ << " steps, "
              << num_bodies_meta_ << " bodies, " << num_joints_meta << " joints (meta)" << std::endl;

    // Copy body data directly (stays in meta order)
    body_pos_w_.resize(num_steps_ * num_bodies_meta_ * 3);
    body_quat_w_.resize(num_steps_ * num_bodies_meta_ * 4);

    const double* bp = body_pos_arr.data<double>();
    const double* bq = body_quat_arr.data<double>();
    std::copy(bp, bp + body_pos_w_.size(), body_pos_w_.data());
    std::copy(bq, bq + body_quat_w_.size(), body_quat_w_.data());

    // Remap joints: meta order → isaac order
    // Find shared joints between meta and isaac
    std::vector<int> src_indices, dest_indices;
    for (const auto& jname : meta_joint_names) {
        auto it = std::find(isaac_joint_names.begin(), isaac_joint_names.end(), jname);
        if (it != isaac_joint_names.end()) {
            int src_idx = static_cast<int>(&jname - &meta_joint_names[0]);
            int dest_idx = static_cast<int>(std::distance(isaac_joint_names.begin(), it));
            src_indices.push_back(src_idx);
            dest_indices.push_back(dest_idx);
        }
    }

    std::cout << "[MotionData] Joint remapping: " << src_indices.size()
              << " shared joints (meta → isaac)" << std::endl;

    // Remap joint_pos to isaac order
    joint_pos_.resize(num_steps_ * num_joints_isaac_, 0.0);
    const double* jp = joint_pos_arr.data<double>();
    for (int t = 0; t < num_steps_; t++) {
        for (size_t k = 0; k < src_indices.size(); k++) {
            joint_pos_[t * num_joints_isaac_ + dest_indices[k]] =
                jp[t * num_joints_meta + src_indices[k]];
        }
    }

    // Find root body index in meta body names
    auto root_it = std::find(meta_body_names.begin(), meta_body_names.end(), "pelvis");
    if (root_it == meta_body_names.end()) {
        throw std::runtime_error("Root body 'pelvis' not found in meta body_names");
    }
    root_body_idx_ = static_cast<int>(std::distance(meta_body_names.begin(), root_it));

    // Map policy body names to meta body indices
    policy_body_indices_.clear();
    for (const auto& name : policy_body_names) {
        auto it = std::find(meta_body_names.begin(), meta_body_names.end(), name);
        if (it == meta_body_names.end()) {
            throw std::runtime_error("Policy body '" + name + "' not found in meta body_names");
        }
        policy_body_indices_.push_back(static_cast<int>(std::distance(meta_body_names.begin(), it)));
    }

    // Map policy joint names to isaac joint indices
    policy_joint_indices_.clear();
    for (const auto& name : policy_joint_names) {
        auto it = std::find(isaac_joint_names.begin(), isaac_joint_names.end(), name);
        if (it == isaac_joint_names.end()) {
            throw std::runtime_error("Policy joint '" + name + "' not found in isaac_joint_names");
        }
        policy_joint_indices_.push_back(static_cast<int>(std::distance(isaac_joint_names.begin(), it)));
    }

    // Store config
    future_steps_ = future_steps;
    n_future_steps_ = static_cast<int>(future_steps.size());
    n_policy_bodies_ = static_cast<int>(policy_body_names.size());
    n_policy_joints_ = static_cast<int>(policy_joint_names.size());
    motion_steps_ = static_cast<int>(motion_duration_sec * 50.0);

    t_ = 0;
    loaded_ = true;

    std::cout << "[MotionData] Ready: " << n_policy_bodies_ << " policy bodies, "
              << n_policy_joints_ << " policy joints, "
              << n_future_steps_ << " future steps, "
              << motion_steps_ << " motion steps" << std::endl;
}

std::vector<float> MotionData::step() {
    assert(loaded_);

    // Advance time (wrapping)
    t_++;
    if (t_ >= num_steps_) {
        t_ = 0;
    }

    // Compute command observation [376]
    // = ref_body_pos_future_local [255] + ref_joint_pos_future [120] + ref_motion_phase [1]

    std::vector<float> command;
    command.reserve(n_future_steps_ * n_policy_bodies_ * 3 + n_future_steps_ * n_policy_joints_ + 1);

    // Get root position and yaw quaternion at first future step
    // Python ref: ref_root_pos_w = motion_data.body_pos_w[:, [0], [root_body_idx], :]
    // get_slice returns data at t + future_steps, so index [0] = t + future_steps[0]
    int cur_t = t_;
    int root_idx = root_body_idx_;
    int root_t = cur_t + future_steps_[0];
    if (root_t >= num_steps_) root_t = num_steps_ - 1;

    double root_pos[3];
    double root_quat[4];
    double root_yaw_quat[4];

    // body_pos_w_[root_t, root_idx, :3]
    for (int i = 0; i < 3; i++) {
        root_pos[i] = body_pos_w_[(root_t * num_bodies_meta_ + root_idx) * 3 + i];
    }
    root_pos[2] = 0.0;  // Zero out z for local frame

    // body_quat_w_[root_t, root_idx, :4]
    for (int i = 0; i < 4; i++) {
        root_quat[i] = body_quat_w_[(root_t * num_bodies_meta_ + root_idx) * 4 + i];
    }
    yaw_quat(root_quat, root_yaw_quat);

    // --- ref_body_pos_future_local [n_future_steps * n_policy_bodies * 3 = 255] ---
    for (int fs = 0; fs < n_future_steps_; fs++) {
        int future_t = cur_t + future_steps_[fs];
        if (future_t >= num_steps_) {
            future_t = num_steps_ - 1;  // Clamp to end
        }

        for (int bi = 0; bi < n_policy_bodies_; bi++) {
            int body_meta_idx = policy_body_indices_[bi];

            // Get body position at future_t
            double body_pos[3];
            for (int i = 0; i < 3; i++) {
                body_pos[i] = body_pos_w_[(future_t * num_bodies_meta_ + body_meta_idx) * 3 + i];
            }

            // Relative position
            double rel_pos[3];
            for (int i = 0; i < 3; i++) {
                rel_pos[i] = body_pos[i] - root_pos[i];
            }

            // Transform to local frame using yaw quaternion
            double local_pos[3];
            quat_rotate_inverse(root_yaw_quat, rel_pos, local_pos);

            command.push_back(static_cast<float>(local_pos[0]));
            command.push_back(static_cast<float>(local_pos[1]));
            command.push_back(static_cast<float>(local_pos[2]));
        }
    }

    // --- ref_joint_pos_future [n_future_steps * n_policy_joints = 120] ---
    for (int fs = 0; fs < n_future_steps_; fs++) {
        int future_t = cur_t + future_steps_[fs];
        if (future_t >= num_steps_) {
            future_t = num_steps_ - 1;
        }

        for (int ji = 0; ji < n_policy_joints_; ji++) {
            int joint_isaac_idx = policy_joint_indices_[ji];
            double jpos = joint_pos_[future_t * num_joints_isaac_ + joint_isaac_idx];
            command.push_back(static_cast<float>(jpos));
        }
    }

    // --- ref_motion_phase [1] ---
    float phase = static_cast<float>((t_ % motion_steps_)) / static_cast<float>(motion_steps_);
    command.push_back(phase);

    return command;
}

void MotionData::reset() {
    t_ = 0;
}

// --- Quaternion math (w,x,y,z convention) ---

void MotionData::yaw_quat(const double* q_in, double* q_out) {
    double qw = q_in[0], qx = q_in[1], qy = q_in[2], qz = q_in[3];
    double yaw = std::atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz));
    q_out[0] = std::cos(yaw / 2.0);
    q_out[1] = 0.0;
    q_out[2] = 0.0;
    q_out[3] = std::sin(yaw / 2.0);
    // Normalize
    double norm = std::sqrt(q_out[0] * q_out[0] + q_out[3] * q_out[3]);
    q_out[0] /= norm;
    q_out[3] /= norm;
}

void MotionData::quat_rotate_inverse(const double* q, const double* v, double* out) {
    double qw = q[0], qx = q[1], qy = q[2], qz = q[3];

    // a = v * (2*qw^2 - 1)
    double s = 2.0 * qw * qw - 1.0;
    double ax = v[0] * s;
    double ay = v[1] * s;
    double az = v[2] * s;

    // b = cross(q_vec, v) * qw * 2
    double bx = (qy * v[2] - qz * v[1]) * 2.0 * qw;
    double by = (qz * v[0] - qx * v[2]) * 2.0 * qw;
    double bz = (qx * v[1] - qy * v[0]) * 2.0 * qw;

    // c = q_vec * dot(q_vec, v) * 2
    double dot = qx * v[0] + qy * v[1] + qz * v[2];
    double cx = qx * dot * 2.0;
    double cy = qy * dot * 2.0;
    double cz = qz * dot * 2.0;

    out[0] = ax - bx + cx;
    out[1] = ay - by + cy;
    out[2] = az - bz + cz;
}
