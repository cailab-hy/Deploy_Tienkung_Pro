/**
 * @file RLControlNewPlugin.cpp
 * @brief ROS2 version of the humanoid RL plugin - Split architecture (Main PC)
 * @version 4.0
 * @date 2025-09-17
 *
 * Main PC side: 400Hz motor control loop.
 * Publishes sensor_state to Jetson Orin (50Hz).
 * Receives action from Orin and applies to motors.
 * Policy inference runs on Orin (hdmi_inference package).
 */

#include "RLControlNewPlugin.h"

#include <pluginlib/class_list_macros.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <stdio.h>
#include <thread>
#include <chrono>
#include <cmath>
#include <iostream>
#include <time.h>
#include <fstream>
#include <regex>
#include "broccoli/core/Time.hpp"
#include "spdlog/async.h"
#include "spdlog/sinks/basic_file_sink.h"
#include "spdlog/sinks/rotating_file_sink.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include <ament_index_cpp/get_package_share_directory.hpp>

#include "funcSPTrans.h" // Serial-parallel conversion

using namespace broccoli::core;

namespace rl_control_new {

// Joint order mapping arrays (HW ↔ Isaac)
const int RLControlNewPlugin::ISAAC_TO_HW[30] = {
    12, 0, 6, 15, 16, 23, 1, 7, 14, 17,
    24, 2, 8, 13, 18, 25, 3, 9, 19, 26,
    4, 10, 20, 27, 5, 11, 21, 28, 22, 29
};

const int RLControlNewPlugin::HW_TO_ISAAC[30] = {
    1, 6, 11, 16, 20, 24, 2, 7, 12, 17,
    21, 25, 0, 13, 8, 3, 4, 9, 14, 18,
    22, 26, 28, 5, 10, 15, 19, 23, 27, 29
};

// Constructor with NodeOptions (for composable nodes)
RLControlNewPlugin::RLControlNewPlugin(const rclcpp::NodeOptions & options)
    : rclcpp::Node("rl_control_new_plugin", options) {
    onInit();
}

bool RLControlNewPlugin::LoadConfig(const std::string &_config_file)
{
    config_ = YAML::LoadFile(_config_file);
    if (!config_)
    {
        RCLCPP_ERROR(this->get_logger(), "Failed to load config file: %s", _config_file.c_str());
        return false;
    }
    action_num = config_["actions_size"].as<int>();
    motor_num = config_["motor_num"].as<int>();
    simulation = config_["simulation"].as<bool>();

    dt = config_["dt"].as<double>();
    ct_scale = Eigen::Map<Eigen::VectorXd>(config_["ct_scale"].as<std::vector<double>>().data(), motor_num);

    return true;
}

void RLControlNewPlugin::onInit()
{
    std::string pkg_path = ament_index_cpp::get_package_share_directory("rl_control_new");
    _config_file = pkg_path + "/config/tg30_config.yaml";
    if (!LoadConfig(_config_file))
    {
        std::cout << "load config file error: " << _config_file << std::endl;
    }

    idMap.bodyCanIdMapInit();
    int whole_joint_num = 36;

    pos_fed_midVec = Eigen::VectorXd::Zero(whole_joint_num);
    vel_fed_midVec = Eigen::VectorXd::Zero(whole_joint_num);
    tau_fed_midVec = Eigen::VectorXd::Zero(whole_joint_num);
    pos_cmd_midVec = Eigen::VectorXd::Zero(whole_joint_num);
    vel_cmd_midVec = Eigen::VectorXd::Zero(whole_joint_num);
    tau_cmd_midVec = Eigen::VectorXd::Zero(whole_joint_num);
    ct_scale_midVec = Eigen::VectorXd::Zero(whole_joint_num);
    temperature_midVec = Eigen::VectorXd::Zero(whole_joint_num);

    pubLegMotorCmd = this->create_publisher<bodyctrl_msgs::msg::CmdMotorCtrl>("/leg/cmd_ctrl", 100);
    pubArmMotorCmd = this->create_publisher<bodyctrl_msgs::msg::CmdMotorCtrl>("/arm/cmd_ctrl", 100);
    pubHeadMotorCmd = this->create_publisher<bodyctrl_msgs::msg::CmdMotorCtrl>("/head/cmd_ctrl", 100);
    pubWaistMotorCmd = this->create_publisher<bodyctrl_msgs::msg::CmdMotorCtrl>("/waist/cmd_ctrl", 100);

    // Reset waist and head
    waists_cmd_pub_ = this->create_publisher<bodyctrl_msgs::msg::CmdSetMotorPosition>("/waist/cmd_pos", 1);

    subLegState = this->create_subscription<bodyctrl_msgs::msg::MotorStatusMsg>(
        "/leg/status", 100,
        std::bind(&RLControlNewPlugin::LegMotorStatusMsg, this, std::placeholders::_1));
    subArmState = this->create_subscription<bodyctrl_msgs::msg::MotorStatusMsg>(
        "/arm/status", 100,
        std::bind(&RLControlNewPlugin::ArmMotorStatusMsg, this, std::placeholders::_1));
    subHeadState = this->create_subscription<bodyctrl_msgs::msg::MotorStatusMsg>(
        "/head/status", 100,
        std::bind(&RLControlNewPlugin::HeadMotorStatusMsg, this, std::placeholders::_1));
    subWaistState = this->create_subscription<bodyctrl_msgs::msg::MotorStatusMsg>(
        "/waist/status", 100,
        std::bind(&RLControlNewPlugin::WaistMotorStatusMsg, this, std::placeholders::_1));
    subImuXsens = this->create_subscription<bodyctrl_msgs::msg::Imu>(
        "/imu/status", 100,
        std::bind(&RLControlNewPlugin::OnXsensImuStatusMsg, this, std::placeholders::_1));
    subJoyCmd = this->create_subscription<sensor_msgs::msg::Joy>(
        "/sbus_data", 100,
        std::bind(&RLControlNewPlugin::xbox_map_read, this, std::placeholders::_1));

    // Orin communication: sensor_state publisher
    pub_sensor_state_ = this->create_publisher<std_msgs::msg::Float64MultiArray>(
        "/hdmi/sensor_state", 10);

    // Orin communication: action subscriber
    sub_action_ = this->create_subscription<std_msgs::msg::Float64MultiArray>(
        "/hdmi/action", 10,
        std::bind(&RLControlNewPlugin::ActionCallback, this, std::placeholders::_1));

    // Orin communication: PD gains subscriber (transient_local for latched delivery)
    auto qos_latched = rclcpp::QoS(1).transient_local();
    sub_pd_gains_ = this->create_subscription<std_msgs::msg::Float64MultiArray>(
        "/hdmi/pd_gains", qos_latched,
        std::bind(&RLControlNewPlugin::PdGainsCallback, this, std::placeholders::_1));

    // Initialize Orin action vectors
    orin_q_d_isaac_ = Eigen::VectorXd::Zero(motor_num);
    orin_qdot_d_isaac_ = Eigen::VectorXd::Zero(motor_num);
    orin_tor_d_isaac_ = Eigen::VectorXd::Zero(motor_num);

    // Initialize member variables
    funS2P = new funcSPTrans;

    pi = 3.14159265358979;
    rpm2rps = pi / 30.0;
    Q_a = Eigen::VectorXd::Zero(motor_num);
    Qdot_a = Eigen::VectorXd::Zero(motor_num);
    Tor_a = Eigen::VectorXd::Zero(motor_num);
    Q_d = Eigen::VectorXd::Zero(motor_num);
    Qdot_d = Eigen::VectorXd::Zero(motor_num);
    Tor_d = Eigen::VectorXd::Zero(motor_num);
    Temperature = Eigen::VectorXd::Zero(motor_num);

    q_a = Eigen::VectorXd::Zero(motor_num);
    qdot_a = Eigen::VectorXd::Zero(motor_num);
    tor_a = Eigen::VectorXd::Zero(motor_num);
    q_d = Eigen::VectorXd::Zero(motor_num);
    qdot_d = Eigen::VectorXd::Zero(motor_num);
    tor_d = Eigen::VectorXd::Zero(motor_num);

    q_a_p = Eigen::VectorXd::Zero(4);
    qdot_a_p = Eigen::VectorXd::Zero(4);
    tor_a_p = Eigen::VectorXd::Zero(4);
    q_d_p = Eigen::VectorXd::Zero(4);
    qdot_d_p = Eigen::VectorXd::Zero(4);
    tor_d_p = Eigen::VectorXd::Zero(4);

    q_a_s = Eigen::VectorXd::Zero(4);
    qdot_a_s = Eigen::VectorXd::Zero(4);
    tor_a_s = Eigen::VectorXd::Zero(4);
    q_d_s = Eigen::VectorXd::Zero(4);
    qdot_d_s = Eigen::VectorXd::Zero(4);
    tor_d_s = Eigen::VectorXd::Zero(4);

    Q_a_last = Eigen::VectorXd::Zero(motor_num);
    Qdot_a_last = Eigen::VectorXd::Zero(motor_num);
    Tor_a_last = Eigen::VectorXd::Zero(motor_num);
    ct_scale_midVec.head(12) << ct_scale.head(12); // leg
    ct_scale_midVec.segment(12, 14) << ct_scale.segment(12, 14);    // arm (14 joints)
    ct_scale_midVec.segment(26, 3) << ct_scale.segment(26, 3);      // head (3 joints)
    ct_scale_midVec(29) = ct_scale(29);                             // waist (1 joint)

    zero_pos = Eigen::VectorXd::Zero(motor_num);
    std::stringstream ss;

    // Zero position adjustment
    if (config_["zero_pos_offset"])
    {
        for (int32_t i = 0; i < motor_num; ++i)
        {
            zero_pos[i] = config_["zero_pos_offset"][i].as<double>();
            ss << zero_pos[i] << "  ";
        }
    }

    init_pos = Eigen::VectorXd::Zero(motor_num);
    motor_dir = Eigen::VectorXd::Ones(motor_num);
    zero_cnt = Eigen::VectorXd::Zero(motor_num);
    zero_offset = Eigen::VectorXd::Zero(motor_num);
    xsense_data = Eigen::VectorXd::Zero(13);
    data = Eigen::VectorXd::Zero(450);

    // Load PD gains from local policy YAML as fallback (before Orin connects)
    kp_hw_ = Eigen::VectorXd::Zero(motor_num);
    kd_hw_ = Eigen::VectorXd::Zero(motor_num);
    {
        std::string policy_yaml = pkg_path + "/config/policy_kbc/policy-fmtq3g5b-final.yaml";
        try {
            YAML::Node policy_config = YAML::LoadFile(policy_yaml);

            // Load isaac joint names
            std::vector<std::string> isaac_joint_names;
            for (const auto& name : policy_config["isaac_joint_names"]) {
                isaac_joint_names.push_back(name.as<std::string>());
            }

            // Parse kp/kd with regex matching (same logic as HdmiPolicyRunner)
            Eigen::VectorXd kp_isaac = Eigen::VectorXd::Zero(motor_num);
            Eigen::VectorXd kd_isaac = Eigen::VectorXd::Zero(motor_num);

            auto kp_node = policy_config["joint_kp"];
            auto kd_node = policy_config["joint_kd"];

            for (int i = 0; i < motor_num; i++) {
                const std::string& jname = isaac_joint_names[i];
                for (auto it = kp_node.begin(); it != kp_node.end(); ++it) {
                    std::regex re(it->first.as<std::string>());
                    if (std::regex_match(jname, re)) {
                        kp_isaac(i) = it->second.as<double>();
                        break;
                    }
                }
                for (auto it = kd_node.begin(); it != kd_node.end(); ++it) {
                    std::regex re(it->first.as<std::string>());
                    if (std::regex_match(jname, re)) {
                        kd_isaac(i) = it->second.as<double>();
                        break;
                    }
                }
            }

            // Convert isaac order → hw order
            for (int i = 0; i < motor_num; i++) {
                int hw_idx = ISAAC_TO_HW[i];
                kp_hw_(hw_idx) = kp_isaac(i);
                kd_hw_(hw_idx) = kd_isaac(i);
            }
            RCLCPP_INFO(this->get_logger(), "[MainPC] Loaded fallback PD gains from policy YAML (kp[0]=%.1f, kd[0]=%.1f)", kp_hw_(0), kd_hw_(0));
        } catch (const std::exception& e) {
            RCLCPP_WARN(this->get_logger(), "[MainPC] Failed to load fallback PD gains: %s. Using zeros until Orin connects.", e.what());
        }
    }

    RCLCPP_INFO(this->get_logger(), "[MainPC] Waiting for Orin action on /hdmi/action ...");

    sleep(1);
    std::thread([this]()
                { rlControl(); })
        .detach();
}

void RLControlNewPlugin::rlControl()
{
    // set sched-strategy
    struct sched_param sched;
    int max_priority;

    max_priority = sched_get_priority_max(SCHED_RR);
    sched.sched_priority = max_priority;

    if (sched_setscheduler(gettid(), SCHED_RR, &sched) == -1)
    {
        printf("Set Scheduler Param, ERROR:%s\n", strerror(errno));
    }
    usleep(1000);

    // joystick init (kept for waist reset detection on Main PC side)
    Joystick_humanoid joystick_humanoid;
    joystick_humanoid.init();

    long count = 0;

    Time start_time;
    Time period(0, dt * 1e9);
    Time sleep2Time;
    Time timer;
    timespec sleep2Time_spec;
    Time timer1, timer2, timer3, total_time;

    // Wait for sensor data
    while (queueLegMotorState.empty() || queueArmMotorState.empty())
    {
        RCLCPP_WARN(this->get_logger(), "[MainPC] queue{arm or leg} is empty");
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    while (queueHeadMotorState.empty() || queueWaistMotorState.empty())
    {
        RCLCPP_WARN(this->get_logger(), "[MainPC] queue{head or waist} is empty");
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    // Drain initial sensor data
    while (!queueLegMotorState.empty()) {
        auto msg = queueLegMotorState.pop();
        for (auto &one : msg->status) {
            int index = idMap.getIndexById(one.name);
            pos_fed_midVec(index) = one.pos;
            vel_fed_midVec(index) = one.speed;
            tau_fed_midVec(index) = one.current * ct_scale_midVec(index);
        }
    }
    while (!queueWaistMotorState.empty()) {
        auto msg = queueWaistMotorState.pop();
        for (auto &one : msg->status) {
            int index = idMap.getIndexById(one.name);
            pos_fed_midVec(index) = one.pos;
            vel_fed_midVec(index) = one.speed;
            tau_fed_midVec(index) = one.current * ct_scale_midVec(index);
        }
    }
    while (!queueHeadMotorState.empty()) {
        auto msg = queueHeadMotorState.pop();
        for (auto &one : msg->status) {
            int index = idMap.getIndexById(one.name);
            pos_fed_midVec(index) = one.pos;
            vel_fed_midVec(index) = one.speed;
            tau_fed_midVec(index) = one.current * ct_scale_midVec(index);
        }
    }
    while (!queueArmMotorState.empty()) {
        auto msg = queueArmMotorState.pop();
        for (auto &one : msg->status) {
            int index = idMap.getIndexById(one.name);
            pos_fed_midVec(index) = one.pos;
            vel_fed_midVec(index) = one.speed;
            tau_fed_midVec(index) = one.current * ct_scale_midVec(index);
        }
    }

    Q_a.head(motor_num) << pos_fed_midVec.head(motor_num);
    Qdot_a.head(motor_num) << vel_fed_midVec.head(motor_num);
    Tor_a.head(motor_num) << tau_fed_midVec.head(motor_num);

    init_pos = Q_a;
    Q_a_last = Q_a;
    Qdot_a_last = Qdot_a;
    Tor_a_last = Tor_a;

    for (int i = 0; i < motor_num; i++)
    {
        q_a(i) = (Q_a(i) - zero_pos(i)) * motor_dir(i) + zero_offset(i);
        zero_cnt(i) = (q_a(i) > pi) ? -1.0 : zero_cnt(i);
        zero_cnt(i) = (q_a(i) < -pi) ? 1.0 : zero_cnt(i);
        q_a(i) += zero_cnt(i) * 2.0 * pi;
    }

    bool is_disable{false};
    xbox_flag flag_;
    flag_.is_disable = 0;

    while (1)
    {
        total_time = timer.currentTime() - start_time;
        start_time = timer.currentTime();

        // ========== Read sensor data (same as before) ==========
        while (!queueLegMotorState.empty()) {
            auto msg = queueLegMotorState.pop();
            for (auto &one : msg->status) {
                int index = idMap.getIndexById(one.name);
                pos_fed_midVec(index) = one.pos;
                vel_fed_midVec(index) = one.speed;
                tau_fed_midVec(index) = one.current * ct_scale_midVec(index);
                temperature_midVec(index) = one.temperature;
            }
        }
        while (!queueWaistMotorState.empty()) {
            auto msg = queueWaistMotorState.pop();
            for (auto &one : msg->status) {
                int index = idMap.getIndexById(one.name);
                pos_fed_midVec(index) = one.pos;
                vel_fed_midVec(index) = one.speed;
                tau_fed_midVec(index) = one.current * ct_scale_midVec(index);
                temperature_midVec(index) = one.temperature;
            }
        }
        while (!queueHeadMotorState.empty()) {
            auto msg = queueHeadMotorState.pop();
            for (auto &one : msg->status) {
                int index = idMap.getIndexById(one.name);
                pos_fed_midVec(index) = one.pos;
                vel_fed_midVec(index) = one.speed;
                tau_fed_midVec(index) = one.current * ct_scale_midVec(index);
                temperature_midVec(index) = one.temperature;
            }
        }
        while (!queueArmMotorState.empty()) {
            auto msg = queueArmMotorState.pop();
            for (auto &one : msg->status) {
                int index = idMap.getIndexById(one.name);
                pos_fed_midVec(index) = one.pos;
                vel_fed_midVec(index) = one.speed;
                tau_fed_midVec(index) = one.current * ct_scale_midVec(index);
                temperature_midVec(index) = one.temperature;
            }
        }
        Q_a.head(motor_num) << pos_fed_midVec.head(motor_num);
        Qdot_a.head(motor_num) << vel_fed_midVec.head(motor_num);
        Tor_a.head(motor_num) << tau_fed_midVec.head(motor_num);
        Temperature.head(motor_num) << temperature_midVec.head(motor_num);

        while (!queueImuXsens.empty())
        {
            auto msg = queueImuXsens.pop();
            xsense_data(0) = msg->euler.yaw;
            xsense_data(1) = msg->euler.pitch;
            xsense_data(2) = msg->euler.roll;
            // Apply roll offset only once when reading new IMU data
            if (config_["xsense_data_roll_offset"]) {
                xsense_data(2) += config_["xsense_data_roll_offset"].as<double>();
            }
            xsense_data(3) = msg->angular_velocity.x;
            xsense_data(4) = msg->angular_velocity.y;
            xsense_data(5) = msg->angular_velocity.z;
            xsense_data(6) = msg->linear_acceleration.x;
            xsense_data(7) = msg->linear_acceleration.y;
            xsense_data(8) = msg->linear_acceleration.z;
        }

        while (!queueJoyCmd.empty()) {
            auto msg = queueJoyCmd.pop();
            if (msg->axes.size() == 12){
                xbox_map.a = msg->axes[8];
                xbox_map.b = msg->axes[9];
                xbox_map.c = msg->axes[10];
                xbox_map.d = msg->axes[11];
                xbox_map.e = msg->axes[4];
                xbox_map.f = msg->axes[7];
                xbox_map.g = msg->axes[5];
                xbox_map.h = msg->axes[6];
                xbox_map.x1 = msg->axes[3];
                xbox_map.x2 = msg->axes[1];
                xbox_map.y1 = msg->axes[2];
                xbox_map.y2 = msg->axes[0];
            }
            else if (msg->axes.size() == 8)
            {
                xbox_map.x1 = msg->axes[0];
                xbox_map.x2 = msg->axes[3];
                xbox_map.y1 = msg->axes[1];
                xbox_map.y2 = msg->axes[4];
                xbox_map.a = msg->buttons[0];
                xbox_map.b = msg->buttons[1];
                xbox_map.c = msg->buttons[3];
                xbox_map.d = msg->buttons[2];
                xbox_map.e = msg->buttons[4];
                xbox_map.f = msg->buttons[5];
                xbox_map.g = msg->buttons[6];
                xbox_map.h = msg->buttons[7];
            }
        }

        // ========== Joint angle processing ==========
        for (int i = 0; i < motor_num; i++)
        {
            if (fabs(Q_a(i) - Q_a_last(i)) > pi)
            {
                Q_a(i) = Q_a_last(i);
                Qdot_a(i) = Qdot_a_last(i);
                Tor_a(i) = Tor_a_last(i);
            }
        }
        Q_a_last = Q_a;
        Qdot_a_last = Qdot_a;
        Tor_a_last = Tor_a;

        for (int i = 0; i < motor_num; i++)
        {
            q_a(i) = (Q_a(i) - zero_pos(i)) * motor_dir(i) + zero_offset(i);
            q_a(i) += zero_cnt(i) * 2.0 * pi;
            qdot_a(i) = Qdot_a(i) * motor_dir(i);
            tor_a(i) = Tor_a(i) * motor_dir(i);
        }

        // ========== S/P conversion (parallel → serial) ==========
        if (!simulation){
            q_a_p << q_a.segment(4, 2), q_a.segment(10, 2);
            qdot_a_p << qdot_a.segment(4, 2), qdot_a.segment(10, 2);
            tor_a_p << tor_a.segment(4, 2), tor_a.segment(10, 2);

            funS2P->setPEst(q_a_p, qdot_a_p, tor_a_p);
            funS2P->calcFK();
            funS2P->calcIK();
            funS2P->getSState(q_a_s, qdot_a_s, tor_a_s);

            q_a.segment(4, 2) = q_a_s.head(2);
            q_a.segment(10, 2) = q_a_s.tail(2);
            qdot_a.segment(4, 2) = qdot_a_s.head(2);
            qdot_a.segment(10, 2) = qdot_a_s.tail(2);
            tor_a.segment(4, 2) = tor_a_s.head(2);
            tor_a.segment(10, 2) = tor_a_s.tail(2);
        }

        // ========== HW → Isaac order conversion ==========
        Eigen::VectorXd q_a_isaac = Eigen::VectorXd::Zero(motor_num);
        for (int hw = 0; hw < motor_num; hw++) {
            int isaac = HW_TO_ISAAC[hw];
            q_a_isaac(isaac) = q_a(hw);
        }

        timer1 = timer.currentTime() - start_time;

        // ========== Publish sensor_state to Orin (every 8th step = 50Hz) ==========
        if (count % 8 == 0) {
            auto sensor_msg = std_msgs::msg::Float64MultiArray();
            sensor_msg.data.resize(39);
            for (int i = 0; i < 30; i++) {
                sensor_msg.data[i] = q_a_isaac(i);
            }
            // IMU: euler(yaw,pitch,roll) + angular_vel(x,y,z) + linear_accel(x,y,z)
            for (int i = 0; i < 9; i++) {
                sensor_msg.data[30 + i] = xsense_data(i);
            }
            pub_sensor_state_->publish(sensor_msg);
        }

        // ========== Apply action from Orin ==========
        Eigen::VectorXd q_d_isaac, qdot_d_isaac, tor_d_isaac;
        int fsm_state_local;
        bool disable_local;
        bool waist_reset_local;
        {
            std::lock_guard<std::mutex> lock(action_mutex_);

            // Safety: timeout check (150ms = 3 cycles at 50Hz + margin)
            if (action_received_) {
                auto now = std::chrono::steady_clock::now();
                auto age = std::chrono::duration_cast<std::chrono::milliseconds>(
                    now - last_action_time_);
                if (age.count() > 150) {
                    // Timeout: force STOP (hold current position)
                    if (orin_fsm_state_ != 0) {
                        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                            "[MainPC] Action timeout (%ldms)! Auto-STOP.", age.count());
                    }
                    orin_q_d_isaac_ = q_a_isaac;
                    orin_qdot_d_isaac_.setZero();
                    orin_tor_d_isaac_.setZero();
                    orin_fsm_state_ = 0;  // STOP
                }
            } else {
                // No action received yet: hold current position
                orin_q_d_isaac_ = q_a_isaac;
                orin_qdot_d_isaac_.setZero();
                orin_tor_d_isaac_.setZero();
            }

            q_d_isaac = orin_q_d_isaac_;
            qdot_d_isaac = orin_qdot_d_isaac_;
            tor_d_isaac = orin_tor_d_isaac_;
            fsm_state_local = orin_fsm_state_;
            disable_local = orin_disable_;
            waist_reset_local = orin_waist_reset_;
        }

        // Waist reset: triggered by Orin when STOP → ZERO transition occurs
        if (waist_reset_local) {
            bodyctrl_msgs::msg::CmdSetMotorPosition waist_pos_msg;
            bodyctrl_msgs::msg::SetMotorPosition cmd;
            cmd.name = 31;
            cmd.pos = 0.0;
            cmd.spd = 0.3;
            cmd.cur = 3;
            waist_pos_msg.header.stamp = rclcpp::Clock().now();
            waist_pos_msg.cmds.push_back(cmd);
            waists_cmd_pub_->publish(waist_pos_msg);

            // Clear the flag after publishing
            {
                std::lock_guard<std::mutex> lock(action_mutex_);
                orin_waist_reset_ = false;
            }
        }

        timer2 = timer.currentTime() - start_time - timer1;

        // ========== Isaac → HW order conversion ==========
        for (int isaac = 0; isaac < motor_num; isaac++) {
            int hw = ISAAC_TO_HW[isaac];
            q_d(hw) = q_d_isaac(isaac);
            qdot_d(hw) = qdot_d_isaac(isaac);
            tor_d(hw) = tor_d_isaac(isaac);
        }

        // ========== S/P conversion (serial → parallel) ==========
        if (!simulation)
        {
            q_d_s << q_d.segment(4, 2), q_d.segment(10, 2);
            qdot_d_s << qdot_d.segment(4, 2), qdot_d.segment(10, 2);
            tor_d_s << tor_d.segment(4, 2), tor_d.segment(10, 2);

            funS2P->setSDes(q_d_s, qdot_d_s, tor_d_s);
            funS2P->calcJointPosRef();
            funS2P->calcJointTorDes();
            funS2P->getPDes(q_d_p, qdot_d_p, tor_d_p);

            q_d.segment(4, 2) = q_d_p.head(2);
            q_d.segment(10, 2) = q_d_p.tail(2);
            qdot_d.segment(4, 2) = qdot_d_p.head(2);
            qdot_d.segment(10, 2) = qdot_d_p.tail(2);
            tor_d.segment(4, 2) = tor_d_p.head(2);
            tor_d.segment(10, 2) = tor_d_p.tail(2);
        }

        // ========== Motor command conversion ==========
        for (int i = 0; i < motor_num; i++)
        {
            Q_d(i) = (q_d(i) - zero_offset(i) - zero_cnt(i) * 2.0 * pi) * motor_dir(i) + zero_pos(i);
            Qdot_d(i) = qdot_d(i) * motor_dir(i);
            Tor_d(i) = tor_d(i) * motor_dir(i);
        }

        // Use local copy of kp/kd (updated by PdGainsCallback or fallback)
        Eigen::VectorXd kp_use = kp_hw_;
        Eigen::VectorXd kd_use = kd_hw_;

        if (disable_local)
        {
            kp_use.setZero();
            kd_use.setZero();
            Tor_d.setZero();
            is_disable = true;
        }

        pos_cmd_midVec.head(motor_num) << Q_d.head(motor_num);
        vel_cmd_midVec.head(motor_num) << Qdot_d.head(motor_num);
        tau_cmd_midVec.head(motor_num) << Tor_d.head(motor_num);

        // ========== Publish motor commands ==========
        // Leg control
        bodyctrl_msgs::msg::CmdMotorCtrl leg_msg;
        leg_msg.header.stamp = this->get_clock()->now();
        for (int i = 0; i < 12; i++)
        {
            bodyctrl_msgs::msg::MotorCtrl cmd;
            cmd.name = idMap.getIdByIndex(i);
            cmd.kp = kp_use(i);
            cmd.kd = kd_use(i);
            cmd.pos = pos_cmd_midVec(i);
            cmd.spd = vel_cmd_midVec(i);
            cmd.tor = tau_cmd_midVec(i);
            leg_msg.cmds.push_back(cmd);
        }
        pubLegMotorCmd->publish(leg_msg);

        // Arm control
        bodyctrl_msgs::msg::CmdMotorCtrl arm_msg;
        arm_msg.header.stamp = this->get_clock()->now();
        std::vector<int> arm_indices = {16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29};
        for (int index : arm_indices)
        {
            bodyctrl_msgs::msg::MotorCtrl cmd_arm;
            cmd_arm.name = idMap.getIdByIndex(index);
            cmd_arm.kp = kp_use(index);
            cmd_arm.kd = kd_use(index);
            cmd_arm.pos = pos_cmd_midVec(index);
            cmd_arm.spd = vel_cmd_midVec(index);
            cmd_arm.tor = tau_cmd_midVec(index);
            arm_msg.cmds.push_back(cmd_arm);
        }
        pubArmMotorCmd->publish(arm_msg);

        // Head control
        bodyctrl_msgs::msg::CmdMotorCtrl head_msg;
        head_msg.header.stamp = this->get_clock()->now();
        std::vector<int> head_indices = {13, 14, 15};
        for (int index : head_indices)
        {
            bodyctrl_msgs::msg::MotorCtrl cmd_head;
            cmd_head.name = idMap.getIdByIndex(index);
            cmd_head.kp = kp_use(index);
            cmd_head.kd = kd_use(index);
            cmd_head.pos = pos_cmd_midVec(index);
            cmd_head.spd = vel_cmd_midVec(index);
            cmd_head.tor = tau_cmd_midVec(index);
            head_msg.cmds.push_back(cmd_head);
        }
        pubHeadMotorCmd->publish(head_msg);

        // Waist control
        bodyctrl_msgs::msg::CmdMotorCtrl waist_msg;
        waist_msg.header.stamp = this->get_clock()->now();
        std::vector<int> waist_indices = {12};
        for (int index : waist_indices)
        {
            bodyctrl_msgs::msg::MotorCtrl cmd_waist;
            cmd_waist.name = idMap.getIdByIndex(index);
            cmd_waist.kp = kp_use(index);
            cmd_waist.kd = kd_use(index);
            cmd_waist.pos = pos_cmd_midVec(index);
            cmd_waist.spd = vel_cmd_midVec(index);
            cmd_waist.tor = tau_cmd_midVec(index);
            waist_msg.cmds.push_back(cmd_waist);
        }
        pubWaistMotorCmd->publish(waist_msg);

        timer3 = timer.currentTime() - start_time - timer1 - timer2;
        sleep2Time = start_time + period;
        sleep2Time_spec = sleep2Time.toTimeSpec();
        clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &(sleep2Time_spec), NULL);
        count++;

        // Periodic log
        if (count % 800 == 0) {
            joystick_humanoid.xbox_flag_update(xbox_map);
            flag_ = joystick_humanoid.get_xbox_flag();
            printXboxFlag(flag_);
            RCLCPP_INFO(this->get_logger(), "[MainPC] FSM=%d, action_received=%d, pd_gains=%d",
                        fsm_state_local, action_received_, pd_gains_received_);
        }

        if (is_disable)
        {
            break;
        }
    }
}

void RLControlNewPlugin::ActionCallback(const std_msgs::msg::Float64MultiArray::SharedPtr msg) {
    if (msg->data.size() < 93) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
            "[MainPC] Action has %zu values, expected 93", msg->data.size());
        return;
    }

    std::lock_guard<std::mutex> lock(action_mutex_);
    for (int i = 0; i < 30; i++) {
        orin_q_d_isaac_(i) = msg->data[i];
        orin_qdot_d_isaac_(i) = msg->data[30 + i];
        orin_tor_d_isaac_(i) = msg->data[60 + i];
    }
    orin_fsm_state_ = static_cast<int>(msg->data[90]);
    orin_disable_ = (msg->data[91] > 0.5);
    orin_waist_reset_ = (msg->data[92] > 0.5);
    last_action_time_ = std::chrono::steady_clock::now();
    action_received_ = true;
}

void RLControlNewPlugin::PdGainsCallback(const std_msgs::msg::Float64MultiArray::SharedPtr msg) {
    if (msg->data.size() < 60) {
        RCLCPP_WARN(this->get_logger(), "[MainPC] PD gains has %zu values, expected 60", msg->data.size());
        return;
    }

    // Convert from isaac order to hw order
    for (int i = 0; i < 30; i++) {
        int hw_idx = ISAAC_TO_HW[i];
        kp_hw_(hw_idx) = msg->data[i];
        kd_hw_(hw_idx) = msg->data[30 + i];
    }
    pd_gains_received_ = true;
    RCLCPP_INFO(this->get_logger(), "[MainPC] Received PD gains from Orin (kp[0]=%.1f, kd[0]=%.1f)", kp_hw_(0), kd_hw_(0));
}

void RLControlNewPlugin::LegMotorStatusMsg(const bodyctrl_msgs::msg::MotorStatusMsg::SharedPtr msg)
{
    queueLegMotorState.push(msg);
}

void RLControlNewPlugin::ArmMotorStatusMsg(const bodyctrl_msgs::msg::MotorStatusMsg::SharedPtr msg)
{
    queueArmMotorState.push(msg);
}

void RLControlNewPlugin::HeadMotorStatusMsg(const bodyctrl_msgs::msg::MotorStatusMsg::SharedPtr msg)
{
    queueHeadMotorState.push(msg);
}

void RLControlNewPlugin::WaistMotorStatusMsg(const bodyctrl_msgs::msg::MotorStatusMsg::SharedPtr msg)
{
    queueWaistMotorState.push(msg);
}

void RLControlNewPlugin::OnXsensImuStatusMsg(const bodyctrl_msgs::msg::Imu::SharedPtr msg)
{
    queueImuXsens.push(msg);
}

void RLControlNewPlugin::xbox_map_read(const sensor_msgs::msg::Joy::SharedPtr msg)
{
    queueJoyCmd.push(msg);
}

void RLControlNewPlugin::printXboxFlag(const xbox_flag& flag) {
    std::cout << "======= Xbox Flag Info =======" << std::endl;
    std::cout << "fsm_state_command: " << flag.fsm_state_command << std::endl;
    std::cout << "is_disable: " << flag.is_disable << std::endl;
  }
} // namespace rl_control_new

// Register composable node
RCLCPP_COMPONENTS_REGISTER_NODE(rl_control_new::RLControlNewPlugin)
