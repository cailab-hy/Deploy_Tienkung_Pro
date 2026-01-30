/**
 * @file RLControlNewPlugin.cpp
 * @brief ROS2 version of the humanoid RL plugin
 * @version 2.0
 * @date 2025-09-17
 *
 * @修改记录1: Revision log 1: zyj, 2025-09-17 — adapted for Tiangong Dex
 *
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
#include "broccoli/core/Time.hpp"
#include "spdlog/async.h"
#include "spdlog/sinks/basic_file_sink.h"
#include "spdlog/sinks/rotating_file_sink.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include <ament_index_cpp/get_package_share_directory.hpp>

#include "funcSPTrans.h" // Serial-parallel conversion

using namespace broccoli::core;

namespace rl_control_new {

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
    dry_run = config_["dry_run"] ? config_["dry_run"].as<bool>() : false;
    if (dry_run)
    {
        RCLCPP_WARN(this->get_logger(), "[DryRun] enabled: motor commands will NOT be published.");
    }

    dt = config_["dt"].as<double>();
    ct_scale = Eigen::Map<Eigen::VectorXd>(config_["ct_scale"].as<std::vector<double>>().data(), motor_num);
    robot_data.config_file_ = _config_file;
    robot_data.joint_kp_p_ = Eigen::Map<Eigen::VectorXd>(config_["joint_kp_p"].as<std::vector<double>>().data(), motor_num);
    robot_data.joint_kd_p_ = Eigen::Map<Eigen::VectorXd>(config_["joint_kd_p"].as<std::vector<double>>().data(), motor_num);
    
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
    ////// (Tienkung Lite)  Leg 12 + Arm  8 + Floating base 6 = 26
    ////// (Tienkung Pro)   Leg 12 + Arm 14 + Head 3 + Waist 1 + Floating base 6 = 36
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
    if (config_["xsense_data_roll_offset"])
    {
        // Log output is useful during debugging, but can be omitted during normal operation
    }

    init_pos = Eigen::VectorXd::Zero(motor_num);
    motor_dir = Eigen::VectorXd::Ones(motor_num);
    zero_cnt = Eigen::VectorXd::Zero(motor_num);
    zero_offset = Eigen::VectorXd::Zero(motor_num);
    xsense_data = Eigen::VectorXd::Zero(13);
    data = Eigen::VectorXd::Zero(450);

    sleep(1);
    std::thread([this]()
                { rlControl(); })
        .detach();
}

// Hybrid mode test
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
    // set sched-strategy

    // joystick init
    Joystick_humanoid joystick_humanoid;
    joystick_humanoid.init();

    // robot FSM init
    RobotFSM *robot_fsm = get_robot_FSM(robot_data);

    // robot_interface init
    RobotInterface *robot_interface = get_robot_interface();
    robot_interface->Init();

    long count = 0;
    double t_now = 0;

    Time start_time;
    Time period(0, dt * 1e9); // Convert dt to nanoseconds
    Time sleep2Time;
    Time timer;
    timespec sleep2Time_spec;
    double timeFSM = 0.0;
    Time timer1, timer2, timer3, total_time;
    while (queueLegMotorState.empty() || queueArmMotorState.empty() )
    {
        RCLCPP_WARN(this->get_logger(), "[RobotFSM] queue{arm or leg} is empty");
        std::this_thread::sleep_for(std::chrono::milliseconds(10)); // Sleep for 10 milliseconds (0.01 seconds)
    }

    while (queueHeadMotorState.empty()|| queueWaistMotorState.empty())
    {
        RCLCPP_WARN(this->get_logger(), "[RobotFSM] queue{head or waist} is empty");
        std::this_thread::sleep_for(std::chrono::milliseconds(10)); // Sleep for 10 milliseconds (0.01 seconds)
    }


    while (!queueLegMotorState.empty())
    {
        auto msg = queueLegMotorState.pop();
        for (auto &one : msg->status)
        {
            int index = idMap.getIndexById(one.name);
            pos_fed_midVec(index) = one.pos;
            vel_fed_midVec(index) = one.speed;
            tau_fed_midVec(index) = one.current * ct_scale_midVec(index);
            temperature_midVec(index) = one.temperature;
        }
    }

    while (!queueWaistMotorState.empty())
    {
        auto msg = queueWaistMotorState.pop();
        for (auto &one : msg->status)
        {
            int index = idMap.getIndexById(one.name);
            pos_fed_midVec(index) = one.pos;
            vel_fed_midVec(index) = one.speed;
            tau_fed_midVec(index) = one.current * ct_scale_midVec(index);
            temperature_midVec(index) = one.temperature;
        }
    }

    while (!queueHeadMotorState.empty())
    {
        auto msg = queueHeadMotorState.pop();
        for (auto &one : msg->status)
        {
            int index = idMap.getIndexById(one.name);
            pos_fed_midVec(index) = one.pos;
            vel_fed_midVec(index) = one.speed;
            tau_fed_midVec(index) = one.current * ct_scale_midVec(index);
            temperature_midVec(index) = one.temperature;
        }
    }
    while (!queueArmMotorState.empty())
    {   
        auto msg = queueArmMotorState.pop();
        for (auto &one : msg->status)
        {
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
    int cnt = 0;
    xbox_flag flag_;
    flag_.is_disable = 0;
    while (1)
    {
        total_time = timer.currentTime() - start_time;
        start_time = timer.currentTime();

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
            // set xsens imu buf
            xsense_data(0) = msg->euler.yaw;
            xsense_data(1) = msg->euler.pitch;
            xsense_data(2) = msg->euler.roll;
            xsense_data(3) = msg->angular_velocity.x;
            xsense_data(4) = msg->angular_velocity.y;
            xsense_data(5) = msg->angular_velocity.z;
            xsense_data(6) = msg->linear_acceleration.x;
            xsense_data(7) = msg->linear_acceleration.y;
            xsense_data(8) = msg->linear_acceleration.z;
            xsense_data(9) = msg->orientation.x;
            xsense_data(10) = msg->orientation.y;
            xsense_data(11) = msg->orientation.z;
            xsense_data(12) = msg->orientation.w;
        }

        double pitch = xsense_data(1);
        double roll = xsense_data(2);
        Eigen::Vector3d gyro_vec = xsense_data.segment<3>(3);
        double gyro_norm = gyro_vec.norm();

        if (std::abs(pitch) >= 0.8 || std::abs(roll) >= 0.8 ||
            gyro_norm > 5.0) {
            // Log output is useful during debugging, but can be omitted during normal operation
            // std::cout << "[IMU] Pitch/Roll/AngularVel 超限，进入STOP" << std::endl;
        }

        while (!queueJoyCmd.empty()) {
            auto msg = queueJoyCmd.pop();
            // set joy cmd buf
            if (msg->axes.size() == 12){
                // YunZhuo
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
                //xbox }
                // {x,y, yaw} command
                xbox_map.x1 = msg->axes[0]; //lx
                xbox_map.x2 = msg->axes[3]; //rx
                xbox_map.y1 = msg->axes[1]; // ly
                xbox_map.y2 = msg->axes[4]; // ry
                // 
                xbox_map.a = msg->buttons[0]; // a
                xbox_map.b = msg->buttons[1]; // b
                xbox_map.c = msg->buttons[3]; // y
                xbox_map.d = msg->buttons[2]; // x
                xbox_map.e = msg->buttons[4]; // lb
                xbox_map.f = msg->buttons[5]; // rb
                xbox_map.g = msg->buttons[6]; // select
                xbox_map.h = msg->buttons[7]; // start
            }
        }

        // calculate Command
        t_now = count * dt;
        robot_data.time_now_ = t_now;

        for (int i = 0; i < motor_num; i++)
        {
            if (fabs(Q_a(i) - Q_a_last(i)) > pi)
            {
                // Error handling: joint angle jumps too large
                Q_a(i) = Q_a_last(i);
                Qdot_a(i) = Qdot_a_last(i);
                Tor_a(i) = Tor_a_last(i);
            }
        }
        Q_a_last = Q_a;
        Qdot_a_last = Qdot_a;
        Tor_a_last = Tor_a;

        // real feedback
        for (int i = 0; i < motor_num; i++)
        {
            q_a(i) = (Q_a(i) - zero_pos(i)) * motor_dir(i) + zero_offset(i);
            q_a(i) += zero_cnt(i) * 2.0 * pi;
            qdot_a(i) = Qdot_a(i) * motor_dir(i);
            tor_a(i) = Tor_a(i) * motor_dir(i);
        }

        if (!simulation){
            // Parallel to serial conversion
            // Extract the two ankle joints of the left and right legs
            q_a_p << q_a.segment(4, 2), q_a.segment(10, 2);
            qdot_a_p << qdot_a.segment(4, 2), qdot_a.segment(10, 2);
            tor_a_p << tor_a.segment(4, 2), tor_a.segment(10, 2);

            // Calculate parallel to serial
            funS2P->setPEst(q_a_p, qdot_a_p, tor_a_p);
            funS2P->calcFK();
            funS2P->calcIK();
            // Obtain results
            funS2P->getSState(q_a_s, qdot_a_s, tor_a_s);

            // Overwrite results
            q_a.segment(4, 2) = q_a_s.head(2);
            q_a.segment(10, 2) = q_a_s.tail(2);
            qdot_a.segment(4, 2) = qdot_a_s.head(2);
            qdot_a.segment(10, 2) = qdot_a_s.tail(2);
            tor_a.segment(4, 2) = tor_a_s.head(2);
            tor_a.segment(10, 2) = tor_a_s.tail(2);
        }

        // add offset
        if (config_["xsense_data_roll_offset"])
        {
            double offset = config_["xsense_data_roll_offset"].as<double>();
            xsense_data(2) += offset;
        }

        // get state (take the last motor_num of whole_joint_num)
        robot_data.q_a_.tail(motor_num) = q_a;
        robot_data.q_dot_a_.tail(motor_num) = qdot_a;
        robot_data.tau_a_.tail(motor_num) = tor_a;
        robot_data.imu_data_ = xsense_data.head(12); // xsense imu

        // get state
        robot_interface->GetState(timeFSM, robot_data);

        joystick_humanoid.xbox_flag_update(xbox_map);

        xbox_flag flag_ = joystick_humanoid.get_xbox_flag();
        printXboxFlag(flag_);
        timer1 = timer.currentTime() - start_time;

        if (robot_fsm->getCurrentState() == FSMStateName::STOP && flag_.fsm_state_command == "gotoZero")
        {
            // Waist return to zero
            bodyctrl_msgs::msg::CmdSetMotorPosition waist_msg;
            for (int i = 0; i < 1; i++)
            {
                bodyctrl_msgs::msg::SetMotorPosition cmd;
                cmd.name = 31;
                cmd.pos = 0.0;
                cmd.spd = 0.3;
                cmd.cur = 3;
                waist_msg.header.stamp = rclcpp::Clock().now();

                waist_msg.cmds.push_back(cmd);
            }
            if (!dry_run)
            {
                waists_cmd_pub_->publish(waist_msg);
            }
        }

        // rl fsm
        robot_fsm->RunFSM(flag_);

        timer2 = timer.currentTime() - start_time - timer1;

        // set command
        robot_interface->SetCommand(robot_data);

        timeFSM += dt;

        q_d = robot_data.q_d_.tail(motor_num);
        qdot_d = robot_data.q_dot_d_.tail(motor_num);
        tor_d = robot_data.tau_d_.tail(motor_num);
        {
            const auto fsm_state = robot_fsm->getCurrentState();
            RCLCPP_INFO_THROTTLE(
                this->get_logger(),
                *this->get_clock(),
                1000,
                "[FSM] cmd=%s state=%d",
                flag_.fsm_state_command.c_str(),
                static_cast<int>(fsm_state));
        }

        if (!simulation)
        {
            // Serial to parallel conversion
            // Extract the two ankle joints
            q_d_s << q_d.segment(4, 2), q_d.segment(10, 2);
            qdot_d_s << qdot_d.segment(4, 2), qdot_d.segment(10, 2);
            tor_d_s << tor_d.segment(4, 2), tor_d.segment(10, 2);

            // Conversion
            funS2P->setSDes(q_d_s, qdot_d_s, tor_d_s);
            funS2P->calcJointPosRef();
            funS2P->calcJointTorDes();
            funS2P->getPDes(q_d_p, qdot_d_p, tor_d_p);

            // Overwrite the original values
            q_d.segment(4, 2) = q_d_p.head(2);
            q_d.segment(10, 2) = q_d_p.tail(2);
            qdot_d.segment(4, 2) = qdot_d_p.head(2);
            qdot_d.segment(10, 2) = qdot_d_p.tail(2);
            tor_d.segment(4, 2) = tor_d_p.head(2);
            tor_d.segment(10, 2) = tor_d_p.tail(2);
        }

        for (int i = 0; i < motor_num; i++)
        {
            Q_d(i) = (q_d(i) - zero_offset(i) - zero_cnt(i) * 2.0 * pi) * motor_dir(i) + zero_pos(i);
            Qdot_d(i) = qdot_d(i) * motor_dir(i);
            Tor_d(i) = tor_d(i) * motor_dir(i);
        }
        {
            static const std::vector<int> arm_indices = {16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29};
            if (arm_indices.size() >= 4)
            {
                RCLCPP_INFO_THROTTLE(
                    this->get_logger(),
                    *this->get_clock(),
                    1000,
                    "[Arm q_d] i16=%.6f i17=%.6f i18=%.6f i19=%.6f",
                    q_d(arm_indices[0]),
                    q_d(arm_indices[1]),
                    q_d(arm_indices[2]),
                    q_d(arm_indices[3]));
            }
        }

        if (robot_fsm->disable_joints_)
        {
            robot_data.joint_kp_p_.setZero();
            robot_data.joint_kd_p_.setZero();
            Tor_d.setZero();
            is_disable = true;
        }

        pos_cmd_midVec.head(motor_num) << Q_d.head(motor_num);
        vel_cmd_midVec.head(motor_num) << Qdot_d.head(motor_num);
        tau_cmd_midVec.head(motor_num) << Tor_d.head(motor_num);

        // Leg control
        bodyctrl_msgs::msg::CmdMotorCtrl leg_msg;
        leg_msg.header.stamp = this->get_clock()->now();
        for (int i = 0; i < 12; i++)    // idx for leg joint
        {   
            bodyctrl_msgs::msg::MotorCtrl cmd;
            cmd.name = idMap.getIdByIndex(i);
            cmd.kp = robot_data.joint_kp_p_(i);
            cmd.kd = robot_data.joint_kd_p_(i);
            cmd.pos = pos_cmd_midVec(i);
            cmd.spd = vel_cmd_midVec(i);
            cmd.tor = tau_cmd_midVec(i);
            leg_msg.cmds.push_back(cmd);
        }
        // pubLegMotorCmd->publish(leg_msg); // -> 주석 처리됨
        if (flag_.fsm_state_command == "gotoZero"){
           if (!dry_run)
           {
               pubLegMotorCmd->publish(leg_msg);
           }
        }
        
        // Arm control
        bodyctrl_msgs::msg::CmdMotorCtrl arm_msg;
        arm_msg.header.stamp = this->get_clock()->now();

        std::vector<int> arm_indices = {16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29}; // idx for arm joint
        for (int index : arm_indices)
        {
            bodyctrl_msgs::msg::MotorCtrl cmd_arm;
            cmd_arm.name = idMap.getIdByIndex(index);
            cmd_arm.kp = robot_data.joint_kp_p_(index);
            cmd_arm.kd = robot_data.joint_kd_p_(index);
            cmd_arm.pos = pos_cmd_midVec(index);
            cmd_arm.spd = vel_cmd_midVec(index);
            cmd_arm.tor = tau_cmd_midVec(index);
            arm_msg.cmds.push_back(cmd_arm);
            // RCLCPP_WARN(this->get_logger(), "[KP] %f [KD] %f [Pose] %f [Speed] %f [Torque] %f",cmd_arm.kp, cmd_arm.kd, cmd_arm.pos,cmd_arm.spd,cmd_arm.tor);
        }
        if (!dry_run)
        {
            pubArmMotorCmd->publish(arm_msg); // -> 주석 처리됨
        }
        if (!arm_msg.cmds.empty())
        {
            const auto &cmd = arm_msg.cmds.front();
            RCLCPP_INFO_THROTTLE(
                this->get_logger(),
                *this->get_clock(),
                100,
                "[ArmCmd] id=%d kp=%.3f kd=%.3f pos=%.6f spd=%.6f tor=%.6f (count=%zu)",
                cmd.name,
                cmd.kp,
                cmd.kd,
                cmd.pos,
                cmd.spd,
                cmd.tor,
                arm_msg.cmds.size());
        }
        
        // Head control
        bodyctrl_msgs::msg::CmdMotorCtrl head_msg;
        head_msg.header.stamp = this->get_clock()->now();

        std::vector<int> head_indices = {13, 14, 15}; // idx for Head joint
        for (int index : head_indices)
        {
            bodyctrl_msgs::msg::MotorCtrl cmd_head;
            cmd_head.name = idMap.getIdByIndex(index);
            cmd_head.kp = robot_data.joint_kp_p_(index);
            cmd_head.kd = robot_data.joint_kd_p_(index);
            cmd_head.pos = pos_cmd_midVec(index);
            cmd_head.spd = vel_cmd_midVec(index);
            cmd_head.tor = tau_cmd_midVec(index);
            head_msg.cmds.push_back(cmd_head);
        }
        // pubHeadMotorCmd->publish(head_msg); // -> 주석 처리됨
        //if (flag_.fsm_state_command == "gotoZero"){
        //   pubHeadMotorCmd->publish(head_msg);
        //}

        // Waist control
        bodyctrl_msgs::msg::CmdMotorCtrl waist_msg;
        waist_msg.header.stamp = this->get_clock()->now();

        std::vector<int> waist_indices = {12};      // idx for Waist joint
        for (int index : waist_indices)
        {
            bodyctrl_msgs::msg::MotorCtrl cmd_waist;
            cmd_waist.name = idMap.getIdByIndex(index);
            cmd_waist.kp = robot_data.joint_kp_p_(index);
            cmd_waist.kd = robot_data.joint_kd_p_(index);
            cmd_waist.pos = pos_cmd_midVec(index);
            cmd_waist.spd = vel_cmd_midVec(index);
            cmd_waist.tor = tau_cmd_midVec(index);
            waist_msg.cmds.push_back(cmd_waist);
        }
        if (!dry_run)
        {
            pubWaistMotorCmd->publish(waist_msg); // -> 주석 처리됨
        }
        if (!waist_msg.cmds.empty())
        {
            const auto &cmd = waist_msg.cmds.front();
            RCLCPP_INFO_THROTTLE(
                this->get_logger(),
                *this->get_clock(),
                1000,
                "[WaistCmd] id=%d kp=%.3f kd=%.3f pos=%.6f spd=%.6f tor=%.6f",
                cmd.name,
                cmd.kp,
                cmd.kd,
                cmd.pos,
                cmd.spd,
                cmd.tor);
        }
        //if (flag_.fsm_state_command == "gotoZero"){
        //   pubWaistMotorCmd->publish(waist_msg);
        //}
        
        timer3 = timer.currentTime() - start_time - timer1 - timer2;
        sleep2Time = start_time + period;
        sleep2Time_spec = sleep2Time.toTimeSpec();
        clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &(sleep2Time_spec), NULL);
        count++;

        if (is_disable)
        {
            break;
        }
    }
}

void RLControlNewPlugin::LegMotorStatusMsg(const bodyctrl_msgs::msg::MotorStatusMsg::SharedPtr msg)
{
    auto wrapper = msg;
    queueLegMotorState.push(wrapper);
}

void RLControlNewPlugin::ArmMotorStatusMsg(const bodyctrl_msgs::msg::MotorStatusMsg::SharedPtr msg)
{
    auto wrapper = msg;
    queueArmMotorState.push(wrapper);
}

void RLControlNewPlugin::HeadMotorStatusMsg(const bodyctrl_msgs::msg::MotorStatusMsg::SharedPtr msg)
{
    auto wrapper = msg;
    queueHeadMotorState.push(wrapper);
}

void RLControlNewPlugin::WaistMotorStatusMsg(const bodyctrl_msgs::msg::MotorStatusMsg::SharedPtr msg)
{
    auto wrapper = msg;
    queueWaistMotorState.push(wrapper);
}

void RLControlNewPlugin::OnXsensImuStatusMsg(const bodyctrl_msgs::msg::Imu::SharedPtr msg)
{
    auto wrapper = msg;
    queueImuXsens.push(wrapper);
}

void RLControlNewPlugin::xbox_map_read(const sensor_msgs::msg::Joy::SharedPtr msg)
{
    auto wrapper = msg;
    queueJoyCmd.push(wrapper);
}

void RLControlNewPlugin::printXboxFlag(const xbox_flag& flag) {
    static std::string last_cmd;
    static bool last_disable = false;
    static bool first = true;
    const bool changed = first || (flag.fsm_state_command != last_cmd) || (flag.is_disable != last_disable);
    if (changed) {
        first = false;
        last_cmd = flag.fsm_state_command;
        last_disable = flag.is_disable;
        RCLCPP_INFO(this->get_logger(), "[XboxFlag] cmd=%s disable=%d",
                    flag.fsm_state_command.c_str(), static_cast<int>(flag.is_disable));
        return;
    }
    RCLCPP_INFO_THROTTLE(
        this->get_logger(),
        *this->get_clock(),
        2000,
        "[XboxFlag] cmd=%s disable=%d",
        flag.fsm_state_command.c_str(),
        static_cast<int>(flag.is_disable));
  }
} // namespace rl_control_new

// Register composable node
RCLCPP_COMPONENTS_REGISTER_NODE(rl_control_new::RLControlNewPlugin)
