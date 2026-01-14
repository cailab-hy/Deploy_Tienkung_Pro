/**
 * @file RLControlNewPlugin.h
 * @brief Header file for the ROS2-based humanoid robot reinforcement learning control plugin
 * @version 2.0
 * @date 2025-09-17
 */

#ifndef RL_CONTROL_NEW_PLUGIN_H
#define RL_CONTROL_NEW_PLUGIN_H

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float64.hpp>
#include <bodyctrl_msgs/msg/motor_status_msg.hpp>
#include <bodyctrl_msgs/msg/imu.hpp>
#include <bodyctrl_msgs/msg/cmd_set_motor_speed.hpp>
#include <bodyctrl_msgs/msg/cmd_set_motor_position.hpp>
#include <bodyctrl_msgs/msg/cmd_motor_ctrl.hpp>
#include <bodyctrl_msgs/msg/motor_name.hpp>
#include <sensor_msgs/msg/joy.hpp>
#include <std_msgs/msg/float32.hpp>

#include <thread>
#include <Eigen/Dense>
#include "util/LockFreeQueue.h"
#include "robot_interface/RobotInterface.h"
#include "robot_FSM/RobotFSM.h"
#include "Joystick.h"
#include "bodyIdMap.h"
#include <yaml-cpp/yaml.h>

// forward declarations
namespace broccoli {
namespace core {
class Time;
}
}

class funcSPTrans;

namespace rl_control_new {

/**
 * @class RLControlNewPlugin
 * @brief RL control plugin class, responsible for handling the humanoid robot’s RL control logic
 */
class RLControlNewPlugin : public rclcpp::Node {
public:
    /**
     * @brief Constructor with NodeOptions (for composable nodes)
     * @param ROS2 node options
     */
    explicit RLControlNewPlugin(const rclcpp::NodeOptions & options);

private:
    /**
     * @brief Initialize the node
     */
    virtual void onInit();

    /**
     * @brief Load configuration file
     * @param Path to the config file
     * @return Whether loading was successful
     */
    bool LoadConfig(const std::string &config_file);

    /**
     * @brief Main loop for reinforcement learning control
     */
    void rlControl();

    /**
     * @brief Handle leg motor status message callback
     * @param msg pointer
     */
    void LegMotorStatusMsg(const bodyctrl_msgs::msg::MotorStatusMsg::SharedPtr msg);

    /**
     * @brief Handle arm motor status message callback
     * @param msg pointer
     */
    void ArmMotorStatusMsg(const bodyctrl_msgs::msg::MotorStatusMsg::SharedPtr msg);

    /**
     * @brief Handle Head motor status message callback for Tienkung Pro
     * @param msg pointer
     */
    void HeadMotorStatusMsg(const bodyctrl_msgs::msg::MotorStatusMsg::SharedPtr msg);

    /**
     * @brief Handle Waist motor status message callback for Tienkung Pro
     * @param msg pointer
     */
    void WaistMotorStatusMsg(const bodyctrl_msgs::msg::MotorStatusMsg::SharedPtr msg);

    /**
     * @brief Handle Inertial Measurement Unit (IMU) data message callback
     * @param msg pointer
     */
    void OnXsensImuStatusMsg(const bodyctrl_msgs::msg::Imu::SharedPtr msg);

    /**
     * @brief Handle game controller input message callback
     * @param msg pointer
     */
    void xbox_map_read(const sensor_msgs::msg::Joy::SharedPtr msg);

    /**
     * @brief Print game controller flag information
     * @param Game controller flag structure
     */
    void printXboxFlag(const xbox_flag& flag);

    // subscribers
    rclcpp::Subscription<bodyctrl_msgs::msg::MotorStatusMsg>::SharedPtr subLegState;
    rclcpp::Subscription<bodyctrl_msgs::msg::MotorStatusMsg>::SharedPtr subArmState;
    rclcpp::Subscription<bodyctrl_msgs::msg::MotorStatusMsg>::SharedPtr subHeadState;
    rclcpp::Subscription<bodyctrl_msgs::msg::MotorStatusMsg>::SharedPtr subWaistState;
    rclcpp::Subscription<bodyctrl_msgs::msg::Imu>::SharedPtr subImuXsens;
    rclcpp::Subscription<sensor_msgs::msg::Joy>::SharedPtr subJoyCmd;

    // publishers
    rclcpp::Publisher<bodyctrl_msgs::msg::CmdMotorCtrl>::SharedPtr pubLegMotorCmd;
    rclcpp::Publisher<bodyctrl_msgs::msg::CmdMotorCtrl>::SharedPtr pubArmMotorCmd;
    rclcpp::Publisher<bodyctrl_msgs::msg::CmdMotorCtrl>::SharedPtr pubHeadMotorCmd;
    rclcpp::Publisher<bodyctrl_msgs::msg::CmdMotorCtrl>::SharedPtr pubWaistMotorCmd;
    rclcpp::Publisher<bodyctrl_msgs::msg::CmdSetMotorPosition>::SharedPtr waists_cmd_pub_;

    // state variables
    Eigen::VectorXd q_a;            // Actual joint angles
    Eigen::VectorXd qdot_a;         // Actual joint velocities
    Eigen::VectorXd tor_a;          // Actual joint torques
    Eigen::VectorXd q_d;            // Target joint angles
    Eigen::VectorXd qdot_d;         // Target joint velocities
    Eigen::VectorXd tor_d;          // Target joint torques
    Eigen::VectorXd Q_a;            // Actual motor angles
    Eigen::VectorXd Qdot_a;         // Actual motor velocities
    Eigen::VectorXd Tor_a;          // Actual motor torques
    Eigen::VectorXd Q_a_last;       // 上一Last actual motor angles
    Eigen::VectorXd Qdot_a_last;    // 上一Last actual motor velocities
    Eigen::VectorXd Tor_a_last;     // 上一Last actual motor torques
    Eigen::VectorXd Q_d;            // Target motor angles
    Eigen::VectorXd Qdot_d;         // Target motor velocities
    Eigen::VectorXd Tor_d;          // Target motor torques
    Eigen::VectorXd ct_scale;       // Current to torque conversion scale
    Eigen::VectorXd data;           // Data buffer
    Eigen::VectorXd zero_pos;       // Zero position
    Eigen::VectorXd zero_offset;    // Zero offset
    Eigen::VectorXd init_pos;       // Initial position
    Eigen::VectorXd motor_dir;      // Motor direction
    Eigen::VectorXd zero_cnt;       // Zero count

    // Variables related to series-parallel conversion
    Eigen::VectorXd q_a_p;
    Eigen::VectorXd qdot_a_p;
    Eigen::VectorXd tor_a_p;
    Eigen::VectorXd q_d_p;
    Eigen::VectorXd qdot_d_p;
    Eigen::VectorXd tor_d_p;
    Eigen::VectorXd q_a_s;
    Eigen::VectorXd qdot_a_s;
    Eigen::VectorXd tor_a_s;
    Eigen::VectorXd q_d_s;
    Eigen::VectorXd qdot_d_s;
    Eigen::VectorXd tor_d_s;

    Eigen::VectorXd xsense_data;  // IMU data
    Eigen::VectorXd kp;           // Proportional gain
    Eigen::VectorXd kd;           // Derivative gain
    Eigen::VectorXd Temperature;  // Temperature data

    // 配置相关
    std::string _config_file;    // Path to configuration file
    YAML::Node config_;          // YAML configuration object
    std::unordered_map<std::string, std::string> _config_map; // Configuration map

    xbox_map_t xbox_map;         // Game controller mapping structure

    double dt;                      // Control period
    int motor_num;                  // Number of motors
    int action_num;                 // Number of actions
    std::map<int, int> motor_id;    // Motor ID mapping
    std::map<int, int> motor_name;  // Motor name mapping
    float rpm2rps;                  // RPM to RPS conversion factor
    funcSPTrans *funS2P;            // Series-parallel conversion function object
    float pi;                       // PI)
    bool simulation;                // Whether in simulation mode

    // message queues
    LockFreeQueue<bodyctrl_msgs::msg::MotorStatusMsg::SharedPtr> queueLegMotorState;
    LockFreeQueue<bodyctrl_msgs::msg::MotorStatusMsg::SharedPtr> queueArmMotorState;
    LockFreeQueue<bodyctrl_msgs::msg::MotorStatusMsg::SharedPtr> queueHeadMotorState;
    LockFreeQueue<bodyctrl_msgs::msg::MotorStatusMsg::SharedPtr> queueWaistMotorState;
    LockFreeQueue<bodyctrl_msgs::msg::Imu::SharedPtr> queueImuXsens;
    LockFreeQueue<sensor_msgs::msg::Joy::SharedPtr> queueJoyCmd;

    // robot data 
    RobotData robot_data;

    // ID mapping
    bodyServoIdMap::BodyServoIdMap idMap;

    // Intermediate variable vectors
    Eigen::VectorXd pos_fed_midVec;
    Eigen::VectorXd vel_fed_midVec;
    Eigen::VectorXd tau_fed_midVec;
    Eigen::VectorXd pos_cmd_midVec;
    Eigen::VectorXd vel_cmd_midVec;
    Eigen::VectorXd tau_cmd_midVec;
    Eigen::VectorXd ct_scale_midVec;
    Eigen::VectorXd temperature_midVec;
};

} // namespace rl_control_new

#endif // RL_CONTROL_NEW_PLUGIN_H
