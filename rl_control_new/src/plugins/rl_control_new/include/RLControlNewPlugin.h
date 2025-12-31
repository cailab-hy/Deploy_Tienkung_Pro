/**
 * @file RLControlNewPlugin.h
 * @brief ROS2版本的人形机器人强化学习控制插件头文件 (Header file for the ROS2-based humanoid robot reinforcement learning control plugin)
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

// 前向声明 (forward declarations)
namespace broccoli {
namespace core {
class Time;
}
}

class funcSPTrans;

namespace rl_control_new {

/**
 * @class RLControlNewPlugin
 * @brief 强化学习控制插件类，负责处理人形机器人的强化学习控制逻辑 (RL control plugin class, responsible for handling the humanoid robot’s RL control logic)
 */
class RLControlNewPlugin : public rclcpp::Node {
public:
    /**
     * @brief 带 NodeOptions 的构造函数（用于可组合节点）(Constructor with NodeOptions (for composable nodes))
     * @param options ROS2节点选项 (ROS2 node options)
     */
    explicit RLControlNewPlugin(const rclcpp::NodeOptions & options);

private:
    /**
     * @brief 初始化节点 (Initialize the node)
     */
    virtual void onInit();

    /**
     * @brief 加载配置文件 (Load configuration file)
     * @param config_file 配置文件路径 (Path to the configuration file)
     * @return 是否加载成功 (Whether loading was successful)
     */
    bool LoadConfig(const std::string &config_file);

    /**
     * @brief 强化学习控制主循环 (Main loop for reinforcement learning control)
     */
    void rlControl();

    /**
     * @brief 处理腿部电机状态消息回调函数 (Handle leg motor status message callback)
     * @param msg 消息指针 (Message pointer)
     */
    void LegMotorStatusMsg(const bodyctrl_msgs::msg::MotorStatusMsg::SharedPtr msg);

    /**
     * @brief 处理手臂电机状态消息回调函数 (Handle arm motor status message callback)
     * @param msg 消息指针 (Message pointer)
     */
    void ArmMotorStatusMsg(const bodyctrl_msgs::msg::MotorStatusMsg::SharedPtr msg);

    /**
     * @brief (Handle Head motor status message callback for Tienkung Pro)
     * @param msg (Message pointer)
     */
    void HeadMotorStatusMsg(const bodyctrl_msgs::msg::MotorStatusMsg::SharedPtr msg);

    /**
     * @brief (Handle Waist motor status message callback for Tienkung Pro)
     * @param msg (Message pointer)
     */
    void WaistMotorStatusMsg(const bodyctrl_msgs::msg::MotorStatusMsg::SharedPtr msg);

    /**
     * @brief 处理惯性测量单元(IMU)数据消息回调函数 (Handle Inertial Measurement Unit (IMU) data message callback)
     * @param msg 消息指针 (Message pointer)
     */
    void OnXsensImuStatusMsg(const bodyctrl_msgs::msg::Imu::SharedPtr msg);

    /**
     * @brief 处理游戏手柄输入消息回调函数 (Handle game controller input message callback)
     * @param msg 消息指针 (Message pointer)
     */
    void xbox_map_read(const sensor_msgs::msg::Joy::SharedPtr msg);

    /**
     * @brief 打印游戏手柄标志信息 (Print game controller flag information)
     * @param flag 游戏手柄标志结构体 (Game controller flag structure)
     */
    void printXboxFlag(const xbox_flag& flag);

    // 订阅器 (subscribers)
    rclcpp::Subscription<bodyctrl_msgs::msg::MotorStatusMsg>::SharedPtr subLegState;
    rclcpp::Subscription<bodyctrl_msgs::msg::MotorStatusMsg>::SharedPtr subArmState;
    rclcpp::Subscription<bodyctrl_msgs::msg::MotorStatusMsg>::SharedPtr subHeadState;
    rclcpp::Subscription<bodyctrl_msgs::msg::MotorStatusMsg>::SharedPtr subWaistState;
    rclcpp::Subscription<bodyctrl_msgs::msg::Imu>::SharedPtr subImuXsens;
    rclcpp::Subscription<sensor_msgs::msg::Joy>::SharedPtr subJoyCmd;

    // 发布器 (publishers)
    rclcpp::Publisher<bodyctrl_msgs::msg::CmdMotorCtrl>::SharedPtr pubLegMotorCmd;
    rclcpp::Publisher<bodyctrl_msgs::msg::CmdMotorCtrl>::SharedPtr pubArmMotorCmd;
    rclcpp::Publisher<bodyctrl_msgs::msg::CmdMotorCtrl>::SharedPtr pubHeadMotorCmd;
    rclcpp::Publisher<bodyctrl_msgs::msg::CmdMotorCtrl>::SharedPtr pubWaistMotorCmd;
    // rclcpp::Publisher<bodyctrl_msgs::msg::CmdSetMotorPosition>::SharedPtr waists_cmd_pub_;

    // 状态变量 (state variables)
    Eigen::VectorXd q_a;            // 关节角度实际值 (Actual joint angles)
    Eigen::VectorXd qdot_a;         // 关节角速度实际值 (Actual joint velocities)
    Eigen::VectorXd tor_a;          // 关节力矩实际值 (Actual joint torques)
    Eigen::VectorXd q_d;            // 关节角度目标值 (Target joint angles)
    Eigen::VectorXd qdot_d;         // 关节角速度目标值 (Target joint velocities)
    Eigen::VectorXd tor_d;          // 关节力矩目标值 (Target joint torques)
    Eigen::VectorXd Q_a;            // 电机角度实际值 (Actual motor angles)
    Eigen::VectorXd Qdot_a;         // 电机角速度实际值 (Actual motor velocities)
    Eigen::VectorXd Tor_a;          // 电机力矩实际值 (Actual motor torques)
    Eigen::VectorXd Q_a_last;       // 上一时刻电机角度实际值 (Last actual motor angles)
    Eigen::VectorXd Qdot_a_last;    // 上一时刻电机角速度实际值 (Last actual motor velocities)
    Eigen::VectorXd Tor_a_last;     // 上一时刻电机力矩实际值 (Last actual motor torques)
    Eigen::VectorXd Q_d;            // 电机角度目标值 (Target motor angles)
    Eigen::VectorXd Qdot_d;         // 电机角速度目标值 (Target motor velocities)
    Eigen::VectorXd Tor_d;          // 电机力矩目标值 (Target motor torques)
    Eigen::VectorXd ct_scale;       // 电流到力矩转换比例 (Current to torque conversion scale)
    Eigen::VectorXd data;           // 数据缓冲区 (Data buffer)
    Eigen::VectorXd zero_pos;       // 零点位置 (Zero position)
    Eigen::VectorXd zero_offset;    // 零点偏移 (Zero offset)
    Eigen::VectorXd init_pos;       // 初始位置 (Initial position)
    Eigen::VectorXd motor_dir;      // 电机方向 (Motor direction)
    Eigen::VectorXd zero_cnt;       // 零点计数 (Zero count)

    // 串联并联转换相关变量 (Variables related to series-parallel conversion)
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

    Eigen::VectorXd xsense_data;  // IMU数据 (IMU data)
    Eigen::VectorXd kp;           // 比例增益 (Proportional gain)
    Eigen::VectorXd kd;           // 微分增益 (Derivative gain)
    Eigen::VectorXd Temperature;  // 温度数据 (Temperature data)

    // 配置相关
    std::string _config_file;    // 配置文件路径 (Path to configuration file)
    YAML::Node config_;          // YAML配置对象 (YAML configuration object)
    std::unordered_map<std::string, std::string> _config_map; // 配置映射 (Configuration map)

    xbox_map_t xbox_map;         // 手柄映射结构体 (Game controller mapping structure)

    double dt;                   // 控制周期 (Control period)
    int motor_num;               // 电机数量 (Number of motors)
    int action_num;              // 动作数量 (Number of actions)
    std::map<int, int> motor_id; // 电机ID映射 (Motor ID mapping)
    std::map<int, int> motor_name; // 电机名称映射 (Motor name mapping)
    float rpm2rps;               // 转速单位转换系数 (RPM to RPS conversion factor)
    funcSPTrans *funS2P;         // 串并联转换功能对象 (Series-parallel conversion function object)
    float pi;                    // 圆周率 (PI)
    bool simulation;             // 是否为仿真模式 (Whether in simulation mode)

    // 消息队列 (message queues)
    LockFreeQueue<bodyctrl_msgs::msg::MotorStatusMsg::SharedPtr> queueLegMotorState;
    LockFreeQueue<bodyctrl_msgs::msg::MotorStatusMsg::SharedPtr> queueArmMotorState;
    LockFreeQueue<bodyctrl_msgs::msg::MotorStatusMsg::SharedPtr> queueHeadMotorState;
    LockFreeQueue<bodyctrl_msgs::msg::MotorStatusMsg::SharedPtr> queueWaistMotorState;
    LockFreeQueue<bodyctrl_msgs::msg::Imu::SharedPtr> queueImuXsens;
    LockFreeQueue<sensor_msgs::msg::Joy::SharedPtr> queueJoyCmd;

    // 机器人数据 (robot data )
    RobotData robot_data;

    // ID映射 (ID mapping )
    bodyServoIdMap::BodyServoIdMap idMap;

    // 中间变量向量 (Intermediate variable vectors)
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