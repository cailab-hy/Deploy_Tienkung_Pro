/**
 * @file hdmi_inference_node.cpp
 * @brief ROS2 node for HDMI policy inference on Jetson Orin (50Hz)
 *
 * Subscribes to sensor state from Main PC, depth camera (local), joystick.
 * Runs HdmiPolicyRunner (ONNX inference + FSM).
 * Publishes desired joint positions back to Main PC.
 */

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <opencv2/imgproc.hpp>
#include <sensor_msgs/msg/joy.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <algorithm>
#include <mutex>
#include <chrono>

#include "HdmiPolicyRunner.h"
#include "ObservationBuilder.h"
#include "Joystick.h"

class HdmiInferenceNode : public rclcpp::Node {
public:
    HdmiInferenceNode() : Node("hdmi_inference") {
        RCLCPP_INFO(get_logger(), "Starting HDMI Inference Node (50Hz)");

        // Initialize policy runner with 4 policies (STAND1 -> PICKANDPLACE -> STAND1 -> RETURN -> repeat)
        std::string pkg_path = ament_index_cpp::get_package_share_directory("hdmi_inference");
        std::string stand1_dir         = pkg_path + "/config/policy_stand1";
        std::string stand1_name        = "policy-b3tvcz86-final";
        std::string pickandplace_dir  = pkg_path + "/config/policy_pickandplace";
        std::string pickandplace_name = "policy-ilv22uqa-final";
        std::string return_dir         = pkg_path + "/config/policy_return";
        std::string return_name        = "policy-zm8eaj0d-4000";
        hdmi_runner_.init(stand1_dir,        stand1_name,
                          pickandplace_dir,  pickandplace_name,
                          stand1_dir,        stand1_name,
                          return_dir,        return_name);

        // Initialize joystick
        joystick_.init();

        // Initialize depth frame
        depth_frame_.resize(ObservationBuilder::DEPTH_SIZE, 0.0f);

        // Initialize sensor data
        q_isaac_ = Eigen::VectorXd::Zero(30);
        imu_data_ = Eigen::VectorXd::Zero(9);
        sensor_received_ = false;

        // --- Publishers ---
        pub_action_ = create_publisher<std_msgs::msg::Float64MultiArray>(
            "/hdmi/action", 10);

        // PD gains: publish once with transient_local durability
        auto qos_latched = rclcpp::QoS(1).transient_local();
        pub_pd_gains_ = create_publisher<std_msgs::msg::Float64MultiArray>(
            "/hdmi/pd_gains", qos_latched);

        // --- Subscribers ---
        sub_sensor_state_ = create_subscription<std_msgs::msg::Float64MultiArray>(
            "/hdmi/sensor_state", 10,
            std::bind(&HdmiInferenceNode::sensorStateCallback, this, std::placeholders::_1));

        sub_depth_image_ = create_subscription<sensor_msgs::msg::Image>(
            "/camera/depth/image_raw", 10,
            std::bind(&HdmiInferenceNode::depthImageCallback, this, std::placeholders::_1));

        sub_joy_ = create_subscription<sensor_msgs::msg::Joy>(
            "/sbus_data", 100,
            std::bind(&HdmiInferenceNode::joyCallback, this, std::placeholders::_1));

        // --- 50Hz Timer ---
        timer_ = create_wall_timer(
            std::chrono::milliseconds(20),
            std::bind(&HdmiInferenceNode::timerCallback, this));

        // Publish PD gains once
        publishPdGains();

        RCLCPP_INFO(get_logger(), "HDMI Inference Node ready. Waiting for sensor data...");
    }

private:
    void sensorStateCallback(const std_msgs::msg::Float64MultiArray::SharedPtr msg) {
        if (msg->data.size() < 39) {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
                "sensor_state has %zu values, expected 39", msg->data.size());
            return;
        }

        std::lock_guard<std::mutex> lock(sensor_mutex_);
        for (int i = 0; i < 30; i++) {
            q_isaac_(i) = msg->data[i];
        }
        for (int i = 0; i < 9; i++) {
            imu_data_(i) = msg->data[30 + i];
        }
        sensor_received_ = true;
    }

    void depthImageCallback(const sensor_msgs::msg::Image::SharedPtr msg) {
        cv::Mat src_f;

        if (msg->encoding == "32FC1") {
            cv::Mat src(msg->height, msg->width, CV_32FC1,
                        const_cast<void*>(static_cast<const void*>(msg->data.data())));
            src.copyTo(src_f);
        } else if (msg->encoding == "16UC1") {
            cv::Mat src16(msg->height, msg->width, CV_16UC1,
                          const_cast<void*>(static_cast<const void*>(msg->data.data())));
            src16.convertTo(src_f, CV_32FC1, 1.0 / 1000.0);
        } else {
            return;
        }

        cv::Mat dst;
        cv::resize(src_f, dst, cv::Size(160, 120), 0, 0, cv::INTER_LINEAR);

        std::lock_guard<std::mutex> lock(depth_mutex_);
        const float* ptr = dst.ptr<float>(0);
        depth_frame_.assign(ptr, ptr + 120 * 160);
    }

    void joyCallback(const sensor_msgs::msg::Joy::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(joy_mutex_);
        if (msg->axes.size() == 12) {
            xbox_map_.a = msg->axes[8];
            xbox_map_.b = msg->axes[9];
            xbox_map_.c = msg->axes[10];
            xbox_map_.d = msg->axes[11];
            xbox_map_.e = msg->axes[4];
            xbox_map_.f = msg->axes[7];
            xbox_map_.g = msg->axes[5];
            xbox_map_.h = msg->axes[6];
            xbox_map_.x1 = msg->axes[3];
            xbox_map_.x2 = msg->axes[1];
            xbox_map_.y1 = msg->axes[2];
            xbox_map_.y2 = msg->axes[0];
        } else if (msg->axes.size() == 8) {
            xbox_map_.x1 = msg->axes[0];
            xbox_map_.x2 = msg->axes[3];
            xbox_map_.y1 = msg->axes[1];
            xbox_map_.y2 = msg->axes[4];
            xbox_map_.a = msg->buttons[0];
            xbox_map_.b = msg->buttons[1];
            xbox_map_.c = msg->buttons[3];
            xbox_map_.d = msg->buttons[2];
            xbox_map_.e = msg->buttons[4];
            xbox_map_.f = msg->buttons[5];
            xbox_map_.g = msg->buttons[6];
            xbox_map_.h = msg->buttons[7];
        }
    }

    void timerCallback() {
        // Skip if no sensor data received yet
        if (!sensor_received_) {
            return;
        }

        // Copy sensor data (thread-safe)
        Eigen::VectorXd q_copy, imu_copy;
        {
            std::lock_guard<std::mutex> lock(sensor_mutex_);
            q_copy = q_isaac_;
            imu_copy = imu_data_;
        }

        // Copy depth (thread-safe)
        std::vector<float> depth_copy;
        {
            std::lock_guard<std::mutex> lock(depth_mutex_);
            depth_copy = depth_frame_;
        }

        // Update joystick flags
        xbox_flag flag;
        {
            std::lock_guard<std::mutex> lock(joy_mutex_);
            joystick_.xbox_flag_update(xbox_map_);
            flag = joystick_.get_xbox_flag();
        }

        // Detect STOP→ZERO transition for waist reset
        HdmiPolicyRunner::State prev_state = hdmi_runner_.getCurrentState();

        // Run policy pipeline
        hdmi_runner_.updateState(q_copy, imu_copy, depth_copy);
        hdmi_runner_.runFSM(flag);

        HdmiPolicyRunner::State cur_state = hdmi_runner_.getCurrentState();
        bool waist_reset = (prev_state == HdmiPolicyRunner::STOP && cur_state == HdmiPolicyRunner::ZERO);

        // Publish action
        auto action_msg = std_msgs::msg::Float64MultiArray();
        action_msg.data.resize(93);

        Eigen::VectorXd q_d = hdmi_runner_.getDesiredPos();
        Eigen::VectorXd qdot_d = hdmi_runner_.getDesiredVel();
        Eigen::VectorXd tor_d = hdmi_runner_.getDesiredTor();

        for (int i = 0; i < 30; i++) {
            action_msg.data[i] = q_d(i);
            action_msg.data[30 + i] = qdot_d(i);
            action_msg.data[60 + i] = tor_d(i);
        }
        action_msg.data[90] = static_cast<double>(cur_state);
        action_msg.data[91] = hdmi_runner_.disableJoints() ? 1.0 : 0.0;
        action_msg.data[92] = waist_reset ? 1.0 : 0.0;

        pub_action_->publish(action_msg);

        // Periodic log
        timer_count_++;
        if (timer_count_ % 250 == 0) {  // Every 5s at 50Hz
            float d_min = 0.0f, d_max = 0.0f;
            {
                std::lock_guard<std::mutex> lock(depth_mutex_);
                if (!depth_frame_.empty()) {
                    d_min = *std::min_element(depth_frame_.begin(), depth_frame_.end());
                    d_max = *std::max_element(depth_frame_.begin(), depth_frame_.end());
                }
            }
            RCLCPP_INFO(get_logger(), "State=%d, Policy=%d, q_d[0]=%.4f, waist_reset=%d, depth=%zu(%.2f~%.2fm)",
                        cur_state, hdmi_runner_.getActivePolicy(), q_d(0), waist_reset, depth_frame_.size(), d_min, d_max);
        }
    }

    void publishPdGains() {
        auto gains_msg = std_msgs::msg::Float64MultiArray();
        gains_msg.data.resize(60);

        const auto& kp = hdmi_runner_.getKp();
        const auto& kd = hdmi_runner_.getKd();

        for (int i = 0; i < 30; i++) {
            gains_msg.data[i] = kp(i);
            gains_msg.data[30 + i] = kd(i);
        }

        pub_pd_gains_->publish(gains_msg);
        RCLCPP_INFO(get_logger(), "Published PD gains (kp[0]=%.1f, kd[0]=%.1f)", kp(0), kd(0));
    }

    // Publishers
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr pub_action_;
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr pub_pd_gains_;

    // Subscribers
    rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr sub_sensor_state_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_depth_image_;
    rclcpp::Subscription<sensor_msgs::msg::Joy>::SharedPtr sub_joy_;

    // Timer
    rclcpp::TimerBase::SharedPtr timer_;
    int timer_count_ = 0;

    // Components
    HdmiPolicyRunner hdmi_runner_;
    Joystick_humanoid joystick_;

    // Thread-safe data
    std::mutex sensor_mutex_;
    Eigen::VectorXd q_isaac_;
    Eigen::VectorXd imu_data_;
    bool sensor_received_ = false;

    std::mutex depth_mutex_;
    std::vector<float> depth_frame_;

    std::mutex joy_mutex_;
    xbox_map_t xbox_map_ = {};
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<HdmiInferenceNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
