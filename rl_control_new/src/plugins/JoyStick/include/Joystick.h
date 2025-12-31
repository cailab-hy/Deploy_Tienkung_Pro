#ifndef JOYSTICK_H_
#define JOYSTICK_H_

#include <mutex>
#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/int32.hpp>
#include <sensor_msgs/msg/joy.hpp>
#include <atomic>
// 通用结构体定义 (General struct definitions)
struct xbox_flag
{
  bool is_disable = false;
  std::string fsm_state_command = "gotoSTOP";
  double command_gait = 1.0;
  double x_speed_command = 0.0;
  double y_speed_command = 0.0;
  double yaw_speed_command = 0.0;
  double x_speed_offset = 0.0;
  double y_speed_offset = 0.0;
};


// 云卓T12手柄模式 (Xbox T12 controller mode)

typedef struct xbox_map
{
  double a;
  double b;
  double c;
  double d;
  double e;
  double f;
  double g;
  double h;
  double x1;
  double x2;
  double y1;
  double y2;
} xbox_map_t;

class Joystick_humanoid
{
public:
  Joystick_humanoid();
  ~Joystick_humanoid();
  int init();
  xbox_flag get_xbox_flag();
  void xbox_flag_update(xbox_map_t xbox_map);
  void xbox_map_read(const sensor_msgs::msg::Joy::ConstSharedPtr &msg);

private:
  std::mutex data_mutex;
  std::shared_ptr<rclcpp::Node> nh_;
  rclcpp::Publisher<std_msgs::msg::Int32>::SharedPtr arm_box_control_pub; // 添加发布器成员变量 (Add publisher member variable)

  // ros::Subscriber sub_;
  xbox_map xbox_map_;
  xbox_flag xbox_flag_;
  float lateral_command_offset_;
  float forward_command_offset_;
  float rotation_command_offset_;
  // void xbox_map_read(const sensor_msgs::Joy::ConstPtr &msg);
};


#endif // JOYSTICK_H_