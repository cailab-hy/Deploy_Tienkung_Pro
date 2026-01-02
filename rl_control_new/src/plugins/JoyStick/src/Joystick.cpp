#include "../include/Joystick.h"
#include <std_msgs/msg/int32.hpp>
#include <yaml-cpp/yaml.h>
#include <ament_index_cpp/get_package_share_directory.hpp>

Joystick_humanoid::Joystick_humanoid()
{
  std::cout << "Joystick Start" << std::endl;

  try
  {
    std::string pkg_path;
    try {
      pkg_path = ament_index_cpp::get_package_share_directory("rl_control_new");
    } catch (const std::exception& e) {
      std::cerr << "[ERROR] Failed to find package: rl_control_new - " << e.what() << std::endl;
      pkg_path = "../rl_control_new";
    }
    YAML::Node config = YAML::LoadFile(pkg_path + "/config/tg30_config.yaml"); // 使用相对路径加载配置文件
    if (!config)
    {
      std::cerr << "[Joystick_humanoid] Failed to load config file: /path/to/config.yaml" << std::endl;
      return; // 或者抛出异常
    }

    auto joystick_cfg = config["joystick"];

    // 示例：加载配置参数
    forward_command_offset_ = joystick_cfg["forward_command_offset"].as<float>();
    lateral_command_offset_ = joystick_cfg["lateral_command_offset"].as<float>();
    rotation_command_offset_ = joystick_cfg["rotation_command_offset"].as<float>();
    
    // 输出读取的参数
    std::cout << ", forward_command_offset: " << forward_command_offset_ 
              << ", lateral_command_offset: " << lateral_command_offset_ 
              << ", rotation_command_offset: " << rotation_command_offset_ << std::endl;

  }
  catch (const std::exception &e)
  {
    std::cerr << "[Joystick_humanoid] YAML load error: " << e.what() << std::endl;
  }


}

Joystick_humanoid::~Joystick_humanoid()
{
  std::cout << "Joystick End" << std::endl;
}

void Joystick_humanoid::xbox_map_read(const sensor_msgs::msg::Joy::ConstSharedPtr &msg)
{
  std::lock_guard<std::mutex> lock(data_mutex);
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
}

void Joystick_humanoid::xbox_flag_update(xbox_map_t xbox_map)
{
  xbox_map_ = xbox_map;
  std::lock_guard<std::mutex> lock(data_mutex);
  
  // Joint enable flag
  if (xbox_map_.e == 1.0 && xbox_map_.b == 1.0)
  {
    xbox_flag_.is_disable = true;
    
  }
  else
  {
    xbox_flag_.is_disable = false;
  }

  // Get mlp state change flag
  xbox_flag_.fsm_state_command = (xbox_map_.d == 1.0)   ? "gotoZero"
                                 : (xbox_map_.c == 1.0) ? "gotoStop"
                                 : (xbox_map_.a == 1.0 && xbox_map_.g == 0.0) ? "gotoMLP"
                                  : xbox_flag_.fsm_state_command;


  xbox_flag_.y_speed_command = xbox_map_.x1 * -0.4 + lateral_command_offset_;

  if (xbox_map_.y1 >= 0)
  {
    xbox_flag_.x_speed_command = xbox_map_.y1 * 0.8 + forward_command_offset_; // 前进快一点
  }
  else
  {
    xbox_flag_.x_speed_command = xbox_map_.y1 * 0.5; // 后退慢一点
  }

  xbox_flag_.yaw_speed_command = xbox_map_.y2 * -0.4 + rotation_command_offset_;

}

xbox_flag Joystick_humanoid::get_xbox_flag()
{
  return xbox_flag_;
}

int Joystick_humanoid::init()
{
  return 0;
}
