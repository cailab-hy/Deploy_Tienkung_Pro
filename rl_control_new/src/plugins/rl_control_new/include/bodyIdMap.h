#ifndef BODY_SERVO_ID_MAP_HPP_
#define BODY_SERVO_ID_MAP_HPP_

#include <iostream>
#include <chrono>
#include <vector>
#include <unordered_map>
#include <string>


namespace bodyServoIdMap 
{

#define BODY_MAX_SERVO 30

class BodyServoIdMap 
{
public:
    BodyServoIdMap() {
        bodyCanIdMapInit();
    }

    void bodyCanIdMapInit()
    {
        // 清空所有映射 (Clear all mappings)    
        idToIndexMap.clear();
        indexToIdMap.clear();
        nameToIndexMap.clear();
        indexToNameMap.clear();

        // 腿部关节映射 (Leg joint mapping) (0-11) 
        std::vector<int> legIds = {51, 52, 53, 54, 55, 56,  // 左腿 (left leg)
                                   61, 62, 63, 64, 65, 66}; // 右腿 (right leg)
        std::vector<std::string> legNames = {
            "l_hip_roll", "l_hip_pitch", "l_hip_yaw", "l_knee", "l_ankle_pitch", "l_ankle_roll",
            "r_hip_roll", "r_hip_pitch", "r_hip_yaw", "r_knee", "r_ankle_pitch", "r_ankle_roll"
        };


        // // 手臂关节映射 (Arm joint mapping for Tienkung Lite) (12-19)
        // std::vector<int> armIds = {11, 12, 13, 14,   // 左臂 (left arm)
        //                            21, 22, 23, 24,}; // 右臂 (right arm)
        // std::vector<std::string> armNames = {
        //    "l_shoulder_pitch", "l_shoulder_roll", "l_shoulder_yaw", "l_elbow",
        //    "r_shoulder_pitch", "r_shoulder_roll", "r_shoulder_yaw", "r_elbow",
        // };

        // 手臂关节映射 (Arm joint mapping for Tienkung Pro | Add wrist rpy joints) (12-25)
        std::vector<int> armIds = {11, 12, 13, 14, 15, 16, 17,   // 左臂 (left arm)
                                   21, 22, 23, 24, 25, 26, 27}; // 右臂 (right arm)
        std::vector<std::string> armNames = {
            "l_shoulder_pitch", "l_shoulder_roll", "l_shoulder_yaw", "l_elbow", "l_wrist_yaw", "l_wrist_pitch", "l_wrist_roll",
            "r_shoulder_pitch", "r_shoulder_roll", "r_shoulder_yaw", "r_elbow", "r_wrist_yaw", "r_wrist_pitch", "r_wrist_roll",
        };

        // (Head joint mapping for Tienkung Pro) (26-28)
        std::vector<int> headIds = {1, 2, 3}; // Head joints
        std::vector<std::string> headNames = {
            "head_roll", "head_pitch", "head_yaw"
        };

        // (Waist joint mapping for Tienkung Pro) (29)
        std::vector<int> waistIds = {31}; // Waist joint
        std::vector<std::string> waistNames = {
            "waist_yaw"
        };

        // """
        // // 合并所有映射 (Combine all mappings)
        // std::vector<int> allIds;
        // std::vector<std::string> allNames;
        
        // allIds.insert(allIds.end(), legIds.begin(), legIds.end());
        // allIds.insert(allIds.end(), armIds.begin(), armIds.end());
        
        // allNames.insert(allNames.end(), legNames.begin(), legNames.end());
        // allNames.insert(allNames.end(), armNames.begin(), armNames.end());
        // """

        // 合并所有映射 (Combine all mappings)
        std::vector<int> allIds;
        std::vector<std::string> allNames;
        
        allIds.insert(allIds.end(), legIds.begin(), legIds.end());
        allIds.insert(allIds.end(), armIds.begin(), armIds.end());
        allIds.insert(allIds.end(), headIds.begin(), headIds.end());
        allIds.insert(allIds.end(), waistIds.begin(), waistIds.end());

        allNames.insert(allNames.end(), legNames.begin(), legNames.end());
        allNames.insert(allNames.end(), armNames.begin(), armNames.end());
        allNames.insert(allNames.end(), headNames.begin(), headNames.end());
        allNames.insert(allNames.end(), waistNames.begin(), waistNames.end());

        // 创建双向映射 (Create bidirectional mappings)
        for (size_t index = 0; index < allIds.size() && index < BODY_MAX_SERVO; ++index) {
            int canId = allIds[index];
            const std::string& name = allNames[index];
            
            idToIndexMap[canId] = static_cast<int>(index);
            indexToIdMap[static_cast<int>(index)] = canId;
            nameToIndexMap[name] = static_cast<int>(index);
            indexToNameMap[static_cast<int>(index)] = name;
        }
    }
    
    /*
    id:
    index: use for vector
    map id to index;    
    */
    int getIndexById(int canId) const
    {
        auto it = idToIndexMap.find(canId);
        if (it != idToIndexMap.end()) {
            return it->second;
        }
        return -1;
    }

    /*
    id: 
    index: use for vector
    map index to id;    
    */
    int getIdByIndex(int index) const
    {
        auto it = indexToIdMap.find(index);
        if (it != indexToIdMap.end()) {
            return it->second;
        }
        return -1;
    }

    /*
    name: joint name
    index: use for vector
    map name to index;
    */
    int getIndexByName(const std::string& name) const
    {
        auto it = nameToIndexMap.find(name);
        if (it != nameToIndexMap.end()) {
            return it->second;
        }
        return -1;
    }

    /*
    index: use for vector
    name: joint name
    map index to name;
    */
    std::string getNameByIndex(int index) const
    {
        auto it = indexToNameMap.find(index);
        if (it != indexToNameMap.end()) {
            return it->second;
        }
        return "";
    }

    // 获取所有映射信息的辅助函数 (Helper functions to get all mapping information)
    const std::unordered_map<int, int>& getIdToIndexMap() const { return idToIndexMap; }
    const std::unordered_map<int, int>& getIndexToIdMap() const { return indexToIdMap; }
    const std::unordered_map<std::string, int>& getNameToIndexMap() const { return nameToIndexMap; }
    const std::unordered_map<int, std::string>& getIndexToNameMap() const { return indexToNameMap; }

private:
    // 使用哈希表提高查找效率 (Using hash tables for efficient lookups)
    std::unordered_map<int, int> idToIndexMap;
    std::unordered_map<int, int> indexToIdMap;
    std::unordered_map<std::string, int> nameToIndexMap;
    std::unordered_map<int, std::string> indexToNameMap;
};

}

#endif //BODY_SERVO_ID_MAP_HPP_