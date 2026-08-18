#include "OnnxInference.h"
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <numeric>
#include <cassert>

using json = nlohmann::json;

void OnnxInference::loadModel(const std::string& onnx_path, const std::string& json_path) {
    // Load JSON metadata
    std::ifstream json_file(json_path);
    if (!json_file.is_open()) {
        throw std::runtime_error("Failed to open JSON metadata: " + json_path);
    }
    json meta;
    json_file >> meta;

    // Parse out_keys to find action index
    auto& out_keys = meta["out_keys"];
    action_output_idx_ = -1;
    for (size_t i = 0; i < out_keys.size(); i++) {
        if (out_keys[i].is_string() && out_keys[i].get<std::string>() == "action") {
            action_output_idx_ = static_cast<int>(i);
            break;
        }
    }
    if (action_output_idx_ < 0) {
        throw std::runtime_error("Could not find 'action' in out_keys");
    }

    // Parse input shapes from JSON
    // in_shapes: [[[1,376],[1,276],[1,1,120,160]]]
    auto& in_shapes = meta["in_shapes"][0];
    input_shapes_.clear();
    for (auto& shape : in_shapes) {
        std::vector<int64_t> s;
        for (auto& dim : shape) {
            s.push_back(dim.get<int64_t>());
        }
        input_shapes_.push_back(s);
    }

    // Initialize ONNX Runtime
    env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "HdmiPolicy");

    session_options_.SetIntraOpNumThreads(1);
    session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    // Try CUDA provider first, fall back to CPU
    try {
        OrtCUDAProviderOptions cuda_options;
        cuda_options.device_id = 0;
        session_options_.AppendExecutionProvider_CUDA(cuda_options);
        std::cout << "[OnnxInference] Using CUDA execution provider" << std::endl;
    } catch (...) {
        std::cout << "[OnnxInference] CUDA not available, using CPU" << std::endl;
    }

    session_ = std::make_unique<Ort::Session>(*env_, onnx_path.c_str(), session_options_);

    // Get input names from ONNX model
    size_t num_inputs = session_->GetInputCount();
    input_names_str_.resize(num_inputs);
    input_names_.resize(num_inputs);
    for (size_t i = 0; i < num_inputs; i++) {
        auto name = session_->GetInputNameAllocated(i, allocator_);
        input_names_str_[i] = name.get();
        input_names_[i] = input_names_str_[i].c_str();
    }

    // Get output names from ONNX model
    size_t num_outputs = session_->GetOutputCount();
    output_names_str_.resize(num_outputs);
    output_names_.resize(num_outputs);
    for (size_t i = 0; i < num_outputs; i++) {
        auto name = session_->GetOutputNameAllocated(i, allocator_);
        output_names_str_[i] = name.get();
        output_names_[i] = output_names_str_[i].c_str();
    }

    std::cout << "[OnnxInference] Model loaded: " << onnx_path << std::endl;
    std::cout << "[OnnxInference] Inputs: " << num_inputs
              << ", Outputs: " << num_outputs
              << ", Action index: " << action_output_idx_ << std::endl;

    loaded_ = true;
}

std::vector<float> OnnxInference::run(const std::vector<float>& command,
                                       const std::vector<float>& policy_obs,
                                       const std::vector<float>& depth) {
    assert(loaded_ && "Model not loaded");

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // Create input tensors
    // input[0]: command [1, 376]
    std::vector<float> command_data(command);
    auto command_tensor = Ort::Value::CreateTensor<float>(
        memory_info, command_data.data(), command_data.size(),
        input_shapes_[0].data(), input_shapes_[0].size());

    // input[1]: policy [1, 276]
    std::vector<float> policy_data(policy_obs);
    auto policy_tensor = Ort::Value::CreateTensor<float>(
        memory_info, policy_data.data(), policy_data.size(),
        input_shapes_[1].data(), input_shapes_[1].size());

    // input[2]: depth [1, 1, 120, 160]
    std::vector<float> depth_data(depth);
    auto depth_tensor = Ort::Value::CreateTensor<float>(
        memory_info, depth_data.data(), depth_data.size(),
        input_shapes_[2].data(), input_shapes_[2].size());

    std::vector<Ort::Value> input_tensors;
    input_tensors.push_back(std::move(command_tensor));
    input_tensors.push_back(std::move(policy_tensor));
    input_tensors.push_back(std::move(depth_tensor));

    // Run inference
    auto outputs = session_->Run(
        Ort::RunOptions{nullptr},
        input_names_.data(), input_tensors.data(), input_tensors.size(),
        output_names_.data(), output_names_.size());

    // Extract action (index 10)
    float* action_data = outputs[action_output_idx_].GetTensorMutableData<float>();
    auto action_shape = outputs[action_output_idx_].GetTensorTypeAndShapeInfo().GetShape();

    // action shape should be [1, 30]
    int action_size = 1;
    for (auto dim : action_shape) {
        action_size *= static_cast<int>(dim);
    }

    return std::vector<float>(action_data, action_data + action_size);
}
