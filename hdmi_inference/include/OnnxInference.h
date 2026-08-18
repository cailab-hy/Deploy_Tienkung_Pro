#ifndef ONNX_INFERENCE_H_
#define ONNX_INFERENCE_H_

#include <string>
#include <vector>
#include <memory>
#include <onnxruntime_cxx_api.h>

class OnnxInference {
public:
    OnnxInference() = default;
    ~OnnxInference() = default;

    /**
     * @brief Load ONNX model and JSON metadata
     * @param onnx_path Path to .onnx file
     * @param json_path Path to .json metadata file (in_keys, out_keys, in_shapes)
     */
    void loadModel(const std::string& onnx_path, const std::string& json_path);

    /**
     * @brief Run inference with 3 inputs
     * @param command Command observation [376]
     * @param policy_obs Policy observation [276]
     * @param depth Depth image [1*120*160 = 19200]
     * @return Action vector [30]
     */
    std::vector<float> run(const std::vector<float>& command,
                           const std::vector<float>& policy_obs,
                           const std::vector<float>& depth);

    bool isLoaded() const { return loaded_; }

private:
    bool loaded_ = false;

    // ONNX Runtime
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;
    Ort::SessionOptions session_options_;
    Ort::AllocatorWithDefaultOptions allocator_;

    // Input/output names from ONNX model
    std::vector<std::string> input_names_str_;
    std::vector<const char*> input_names_;
    std::vector<std::string> output_names_str_;
    std::vector<const char*> output_names_;

    // Input shapes
    std::vector<std::vector<int64_t>> input_shapes_;

    // Action output index (index 10 in out_keys)
    int action_output_idx_ = 10;
};

#endif // ONNX_INFERENCE_H_
