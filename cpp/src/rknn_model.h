#pragma once
#include "rknn_rt.h"
#include <string>
#include <vector>

namespace depth {

/**
 * High-level wrapper for YOLO26-Depth RKNN inference on RK3588 NPU.
 *
 * Uses pre-allocated output buffer to avoid per-frame allocation + memcpy.
 *
 * Usage:
 *   auto model = std::make_unique<DepthModel>("yolo26n-depth-float.rknn", NPU_CORE_AUTO);
 *   auto input = preprocess("bus.jpg", model->input_w(), model->input_h());
 *   auto depth = model->infer(input);  // (output_h, output_w) float32 at model size
 *   // resize back to original image size as needed
 */
class DepthModel {
public:
    explicit DepthModel(const std::string& model_path, rknn_core_mask core = NPU_CORE_AUTO);
    ~DepthModel();

    // Non-copyable
    DepthModel(const DepthModel&) = delete;
    DepthModel& operator=(const DepthModel&) = delete;

    int input_w() const { return input_w_; }
    int input_h() const { return input_h_; }
    int output_w() const { return output_w_; }
    int output_h() const { return output_h_; }

    /**
     * Run inference on a preprocessed NHWC RGB8 buffer (H*W*3 bytes).
     * Returns depth map at model output resolution (output_h_ * output_w_ float32).
     * The buffer is reused across calls — data is valid until the next infer().
     */
    const std::vector<float>& infer(const std::vector<uint8_t>& input);

private:
    void query_io();

    rknn_context context_ = 0;
    int input_w_ = 0;
    int input_h_ = 0;
    int output_w_ = 0;
    int output_h_ = 0;

    // Pre-allocated output buffer — avoids rknn_outputs_get allocation + memcpy
    std::vector<float> output_buffer_;
};

} // namespace depth
