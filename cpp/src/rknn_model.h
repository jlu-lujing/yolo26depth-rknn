#pragma once
#include "rknn_rt.h"
#include <string>
#include <vector>

namespace depth {

/**
 * High-level wrapper for YOLO26-Depth RKNN inference on RK3588 NPU.
 *
 * Usage:
 *   auto model = std::make_unique<DepthModel>("yolo26n-depth-float.rknn", NPU_CORE_AUTO);
 *   auto input = preprocess("bus.jpg", model->input_w(), model->input_h());
 *   auto depth = model->infer(input, src_w, src_h);  // (H,W) float32 at original image size
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
     * Returns depth map resized back to the original image size.
     */
    std::vector<float> infer(const std::vector<uint8_t>& input, int src_w, int src_h);

private:
    void query_io();

    rknn_context context_ = 0;
    int input_w_ = 0;
    int input_h_ = 0;
    int output_w_ = 0;
    int output_h_ = 0;
};

} // namespace depth
