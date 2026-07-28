#include "rknn_model.h"
#include <fstream>
#include <iostream>
#include <cstring>
#include <stdexcept>
#include <algorithm>
#include <opencv2/opencv.hpp>

namespace depth {

static void check_rknn(int ret, const std::string& msg) {
    if (ret != RKNN_SUCC) {
        throw std::runtime_error("RKNN error (" + msg + "): code=" + std::to_string(ret));
    }
}

DepthModel::DepthModel(const std::string& model_path, rknn_core_mask core) {
    // Init RKNN context with file path (size=0 means load from disk)
    check_rknn(rknn_init(&context_, model_path.c_str(), 0, 0, nullptr), "rknn_init");

    // Set NPU core mask if not auto
    if (core != NPU_CORE_AUTO) {
        int mask = static_cast<int>(core);
        check_rknn(rknn_set_core_mask(context_, static_cast<uint32_t>(mask)), "rknn_set_core_mask");
    }

    query_io();

    std::cout << "RKNN model loaded (imgsz=(" << input_h_ << "," << input_w_
              << "), core=" << static_cast<int>(core) << ")" << std::endl;
}

DepthModel::~DepthModel() {
    if (context_) {
        rknn_destroy(context_);
    }
}

void DepthModel::query_io() {
    // Query I/O count
    rknn_input_output_num io_num = {};
    check_rknn(rknn_query(context_, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num)),
               "query in/out num");

    if (io_num.n_input < 1 || io_num.n_output < 1) {
        throw std::runtime_error("Invalid model: expected at least 1 input and 1 output");
    }

    // Query input tensor attribute (index=0, NHWC layout)
    rknn_tensor_attr in_attr = {};
    in_attr.index = 0;
    check_rknn(rknn_query(context_, RKNN_QUERY_INPUT_ATTR, &in_attr, sizeof(in_attr)),
               "query input attr");

    // Input is NHWC: [N, H, W, C] -> dims[1]=H, dims[2]=W
    input_h_ = static_cast<int>(in_attr.dims[1]);
    input_w_ = static_cast<int>(in_attr.dims[2]);

    // Query output tensor attribute (index=0, NCHW layout)
    rknn_tensor_attr out_attr = {};
    out_attr.index = 0;
    check_rknn(rknn_query(context_, RKNN_QUERY_OUTPUT_ATTR, &out_attr, sizeof(out_attr)),
               "query output attr");

    // Output is NCHW: [N, C, H, W] -> dims[2]=H, dims[3]=W
    output_h_ = static_cast<int>(out_attr.dims[2]);
    output_w_ = static_cast<int>(out_attr.dims[3]);
}

std::vector<float> DepthModel::infer(const std::vector<uint8_t>& input,
                                     int src_w, int src_h) {
    if (static_cast<int>(input.size()) != input_h_ * input_w_ * 3) {
        throw std::runtime_error("Input size mismatch: expected "
                                 + std::to_string(input_h_) + "x"
                                 + std::to_string(input_w_) + "*3="
                                 + std::to_string(input_h_ * input_w_ * 3)
                                 + " bytes, got " + std::to_string(input.size()));
    }

    // Set input (NHWC RGB8 buffer)
    rknn_input inputs[1];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].buf = const_cast<uint8_t*>(input.data());
    inputs[0].size = static_cast<uint32_t>(input.size());
    inputs[0].pass_through = 0;
    inputs[0].type_ = RKNN_TENSOR_UINT8;
    inputs[0].fmt = RKNN_TENSOR_NHWC;

    check_rknn(rknn_inputs_set(context_, 1, inputs), "inputs_set");

    // Run inference
    check_rknn(rknn_run(context_, nullptr), "run");

    // Get output (float32 depth map)
    rknn_output outputs[1];
    memset(outputs, 0, sizeof(outputs));
    outputs[0].want_float = 1;     // convert to float if needed
    outputs[0].is_prealloc = 0;    // runtime allocates buffer
    outputs[0].index = 0;

    check_rknn(rknn_outputs_get(context_, 1, outputs, nullptr), "outputs_get");

    uint32_t out_bytes = outputs[0].size;
    size_t n_pixels = static_cast<size_t>(output_h_) * output_w_;

    if (out_bytes < n_pixels * sizeof(float)) {
        rknn_outputs_release(context_, 1, outputs);
        throw std::runtime_error("Unexpected output size: expected " + std::to_string(n_pixels * sizeof(float))
                                 + ", got " + std::to_string(out_bytes));
    }

    // Copy output to vector (NCHW -> flat H*W)
    std::vector<float> depth_raw(n_pixels);
    std::memcpy(depth_raw.data(), outputs[0].buf, n_pixels * sizeof(float));

    rknn_outputs_release(context_, 1, outputs);

    // Resize to original image size using bilinear interpolation
    if (output_h_ != src_h || output_w_ != src_w) {
        cv::Mat raw_map(output_h_, output_w_, CV_32F, depth_raw.data());
        cv::Mat resized;
        cv::resize(raw_map, resized, cv::Size(src_w, src_h), 0, 0, cv::INTER_LINEAR);
        depth_raw.assign(resized.begin<float>(), resized.end<float>());
    }

    return depth_raw; // (src_h * src_w) float32 in row-major order
}

} // namespace depth
