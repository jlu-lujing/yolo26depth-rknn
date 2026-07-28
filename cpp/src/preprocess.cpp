#include "preprocess.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <algorithm>
#include <cmath>

namespace depth {

std::vector<uint8_t> preprocess(const std::string& img_path,
                                int target_w, int target_h) {
    cv::Mat img = cv::imread(img_path);
    if (img.empty()) {
        throw std::runtime_error("Cannot load image: " + img_path);
    }

    int src_h = img.rows;
    int src_w = img.cols;

    // Compute rect input size from image aspect ratio
    int model_max_dim = std::max(target_h, target_w);
    auto [rect_h, rect_w] = compute_rect_input(src_h, src_w, model_max_dim);

    int rw, rh;
    if (rect_h == target_h && rect_w == target_w) {
        rw = target_w;
        rh = target_h;
    } else {
        std::cerr << "[WARN] Image aspect ratio (" << src_w << ":" << src_h
                  << ") does not match model aspect ratio ("
                  << target_w << ":" << target_h << ").\n"
                  << "       Rect input would be " << rect_w << "x" << rect_h
                  << " but model expects " << target_w << "x" << target_h << ".\n"
                  << "       Results will differ from ultralytics PT predict.\n"
                  << "       Export a rect model with --imgsz " << rect_w
                  << "x" << rect_h << " for best accuracy." << std::endl;
        // Fallback: stretch to model size
        rw = target_w;
        rh = target_h;
    }

    // Resize using OpenCV INTER_LINEAR (matches Python/Rust bilinear)
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(rw, rh), 0, 0, cv::INTER_LINEAR);
    cv::Mat rgb;
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

    // Return as flat NHWC buffer (H*W*3)
    std::vector<uint8_t> buf(rgb.data, rgb.data + rgb.total() * 3);
    return buf;
}

std::pair<int, int> load_image_size(const std::string& img_path) {
    cv::Mat img = cv::imread(img_path);
    if (img.empty()) {
        throw std::runtime_error("Cannot load image: " + img_path);
    }
    return {img.cols, img.rows}; // w, h
}

} // namespace depth
