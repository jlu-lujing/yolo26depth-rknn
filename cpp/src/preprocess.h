#pragma once
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>

namespace depth {

/**
 * Compute rect input size matching ultralytics PT predict behavior.
 * The image is scaled so the longer side equals model_max_dim while
 * keeping the aspect ratio, then each dimension is rounded to stride-32.
 */
inline std::pair<int, int> compute_rect_input(int src_h, int src_w, int model_max_dim) {
    double scale = (double)model_max_dim / std::max(src_h, src_w);
    int new_h = std::max((int)(std::round((src_h * scale) / 32.0) * 32), 32);
    int new_w = std::max((int)(std::round((src_w * scale) / 32.0) * 32), 32);
    return {new_h, new_w};
}

/**
 * Load an image from file and preprocess it for RKNN inference.
 *
 * Uses rect preprocessing: computes the expected input size from the image
 * aspect ratio; if it matches (target_w, target_h) the resize is direct.
 * Otherwise a warning is printed and the image is stretched.
 *
 * Returns a flat RGB8 buffer in NHWC layout (width*height*3 bytes).
 */
std::vector<uint8_t> preprocess(const std::string& img_path,
                                int target_w, int target_h);

/**
 * Load an image and return its raw dimensions without preprocessing.
 */
std::pair<int, int> load_image_size(const std::string& img_path);

} // namespace depth
