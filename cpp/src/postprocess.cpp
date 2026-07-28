#include "postprocess.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <cstring>

namespace depth {

// ============================================================================
// Exact OpenCV COLORMAP_JET LUT (BGR) — extracted from cv2.applyColorMap
// ============================================================================
static const uint8_t JET_LUT[256][3] = {
    {128,0,0},{132,0,0},{136,0,0},{140,0,0},{144,0,0},{148,0,0},{152,0,0},{156,0,0},
    {160,0,0},{164,0,0},{168,0,0},{172,0,0},{176,0,0},{180,0,0},{184,0,0},{188,0,0},
    {192,0,0},{196,0,0},{200,0,0},{204,0,0},{208,0,0},{212,0,0},{216,0,0},{220,0,0},
    {224,0,0},{228,0,0},{232,0,0},{236,0,0},{240,0,0},{244,0,0},{248,0,0},{252,0,0},
    {255,0,0},{255,4,0},{255,8,0},{255,12,0},{255,16,0},{255,20,0},{255,24,0},{255,28,0},
    {255,32,0},{255,36,0},{255,40,0},{255,44,0},{255,48,0},{255,52,0},{255,56,0},{255,60,0},
    {255,64,0},{255,68,0},{255,72,0},{255,76,0},{255,80,0},{255,84,0},{255,88,0},{255,92,0},
    {255,96,0},{255,100,0},{255,104,0},{255,108,0},{255,112,0},{255,116,0},{255,120,0},{255,124,0},
    {255,128,0},{255,132,0},{255,136,0},{255,140,0},{255,144,0},{255,148,0},{255,152,0},{255,156,0},
    {255,160,0},{255,164,0},{255,168,0},{255,172,0},{255,176,0},{255,180,0},{255,184,0},{255,188,0},
    {255,192,0},{255,196,0},{255,200,0},{255,204,0},{255,208,0},{255,212,0},{255,216,0},{255,220,0},
    {255,224,0},{255,228,0},{255,232,0},{255,236,0},{255,240,0},{255,244,0},{255,248,0},{255,252,0},
    {254,255,2},{250,255,6},{246,255,10},{242,255,14},{238,255,18},{234,255,22},{230,255,26},{226,255,30},
    {222,255,34},{218,255,38},{214,255,42},{210,255,46},{206,255,50},{202,255,54},{198,255,58},{194,255,62},
    {190,255,66},{186,255,70},{182,255,74},{178,255,78},{174,255,82},{170,255,86},{166,255,90},{162,255,94},
    {158,255,98},{154,255,102},{150,255,106},{146,255,110},{142,255,114},{138,255,118},{134,255,122},{130,255,126},
    {126,255,130},{122,255,134},{118,255,138},{114,255,142},{110,255,146},{106,255,150},{102,255,154},{98,255,158},
    {94,255,162},{90,255,166},{86,255,170},{82,255,174},{78,255,178},{74,255,182},{70,255,186},{66,255,190},
    {62,255,194},{58,255,198},{54,255,202},{50,255,206},{46,255,210},{42,255,214},{38,255,218},{34,255,222},
    {30,255,226},{26,255,230},{22,255,234},{18,255,238},{14,255,242},{10,255,246},{6,255,250},{1,255,254},
    {0,252,255},{0,248,255},{0,244,255},{0,240,255},{0,236,255},{0,232,255},{0,228,255},{0,224,255},
    {0,220,255},{0,216,255},{0,212,255},{0,208,255},{0,204,255},{0,200,255},{0,196,255},{0,192,255},
    {0,188,255},{0,184,255},{0,180,255},{0,176,255},{0,172,255},{0,168,255},{0,164,255},{0,160,255},
    {0,156,255},{0,152,255},{0,148,255},{0,144,255},{0,140,255},{0,136,255},{0,132,255},{0,128,255},
    {0,124,255},{0,120,255},{0,116,255},{0,112,255},{0,108,255},{0,104,255},{0,100,255},{0,96,255},
    {0,92,255},{0,88,255},{0,84,255},{0,80,255},{0,76,255},{0,72,255},{0,68,255},{0,64,255},
    {0,60,255},{0,56,255},{0,52,255},{0,48,255},{0,44,255},{0,40,255},{0,36,255},{0,32,255},
    {0,28,255},{0,24,255},{0,20,255},{0,16,255},{0,12,255},{0,8,255},{0,4,255},{0,0,255},
    {0,0,252},{0,0,248},{0,0,244},{0,0,240},{0,0,236},{0,0,232},{0,0,228},{0,0,224},
    {0,0,220},{0,0,216},{0,0,212},{0,0,208},{0,0,204},{0,0,200},{0,0,196},{0,0,192},
    {0,0,188},{0,0,184},{0,0,180},{0,0,176},{0,0,172},{0,0,168},{0,0,164},{0,0,160},
    {0,0,156},{0,0,152},{0,0,148},{0,0,144},{0,0,140},{0,0,136},{0,0,132},{0,0,128},
};

/// Linear-interpolation percentile (matches numpy np.percentile).
static float percentile_linear(const std::vector<float>& sorted, float p) {
    if (sorted.empty()) return 0.0f;
    float idx = p / 100.0f * (sorted.size() - 1);
    size_t lo = static_cast<size_t>(idx);
    size_t hi = std::min(lo + 1, sorted.size() - 1);
    float frac = idx - static_cast<float>(lo);
    return sorted[lo] * (1.0f - frac) + sorted[hi] * frac;
}

std::vector<uint8_t> colorize_depth(const std::vector<float>& depth,
                                    int w, int h,
                                    const char* mode) {
    // ==========================================================================
    // Step 1: compute normalized value v ∈ [0, 1] for each valid pixel
    // ==========================================================================
    std::vector<float> v(depth.size(), 0.0f);
    float lo = 0.0f, hi = 1.0f;

    if (std::string(mode) == "disparity") {
        // Disparity: 1/d with 2%-98% percentile clipping
        std::vector<float> valid_disp;
        for (size_t i = 0; i < depth.size(); ++i) {
            if (depth[i] > 0.0f && std::isfinite(depth[i])) {
                float disp = 1.0f / depth[i];
                v[i] = disp;
                valid_disp.push_back(disp);
            }
        }

        if (valid_disp.empty()) {
            return std::vector<uint8_t>(w * h * 3, 0);
        }

        std::sort(valid_disp.begin(), valid_disp.end());
        lo = percentile_linear(valid_disp, 2.0f);
        hi = percentile_linear(valid_disp, 98.0f);
    } else {
        // Metric: linear over valid depth range
        std::vector<float> valid_depth;
        for (float d : depth) {
            if (d > 0.0f && std::isfinite(d)) {
                valid_depth.push_back(d);
            }
        }
        if (valid_depth.empty()) {
            return std::vector<uint8_t>(w * h * 3, 0);
        }
        auto [mn, mx] = std::minmax_element(valid_depth.begin(), valid_depth.end());
        lo = *mn;
        hi = *mx;
        // v stays 0 for invalid pixels; copy depth values for valid ones
        for (size_t i = 0; i < depth.size(); ++i) {
            if (depth[i] > 0.0f && std::isfinite(depth[i])) {
                v[i] = depth[i];
            }
        }
    }

    float range = (hi - lo > 1e-8f) ? (hi - lo) : 1e-8f;

    // ==========================================================================
    // Step 2: apply JET LUT
    // ==========================================================================
    std::vector<uint8_t> result(w * h * 3, 0);
    for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j) {
            size_t idx = static_cast<size_t>(i) * w + j;
            if (depth[idx] > 0.0f && std::isfinite(depth[idx])) {
                float t = (v[idx] - lo) / range;
                t = std::max(0.0f, std::min(1.0f, t));
                int lut_idx = static_cast<int>(t * 255.0f);  // truncation, matches Python astype(np.uint8)
                lut_idx = std::min(255, std::max(0, lut_idx));
                size_t pixel = idx * 3;
                result[pixel + 0] = JET_LUT[lut_idx][0]; // B
                result[pixel + 1] = JET_LUT[lut_idx][1]; // G
                result[pixel + 2] = JET_LUT[lut_idx][2]; // R
            }
            // else: already 0
        }
    }

    return result;
}

std::vector<uint8_t> blend_overlay(const std::vector<uint8_t>& src_bgr,
                                   const std::vector<uint8_t>& heat_bgr,
                                   int w, int h,
                                   double alpha) {
    cv::Mat src(h, w, CV_8UC3, const_cast<uint8_t*>(src_bgr.data()));
    cv::Mat heat(h, w, CV_8UC3, const_cast<uint8_t*>(heat_bgr.data()));
    cv::Mat overlay;
    cv::addWeighted(src, 1.0 - alpha, heat, alpha, 0.0, overlay);
    return std::vector<uint8_t>(overlay.data, overlay.data + overlay.total() * 3);
}

void save_image(const std::string& path, const std::vector<uint8_t>& bgr, int w, int h) {
    cv::Mat img(h, w, CV_8UC3, const_cast<uint8_t*>(bgr.data()));
    if (!cv::imwrite(path, img)) {
        throw std::runtime_error("Failed to save image: " + path);
    }
}

void print_depth_stats(const std::vector<float>& depth) {
    std::vector<float> valid;
    valid.reserve(depth.size());
    for (float d : depth) {
        if (d > 0 && std::isfinite(d)) {
            valid.push_back(d);
        }
    }

    if (valid.empty()) {
        std::cout << "  No valid depth values!" << std::endl;
        return;
    }

    std::sort(valid.begin(), valid.end());
    size_t n = valid.size();

    double median = valid[n / 2];
    double p5   = valid[static_cast<size_t>(n * 0.05)];
    double p95  = valid[static_cast<size_t>(n * 0.95)];

    std::cout << "  Median: " << median << " m" << std::endl;
    std::cout << "  5%-95%: " << p5 << " ~ " << p95 << " m" << std::endl;
}

} // namespace depth
