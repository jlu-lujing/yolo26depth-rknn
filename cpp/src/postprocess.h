#pragma once
#include <vector>
#include <string>
#include <cmath>

namespace depth {

/**
 * Convert a raw (H, W) float32 depth map to an 8-bit JET colormap image.
 *
 * Mode "disparity" uses 1/d with percentile clipping (matches Python/Rust).
 * Mode "metric" uses linear mapping from min/max depth.
 */
std::vector<uint8_t> colorize_depth(const std::vector<float>& depth,
                                    int w, int h,
                                    const char* mode = "disparity");

/**
 * Blend a BGR source image with a BGR heatmap into an overlay.
 *
 * result = (1-alpha)*src + alpha*heatmap  (default alpha=0.5)
 */
std::vector<uint8_t> blend_overlay(const std::vector<uint8_t>& src_bgr,
                                   const std::vector<uint8_t>& heat_bgr,
                                   int w, int h,
                                   double alpha = 0.5);

/**
 * Save a BGR image to file (PNG/JPG).
 */
void save_image(const std::string& path, const std::vector<uint8_t>& bgr, int w, int h);

/**
 * Print depth statistics to stdout.
 */
void print_depth_stats(const std::vector<float>& depth);

} // namespace depth
