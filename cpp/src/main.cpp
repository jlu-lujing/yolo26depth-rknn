/**
 * YOLO26-Depth RKNN inference — C++ implementation for RK3588 NPU.
 *
 * Usage:
 *   ./yolo26depth-cpp --model yolo26n-depth-float.rknn --image bus.jpg \
 *       --save result.png
 *
 *   # Save heatmap + raw depth
 *   ./yolo26depth-cpp --model yolo26n-depth_768x576-float.rknn --image bus.jpg \
 *       --save-heat heatmap.png --save-depth depth.raw
 *
 *   # Benchmark
 *   ./yolo26depth-cpp --model yolo26n-depth-float.rknn --image bus.jpg \
 *       --benchmark 100
 *
 * Build on RK3588:
 *   cd src/cpp && cmake -B build && cmake --build build -j
 */

#include "rknn_model.h"
#include "preprocess.h"
#include "postprocess.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <chrono>
#include <cstring>
#include <algorithm>
#include <stdexcept>

// Minimal argument parser
struct Args {
    std::string model;
    std::string image;
    std::string save;
    std::string save_heat;
    std::string save_depth;
    std::string mode = "disparity";
    int benchmark = 0;
    int warmup = 3;
    rknn_core_mask core = NPU_CORE_AUTO;

    void parse(int argc, char** argv) {
        auto get_arg = [&](const char* flag) -> const char* {
            for (int i = 1; i < argc - 1; ++i) {
                if (std::strcmp(argv[i], flag) == 0) return argv[i + 1];
            }
            return nullptr;
        };

        auto has_flag = [&](const char* flag) -> bool {
            for (int i = 1; i < argc; ++i) {
                if (std::strcmp(argv[i], flag) == 0) return true;
            }
            return false;
        };

        const char* v = nullptr;

        if ((v = get_arg("--model")) != nullptr)     model       = v;
        else if (!has_flag("--help")) usage();

        if ((v = get_arg("--image")) != nullptr)      image       = v;
        else usage("missing --image");

        if ((v = get_arg("--save")) != nullptr)       save        = v;
        if ((v = get_arg("--save-heat")) != nullptr)  save_heat   = v;
        if ((v = get_arg("--save-depth")) != nullptr) save_depth  = v;
        if ((v = get_arg("--mode")) != nullptr) {
            mode = std::string(v);
            if (mode != "disparity" && mode != "metric")
                usage("--mode must be 'disparity' or 'metric'");
        }
        if ((v = get_arg("--benchmark")) != nullptr)  benchmark   = std::atoi(v);
        if ((v = get_arg("--warmup")) != nullptr)     warmup      = std::atoi(v);

        if ((v = get_arg("--core")) != nullptr) {
            std::string c = v;
            if (c == "auto")   core = NPU_CORE_AUTO;
            else if (c == "0") core = NPU_CORE_0;
            else if (c == "1") core = NPU_CORE_1;
            else if (c == "2") core = NPU_CORE_2;
            else if (c == "012") core = NPU_CORE_0_1_2;
            else usage("invalid --core");
        }

        if (has_flag("--help")) {
            usage();
            exit(0);
        }
    }

private:
    void usage(const char* msg = nullptr) {
        if (msg) std::cerr << "[ERROR] " << msg << "\n\n";
        std::cerr << R"(Usage: yolo26depth-cpp [OPTIONS] --model MODEL --image IMAGE

Options:
  --model MODEL      Path to .rknn model file (required)
  --image IMAGE      Input image file (required)
  --save PATH        Save overlay (original image + depth heatmap blended)
  --save-heat PATH   Save standalone depth heatmap
  --save-depth PATH  Save raw depth map as binary float32 (H*W bytes)
  --mode MODE        Colorize mode: 'disparity' (default) or 'metric'
  --benchmark N      Run benchmark for N iterations
  --warmup N         Number of warmup runs (default: 3)
  --core CORE        NPU core selection: auto, 0, 1, 2, 012 (default: auto)
  --help             Show this help message)" << std::endl;
        exit(1);
    }
};

int main(int argc, char** argv) {
    Args args;
    args.parse(argc, argv);

    try {
        // Load model
        auto model = std::make_unique<depth::DepthModel>(args.model, args.core);
        int iw = model->input_w();
        int ih = model->input_h();

        // Get original image size
        auto [src_w, src_h] = depth::load_image_size(args.image);
        std::cout << "Input image: " << src_w << "x" << src_h << "\n";

        // Preprocess ONCE (for benchmark, reuse the buffer)
        std::vector<uint8_t> input_data = depth::preprocess(args.image, iw, ih);

        // Warmup
        for (int i = 0; i < args.warmup; ++i) {
            model->infer(input_data, src_w, src_h);
        }

        if (args.benchmark > 0) {
            // Benchmark mode
            std::vector<double> latencies;
            for (int i = 0; i < args.benchmark; ++i) {
                auto t0 = std::chrono::high_resolution_clock::now();
                model->infer(input_data, src_w, src_h);
                auto t1 = std::chrono::high_resolution_clock::now();
                double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
                latencies.push_back(ms);
            }

            std::sort(latencies.begin(), latencies.end());
            size_t n = latencies.size();
            double avg = 0.0;
            for (double l : latencies) avg += l;
            avg /= n;

            std::cout << "\nBenchmark (" << args.benchmark << " iterations):\n";
            std::cout << "  Latency: " << avg << " ms (" << (1000.0 / avg) << " FPS)\n";
            std::cout << "  Min: " << latencies.front() << " ms  Max: " << latencies.back() << " ms\n";
        } else {
            // Single inference
            auto t0 = std::chrono::high_resolution_clock::now();
            std::vector<float> depth_map = model->infer(input_data, src_w, src_h);
            auto t1 = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

            std::cout << "\nInference: " << ms << " ms\n";
            depth::print_depth_stats(depth_map);

            // Save outputs
            if (!args.save.empty() || !args.save_heat.empty()) {
                // Load original BGR image for overlay
                cv::Mat img = cv::imread(args.image);
                std::vector<uint8_t> src_bgr(img.data, img.data + img.total() * 3);

                std::vector<uint8_t> heat = depth::colorize_depth(depth_map, src_w, src_h, args.mode.c_str());

                if (!args.save_heat.empty()) {
                    depth::save_image(args.save_heat, heat, src_w, src_h);
                    std::cout << "  Heatmap saved: " << args.save_heat << "\n";
                }

                if (!args.save.empty()) {
                    std::vector<uint8_t> overlay = depth::blend_overlay(src_bgr, heat, src_w, src_h);
                    depth::save_image(args.save, overlay, src_w, src_h);
                    std::cout << "  Overlay saved: " << args.save << "\n";
                }
            }

            if (!args.save_depth.empty()) {
                // Save as raw binary float32 (H*W bytes)
                std::ofstream out(args.save_depth, std::ios::binary);
                if (!out.is_open()) {
                    throw std::runtime_error("Failed to open " + args.save_depth);
                }
                out.write(reinterpret_cast<const char*>(depth_map.data()),
                          depth_map.size() * sizeof(float));
                out.close();
                std::cout << "  Depth saved: " << args.save_depth << "\n";
            }
        }

    } catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << "\n";
        return 1;
    }

    return 0;
}
