//! YOLO26-Depth RKNN — CLI entry point.

use std::time::Instant;

use clap::Parser;
use image::GenericImageView;

use yolo26depth_rknn::cli::Args;
use yolo26depth_rknn::error::Result;
use yolo26depth_rknn::postprocess::{print_stats, resize_depth, save_colormap};
use yolo26depth_rknn::pointcloud::{save_depth_binary, save_pointcloud};
use yolo26depth_rknn::preprocess::{load_image, preprocess};
use yolo26depth_rknn::rknn::DepthModel;

fn run(args: &Args) -> Result<()> {
    // Load image
    let orig_img = load_image(&args.image)?;
    let (orig_w, orig_h) = orig_img.dimensions();
    log::info!("Image: {}x{}", orig_w, orig_h);

    // Load model
    log::info!("Loading: {}", args.model);
    let model = DepthModel::load(&args.model, args.core)?;

    // Preprocess
    let rgb_data = preprocess(&args.image, model.input_w, model.input_h)?;

    // Benchmark or single inference
    if args.benchmark > 0 {
        run_benchmark(&model, &rgb_data, args.benchmark)
    } else {
        run_inference(&model, &rgb_data, &orig_img, args)
    }
}

fn run_benchmark(model: &DepthModel, rgb_data: &[u8], iterations: u32) -> Result<()> {
    // Warmup
    log::info!("Warmup (3)...");
    for _ in 0..3 {
        model.infer(rgb_data)?;
    }

    // Benchmark
    let mut times = Vec::with_capacity(iterations as usize);
    for _ in 0..iterations {
        let start = Instant::now();
        model.infer(rgb_data)?;
        times.push(start.elapsed().as_secs_f64() * 1000.0);
    }

    let avg = times.iter().sum::<f64>() / times.len() as f64;
    let min_t = times.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_t = times.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    log::info!("\nBenchmark ({} iters):", iterations);
    log::info!("  Latency: {:.1} ms ({:.2} FPS)", avg, 1000.0 / avg);
    log::info!("  Min: {:.1} ms  Max: {:.1} ms", min_t, max_t);
    Ok(())
}

fn run_inference(
    model: &DepthModel,
    rgb_data: &[u8],
    orig_img: &image::DynamicImage,
    args: &Args,
) -> Result<()> {
    let (orig_w, orig_h) = orig_img.dimensions();

    // Inference
    log::info!("Inference...");
    let start = Instant::now();
    let depth_raw = model.infer(rgb_data)?;
    let elapsed = start.elapsed();
    log::info!(
        "Inference: {:.1} ms",
        elapsed.as_secs_f64() * 1000.0
    );

    // Resize depth to original image size
    let depth = resize_depth(
        &depth_raw,
        model.output_w,
        model.output_h,
        orig_w as usize,
        orig_h as usize,
    );

    // Stats
    print_stats(&depth);

    // Save colormap
    save_colormap(&depth, orig_w as usize, orig_h as usize, &args.output, args.mode)?;

    // Optional: save raw depth
    if let Some(ref path) = args.save_depth {
        save_depth_binary(&depth, path)?;
    }

    // Optional: save point cloud
    if let Some(ref ply_path) = args.save_ply {
        save_pointcloud(
            &depth,
            orig_img,
            orig_w as usize,
            orig_h as usize,
            ply_path,
            args.pc_downsample,
        )?;
    }

    Ok(())
}

fn main() -> Result<()> {
    // Default to info level so progress messages are visible without RUST_LOG
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format_timestamp(None)
        .format_target(false)
        .init();
    let args = Args::parse();
    run(&args)
}
