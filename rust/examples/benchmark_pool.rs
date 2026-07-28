//! Multi-core NPU throughput benchmark: worker threads + frame queue.
//!
//! Measures pure inference throughput:
//! - The input image is decoded + preprocessed ONCE; each "frame" is a
//!   clone of the ready-to-run uint8 NHWC buffer (no repeated decode/resize).
//! - Workers pull frames from a shared queue and only call RKNN inference.
//! - Default 6 workers = 2 RKNN instances per NPU core (RK3588 has 3 cores),
//!   so the NPU never idles while a worker is busy on the CPU side.
//!
//! Usage (on RK3588):
//!     cargo build --release --example benchmark_pool
//!     ./target/release/examples/benchmark_pool \
//!         --model yolo26n-depth-float.rknn --image bus.jpg --frames 180
//!
//!     # Single worker / single core baseline
//!     ./target/release/examples/benchmark_pool \
//!         --model yolo26n-depth-float.rknn --image bus.jpg \
//!         --frames 30 --workers 1

use std::sync::mpsc;
use std::sync::{Arc, Barrier, Mutex};
use std::thread;
use std::time::Instant;

use clap::Parser;

use yolo26depth_rknn::preprocess::preprocess;
use yolo26depth_rknn::rknn::{DepthModel, NpuCore};

#[derive(Parser, Debug)]
#[command(name = "benchmark_pool", about = "Queue-based multi-core NPU benchmark")]
struct Args {
    /// Path to the .rknn model file
    #[arg(short, long)]
    model: String,

    /// Path to the input image (decoded once)
    #[arg(short, long)]
    image: String,

    /// Total frames to process
    #[arg(long, default_value = "180")]
    frames: usize,

    /// Worker threads / RKNN instances (default: 6 = 2 per core)
    #[arg(long, default_value = "6")]
    workers: usize,

    /// Warmup runs per worker
    #[arg(long, default_value = "3")]
    warmup: usize,
}

const CORES: [NpuCore; 3] = [NpuCore::Core0, NpuCore::Core1, NpuCore::Core2];

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn"))
        .format_timestamp(None)
        .format_target(false)
        .init();
    let args = Args::parse();

    // Load model once on the main thread just to get input dims
    let probe = DepthModel::load(&args.model, NpuCore::Auto).expect("load model");
    let (input_w, input_h) = (probe.input_w, probe.input_h);
    drop(probe);

    // Decode + preprocess ONCE; frames are clones of the ready buffer
    let rgb_data = preprocess(&args.image, input_w, input_h).expect("preprocess");

    let (tx, rx) = mpsc::channel::<Arc<Vec<u8>>>();
    let rx = Arc::new(Mutex::new(rx));
    // +1 for the main thread: workers park here after load + warmup
    let barrier = Arc::new(Barrier::new(args.workers + 1));

    let mut handles = Vec::new();
    for i in 0..args.workers {
        let core = CORES[i % CORES.len()];
        let model_path = args.model.clone();
        let warmup_data = rgb_data.clone();
        let warmup = args.warmup;
        let rx = Arc::clone(&rx);
        let barrier = Arc::clone(&barrier);

        handles.push(thread::spawn(move || {
            let model = DepthModel::load(&model_path, core).expect("load model");
            for _ in 0..warmup {
                model.infer(&warmup_data).expect("warmup");
            }
            barrier.wait();

            let mut latencies = Vec::new();
            loop {
                // Lock only to receive; inference runs unlocked
                let frame = match rx.lock().unwrap().recv() {
                    Ok(f) => f,
                    Err(_) => break, // channel closed: no more frames
                };
                let t0 = Instant::now();
                model.infer(&frame).expect("infer");
                latencies.push(t0.elapsed().as_secs_f64() * 1000.0);
            }
            (core, latencies)
        }));
    }

    barrier.wait(); // all workers loaded + warmed up
    let t0 = Instant::now();
    for _ in 0..args.frames {
        tx.send(Arc::new(rgb_data.clone())).unwrap();
    }
    drop(tx); // close the queue: workers exit when drained

    let results: Vec<(NpuCore, Vec<f64>)> =
        handles.into_iter().map(|h| h.join().unwrap()).collect();
    let wall = t0.elapsed().as_secs_f64();

    let mut latencies: Vec<f64> = results.iter().flat_map(|(_, l)| l.iter().copied()).collect();
    latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let per_worker: Vec<(NpuCore, usize)> =
        results.iter().map(|(c, l)| (*c, l.len())).collect();

    println!("\nQueue benchmark: {} frames, {} workers", args.frames, args.workers);
    println!("  Wall time:  {:.2} s", wall);
    println!("  Throughput: {:.2} FPS (aggregate, pure inference)", args.frames as f64 / wall);
    println!(
        "  Latency:    median {:.1} ms, p95 {:.1} ms",
        latencies[latencies.len() / 2],
        latencies[(latencies.len() as f64 * 0.95) as usize]
    );
    println!("  Frames per worker (core, count): {:?}", per_worker);
}
