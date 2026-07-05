use anyhow::Result;
use clap::Parser;
use std::time::Instant;

/// Profile RIFE-Lite forward pass by component.
#[derive(Parser)]
struct Args {
    #[arg(long)]
    weights: String,
    #[arg(long, default_value = "1080")]
    h: usize,
    #[arg(long, default_value = "1920")]
    w: usize,
    #[arg(long, default_value = "20")]
    iters: usize,
    #[arg(long, default_value = "1.0")]
    scale: f64,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let dev = candle_core::Device::cuda_if_available(0)?;
    println!("device: {:?}", dev);
    let dtype = match &dev {
        candle_core::Device::Cuda(_) => candle_core::DType::F16,
        _ => candle_core::DType::F32,
    };
    println!("dtype: {:?}", dtype);

    let rife = framegen::RifeCandle::load(&args.weights, dtype, &dev)?;

    let img0 = candle_core::Tensor::rand(0f32, 1f32, (1, 3, args.h, args.w), &dev)?;
    let img1 = candle_core::Tensor::rand(0f32, 1f32, (1, 3, args.h, args.w), &dev)?;

    // warmup
    let _ = rife.interpolate_scaled(&img0, &img1, 0.5, 1.0)?;
    let _ = rife.interpolate_scaled(&img0, &img1, 0.5, 1.0)?;

    // measure full interpolate (includes padding + forward)
    let mut times = Vec::new();
    for _ in 0..args.iters {
        let t = Instant::now();
        let _ = rife.interpolate_scaled(&img0, &img1, 0.5, args.scale)?;
        dev.synchronize()?;
        times.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p50 = times[times.len() / 2];
    let p10 = times[times.len() / 10];
    let p90 = times[times.len() * 9 / 10];
    let mean = times.iter().sum::<f64>() / times.len() as f64;
    println!("\n=== {}x{} ({} iters, warm) ===", args.h, args.w, args.iters);
    println!("full interpolate: p50={p50:.1}ms  p10={p10:.1}ms  p90={p90:.1}ms  mean={mean:.1}ms");

    // measure CPU->GPU upload (simulate ffmpeg decode -> tensor -> GPU)
    let raw: Vec<u8> = vec![128u8; args.h * args.w * 3];
    let mut upload_times = Vec::new();
    for _ in 0..args.iters {
        let t = Instant::now();
        let _t = framegen::imgutil::raw_rgb24_to_tensor(&raw, args.w, args.h, &dev)?;
        dev.synchronize()?;
        upload_times.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    upload_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let up_p50 = upload_times[upload_times.len() / 2];
    println!("cpu->gpu upload:   p50={up_p50:.1}ms");

    // measure GPU->CPU download (tensor -> bytes for ffmpeg encode)
    let out = rife.interpolate_scaled(&img0, &img1, 0.5, 1.0)?;
    let mut dl_times = Vec::new();
    for _ in 0..args.iters {
        let t = Instant::now();
        let _bytes = framegen::imgutil::tensor_to_rgb24(&out)?;
        dev.synchronize()?;
        dl_times.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    dl_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let dl_p50 = dl_times[dl_times.len() / 2];
    println!("gpu->cpu download: p50={dl_p50:.1}ms");

    let overhead = up_p50 + dl_p50;
    println!("\ntotal pipeline budget: {:.1}ms (inference {p50:.1} + transfer {overhead:.1})", p50 + overhead);
    println!("real-time 60fps needs: 16.6ms total");
    println!("real-time 48fps needs: 20.8ms total (2x @ 24fps)");

    Ok(())
}
