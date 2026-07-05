use anyhow::Result;
use clap::Parser;
use std::time::Instant;

/// RIFE-Lite smoke test: load weights, run one forward on a random tensor, print shapes.
#[derive(Parser)]
struct Args {
    /// Path to rife_lite.safetensors (from tools/convert_weights.py)
    #[arg(long)]
    weights: String,
    /// Tensor height (default 256)
    #[arg(long, default_value = "256")]
    h: usize,
    /// Tensor width (default 448)
    #[arg(long, default_value = "448")]
    w: usize,
    /// Use CUDA GPU if available (otherwise CPU)
    #[arg(long, default_value = "true")]
    gpu: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let dev = if args.gpu {
        candle_core::Device::cuda_if_available(0)?
    } else {
        candle_core::Device::Cpu
    };
    println!("device: {:?}", dev);
    let dtype = match &dev {
        candle_core::Device::Cuda(_) => candle_core::DType::F16,
        _ => candle_core::DType::F32,
    };
    println!("dtype: {:?}", dtype);

    let t0 = Instant::now();
    let rife = framegen::RifeCandle::load(&args.weights, dtype, &dev)?;
    println!("loaded weights in {:.2?}", t0.elapsed());

    let img0 = candle_core::Tensor::rand(0f32, 1f32, (1, 3, args.h, args.w), &dev)?;
    let img1 = candle_core::Tensor::rand(0f32, 1f32, (1, 3, args.h, args.w), &dev)?;

    let t1 = Instant::now();
    let out = rife.interpolate_scaled(&img0, &img1, 0.5, 1.0)?;
    let dt = t1.elapsed();
    println!("img0 {:?} img1 {:?}", img0.shape(), img1.shape());
    println!("out  {:?} ({:?})", out.shape(), dt);

    // second pass - is it faster after CUDA warmup?
    let t2 = Instant::now();
    let _out2 = rife.interpolate_scaled(&img0, &img1, 0.5, 1.0)?;
    println!("2nd pass: {:?}", t2.elapsed());

    let t3 = Instant::now();
    let _out3 = rife.interpolate_scaled(&img0, &img1, 0.5, 1.0)?;
    println!("3rd pass: {:?}", t3.elapsed());

    let _ = out.flatten_all()?.to_vec1::<f32>()?;
    println!("forward ok, output range sampled.");
    Ok(())
}
