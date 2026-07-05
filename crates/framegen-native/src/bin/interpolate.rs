use anyhow::{anyhow, Result};
use clap::{Parser, Subcommand};
use image::GenericImageView;
use std::path::PathBuf;
use std::time::Instant;

/// RIFE-Lite offline interpolation (image pair or mp4).
#[derive(Parser)]
#[command(name = "rife-interpolate", version)]
struct Args {
    /// Path to rife_lite.safetensors (from tools/convert_weights.py)
    #[arg(long)]
    weights: PathBuf,
    /// Use CUDA GPU if available (requires --features cuda)
    #[arg(long, default_value = "false")]
    cpu: bool,

    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Interpolate between two images -> one PNG
    Img {
        #[arg(long)]
        img0: PathBuf,
        #[arg(long)]
        img1: PathBuf,
        #[arg(long, default_value = "out.png")]
        out: PathBuf,
        /// 0.0 = img0, 1.0 = img1
        #[arg(long, default_value = "0.5")]
        timestep: f64,
        #[arg(long, default_value = "1.0")]
        scale: f64,
    },
    /// Interpolate an mp4 video -> N× fps mp4 (candle backend)
    Video {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long, default_value = "2")]
        times: u32,
        #[arg(long, default_value = "1.0")]
        scale: f64,
    },
}

fn main() -> Result<()> {
    let args = Args::parse();
    let dev = if args.cpu {
        candle_core::Device::Cpu
    } else {
        #[cfg(feature = "cuda")]
        {
            match candle_core::Device::cuda_if_available(0) {
                Ok(d) => d,
                Err(e) => {
                    eprintln!("CUDA unavailable ({e}), falling back to CPU");
                    candle_core::Device::Cpu
                }
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            candle_core::Device::Cpu
        }
    };
    println!("device: {:?}", dev);
    let dtype = match &dev {
        candle_core::Device::Cuda(_) => candle_core::DType::F16,
        _ => candle_core::DType::F32,
    };
    let rife = framegen::RifeCandle::load(&args.weights, dtype, &dev)?;

    match args.cmd {
        Cmd::Img { img0, img1, out, timestep, scale } => {
            let i0 = image::open(&img0).map_err(|e| anyhow!("open {}: {e}", img0.display()))?;
            let i1 = image::open(&img1).map_err(|e| anyhow!("open {}: {e}", img1.display()))?;
            if i0.dimensions() != i1.dimensions() {
                return Err(anyhow!("img0 and img1 must have identical dimensions"));
            }
            let t0 = framegen::imgutil::image_to_tensor(&i0, &dev)?;
            let t1 = framegen::imgutil::image_to_tensor(&i1, &dev)?;
            let start = Instant::now();
            let out_t = rife.interpolate_scaled(&t0, &t1, timestep, scale)?;
            println!("interpolate {:?} -> {:?} scale={} ({:?})", t0.shape(), out_t.shape(), scale, start.elapsed());
            let img = framegen::imgutil::tensor_to_image(&out_t)?;
            img.save(&out).map_err(|e| anyhow!("save {}: {e}", out.display()))?;
            println!("wrote {}", out.display());
            Ok(())
        }
        Cmd::Video { input, output, times, scale } => {
            framegen::io::video::interpolate_video(&rife, &input, &output, times, scale, &dev)
                .map(|_| ())
                .map_err(|e| anyhow!("video: {e}"))
        }
    }
}
