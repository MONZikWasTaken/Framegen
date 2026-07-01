//! Native TensorRT video interpolation CLI (no Python). mp4 -> 2x -> mp4, in-process.
use anyhow::Result;
use clap::Parser;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "rife-trt", version)]
struct Args {
    /// Serialized TensorRT engine (from tools/build_trt_engine.py / build_trt_int8.py)
    #[arg(long)]
    engine: PathBuf,
    #[arg(long)]
    input: PathBuf,
    #[arg(long)]
    output: PathBuf,
    /// Only 2 is supported (engine has a fixed timestep=0.5)
    #[arg(long, default_value = "2")]
    times: u32,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let rife = framecast::trt::RifeTrt::load(&args.engine)?;
    eprintln!("engine loaded: C={} EH={} EW={}", rife.c, rife.eh, rife.ew);
    framecast::io::video_trt::interpolate_video_trt(&rife, &args.input, &args.output, args.times)?;
    Ok(())
}
