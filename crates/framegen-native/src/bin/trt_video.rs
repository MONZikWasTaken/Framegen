//! Native TensorRT video interpolation CLI (no Python). mp4 -> 2x -> mp4, in-process.
use anyhow::Result;
use clap::Parser;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "rife-trt", version)]
struct Args {
    /// Serialized TensorRT engine (from tools/build_trt_engine.py)
    #[arg(long)]
    engine: PathBuf,
    #[arg(long)]
    input: PathBuf,
    #[arg(long)]
    output: PathBuf,
    /// Output fps multiplier. 2 works on any engine; >2 needs a variable-timestep
    /// engine (tools/export_u8.py)
    #[arg(long, default_value = "2")]
    times: u32,
    /// Skip inference when two frames are (nearly) identical - the previous frame is
    /// duplicated instead. Conservative threshold; big win on anime/screencasts.
    #[arg(long)]
    skip_static: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let rife = framegen::trt::RifeTrt::load(&args.engine)?;
    eprintln!("engine loaded: C={} EH={} EW={}", rife.c, rife.eh, rife.ew);
    framegen::io::video_trt::interpolate_video_trt(
        &rife, &args.input, &args.output, args.times, args.skip_static)?;
    Ok(())
}
