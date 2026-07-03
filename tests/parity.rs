// Runs the SAME frame pair through the candle oracle and the TensorRT engine and
// asserts they agree within tolerance. Gated: needs models + a built engine (both
// gitignored). Run manually (with the TRT/CUDA DLLs on PATH):
//   cargo test --features trt --test parity -- --ignored --nocapture
#![cfg(feature = "trt")]

use framecast::trt::RifeTrt;
use framecast::{FrameInterpolator, RifeCandle};
use rife_core::Frame;

/// Decode the first two frames of `demo/test_720p.mp4` as RGB24 at native size.
/// Real frames are the meaningful oracle input (the ramp of a synthetic pattern
/// produces sharp discontinuities that fp16 warp exaggerates at isolated pixels).
fn real_pair() -> (Frame, Frame) {
    let (w, h) = (1280u32, 720u32);
    let fb = (w * h * 3) as usize;
    let raw = std::process::Command::new("ffmpeg")
        .args([
            "-v", "error", "-i", "demo/test_720p.mp4",
            "-frames:v", "2", "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
        ])
        .output()
        .expect("ffmpeg decode (is ffmpeg on PATH?)")
        .stdout;
    assert!(raw.len() >= 2 * fb, "expected >=2 frames, got {} bytes", raw.len());
    let f0 = Frame { w, h, rgb: raw[..fb].to_vec() };
    let f1 = Frame { w, h, rgb: raw[fb..2 * fb].to_vec() };
    (f0, f1)
}

#[test]
#[ignore]
fn candle_and_trt_agree() {
    let dev = candle_core::Device::Cpu;
    let candle =
        RifeCandle::load("models/rife_lite.safetensors", candle_core::DType::F32, &dev).unwrap();
    let trt = RifeTrt::load(std::path::Path::new("assets/rife_lite_trt_fp16.engine")).unwrap();

    // Native 720p; both backends pad internally to /32 (736) - exercises the pad path.
    let (f0, f1) = real_pair();

    let c = candle.interpolate(&f0, &f1, 0.5).unwrap();
    let t = trt.interpolate(&f0, &f1, 0.5).unwrap();

    assert_eq!((c.w, c.h), (t.w, t.h));
    let n = c.rgb.len();
    let mut sum_abs = 0u64;
    let mut max_abs = 0u8;
    for i in 0..n {
        let d = c.rgb[i].abs_diff(t.rgb[i]);
        sum_abs += d as u64;
        if d > max_abs {
            max_abs = d;
        }
    }
    let mean = sum_abs as f64 / n as f64;
    eprintln!("parity: mean|delta|={mean:.3} max|delta|={max_abs} over {n} bytes");
    // Observed on 720p real frames (fp16 engine vs fp32 candle): mean~0.36, max~43.
    // The mean is the regression signal - a swapped/broken pre/post spikes it into the
    // tens. `max` is just fp16 rounding amplified at sharp warped edges; keep headroom.
    assert!(mean < 1.0, "mean abs diff too high: {mean} (pre/post regression?)");
    assert!(max_abs < 64, "max abs diff too high: {max_abs}");
}
