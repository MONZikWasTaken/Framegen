//! Measure in-process TensorRT inference latency from Rust (no Python subprocess).
use anyhow::Result;
use std::path::PathBuf;
use std::time::Instant;

fn main() -> Result<()> {
    let engine = std::env::args().nth(1).unwrap_or_else(|| {
        "assets/rife_lite_trt_fp16.engine".to_string()
    });
    let rife = framegen::trt::RifeTrt::load(&PathBuf::from(&engine))?;
    println!("engine {engine}: C={} EH={} EW={}", rife.c, rife.eh, rife.ew);

    let n = rife.elems();
    let in0 = vec![0.5f32; n];
    let in1 = vec![0.5f32; n];
    let mut out = vec![0f32; n];

    for _ in 0..5 {
        rife.infer(&in0, &in1, &mut out)?;
    }
    let iters = 50;
    let mut ts = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t = Instant::now();
        rife.infer(&in0, &in1, &mut out)?;
        ts.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p50 = ts[iters / 2];
    let p10 = ts[iters / 10];
    let mean_out: f64 = out.iter().map(|&x| x as f64).sum::<f64>() / n as f64;

    println!("in-process TRT infer ({iters} iters): p50={p50:.1}ms  p10={p10:.1}ms  fps={:.1}", 1000.0 / p50);
    println!("48fps (2x@24): {}   60fps: {}",
        if p50 <= 41.7 { "PASS" } else { "FAIL" },
        if p50 <= 16.6 { "PASS" } else { "FAIL" });
    println!("out mean={mean_out:.4} (sanity)");
    Ok(())
}
