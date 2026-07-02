//! Native TensorRT video pipeline: mp4 -> 2x -> mp4, fully in-process (no Python).
//! ffmpeg decodes to raw RGB24, `RifeTrt` interpolates in-process, ffmpeg encodes.
use anyhow::{anyhow, Result};
use std::io::Write;
use std::time::Instant;

use crate::io::ffmpeg::{read_exact_or_eof, spawn_decoder, spawn_encoder};
use crate::io::video::probe;
use crate::trt::RifeTrt;
use rife_core::prepost::{from_output_into, to_input_into};

pub struct Stats {
    pub in_frames: u64,
    pub out_frames: u64,
    pub elapsed: std::time::Duration,
    pub ms_per_infer: f64,
}

/// Interpolate `input` by `times` (only 2 supported: engine is fixed timestep=0.5).
pub fn interpolate_video_trt(
    rife: &RifeTrt,
    input: &std::path::Path,
    output: &std::path::Path,
    times: u32,
) -> Result<Stats> {
    if times != 2 {
        return Err(anyhow!("TRT engine supports only --times 2 (fixed timestep=0.5)"));
    }
    let meta = probe(input)?;
    let (w, h) = (meta.width as usize, meta.height as usize);
    if h > rife.eh || w > rife.ew {
        return Err(anyhow!("video {w}x{h} exceeds engine {}x{}", rife.ew, rife.eh));
    }
    let frame_bytes = w * h * 3;
    eprintln!(
        "input: {}x{} @ {:.3} fps, {} frames -> {}x (engine {}x{})",
        w, h, meta.fps,
        meta.frame_count.map(|n| n.to_string()).unwrap_or_else(|| "?".into()),
        times, rife.ew, rife.eh
    );

    let mut dec = spawn_decoder(input)?;
    let mut dec_out = dec.stdout.take().unwrap();

    let out_fps = meta.fps * times as f64;
    let mut enc = spawn_encoder(output, w, h, out_fps)?;
    let mut enc_in = enc.stdin.take().unwrap();

    // Reusable buffers (no per-frame allocation).
    let n = rife.elems();
    let mut in0 = vec![0f32; n];
    let mut in1 = vec![0f32; n];
    let mut out = vec![0f32; n];
    let mut mid = vec![0u8; frame_bytes];

    let start = Instant::now();
    let (mut in_frames, mut out_frames, mut infers) = (0u64, 0u64, 0u64);
    let mut infer_time = std::time::Duration::ZERO;

    let mut buf_prev = vec![0u8; frame_bytes];
    if read_exact_or_eof(&mut dec_out, &mut buf_prev)?.is_none() {
        return Err(anyhow!("empty video"));
    }
    in_frames += 1;
    let mut buf_cur = vec![0u8; frame_bytes];

    while read_exact_or_eof(&mut dec_out, &mut buf_cur)?.is_some() {
        in_frames += 1;
        enc_in.write_all(&buf_prev)?; // original frame
        out_frames += 1;

        to_input_into(&buf_prev, w as u32, h as u32, rife.ew as u32, rife.eh as u32, &mut in0);
        to_input_into(&buf_cur, w as u32, h as u32, rife.ew as u32, rife.eh as u32, &mut in1);
        let t = Instant::now();
        rife.infer(&in0, &in1, &mut out)?;
        infer_time += t.elapsed();
        infers += 1;
        from_output_into(&out, w as u32, h as u32, rife.ew as u32, rife.eh as u32, &mut mid);

        enc_in.write_all(&mid)?; // interpolated frame
        out_frames += 1;

        std::mem::swap(&mut buf_prev, &mut buf_cur);
        if infers % 30 == 0 {
            eprintln!("  {infers} frames, {:.1} ms/infer", infer_time.as_secs_f64() * 1000.0 / infers as f64);
        }
    }
    enc_in.write_all(&buf_prev)?; // last frame
    out_frames += 1;

    drop(enc_in);
    let enc_status = enc.wait()?;
    let _ = dec.wait()?;
    if !enc_status.success() {
        return Err(anyhow!("ffmpeg encode exit {}", enc_status));
    }
    let elapsed = start.elapsed();
    let ms_per_infer = if infers > 0 { infer_time.as_secs_f64() * 1000.0 / infers as f64 } else { 0.0 };
    eprintln!(
        "done: {in_frames} in -> {out_frames} out, {:.2?} ({:.1} fps wall), {:.1} ms/infer",
        elapsed, out_frames as f64 / elapsed.as_secs_f64(), ms_per_infer
    );
    Ok(Stats { in_frames, out_frames, elapsed, ms_per_infer })
}
