//! Native TensorRT video pipeline: mp4 -> 2x -> mp4, fully in-process (no Python).
//! ffmpeg decodes to raw RGB24, `RifeTrt` interpolates in-process, ffmpeg encodes.
//!
//! Overlap layout: a reader thread fills a bounded frame queue and a writer thread
//! drains a bounded output queue, so decode and encode I/O run concurrently with
//! prepost+inference. Each frame is preprocessed once (the f32 input buffers swap
//! together with the frames instead of being recomputed as "prev").
use anyhow::{anyhow, Result};
use std::io::Write;
use std::sync::mpsc;
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
    pub skipped_static: u64,
}

/// True when two frames are near-identical (sampled mean AND max byte diff both tiny).
/// The max guard keeps small localized motion (a moving cursor, blinking eyes) from
/// being mistaken for a static pair just because the global mean is low.
fn is_static_pair(a: &[u8], b: &[u8]) -> bool {
    const STRIDE: usize = 1009; // prime, ~2700 samples on a 720p frame
    let (mut sum, mut mx, mut cnt) = (0u64, 0u8, 0u64);
    let mut i = 0;
    while i < a.len() {
        let d = a[i].abs_diff(b[i]);
        sum += d as u64;
        if d > mx {
            mx = d;
        }
        cnt += 1;
        i += STRIDE;
    }
    (sum as f64 / cnt as f64) < 1.0 && mx < 16
}

/// Interpolate `input` by `times` (2 = midpoint; >2 needs a variable-timestep engine,
/// see tools/export_u8.py — each pair then gets mids at t = k/times).
/// `skip_static`: duplicate the previous frame instead of inferring on near-identical pairs.
pub fn interpolate_video_trt(
    rife: &RifeTrt,
    input: &std::path::Path,
    output: &std::path::Path,
    times: u32,
    skip_static: bool,
) -> Result<Stats> {
    if times < 2 {
        return Err(anyhow!("--times must be >= 2"));
    }
    if times != 2 && !rife.has_timestep {
        return Err(anyhow!(
            "this engine has a baked timestep=0.5 (2x only); export a variable-t engine \
             via tools/export_u8.py for --times {times}"
        ));
    }
    let meta = probe(input)?;
    let (w, h) = (meta.width as usize, meta.height as usize);
    if rife.is_u8 {
        // u8 engines carry prepost (incl. padding) in-graph and are exact-size
        if w != rife.ew || h != rife.eh {
            return Err(anyhow!("u8 engine expects exactly {}x{}, video is {w}x{h}", rife.ew, rife.eh));
        }
    } else if h > rife.eh || w > rife.ew {
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
    let mut enc = spawn_encoder(output, w, h, out_fps, Some(input))?;
    let mut enc_in = enc.stdin.take().unwrap();

    // Decoded frames flow through a small bounded queue; encoding drains another.
    // Queue depth 4 caps memory at a few frames while decoupling the pipe I/O.
    let (frame_tx, frame_rx) = mpsc::sync_channel::<Vec<u8>>(4);
    let reader = std::thread::spawn(move || -> Result<()> {
        loop {
            let mut buf = vec![0u8; frame_bytes];
            match read_exact_or_eof(&mut dec_out, &mut buf)? {
                Some(_) => {
                    if frame_tx.send(buf).is_err() {
                        break; // main loop gone (error path) — stop reading
                    }
                }
                None => break, // EOF
            }
        }
        Ok(())
    });

    let (out_tx, out_rx) = mpsc::sync_channel::<Vec<u8>>(4);
    let writer = std::thread::spawn(move || -> Result<()> {
        for f in out_rx {
            enc_in.write_all(&f)?;
        }
        drop(enc_in); // close stdin so ffmpeg finalizes the file
        Ok(())
    });

    // Reusable f32 buffers (no per-frame allocation on the hot path); unused on the u8 path.
    let n = if rife.is_u8 { 0 } else { rife.elems() };
    let mut in0 = vec![0f32; n];
    let mut in1 = vec![0f32; n];
    let mut out = vec![0f32; n];

    let start = Instant::now();
    let (mut in_frames, mut out_frames, mut infers) = (0u64, 0u64, 0u64);
    let mut skipped_static = 0u64;
    let mut infer_time = std::time::Duration::ZERO;

    let mut prev = match frame_rx.recv() {
        Ok(f) => f,
        Err(_) => return Err(anyhow!("empty video")),
    };
    in_frames += 1;
    if !rife.is_u8 {
        to_input_into(&prev, w as u32, h as u32, rife.ew as u32, rife.eh as u32, &mut in0);
    }

    for cur in frame_rx.iter() {
        in_frames += 1;

        if skip_static && is_static_pair(&prev, &cur) {
            // near-identical pair: duplicate instead of inferring. On the f32 path the
            // skipped swap breaks the "in0 holds prev's input" invariant — restore it.
            for _ in 1..times {
                out_tx.send(prev.clone())?;
                out_frames += 1;
            }
            out_tx.send(prev)?;
            out_frames += 1;
            skipped_static += 1;
            if !rife.is_u8 {
                to_input_into(&cur, w as u32, h as u32, rife.ew as u32, rife.eh as u32, &mut in0);
            }
            prev = cur;
            continue;
        }

        if !rife.is_u8 {
            // prepost for `cur` only: `prev`'s f32 input was computed on its own iteration
            to_input_into(&cur, w as u32, h as u32, rife.ew as u32, rife.eh as u32, &mut in1);
        }

        let mut mids: Vec<Vec<u8>> = Vec::with_capacity(times as usize - 1);
        for k in 1..times {
            let timestep = k as f32 / times as f32;
            let mut mid = vec![0u8; frame_bytes];
            let t = Instant::now();
            if rife.is_u8 {
                // raw frame bytes in, raw frame bytes out — prepost fused into the engine
                rife.infer_u8_t(&prev, &cur, &mut mid, timestep)?;
            } else {
                rife.infer_t(&in0, &in1, &mut out, timestep)?;
                from_output_into(&out, w as u32, h as u32, rife.ew as u32, rife.eh as u32, &mut mid);
            }
            infer_time += t.elapsed();
            infers += 1;
            mids.push(mid);
        }
        if !rife.is_u8 {
            std::mem::swap(&mut in0, &mut in1);
        }

        out_tx.send(prev)?; // original frame
        out_frames += 1;
        for mid in mids {
            out_tx.send(mid)?; // interpolated frame(s) at t = k/times
            out_frames += 1;
        }

        prev = cur;
        if infers % 30 == 0 {
            eprintln!("  {infers} frames, {:.1} ms/infer", infer_time.as_secs_f64() * 1000.0 / infers as f64);
        }
    }
    out_tx.send(prev)?; // last frame
    out_frames += 1;
    drop(out_tx); // close the queue so the writer finishes

    reader
        .join()
        .map_err(|_| anyhow!("reader thread panicked"))??;
    writer
        .join()
        .map_err(|_| anyhow!("writer thread panicked"))??;

    let enc_status = enc.wait()?;
    let _ = dec.wait()?;
    if !enc_status.success() {
        return Err(anyhow!("ffmpeg encode exit {}", enc_status));
    }
    let elapsed = start.elapsed();
    let ms_per_infer = if infers > 0 { infer_time.as_secs_f64() * 1000.0 / infers as f64 } else { 0.0 };
    eprintln!(
        "done: {in_frames} in -> {out_frames} out, {:.2?} ({:.1} fps wall), {:.1} ms/infer, {skipped_static} static pairs skipped",
        elapsed, out_frames as f64 / elapsed.as_secs_f64(), ms_per_infer
    );
    Ok(Stats { in_frames, out_frames, elapsed, ms_per_infer, skipped_static })
}
