use anyhow::{anyhow, Context, Result};
use std::io::Write;
use std::process::Command;
use std::time::Instant;

use candle_core::Device;

use crate::imgutil;
use crate::io::ffmpeg::{read_exact_or_eof, spawn_decoder, spawn_encoder};
use crate::RifeCandle;

pub struct VideoMeta {
    pub width: u32,
    pub height: u32,
    pub fps: f64,
    pub frame_count: Option<u64>,
}

pub struct Stats {
    pub in_frames: u64,
    pub out_frames: u64,
    pub elapsed: std::time::Duration,
    pub ms_per_intermediate: f64,
}

/// Probe an input video via ffprobe.
pub fn probe(path: &std::path::Path) -> Result<VideoMeta> {
    let out = Command::new("ffprobe")
        .args([
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate,nb_frames",
            "-of", "default=noprint_wrappers=1:nokey=0",
            path.to_str().unwrap(),
        ])
        .output()
        .context("ffprobe не запустился — проверь, что ffmpeg/ffprobe в PATH")?;
    if !out.status.success() {
        return Err(anyhow!(
            "ffprobe failed: {}",
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    let txt = String::from_utf8_lossy(&out.stdout);
    let mut width = 0u32;
    let mut height = 0u32;
    let mut fps = 0.0f64;
    let mut frame_count = None;
    for line in txt.lines() {
        let (k, v) = match line.split_once('=') {
            Some(p) => p,
            None => continue,
        };
        match k.trim() {
            "width" => width = v.trim().parse().unwrap_or(0),
            "height" => height = v.trim().parse().unwrap_or(0),
            "r_frame_rate" => fps = parse_fraction(v.trim()),
            "nb_frames" => {
                if let Ok(n) = v.trim().parse::<u64>() {
                    if n > 0 { frame_count = Some(n); }
                }
            }
            _ => {}
        }
    }
    if width == 0 || height == 0 || fps == 0.0 {
        return Err(anyhow!("ffprobe: не удалось прочитать w/h/fps из [{txt}]"));
    }
    Ok(VideoMeta { width, height, fps, frame_count })
}

fn parse_fraction(s: &str) -> f64 {
    match s.split_once('/') {
        Some((n, d)) => {
            let n: f64 = n.parse().unwrap_or(0.0);
            let d: f64 = d.parse().unwrap_or(1.0);
            if d == 0.0 { 0.0 } else { n / d }
        }
        None => s.parse().unwrap_or(0.0),
    }
}

/// Interpolate `input` mp4 by `times` (2 = double fps), writing `output` mp4.
/// Streams raw RGB24 through pipes — no temp files on disk.
pub fn interpolate_video(
    rife: &RifeCandle,
    input: &std::path::Path,
    output: &std::path::Path,
    times: u32,
    scale: f64,
    device: &Device,
) -> Result<Stats> {
    if times < 2 {
        return Err(anyhow!("--times должен быть >= 2 (получили {times})"));
    }
    let meta = probe(input)?;
    let w = meta.width as usize;
    let h = meta.height as usize;
    let frame_bytes = w * h * 3;
    eprintln!(
        "input: {}x{} @ {:.3} fps, {} frames → {}x ({} fps out), scale={}",
        w, h, meta.fps, meta.frame_count.map(|n| n.to_string()).unwrap_or_else(|| "?".into()),
        times, meta.fps * times as f64, scale
    );

    let mut dec = spawn_decoder(input)?;
    let mut dec_out = dec.stdout.take().unwrap();

    let out_fps = meta.fps * times as f64;
    let mut enc = spawn_encoder(output, w, h, out_fps)?;
    let mut enc_in = enc.stdin.take().unwrap();

    let start = Instant::now();
    let mut in_frames: u64 = 0;
    let mut out_frames: u64 = 0;
    let mut inter_count: u64 = 0;
    let mut inter_total: std::time::Duration = std::time::Duration::ZERO;

    // read first frame
    let mut buf_prev = vec![0u8; frame_bytes];
    if read_exact_or_eof(&mut dec_out, &mut buf_prev)?.is_none() {
        return Err(anyhow!("пустое видео — ни одного кадра"));
    }
    in_frames += 1;

    let mut buf_cur = vec![0u8; frame_bytes];
    loop {
        let Some(_) = read_exact_or_eof(&mut dec_out, &mut buf_cur)? else {
            break;
        };
        in_frames += 1;

        // emit prev frame
        enc_in.write_all(&buf_prev)?;
        out_frames += 1;

        // emit (times-1) intermediates between prev and cur
        let t0 = imgutil::raw_rgb24_to_tensor(&buf_prev, w, h, device)?;
        let t1 = imgutil::raw_rgb24_to_tensor(&buf_cur, w, h, device)?;
        for k in 1..times {
            let timestep = k as f64 / times as f64;
            let it = Instant::now();
            let mid = rife.interpolate_scaled(&t0, &t1, timestep, scale)?;
            inter_total += it.elapsed();
            inter_count += 1;
            let bytes = imgutil::tensor_to_rgb24(&mid)?;
            enc_in.write_all(&bytes)?;
            out_frames += 1;
            if inter_count % 10 == 0 {
                eprintln!(
                    "  {inter_count} intermediates, {:.1} ms/each",
                    inter_total.as_secs_f64() * 1000.0 / inter_count as f64
                );
            }
        }
        std::mem::swap(&mut buf_prev, &mut buf_cur);
    }
    // emit last frame
    enc_in.write_all(&buf_prev)?;
    out_frames += 1;
    drop(enc_in); // close stdin -> encode finishes
    let enc_status = enc.wait()?;
    let _ = dec.wait()?;
    if !enc_status.success() {
        return Err(anyhow!("ffmpeg encode завершился с кодом {}", enc_status));
    }
    let elapsed = start.elapsed();
    let ms_per_intermediate = if inter_count > 0 {
        inter_total.as_secs_f64() * 1000.0 / inter_count as f64
    } else { 0.0 };
    eprintln!(
        "done: {in_frames} in -> {out_frames} out, {:.2?} total, {:.1} ms/intermediate",
        elapsed, ms_per_intermediate
    );
    Ok(Stats { in_frames, out_frames, elapsed, ms_per_intermediate })
}
