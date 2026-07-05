use anyhow::{anyhow, Context, Result};
use std::io::Read;
use std::path::Path;
use std::process::{Child, Command, Stdio};

/// Read exactly `buf.len()` bytes, or return `None` on a clean EOF at a frame boundary.
pub fn read_exact_or_eof<R: Read>(r: &mut R, buf: &mut [u8]) -> Result<Option<()>> {
    let mut filled = 0;
    while filled < buf.len() {
        let n = r.read(&mut buf[filled..]).context("read frame from ffmpeg")?;
        if n == 0 {
            if filled == 0 {
                return Ok(None);
            }
            return Err(anyhow!("short read: {}/{} bytes (corrupt stream?)", filled, buf.len()));
        }
        filled += n;
    }
    Ok(Some(()))
}

/// ffmpeg: decode `input` to raw RGB24 on stdout.
pub fn spawn_decoder(input: &Path) -> Result<Child> {
    Command::new("ffmpeg")
        .args(["-v", "error", "-i", input.to_str().unwrap(), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .context("ffmpeg decode failed to start")
}

/// ffmpeg: encode raw RGB24 on stdin to h264 mp4 at `out_fps`.
/// When `audio_from` is set, its audio track (if any) is copied into the output -
/// interpolation keeps the duration, so the original audio stays in sync.
pub fn spawn_encoder(output: &Path, w: usize, h: usize, out_fps: f64, audio_from: Option<&Path>) -> Result<Child> {
    let mut cmd = Command::new("ffmpeg");
    cmd.args([
        "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", &format!("{w}x{h}"), "-r", &format!("{out_fps:.6}"), "-i", "-",
    ]);
    if let Some(src) = audio_from {
        cmd.args([
            "-i", src.to_str().unwrap(),
            "-map", "0:v", "-map", "1:a?", "-c:a", "copy", "-shortest",
        ]);
    }
    cmd.args([
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "fast", "-crf", "18",
        output.to_str().unwrap(),
    ]);
    cmd.stdin(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .context("ffmpeg encode failed to start")
}
