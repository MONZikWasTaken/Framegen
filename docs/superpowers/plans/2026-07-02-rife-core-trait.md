# rife-core + FrameInterpolator + single-source pre/post — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract a backend-neutral `rife-core` crate (Frame, `FrameInterpolator` trait, single-source pre/post) and make both the candle and TensorRT backends implement the trait through one shared color/normalization/padding path.

**Architecture:** New `crates/rife-core` holds `Frame { w,h,rgb }`, the trait, and pure `to_input`/`from_output` functions (BGR, /255, pad-to-32, crop). The existing `framecast` crate stays at the repo root as a workspace member, depends on `rife-core`, and rewrites `RifeTrt` + `RifeCandle` (renamed from `RifeLite`) + `imgutil` to route all color/pad logic through `rife-core`. ffmpeg reader/spawn helpers get deduped into one module.

**Tech Stack:** Rust 2021, cargo workspace, candle 0.11, TensorRT FFI (feature `trt`), anyhow.

## Global Constraints

- Rust edition 2021, `rust-version = "1.80"`.
- `rife-core` must NOT depend on candle, cuda, image, or clap — only `anyhow`. It has to stay usable from a future wasm/wgpu crate.
- Color/normalization semantics are FIXED and must match the current code exactly: channel order **BGR**, scale **/255.0**, zero-pad **bottom/right**, crop **top-left**, output cast is **truncation** (`as u8` after `clamp(0,255)*255`). Do not "fix" or round — bit-parity with the current pipeline is the acceptance bar.
- `cargo clippy --all-targets -- -D warnings` must stay clean.
- No comments in source unless marking a genuine port deviation/TODO (repo convention).
- Commit after each task. Work happens on branch `phase0-rife-core` (already checked out).

---

### Task 1: Cargo workspace + `rife-core` skeleton (Frame, trait, pad32)

**Files:**
- Create: `crates/rife-core/Cargo.toml`
- Create: `crates/rife-core/src/lib.rs`
- Modify: `Cargo.toml` (repo root — add `[workspace]`)

**Interfaces:**
- Produces: `rife_core::Frame { pub w: u32, pub h: u32, pub rgb: Vec<u8> }`; `rife_core::pad32(u32) -> u32`; `pub trait FrameInterpolator { fn interpolate(&self, f0: &Frame, f1: &Frame, timestep: f32) -> anyhow::Result<Frame>; }`

- [ ] **Step 1: Make the root a workspace.** Add this block to the TOP of the root `Cargo.toml` (before `[package]`):

```toml
[workspace]
members = [".", "crates/rife-core"]
```

- [ ] **Step 2: Create `crates/rife-core/Cargo.toml`:**

```toml
[package]
name = "rife-core"
version = "0.1.0"
edition = "2021"
rust-version = "1.80"
description = "Backend-neutral RIFE frame type, interpolation trait, and pre/post"

[dependencies]
anyhow = "1"
```

- [ ] **Step 3: Write the failing test** in `crates/rife-core/src/lib.rs`:

```rust
pub mod prepost;

pub struct Frame {
    pub w: u32,
    pub h: u32,
    pub rgb: Vec<u8>,
}

pub trait FrameInterpolator {
    fn interpolate(&self, f0: &Frame, f1: &Frame, timestep: f32) -> anyhow::Result<Frame>;
}

/// Round `x` up to the next multiple of 32 (RIFE requires /32 input dims).
pub fn pad32(x: u32) -> u32 {
    x.div_ceil(32) * 32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pad32_rounds_up_to_multiple_of_32() {
        assert_eq!(pad32(720), 736);
        assert_eq!(pad32(1280), 1280);
        assert_eq!(pad32(1), 32);
        assert_eq!(pad32(32), 32);
        assert_eq!(pad32(33), 64);
    }
}
```

Note: `prepost` module is created in Task 2. For this task, temporarily comment out `pub mod prepost;` so the crate compiles, OR do Step 3 of Task 2 first. Simplest: create an empty `crates/rife-core/src/prepost.rs` with just `// filled in Task 2` now.

- [ ] **Step 4: Create empty `crates/rife-core/src/prepost.rs`:**

```rust
// Pre/post conversion — implemented in Task 2.
```

- [ ] **Step 5: Run the test, verify it passes:**

Run: `cargo test -p rife-core pad32`
Expected: PASS (1 test).

- [ ] **Step 6: Verify the whole workspace still builds:**

Run: `cargo check`
Expected: framecast + rife-core compile, no errors.

- [ ] **Step 7: Commit:**

```bash
git add Cargo.toml crates/rife-core
git commit -m "feat(rife-core): workspace member with Frame, FrameInterpolator, pad32"
```

---

### Task 2: Single-source pre/post (`to_input` / `from_output`)

**Files:**
- Modify: `crates/rife-core/src/prepost.rs`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `pub fn to_input_into(rgb: &[u8], w: u32, h: u32, pw: u32, ph: u32, dst: &mut [f32])`
  - `pub fn to_input(rgb: &[u8], w: u32, h: u32, pw: u32, ph: u32) -> Vec<f32>`
  - `pub fn from_output_into(chw: &[f32], w: u32, h: u32, pw: u32, ph: u32, dst: &mut [u8])`
  - `pub fn from_output(chw: &[f32], w: u32, h: u32, pw: u32, ph: u32) -> Vec<u8>`
  - Layout contract: input/output buffers are CHW f32 of length `3 * pw * ph`, channel order **BGR**; `rgb` buffers are HWC u8 of length `3 * w * h`. `w <= pw`, `h <= ph`.

Rationale for the `_into` variants: the real-time TensorRT pipeline reuses fixed buffers per frame (zero heap alloc on the hot path). The allocating wrappers are for the trait impls and one-shot callers.

- [ ] **Step 1: Write the failing tests** in `crates/rife-core/src/prepost.rs`:

```rust
/// RGB8 HWC -> CHW f32, BGR order, /255, zero-padded bottom/right to pw x ph.
/// `dst` length must be 3*pw*ph. Matches the legacy trt `fill_input` / candle imgutil.
pub fn to_input_into(rgb: &[u8], w: u32, h: u32, pw: u32, ph: u32, dst: &mut [f32]) {
    unimplemented!()
}

pub fn to_input(rgb: &[u8], w: u32, h: u32, pw: u32, ph: u32) -> Vec<f32> {
    let mut dst = vec![0f32; 3 * (pw * ph) as usize];
    to_input_into(rgb, w, h, pw, ph, &mut dst);
    dst
}

/// CHW f32 (BGR, padded pw x ph) -> RGB8 HWC, crop top-left w x h, clamp*255 truncate.
/// `dst` length must be 3*w*h. Matches the legacy trt `read_output` / candle imgutil.
pub fn from_output_into(chw: &[f32], w: u32, h: u32, pw: u32, ph: u32, dst: &mut [u8]) {
    unimplemented!()
}

pub fn from_output(chw: &[f32], w: u32, h: u32, pw: u32, ph: u32) -> Vec<u8> {
    let mut dst = vec![0u8; 3 * (w * h) as usize];
    from_output_into(chw, w, h, pw, ph, &mut dst);
    dst
}

#[cfg(test)]
mod tests {
    use super::*;

    // One 2x1 RGB image, padded to 4x2. Verify BGR planar layout + zero pad.
    #[test]
    fn to_input_bgr_and_pad() {
        // pixel0 = (R=10,G=20,B=30), pixel1 = (R=40,G=50,B=60)
        let rgb = [10u8, 20, 30, 40, 50, 60];
        let (w, h, pw, ph) = (2u32, 1, 4u32, 2);
        let out = to_input(&rgb, w, h, pw, ph);
        let plane = (pw * ph) as usize; // 8
        // B plane row0: [30/255, 60/255, 0, 0], rest zero
        assert!((out[0] - 30.0 / 255.0).abs() < 1e-6);
        assert!((out[1] - 60.0 / 255.0).abs() < 1e-6);
        assert_eq!(out[2], 0.0);
        assert_eq!(out[3], 0.0);
        // G plane starts at `plane`
        assert!((out[plane] - 20.0 / 255.0).abs() < 1e-6);
        assert!((out[plane + 1] - 50.0 / 255.0).abs() < 1e-6);
        // R plane starts at 2*plane
        assert!((out[2 * plane] - 10.0 / 255.0).abs() < 1e-6);
        assert!((out[2 * plane + 1] - 40.0 / 255.0).abs() < 1e-6);
        // padded row1 all zero
        assert_eq!(out[4], 0.0);
    }

    // from_output must invert to_input for in-range 8-bit values (crop back).
    #[test]
    fn round_trip_crop() {
        let rgb = [10u8, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]; // 2x2
        let (w, h, pw, ph) = (2u32, 2, 32u32, 32);
        let chw = to_input(&rgb, w, h, pw, ph);
        let back = from_output(&chw, w, h, pw, ph);
        assert_eq!(back, rgb);
    }

    #[test]
    fn from_output_clamps_and_truncates() {
        // one pixel, pw=ph=1, chw BGR = [2.0, -1.0, 0.5]
        let chw = [2.0f32, -1.0, 0.5];
        let out = from_output(&chw, 1, 1, 1, 1);
        // R = clamp(0.5)*255 = 127 (trunc), G = clamp(-1)=0, B = clamp(2)=255
        assert_eq!(out, vec![127u8, 0, 255]);
    }
}
```

- [ ] **Step 2: Run tests, verify they fail:**

Run: `cargo test -p rife-core prepost`
Expected: FAIL / panic `unimplemented!()`.

- [ ] **Step 3: Implement** `to_input_into` and `from_output_into` (replace the two `unimplemented!()` bodies):

```rust
pub fn to_input_into(rgb: &[u8], w: u32, h: u32, pw: u32, ph: u32, dst: &mut [f32]) {
    debug_assert_eq!(rgb.len(), (w * h * 3) as usize);
    debug_assert_eq!(dst.len(), (3 * pw * ph) as usize);
    let (w, h, pw, ew) = (w as usize, h as usize, pw as usize, pw as usize);
    let ph = ph as usize;
    let plane = ph * ew;
    dst.fill(0.0);
    for y in 0..h {
        let row = y * ew;
        let srow = y * w * 3;
        for x in 0..w {
            let s = srow + x * 3;
            let o = row + x;
            dst[o] = rgb[s + 2] as f32 / 255.0; // B
            dst[plane + o] = rgb[s + 1] as f32 / 255.0; // G
            dst[2 * plane + o] = rgb[s] as f32 / 255.0; // R
        }
    }
    let _ = pw;
}

pub fn from_output_into(chw: &[f32], w: u32, h: u32, pw: u32, ph: u32, dst: &mut [u8]) {
    debug_assert_eq!(chw.len(), (3 * pw * ph) as usize);
    debug_assert_eq!(dst.len(), (w * h * 3) as usize);
    let (w, h, ew, ph) = (w as usize, h as usize, pw as usize, ph as usize);
    let plane = ph * ew;
    let px = |v: f32| (v * 255.0).clamp(0.0, 255.0) as u8;
    for y in 0..h {
        let row = y * ew;
        let drow = y * w * 3;
        for x in 0..w {
            let o = row + x;
            let d = drow + x * 3;
            dst[d] = px(chw[2 * plane + o]); // R
            dst[d + 1] = px(chw[plane + o]); // G
            dst[d + 2] = px(chw[o]); // B
        }
    }
}
```

Note: the `let (w,h,pw,ew) = ... ; let _ = pw;` shuffle is to keep `pw` bound but unused (stride is `ew == pw`). Cleaner: just bind `let ew = pw as usize;` and drop `pw`. Use whichever keeps clippy quiet — the stride is `pw`.

- [ ] **Step 4: Run tests, verify they pass:**

Run: `cargo test -p rife-core`
Expected: PASS (all prepost tests + pad32).

- [ ] **Step 5: Clippy:**

Run: `cargo clippy -p rife-core --all-targets -- -D warnings`
Expected: clean.

- [ ] **Step 6: Commit:**

```bash
git add crates/rife-core/src/prepost.rs
git commit -m "feat(rife-core): single-source to_input/from_output pre/post + tests"
```

Note: `tools/build_trt_int8.py` keeps its own Python copy of this logic (can't call Rust across the FFI boundary at calibration build time). That's an accepted, documented exception — not part of this Rust consolidation.

---

### Task 3: `framecast` depends on rife-core; `RifeTrt` implements the trait

**Files:**
- Modify: `Cargo.toml` (root — add rife-core dep)
- Modify: `src/trt.rs` (impl trait, delete `fill_input`/`read_output`/`interpolate_rgb`)
- Modify: `src/lib.rs` (re-export trait/Frame for convenience)
- Modify: `src/io/video_trt.rs` (use `to_input_into`/`from_output_into`)

**Interfaces:**
- Consumes: `rife_core::{Frame, FrameInterpolator, prepost::{to_input_into, from_output_into}}`.
- Produces: `impl FrameInterpolator for RifeTrt`; `RifeTrt` keeps `load`, `elems`, `infer`, and fields `c/eh/ew`.

- [ ] **Step 1: Add the dependency** to the root `Cargo.toml` `[dependencies]`:

```toml
rife-core = { path = "crates/rife-core" }
```

- [ ] **Step 2: Re-export from `src/lib.rs`** (add near the top, after existing `pub use`):

```rust
pub use rife_core::{Frame, FrameInterpolator};
```

- [ ] **Step 3: In `src/trt.rs`, delete** the three methods `fill_input`, `read_output`, and `interpolate_rgb` entirely. Add the imports at the top:

```rust
use rife_core::prepost::{from_output_into, to_input_into};
use rife_core::{Frame, FrameInterpolator};
```

- [ ] **Step 4: Add the trait impl** to `src/trt.rs` (after the `impl RifeTrt` block):

```rust
impl FrameInterpolator for RifeTrt {
    fn interpolate(&self, f0: &Frame, f1: &Frame, _timestep: f32) -> Result<Frame> {
        if f0.w != f1.w || f0.h != f1.h {
            return Err(anyhow!("frame size mismatch: {}x{} vs {}x{}", f0.w, f0.h, f1.w, f1.h));
        }
        let (w, h) = (f0.w, f0.h);
        if h as usize > self.eh || w as usize > self.ew {
            return Err(anyhow!("frame {w}x{h} exceeds engine {}x{}", self.ew, self.eh));
        }
        let (ew, eh) = (self.ew as u32, self.eh as u32);
        let n = self.elems();
        let mut in0 = vec![0f32; n];
        let mut in1 = vec![0f32; n];
        let mut out = vec![0f32; n];
        to_input_into(&f0.rgb, w, h, ew, eh, &mut in0);
        to_input_into(&f1.rgb, w, h, ew, eh, &mut in1);
        self.infer(&in0, &in1, &mut out)?;
        let rgb = rife_core::prepost::from_output(&out, w, h, ew, eh);
        Ok(Frame { w, h, rgb })
    }
}
```

(The engine is fixed timestep=0.5, so `_timestep` is ignored — documented behavior.)

- [ ] **Step 5: Update `src/io/video_trt.rs`** to use the shared pre/post with its reusable buffers. Replace the two lines:

```rust
        rife.fill_input(&buf_prev, w, h, &mut in0);
        rife.fill_input(&buf_cur, w, h, &mut in1);
```

with:

```rust
        to_input_into(&buf_prev, w as u32, h as u32, rife.ew as u32, rife.eh as u32, &mut in0);
        to_input_into(&buf_cur, w as u32, h as u32, rife.ew as u32, rife.eh as u32, &mut in1);
```

and replace:

```rust
        rife.read_output(&out, w, h, &mut mid);
```

with:

```rust
        from_output_into(&out, w as u32, h as u32, rife.ew as u32, rife.eh as u32, &mut mid);
```

Add the import at the top of `src/io/video_trt.rs`:

```rust
use rife_core::prepost::{from_output_into, to_input_into};
```

- [ ] **Step 6: Build the trt path:**

Run: `cargo build --release --features trt --bin rife-trt`
Expected: compiles (needs MSVC + CUDA env; this is the machine that has them).

- [ ] **Step 7: Clippy on the trt path:**

Run: `cargo clippy --features trt --all-targets -- -D warnings`
Expected: clean.

- [ ] **Step 8: Behavioral check** — run the real-time pipeline and confirm it still works (set the DLL PATH per AGENTS.md first):

```pwsh
$env:PATH="C:\Users\MONZik\AppData\Roaming\Python\Python312\site-packages\tensorrt_libs;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin;$env:PATH"
.\target\release\rife-trt.exe --engine assets\rife_lite_trt_fp16.engine --input demo\test_720p.mp4 --output out_after.mp4
```
Expected: exit 0, "done: … ms/infer" printed, `out_after.mp4` written. If a `demo/out_before.mp4` from before the refactor exists, the two must be byte-identical (`fc /b`), since pre/post semantics are unchanged.

- [ ] **Step 9: Commit:**

```bash
git add Cargo.toml src/trt.rs src/lib.rs src/io/video_trt.rs
git commit -m "feat(trt): RifeTrt implements FrameInterpolator via rife-core pre/post"
```

---

### Task 4: Rename `RifeLite` -> `RifeCandle`, implement the trait

**Files:**
- Modify: `src/lib.rs` (rename struct + impl, add trait impl)
- Modify: `src/bin/interpolate.rs`, `src/bin/profile.rs`, `src/bin/smoke.rs` (rename references)
- Modify: `src/io/video.rs` (rename type in signature)

**Interfaces:**
- Consumes: `rife_core::{Frame, FrameInterpolator, pad32, prepost::{to_input, from_output}}`; `candle_core::Tensor`.
- Produces: `pub struct RifeCandle`; inherent `load`, `interpolate`, `interpolate_scaled` (unchanged Tensor API); `impl FrameInterpolator for RifeCandle`.

- [ ] **Step 1: Rename in `src/lib.rs`** — change `pub struct RifeLite {` to `pub struct RifeCandle {` and `impl RifeLite {` to `impl RifeCandle {`. Keep all inherent methods (`load`, `interpolate`, `interpolate_scaled`) exactly as they are.

- [ ] **Step 2: Add the trait impl** at the bottom of `src/lib.rs`:

```rust
impl FrameInterpolator for RifeCandle {
    fn interpolate(&self, f0: &Frame, f1: &Frame, timestep: f32) -> Result<Frame> {
        use rife_core::{pad32, prepost};
        if f0.w != f1.w || f0.h != f1.h {
            return Err(candle_core::Error::Msg(format!(
                "frame size mismatch: {}x{} vs {}x{}", f0.w, f0.h, f1.w, f1.h
            )));
        }
        let (w, h) = (f0.w, f0.h);
        let (pw, ph) = (pad32(w), pad32(h));
        let dev = &self.net_device();
        let dt = self.net.dtype();
        let mk = |rgb: &[u8]| -> Result<Tensor> {
            let planar = prepost::to_input(rgb, w, h, pw, ph); // CHW BGR /255, padded
            Tensor::from_vec(planar, (1, 3, ph as usize, pw as usize), dev)?.to_dtype(dt)
        };
        let t0 = mk(&f0.rgb)?;
        let t1 = mk(&f1.rgb)?;
        let imgs = Tensor::cat(&[&t0, &t1], 1)?;
        let scale_list = [4.0, 2.0, 1.0];
        let out = self.net.forward(&imgs, &scale_list, timestep as f64)?;
        let out = out
            .narrow(2, 0, h as usize)?
            .narrow(3, 0, w as usize)?
            .to_dtype(DType::F32)?
            .contiguous()?;
        let chw: Vec<f32> = out.squeeze(0)?.flatten_all()?.to_vec1::<f32>()?;
        let rgb = prepost::from_output(&chw, w, h, w, h); // already cropped, no pad
        Ok(Frame { w, h, rgb })
    }
}
```

Note: this needs the model's device. Add a small accessor. In `src/model.rs` (or wherever `IfNetM` lives), if there is no `device()` accessor, add one; otherwise store the device on `RifeCandle` at `load`. Simplest: add a `device: Device` field to `RifeCandle`, set it in `load` (it already has `device` in scope), and replace `self.net_device()` above with `&self.device`. Do that:
  - In `load`, change `Ok(Self { net })` to `Ok(Self { net, device: device.clone() })`.
  - Add `device: Device,` to the struct.
  - Replace `let dev = &self.net_device();` with `let dev = &self.device;`.

- [ ] **Step 3: Update the three bins** — replace `framecast::RifeLite::load` with `framecast::RifeCandle::load` in `src/bin/interpolate.rs:76`, `src/bin/profile.rs:30`, `src/bin/smoke.rs:37`.

- [ ] **Step 4: Update `src/io/video.rs`** — change the `rife: &RifeLite` parameter type in `interpolate_video` to `rife: &RifeCandle`, and the `use crate::RifeLite;` import to `use crate::RifeCandle;`.

- [ ] **Step 5: Write a failing unit test** for the candle trait impl in `src/lib.rs` `#[cfg(test)]` (gated — needs weights; mark `#[ignore]`):

```rust
#[cfg(test)]
mod trait_tests {
    use super::*;
    use rife_core::{Frame, FrameInterpolator};

    #[test]
    #[ignore] // needs models/rife_lite.safetensors
    fn candle_interpolate_returns_same_size_frame() {
        let dev = candle_core::Device::Cpu;
        let rife = RifeCandle::load("models/rife_lite.safetensors", DType::F32, &dev).unwrap();
        let f0 = Frame { w: 64, h: 64, rgb: vec![128u8; 64 * 64 * 3] };
        let f1 = Frame { w: 64, h: 64, rgb: vec![130u8; 64 * 64 * 3] };
        let out = rife.interpolate(&f0, &f1, 0.5).unwrap();
        assert_eq!((out.w, out.h), (64, 64));
        assert_eq!(out.rgb.len(), 64 * 64 * 3);
    }
}
```

- [ ] **Step 6: Build + run the gated test:**

Run: `cargo build` then `cargo test candle_interpolate_returns_same_size_frame -- --ignored`
Expected: PASS (a mid-gray frame between two near-constant inputs; size preserved).

- [ ] **Step 7: Clippy (default features):**

Run: `cargo clippy --all-targets -- -D warnings`
Expected: clean.

- [ ] **Step 8: Commit:**

```bash
git add src/lib.rs src/model.rs src/bin/interpolate.rs src/bin/profile.rs src/bin/smoke.rs src/io/video.rs
git commit -m "feat(candle): rename RifeLite->RifeCandle, implement FrameInterpolator"
```

---

### Task 5: Shrink `imgutil` to thin adapters over rife-core (no color loops)

**Files:**
- Modify: `src/imgutil.rs`

**Interfaces:**
- Consumes: `rife_core::prepost::{to_input, from_output}`.
- Produces: same 4 public fns, same signatures — `image_to_tensor`, `raw_rgb24_to_tensor`, `tensor_to_image`, `tensor_to_rgb24`. Only internals change: they delegate to rife-core instead of hand-writing the BGR/normalize loops.

The point: these functions must produce **identical bytes** to today. They keep candle `Tensor`/`image` I/O, but the color/normalize step comes from `prepost`.

- [ ] **Step 1: Rewrite `image_to_tensor`** to go PNG -> RGB bytes -> `to_input` (no pad) -> Tensor:

```rust
pub fn image_to_tensor(img: &image::DynamicImage, device: &Device) -> Result<Tensor> {
    let rgb = img.to_rgb8();
    let (w, h) = (rgb.width(), rgb.height());
    let planar = rife_core::prepost::to_input(rgb.as_raw(), w, h, w, h); // CHW BGR /255, no pad
    Tensor::from_vec(planar, (1, 3, h as usize, w as usize), device)?
        .to_dtype(DType::F32)?
        .contiguous()
        .map_err(|e| anyhow!("tensor build: {e}"))
}
```

- [ ] **Step 2: Rewrite `raw_rgb24_to_tensor`** the same way:

```rust
pub fn raw_rgb24_to_tensor(bytes: &[u8], w: usize, h: usize, device: &Device) -> Result<Tensor> {
    let planar = rife_core::prepost::to_input(bytes, w as u32, h as u32, w as u32, h as u32);
    Tensor::from_vec(planar, (1, 3, h, w), device)?
        .to_dtype(DType::F32)?
        .contiguous()
        .map_err(|e| anyhow!("{e}"))
}
```

- [ ] **Step 3: Rewrite `tensor_to_rgb24`** to download CHW f32 and call `from_output`:

```rust
pub fn tensor_to_rgb24(t: &Tensor) -> Result<Vec<u8>> {
    let t = if t.dims().len() == 4 { t.squeeze(0)? } else { t.clone() };
    let (_c, h, w) = t.dims3().map_err(|e| anyhow!("{e}"))?;
    let chw: Vec<f32> = t.to_dtype(DType::F32)?.contiguous()?.flatten_all()?.to_vec1::<f32>()?;
    Ok(rife_core::prepost::from_output(&chw, w as u32, h as u32, w as u32, h as u32))
}
```

- [ ] **Step 4: Rewrite `tensor_to_image`** via `tensor_to_rgb24` + image:

```rust
pub fn tensor_to_image(t: &Tensor) -> Result<image::DynamicImage> {
    let t = if t.dims().len() == 4 { t.squeeze(0)? } else { t.clone() };
    let (_c, h, w) = t.dims3().map_err(|e| anyhow!("{e}"))?;
    let rgb = tensor_to_rgb24(&t)?;
    let buf = image::RgbImage::from_raw(w as u32, h as u32, rgb)
        .ok_or_else(|| anyhow!("rgb buffer size mismatch"))?;
    Ok(image::DynamicImage::ImageRgb8(buf))
}
```

- [ ] **Step 5: Remove now-unused imports** (`candle_core::Device` may still be needed; drop what clippy flags). Keep the top-of-file comment about BGR/truncation but trim it to note the logic now lives in `rife_core::prepost`.

- [ ] **Step 6: Behavioral parity test** — regenerate a PNG that the repo already has a reference for. Run the single-image CLI on the demo inputs and confirm the middle frame is unchanged vs the committed `demo/mid_pytorch.*` reference (or vs a copy made before this task):

```pwsh
.\target\release\rife-interpolate.exe --weights models\rife_lite.safetensors --img0 demo\I0_0.png --img1 demo\I0_1.png --out mid_after.png
```
Expected: exit 0. Compare `mid_after.png` to a pre-refactor `mid_before.png` — must be byte-identical (`fc /b mid_before.png mid_after.png`). (Make `mid_before.png` by running the same command on `main` before starting, or trust the existing `demo/mid_pytorch.png` if it was produced by this exact CLI.)

- [ ] **Step 7: Clippy + build:**

Run: `cargo clippy --all-targets -- -D warnings` and `cargo build --release`
Expected: clean, builds.

- [ ] **Step 8: Commit:**

```bash
git add src/imgutil.rs
git commit -m "refactor(imgutil): delegate color/normalize to rife-core prepost"
```

---

### Task 6: Dedup the ffmpeg reader/spawn into one module

**Files:**
- Create: `src/io/ffmpeg.rs`
- Modify: `src/io/mod.rs` (add `pub mod ffmpeg;`)
- Modify: `src/io/video.rs` (use shared helpers, delete local `read_exact_or_eof`)
- Modify: `src/io/video_trt.rs` (use shared helpers, delete local `read_exact_or_eof`)

**Interfaces:**
- Produces:
  - `pub fn read_exact_or_eof<R: std::io::Read>(r: &mut R, buf: &mut [u8]) -> anyhow::Result<Option<()>>`
  - `pub fn spawn_decoder(input: &std::path::Path) -> anyhow::Result<std::process::Child>` — ffmpeg mp4 -> rawvideo rgb24 on stdout.
  - `pub fn spawn_encoder(output: &std::path::Path, w: usize, h: usize, out_fps: f64) -> anyhow::Result<std::process::Child>` — rawvideo rgb24 stdin -> h264 mp4.

- [ ] **Step 1: Create `src/io/ffmpeg.rs`** with the three helpers (bodies lifted verbatim from the current duplicated code so behavior is identical):

```rust
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
pub fn spawn_encoder(output: &Path, w: usize, h: usize, out_fps: f64) -> Result<Child> {
    Command::new("ffmpeg")
        .args([
            "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", &format!("{w}x{h}"), "-r", &format!("{out_fps:.6}"), "-i", "-",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "fast", "-crf", "18",
            output.to_str().unwrap(),
        ])
        .stdin(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .context("ffmpeg encode failed to start")
}
```

- [ ] **Step 2: Register the module** — add to `src/io/mod.rs`:

```rust
pub mod ffmpeg;
```

- [ ] **Step 3: Update `src/io/video.rs`** — delete its local `read_exact_or_eof` fn and the inline `Command::new("ffmpeg")` decode/encode spawn blocks; replace with calls to `crate::io::ffmpeg::{spawn_decoder, spawn_encoder, read_exact_or_eof}`. The decode block becomes:

```rust
    let mut dec = crate::io::ffmpeg::spawn_decoder(input)?;
    let mut dec_out = dec.stdout.take().unwrap();
    let out_fps = meta.fps * times as f64;
    let mut enc = crate::io::ffmpeg::spawn_encoder(output, w, h, out_fps)?;
    let mut enc_in = enc.stdin.take().unwrap();
```

and all `read_exact_or_eof(...)` call sites now resolve to the `crate::io::ffmpeg::` one (either `use crate::io::ffmpeg::read_exact_or_eof;` at the top or fully-qualify).

- [ ] **Step 4: Update `src/io/video_trt.rs`** the same way — delete its local `read_exact_or_eof` and the two spawn blocks, use the shared helpers.

- [ ] **Step 5: Build both configs:**

Run: `cargo build --release` and `cargo build --release --features trt --bin rife-trt`
Expected: both compile, no duplicate-fn or unused-import warnings.

- [ ] **Step 6: Behavioral check** — re-run the trt pipeline from Task 3 Step 8; output must still be byte-identical to `out_after.mp4`:

```pwsh
.\target\release\rife-trt.exe --engine assets\rife_lite_trt_fp16.engine --input demo\test_720p.mp4 --output out_dedup.mp4
fc /b out_after.mp4 out_dedup.mp4
```
Expected: "no differences encountered".

- [ ] **Step 7: Clippy both:**

Run: `cargo clippy --all-targets -- -D warnings` and `cargo clippy --features trt --all-targets -- -D warnings`
Expected: clean.

- [ ] **Step 8: Commit:**

```bash
git add src/io/ffmpeg.rs src/io/mod.rs src/io/video.rs src/io/video_trt.rs
git commit -m "refactor(io): share ffmpeg reader/spawn between candle and trt pipelines"
```

---

### Task 7: Candle-vs-TensorRT parity integration test (the Phase-0 gate)

**Files:**
- Create: `tests/parity.rs`

**Interfaces:**
- Consumes: `framecast::{RifeCandle, FrameInterpolator}`, `framecast::trt::RifeTrt`, `rife_core::Frame`.

- [ ] **Step 1: Write the gated parity test** in `tests/parity.rs`:

```rust
// Runs the SAME frame pair through the candle oracle and the TensorRT engine
// and asserts they agree within tolerance. Gated: needs models + a built engine
// (both gitignored). Run manually:
//   cargo test --features trt --test parity -- --ignored --nocapture
#![cfg(feature = "trt")]

use framecast::trt::RifeTrt;
use framecast::{FrameInterpolator, RifeCandle};
use rife_core::Frame;

fn synthetic_pair(w: u32, h: u32) -> (Frame, Frame) {
    let mut a = vec![0u8; (w * h * 3) as usize];
    let mut b = vec![0u8; (w * h * 3) as usize];
    for y in 0..h {
        for x in 0..w {
            let i = ((y * w + x) * 3) as usize;
            a[i] = (x % 256) as u8;
            a[i + 1] = (y % 256) as u8;
            a[i + 2] = 128;
            b[i] = ((x + 4) % 256) as u8;
            b[i + 1] = ((y + 4) % 256) as u8;
            b[i + 2] = 130;
        }
    }
    (Frame { w, h, rgb: a }, Frame { w, h, rgb: b })
}

#[test]
#[ignore]
fn candle_and_trt_agree() {
    let dev = candle_core::Device::Cpu;
    let candle = RifeCandle::load("models/rife_lite.safetensors", candle_core::DType::F32, &dev).unwrap();
    let trt = RifeTrt::load(std::path::Path::new("assets/rife_lite_trt_fp16.engine")).unwrap();

    // Use a size the engine supports (<= engine ew/eh), multiple of 32 to avoid pad edges.
    let (w, h) = (trt.ew as u32, trt.eh as u32);
    let (f0, f1) = synthetic_pair(w, h);

    let c = candle.interpolate(&f0, &f1, 0.5).unwrap();
    let t = trt.interpolate(&f0, &f1, 0.5).unwrap();

    assert_eq!((c.w, c.h), (t.w, t.h));
    let n = c.rgb.len();
    let mut sum_abs = 0u64;
    let mut max_abs = 0u8;
    for i in 0..n {
        let d = c.rgb[i].abs_diff(t.rgb[i]);
        sum_abs += d as u64;
        if d > max_abs { max_abs = d; }
    }
    let mean = sum_abs as f64 / n as f64;
    eprintln!("parity: mean|Δ|={mean:.3} max|Δ|={max_abs} over {n} bytes");
    // fp16 engine vs fp32 candle: expect small but non-zero diff.
    assert!(mean < 3.0, "mean abs diff too high: {mean}");
    assert!(max_abs < 32, "max abs diff too high: {max_abs}");
}
```

- [ ] **Step 2: Build the test (compile-only, without running):**

Run: `cargo test --features trt --test parity --no-run`
Expected: compiles.

- [ ] **Step 3: Run the gated test** (needs the engine on `assets/` + DLL PATH set):

```pwsh
$env:PATH="C:\Users\MONZik\AppData\Roaming\Python\Python312\site-packages\tensorrt_libs;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin;$env:PATH"
cargo test --features trt --test parity -- --ignored --nocapture
```
Expected: PASS, prints `parity: mean|Δ|=… max|Δ|=…`. If the tolerances are wrong for the real model, adjust the thresholds to match observed values (document the observed numbers in the commit message) — the goal is a regression guard, not a fixed magic number.

- [ ] **Step 4: Commit:**

```bash
git add tests/parity.rs
git commit -m "test: candle-vs-trt parity gate (Phase 0 done-when)"
```

---

## Post-plan verification (whole feature)

- [ ] `cargo check` and `cargo clippy --all-targets -- -D warnings` — clean (default features).
- [ ] `cargo clippy --features trt --all-targets -- -D warnings` — clean.
- [ ] `cargo test` — rife-core unit tests pass; gated tests skipped by default.
- [ ] `rife-trt` on `demo/test_720p.mp4` produces byte-identical output to pre-refactor.
- [ ] `rife-interpolate` on `demo/I0_*.png` produces byte-identical output to pre-refactor.
- [ ] Update `AGENTS.md` Layout section: add `crates/rife-core/` and `src/io/ffmpeg.rs`, rename `RifeLite`->`RifeCandle` in the description. (Fold into the last commit or a small docs commit.)
- [ ] Update memory `intermodule-direction.md`: Phase 0 trait/core done.

## Self-review notes

- Spec §1 layout → Task 1. §2 rife-core API → Tasks 1–2. §3 backends impl trait → Tasks 3 (trt), 4 (candle), 5 (imgutil). §4 ffmpeg dedup → Task 6. §5 done-when parity → Task 7. All spec sections covered.
- Out-of-scope (CUDA autodiscovery / bootstrap script) correctly excluded.
- Type names consistent across tasks: `Frame`, `FrameInterpolator::interpolate(&self,&Frame,&Frame,f32)`, `to_input`/`to_input_into`/`from_output`/`from_output_into`, `RifeCandle`, `RifeTrt`.
