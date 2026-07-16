//! Single source of truth for RIFE pre/post: the BGR / (/255) / pad-to-32 / crop
//! logic that used to be duplicated in candle `imgutil`, the trt shim wrapper, and
//! the Python INT8 calibrator. Layout contract:
//! - `rgb`: HWC u8, length `3*w*h`.
//! - model buffer: CHW f32, channel order **BGR**, length `3*pw*ph`, `w<=pw`, `h<=ph`.

/// RGB8 HWC -> CHW f32, BGR order, /255, zero-padded bottom/right to `pw`x`ph`.
/// `dst` length must be `3*pw*ph`.
pub fn to_input_into(rgb: &[u8], w: u32, h: u32, pw: u32, ph: u32, dst: &mut [f32]) {
    debug_assert_eq!(rgb.len(), (w * h * 3) as usize);
    debug_assert_eq!(dst.len(), (3 * pw * ph) as usize);
    let (w, h) = (w as usize, h as usize);
    let ew = pw as usize; // row stride in the padded plane
    let plane = ph as usize * ew;
    // zero the PAD only - the interior is fully overwritten below, and the old
    // full-plane fill was an ~11MB memset per 720p frame. Contract unchanged:
    // pad right columns + bottom rows are 0 on return.
    if w < ew || (h as u32) < ph {
        for p in 0..3 {
            let base = p * plane;
            for y in 0..h {
                dst[base + y * ew + w..base + (y + 1) * ew].fill(0.0);
            }
            dst[base + h * ew..base + plane].fill(0.0);
        }
    }
    // /255 through a 256-entry table: same single-rounded f32 values, the
    // per-pixel convert+divide leaves the hot loop
    let lut: [f32; 256] = std::array::from_fn(|v| v as f32 / 255.0);
    for y in 0..h {
        let row = y * ew;
        let srow = y * w * 3;
        for x in 0..w {
            let s = srow + x * 3;
            let o = row + x;
            dst[o] = lut[rgb[s + 2] as usize]; // B
            dst[plane + o] = lut[rgb[s + 1] as usize]; // G
            dst[2 * plane + o] = lut[rgb[s] as usize]; // R
        }
    }
}

pub fn to_input(rgb: &[u8], w: u32, h: u32, pw: u32, ph: u32) -> Vec<f32> {
    let mut dst = vec![0f32; 3 * (pw * ph) as usize];
    to_input_into(rgb, w, h, pw, ph, &mut dst);
    dst
}

/// CHW f32 (BGR, padded `pw`x`ph`) -> RGB8 HWC, crop top-left `w`x`h`,
/// clamp to [0,255] * 255 with truncation (matches PyTorch `.byte()`).
/// `dst` length must be `3*w*h`.
pub fn from_output_into(chw: &[f32], w: u32, h: u32, pw: u32, ph: u32, dst: &mut [u8]) {
    debug_assert_eq!(chw.len(), (3 * pw * ph) as usize);
    debug_assert_eq!(dst.len(), (w * h * 3) as usize);
    let (w, h) = (w as usize, h as usize);
    let ew = pw as usize;
    let plane = ph as usize * ew;
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

pub fn from_output(chw: &[f32], w: u32, h: u32, pw: u32, ph: u32) -> Vec<u8> {
    let mut dst = vec![0u8; 3 * (w * h) as usize];
    from_output_into(chw, w, h, pw, ph, &mut dst);
    dst
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn to_input_bgr_and_pad() {
        // pixel0 = (R=10,G=20,B=30), pixel1 = (R=40,G=50,B=60)
        let rgb = [10u8, 20, 30, 40, 50, 60];
        let (w, h, pw, ph) = (2u32, 1, 4u32, 2);
        let out = to_input(&rgb, w, h, pw, ph);
        let plane = (pw * ph) as usize; // 8
        assert!((out[0] - 30.0 / 255.0).abs() < 1e-6); // B
        assert!((out[1] - 60.0 / 255.0).abs() < 1e-6);
        assert_eq!(out[2], 0.0);
        assert_eq!(out[3], 0.0);
        assert!((out[plane] - 20.0 / 255.0).abs() < 1e-6); // G
        assert!((out[plane + 1] - 50.0 / 255.0).abs() < 1e-6);
        assert!((out[2 * plane] - 10.0 / 255.0).abs() < 1e-6); // R
        assert!((out[2 * plane + 1] - 40.0 / 255.0).abs() < 1e-6);
        assert_eq!(out[4], 0.0); // padded row1
    }

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
        let chw = [2.0f32, -1.0, 0.5]; // one pixel BGR
        let out = from_output(&chw, 1, 1, 1, 1);
        assert_eq!(out, vec![127u8, 0, 255]); // R=clamp(.5)*255=127, G=0, B=255
    }
}
