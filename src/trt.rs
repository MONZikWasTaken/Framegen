//! In-process TensorRT inference via the C++ shim in `csrc/trt_shim.cpp`.
//! No Python: loads a serialized engine, owns the CUDA context/buffers, runs `enqueueV3`.
use anyhow::{anyhow, Result};
use std::ffi::CString;
use std::os::raw::{c_char, c_int, c_void};
use std::path::Path;

extern "C" {
    fn trt_create(engine_path: *const c_char) -> *mut c_void;
    fn trt_dims(h: *mut c_void, c: *mut c_int, eh: *mut c_int, ew: *mut c_int);
    fn trt_infer(h: *mut c_void, in0: *const f32, in1: *const f32, out: *mut f32) -> c_int;
    fn trt_destroy(h: *mut c_void);
}

/// A loaded TensorRT engine. Inputs/outputs are engine-sized (padded /32) CHW float32.
pub struct RifeTrt {
    handle: *mut c_void,
    pub c: usize,
    pub eh: usize,
    pub ew: usize,
}

// The underlying context is only touched through &self methods that serialize on the
// internal CUDA stream; safe to move across threads (but not share without external sync).
unsafe impl Send for RifeTrt {}

impl RifeTrt {
    pub fn load(engine: &Path) -> Result<Self> {
        let cpath = CString::new(engine.to_string_lossy().as_bytes())?;
        let handle = unsafe { trt_create(cpath.as_ptr()) };
        if handle.is_null() {
            return Err(anyhow!("trt_create failed for {}", engine.display()));
        }
        let (mut c, mut eh, mut ew) = (0, 0, 0);
        unsafe { trt_dims(handle, &mut c, &mut eh, &mut ew) };
        Ok(Self { handle, c: c as usize, eh: eh as usize, ew: ew as usize })
    }

    pub fn elems(&self) -> usize {
        self.c * self.eh * self.ew
    }

    /// Native RGB24 (nw*nh*3) -> engine-sized CHW f32 (`dst` len == elems()):
    /// RGB->BGR, /255, zero-padded bottom/right (matches tools/compare_pytorch.py).
    pub fn fill_input(&self, rgb: &[u8], nw: usize, nh: usize, dst: &mut [f32]) {
        debug_assert_eq!(rgb.len(), nw * nh * 3);
        debug_assert_eq!(dst.len(), self.elems());
        let (ew, eh) = (self.ew, self.eh);
        let plane = eh * ew;
        dst.fill(0.0);
        for y in 0..nh {
            let row = y * ew;
            let srow = y * nw * 3;
            for x in 0..nw {
                let s = srow + x * 3;
                let o = row + x;
                dst[o] = rgb[s + 2] as f32 / 255.0;             // B
                dst[plane + o] = rgb[s + 1] as f32 / 255.0;     // G
                dst[2 * plane + o] = rgb[s] as f32 / 255.0;     // R
            }
        }
    }

    /// Engine-sized CHW f32 output -> native RGB24 (crop top-left, BGR->RGB, *255).
    pub fn read_output(&self, out: &[f32], nw: usize, nh: usize, rgb: &mut [u8]) {
        debug_assert_eq!(out.len(), self.elems());
        debug_assert_eq!(rgb.len(), nw * nh * 3);
        let (ew, eh) = (self.ew, self.eh);
        let plane = eh * ew;
        let px = |v: f32| (v * 255.0).clamp(0.0, 255.0) as u8;
        for y in 0..nh {
            let row = y * ew;
            let drow = y * nw * 3;
            for x in 0..nw {
                let o = row + x;
                let d = drow + x * 3;
                rgb[d] = px(out[2 * plane + o]);     // R
                rgb[d + 1] = px(out[plane + o]);     // G
                rgb[d + 2] = px(out[o]);             // B
            }
        }
    }

    /// Convenience: native RGB24 pair -> interpolated native RGB24 (allocates).
    pub fn interpolate_rgb(&self, rgb0: &[u8], rgb1: &[u8], nw: usize, nh: usize) -> Result<Vec<u8>> {
        if nh > self.eh || nw > self.ew {
            return Err(anyhow!("native {nw}x{nh} exceeds engine {}x{}", self.ew, self.eh));
        }
        let n = self.elems();
        let mut in0 = vec![0f32; n];
        let mut in1 = vec![0f32; n];
        let mut out = vec![0f32; n];
        self.fill_input(rgb0, nw, nh, &mut in0);
        self.fill_input(rgb1, nw, nh, &mut in1);
        self.infer(&in0, &in1, &mut out)?;
        let mut rgb = vec![0u8; nw * nh * 3];
        self.read_output(&out, nw, nh, &mut rgb);
        Ok(rgb)
    }

    /// Run inference. `in0`/`in1`/`out` must each be `elems()` long (CHW, BGR, /255, zero-padded).
    pub fn infer(&self, in0: &[f32], in1: &[f32], out: &mut [f32]) -> Result<()> {
        let n = self.elems();
        if in0.len() != n || in1.len() != n || out.len() != n {
            return Err(anyhow!("buffer size mismatch (expected {n})"));
        }
        let r = unsafe { trt_infer(self.handle, in0.as_ptr(), in1.as_ptr(), out.as_mut_ptr()) };
        if r != 0 {
            return Err(anyhow!("trt_infer failed ({r})"));
        }
        Ok(())
    }
}

impl Drop for RifeTrt {
    fn drop(&mut self) {
        unsafe { trt_destroy(self.handle) };
    }
}
