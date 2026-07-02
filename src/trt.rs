//! In-process TensorRT inference via the C++ shim in `csrc/trt_shim.cpp`.
//! No Python: loads a serialized engine, owns the CUDA context/buffers, runs `enqueueV3`.
use anyhow::{anyhow, Result};
use rife_core::prepost::{from_output, to_input_into};
use rife_core::{Frame, FrameInterpolator};
use std::ffi::CString;
use std::os::raw::{c_char, c_int, c_void};
use std::path::Path;

extern "C" {
    fn trt_create(engine_path: *const c_char) -> *mut c_void;
    fn trt_dims(h: *mut c_void, c: *mut c_int, eh: *mut c_int, ew: *mut c_int);
    fn trt_is_u8(h: *mut c_void) -> c_int;
    fn trt_has_timestep(h: *mut c_void) -> c_int;
    fn trt_io_bytes(h: *mut c_void, in_bytes: *mut usize, out_bytes: *mut usize);
    fn trt_infer(
        h: *mut c_void,
        in0: *const c_void,
        in1: *const c_void,
        out: *mut c_void,
        timestep: f32,
    ) -> c_int;
    fn trt_destroy(h: *mut c_void);
}

/// A loaded TensorRT engine.
/// f32 engines: I/O is engine-sized (padded /32) CHW float32, prepost on the CPU.
/// u8 engines (tools/export_u8.py): I/O is raw RGB frame bytes, prepost fused in-graph —
/// `eh`/`ew` are then the exact (unpadded) frame dims.
pub struct RifeTrt {
    handle: *mut c_void,
    pub c: usize,
    pub eh: usize,
    pub ew: usize,
    pub is_u8: bool,
    pub has_timestep: bool,
    in_bytes: usize,
    out_bytes: usize,
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
        let is_u8 = unsafe { trt_is_u8(handle) } != 0;
        let has_timestep = unsafe { trt_has_timestep(handle) } != 0;
        let (mut in_bytes, mut out_bytes) = (0usize, 0usize);
        unsafe { trt_io_bytes(handle, &mut in_bytes, &mut out_bytes) };
        Ok(Self {
            handle,
            c: c as usize,
            eh: eh as usize,
            ew: ew as usize,
            is_u8,
            has_timestep,
            in_bytes,
            out_bytes,
        })
    }

    pub fn elems(&self) -> usize {
        self.c * self.eh * self.ew
    }

    fn check_timestep(&self, t: f32) -> Result<()> {
        if !self.has_timestep && (t - 0.5).abs() > 1e-6 {
            return Err(anyhow!("engine has a baked timestep=0.5, cannot use t={t}"));
        }
        Ok(())
    }

    /// Run f32 inference. `in0`/`in1`/`out` must each be `elems()` long (CHW, BGR, /255, zero-padded).
    /// `timestep` requires a variable-t engine unless it is 0.5.
    pub fn infer_t(&self, in0: &[f32], in1: &[f32], out: &mut [f32], timestep: f32) -> Result<()> {
        if self.is_u8 {
            return Err(anyhow!("u8 engine: use infer_u8 (raw frame bytes)"));
        }
        self.check_timestep(timestep)?;
        let n = self.elems();
        if in0.len() != n || in1.len() != n || out.len() != n {
            return Err(anyhow!("buffer size mismatch (expected {n})"));
        }
        let r = unsafe {
            trt_infer(
                self.handle,
                in0.as_ptr() as *const c_void,
                in1.as_ptr() as *const c_void,
                out.as_mut_ptr() as *mut c_void,
                timestep,
            )
        };
        if r != 0 {
            return Err(anyhow!("trt_infer failed ({r})"));
        }
        Ok(())
    }

    /// f32 inference at the midpoint (timestep 0.5).
    pub fn infer(&self, in0: &[f32], in1: &[f32], out: &mut [f32]) -> Result<()> {
        self.infer_t(in0, in1, out, 0.5)
    }

    /// Run u8 inference: raw RGB frame bytes in, raw RGB frame bytes out (prepost in-graph).
    /// Buffers are exactly `eh*ew*3` bytes (the engine's frame size, no padding).
    pub fn infer_u8_t(&self, in0: &[u8], in1: &[u8], out: &mut [u8], timestep: f32) -> Result<()> {
        if !self.is_u8 {
            return Err(anyhow!("f32 engine: use infer (preprocessed f32 buffers)"));
        }
        self.check_timestep(timestep)?;
        if in0.len() != self.in_bytes || in1.len() != self.in_bytes || out.len() != self.out_bytes {
            return Err(anyhow!(
                "buffer size mismatch (expected in {} / out {} bytes)",
                self.in_bytes, self.out_bytes
            ));
        }
        let r = unsafe {
            trt_infer(
                self.handle,
                in0.as_ptr() as *const c_void,
                in1.as_ptr() as *const c_void,
                out.as_mut_ptr() as *mut c_void,
                timestep,
            )
        };
        if r != 0 {
            return Err(anyhow!("trt_infer failed ({r})"));
        }
        Ok(())
    }

    /// u8 inference at the midpoint (timestep 0.5).
    pub fn infer_u8(&self, in0: &[u8], in1: &[u8], out: &mut [u8]) -> Result<()> {
        self.infer_u8_t(in0, in1, out, 0.5)
    }
}

impl FrameInterpolator for RifeTrt {
    /// Variable-t engines honor `timestep`; fixed engines accept only 0.5.
    fn interpolate(&self, f0: &Frame, f1: &Frame, timestep: f32) -> Result<Frame> {
        if f0.w != f1.w || f0.h != f1.h {
            return Err(anyhow!(
                "frame size mismatch: {}x{} vs {}x{}",
                f0.w, f0.h, f1.w, f1.h
            ));
        }
        let (w, h) = (f0.w, f0.h);
        if self.is_u8 {
            // u8 engines are exact-size: prepost (incl. padding) lives in the graph
            if w as usize != self.ew || h as usize != self.eh {
                return Err(anyhow!("u8 engine expects exactly {}x{}, got {w}x{h}", self.ew, self.eh));
            }
            let mut rgb = vec![0u8; self.out_bytes];
            self.infer_u8_t(&f0.rgb, &f1.rgb, &mut rgb, timestep)?;
            return Ok(Frame { w, h, rgb });
        }
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
        self.infer_t(&in0, &in1, &mut out, timestep)?;
        let rgb = from_output(&out, w, h, ew, eh);
        Ok(Frame { w, h, rgb })
    }
}

impl Drop for RifeTrt {
    fn drop(&mut self) {
        unsafe { trt_destroy(self.handle) };
    }
}
