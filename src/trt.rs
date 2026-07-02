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

impl FrameInterpolator for RifeTrt {
    /// The engine is fixed at timestep=0.5, so `_timestep` is ignored.
    fn interpolate(&self, f0: &Frame, f1: &Frame, _timestep: f32) -> Result<Frame> {
        if f0.w != f1.w || f0.h != f1.h {
            return Err(anyhow!(
                "frame size mismatch: {}x{} vs {}x{}",
                f0.w, f0.h, f1.w, f1.h
            ));
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
        let rgb = from_output(&out, w, h, ew, eh);
        Ok(Frame { w, h, rgb })
    }
}

impl Drop for RifeTrt {
    fn drop(&mut self) {
        unsafe { trt_destroy(self.handle) };
    }
}
