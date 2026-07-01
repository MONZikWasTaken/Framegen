pub mod model;
pub mod warp;

#[cfg(feature = "trt")]
pub mod trt;

pub use model::IfNetM;

#[cfg(feature = "bin")]
pub mod imgutil;

#[cfg(feature = "bin")]
pub mod io;

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::VarBuilder;

pub const DEFAULT_SCALE: [f64; 3] = [4.0, 2.0, 1.0];

pub struct RifeLite {
    net: IfNetM,
}

impl RifeLite {
    /// Load RIFE-Lite (RIFEm) weights from a safetensors file produced by
    /// `tools/convert_weights.py`. Pass DType::F16 for half-precision (faster on GPU).
    pub fn load<P: AsRef<std::path::Path>>(path: P, dtype: DType, device: &Device) -> Result<Self> {
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[path], dtype, device)? };
        let net = IfNetM::new(vb)?;
        Ok(Self { net })
    }

    /// Interpolate between two images. img0/img1: [B,3,H,W] float in [0,1].
    /// timestep in [0,1] (0.5 = halfway). Returns [B,3,H,W] in [0,1].
    /// `scale` < 1.0 processes at lower resolution (faster, lower quality).
    pub fn interpolate(&self, img0: &Tensor, img1: &Tensor, timestep: f64) -> Result<Tensor> {
        self.interpolate_scaled(img0, img1, timestep, 1.0)
    }

    /// Interpolate with explicit scale factor. scale=0.5 → process at half res.
    pub fn interpolate_scaled(
        &self,
        img0: &Tensor,
        img1: &Tensor,
        timestep: f64,
        scale: f64,
    ) -> Result<Tensor> {
        let (b, c, h, w) = img0.dims4()?;
        let dev = img0.device();
        let dt = self.net.dtype();

        let (proc_h, proc_w) = if scale < 1.0 {
            (((h as f64 * scale).floor() as usize).max(32), ((w as f64 * scale).floor() as usize).max(32))
        } else {
            (h, w)
        };

        // Resize input to processing resolution
        let (t0, t1) = if scale < 1.0 {
            (
                img0.to_dtype(dt)?.upsample_bilinear2d(proc_h, proc_w, false)?,
                img1.to_dtype(dt)?.upsample_bilinear2d(proc_h, proc_w, false)?,
            )
        } else {
            (img0.to_dtype(dt)?, img1.to_dtype(dt)?)
        };

        // Pad to multiple of 32
        let ph = ((proc_h - 1) / 32 + 1) * 32;
        let pw = ((proc_w - 1) / 32 + 1) * 32;
        let need_pad = ph != proc_h || pw != proc_w;

        let (t0p, t1p) = if need_pad {
            let pad = |t: &Tensor| -> Result<Tensor> {
                let padded_w = Tensor::zeros((b, c, proc_h, pw), dt, dev)?;
                padded_w.slice_set(t, 3, 0)?;
                let padded = Tensor::zeros((b, c, ph, pw), dt, dev)?;
                padded.slice_set(&padded_w, 2, 0)?;
                Ok(padded)
            };
            (pad(&t0)?, pad(&t1)?)
        } else {
            (t0, t1)
        };

        // Adjusted scale list (matches RIFE inference: scale_list[i] /= scale)
        let scale_list = [4.0 / scale, 2.0 / scale, 1.0 / scale];

        let imgs = Tensor::cat(&[&t0p, &t1p], 1)?;
        let out = self.net.forward(&imgs, &scale_list, timestep)?;

        // Crop padding
        let out = if need_pad {
            out.narrow(2, 0, proc_h)?.narrow(3, 0, proc_w)?.contiguous()?
        } else {
            out
        };

        // Upscale back to original resolution if we downscaled
        let out = if scale < 1.0 {
            out.upsample_bilinear2d(h, w, false)?
        } else {
            out
        };

        out.to_dtype(DType::F32)
    }
}
