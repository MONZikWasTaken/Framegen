pub mod model;
pub mod warp;

#[cfg(feature = "trt")]
pub mod trt;

pub use model::IfNetM;
pub use rife_core::{Frame, FrameInterpolator};

#[cfg(feature = "bin")]
pub mod imgutil;

#[cfg(feature = "bin")]
pub mod io;

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::VarBuilder;

pub const DEFAULT_SCALE: [f64; 3] = [4.0, 2.0, 1.0];

pub struct RifeCandle {
    net: IfNetM,
    device: Device,
}

impl RifeCandle {
    /// Load RIFE-Lite (RIFEm) weights from a safetensors file produced by
    /// `tools/convert_weights.py`. Pass DType::F16 for half-precision (faster on GPU).
    pub fn load<P: AsRef<std::path::Path>>(path: P, dtype: DType, device: &Device) -> Result<Self> {
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[path], dtype, device)? };
        let net = IfNetM::new(vb)?;
        Ok(Self { net, device: device.clone() })
    }

    /// Interpolate two candle tensors. img0/img1: [B,3,H,W] float in [0,1], BGR.
    /// timestep in [0,1] (0.5 = halfway). Returns [B,3,H,W] in [0,1].
    /// `scale` < 1.0 processes at lower resolution (faster, lower quality).
    /// The Frame-based API is `FrameInterpolator::interpolate`.
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

impl FrameInterpolator for RifeCandle {
    fn interpolate(&self, f0: &Frame, f1: &Frame, timestep: f32) -> anyhow::Result<Frame> {
        use rife_core::{pad32, prepost};
        if f0.w != f1.w || f0.h != f1.h {
            anyhow::bail!("frame size mismatch: {}x{} vs {}x{}", f0.w, f0.h, f1.w, f1.h);
        }
        let (w, h) = (f0.w, f0.h);
        let (pw, ph) = (pad32(w), pad32(h));
        let dev = &self.device;
        let dt = self.net.dtype();
        let mk = |rgb: &[u8]| -> anyhow::Result<Tensor> {
            let planar = prepost::to_input(rgb, w, h, pw, ph);
            Ok(Tensor::from_vec(planar, (1, 3, ph as usize, pw as usize), dev)?.to_dtype(dt)?)
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
        let rgb = prepost::from_output(&chw, w, h, w, h);
        Ok(Frame { w, h, rgb })
    }
}

#[cfg(test)]
mod trait_tests {
    use super::*;

    #[test]
    #[ignore] // needs models/rife_lite.safetensors
    fn candle_interpolate_returns_same_size_frame() {
        let dev = Device::Cpu;
        let rife = RifeCandle::load("models/rife_lite.safetensors", DType::F32, &dev).unwrap();
        let f0 = Frame { w: 64, h: 64, rgb: vec![128u8; 64 * 64 * 3] };
        let f1 = Frame { w: 64, h: 64, rgb: vec![130u8; 64 * 64 * 3] };
        let out = rife.interpolate(&f0, &f1, 0.5).unwrap();
        assert_eq!((out.w, out.h), (64, 64));
        assert_eq!(out.rgb.len(), 64 * 64 * 3);
    }
}
