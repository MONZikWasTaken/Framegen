use anyhow::{anyhow, Result};
use candle_core::{DType, Device, Tensor};

// Candle <-> image/raw glue. The BGR / (/255) / clamp / truncation logic lives in
// `rife_core::prepost` (single source of truth); these functions only move bytes
// between that representation and candle tensors / `image` buffers.

// PNG (RGB) -> [1,3,H,W] f32 BGR in [0,1].
pub fn image_to_tensor(img: &image::DynamicImage, device: &Device) -> Result<Tensor> {
    let rgb = img.to_rgb8();
    let (w, h) = (rgb.width(), rgb.height());
    let planar = rife_core::prepost::to_input(rgb.as_raw(), w, h, w, h); // CHW BGR /255, no pad
    Tensor::from_vec(planar, (1, 3, h as usize, w as usize), device)?
        .to_dtype(DType::F32)?
        .contiguous()
        .map_err(|e| anyhow!("tensor build: {e}"))
}

// raw RGB24 (HxWx3, row-major, from ffmpeg) -> [1,3,H,W] f32 BGR in [0,1].
pub fn raw_rgb24_to_tensor(bytes: &[u8], w: usize, h: usize, device: &Device) -> Result<Tensor> {
    let planar = rife_core::prepost::to_input(bytes, w as u32, h as u32, w as u32, h as u32);
    Tensor::from_vec(planar, (1, 3, h, w), device)?
        .to_dtype(DType::F32)?
        .contiguous()
        .map_err(|e| anyhow!("{e}"))
}

// [1,3,H,W] f32/f16 BGR in [0,1] -> raw RGB24 bytes (HxWx3, row-major, for ffmpeg).
pub fn tensor_to_rgb24(t: &Tensor) -> Result<Vec<u8>> {
    let t = if t.dims().len() == 4 { t.squeeze(0)? } else { t.clone() };
    let (_c, h, w) = t.dims3().map_err(|e| anyhow!("{e}"))?;
    let chw: Vec<f32> = t
        .to_dtype(DType::F32)?
        .contiguous()?
        .flatten_all()?
        .to_vec1::<f32>()
        .map_err(|e| anyhow!("{e}"))?;
    Ok(rife_core::prepost::from_output(&chw, w as u32, h as u32, w as u32, h as u32))
}

// [1,3,H,W] f32 BGR in [0,1] -> PNG (RGB).
pub fn tensor_to_image(t: &Tensor) -> Result<image::DynamicImage> {
    let t = if t.dims().len() == 4 { t.squeeze(0)? } else { t.clone() };
    let (_c, h, w) = t.dims3().map_err(|e| anyhow!("{e}"))?;
    let rgb = tensor_to_rgb24(&t)?;
    let buf = image::RgbImage::from_raw(w as u32, h as u32, rgb)
        .ok_or_else(|| anyhow!("rgb buffer size mismatch"))?;
    Ok(image::DynamicImage::ImageRgb8(buf))
}
