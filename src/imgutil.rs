use anyhow::{anyhow, Result};
use candle_core::{DType, Device, Tensor};

// RIFE was trained on cv2 BGR images. All external I/O here is RGB, so we swap
// channels 0 and 2 at the boundary to feed BGR to the model and return RGB to
// the caller. PyTorch uses (x*255).byte() = truncation; we match with `as u8`.

// PNG (RGB) -> [1,3,H,W] f32 BGR in [0,1].
pub fn image_to_tensor(img: &image::DynamicImage, device: &Device) -> Result<Tensor> {
    let rgb = img.to_rgb8();
    let (w, h) = (rgb.width() as usize, rgb.height() as usize);
    let mut data = Vec::with_capacity(w * h * 3);
    for p in rgb.pixels() {
        data.push(p[2] as f32 / 255.0); // B
        data.push(p[1] as f32 / 255.0); // G
        data.push(p[0] as f32 / 255.0); // R
    }
    Tensor::from_vec(data, (h, w, 3), device)?
        .permute((2, 0, 1))?
        .unsqueeze(0)?
        .to_dtype(DType::F32)?
        .contiguous()
        .map_err(|e| anyhow!("tensor build: {e}"))
}

// [1,3,H,W] f32 BGR in [0,1] -> PNG (RGB).
pub fn tensor_to_image(t: &Tensor) -> Result<image::DynamicImage> {
    let t = if t.dims().len() == 4 { t.squeeze(0)? } else { t.clone() };
    let t = t.clamp(0.0, 1.0)?.permute((1, 2, 0))?.to_dtype(DType::F32)?;
    let (h, w, _c) = t.dims3().map_err(|e| anyhow!("{e}"))?;
    let mut buf = image::RgbImage::new(w as u32, h as u32);
    let pixels: Vec<Vec<Vec<f32>>> = t.to_vec3().map_err(|e| anyhow!("{e}"))?;
    for (y, row) in pixels.iter().enumerate() {
        for (x, px) in row.iter().enumerate() {
            let b = (px[0] * 255.0) as u8;
            let g = (px[1] * 255.0) as u8;
            let r = (px[2] * 255.0) as u8;
            buf.put_pixel(x as u32, y as u32, image::Rgb([r, g, b]));
        }
    }
    Ok(image::DynamicImage::ImageRgb8(buf))
}

// raw RGB24 (HxWx3, row-major, from ffmpeg) -> [1,3,H,W] f32 BGR in [0,1].
pub fn raw_rgb24_to_tensor(bytes: &[u8], w: usize, h: usize, device: &Device) -> Result<Tensor> {
    let mut data = Vec::with_capacity(w * h * 3);
    for chunk in bytes.chunks_exact(3) {
        data.push(chunk[2] as f32 / 255.0); // B
        data.push(chunk[1] as f32 / 255.0); // G
        data.push(chunk[0] as f32 / 255.0); // R
    }
    Tensor::from_vec(data, (h, w, 3), device)?
        .permute((2, 0, 1))?
        .unsqueeze(0)?
        .to_dtype(DType::F32)?
        .contiguous()
        .map_err(|e| anyhow!("{e}"))
}

// [1,3,H,W] f32/f16 BGR in [0,1] -> raw RGB24 bytes (HxWx3, row-major, for ffmpeg).
// All math (clamp, *255, BGR->RGB, cast to u8) done on GPU; only u8 bytes downloaded.
pub fn tensor_to_rgb24(t: &Tensor) -> Result<Vec<u8>> {
    let t = if t.dims().len() == 4 { t.squeeze(0)? } else { t.clone() };
    // clamp + scale to [0,255] on GPU
    let t = t.clamp(0.0, 1.0)?.affine(255.0, 0.0)?;
    // BGR -> RGB: cat channels in reversed order
    let t = Tensor::cat(&[
        &t.narrow(0, 2, 1)?,  // R (was channel 2 in BGR)
        &t.narrow(0, 1, 1)?,  // G
        &t.narrow(0, 0, 1)?,  // B (was channel 0 in BGR)
    ], 0)?;
    // permute to HWC and cast to u8 on GPU (truncation matches PyTorch .byte())
    let t = t.permute((1, 2, 0))?.to_dtype(DType::U8)?.contiguous()?;
    t.flatten_all()?.to_vec1::<u8>().map_err(|e| anyhow!("{e}"))
}
