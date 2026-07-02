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
