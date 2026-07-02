//! Correctness + bench of the native wgpu runtime against the browser-verified reference.
//! Run from the repo root:  cargo run --release -p rife-wgpu --example verify
use rife_wgpu::RifeWgpu;
use std::time::Instant;

fn rgba_of(path: &str, w: u32, h: u32) -> Vec<u8> {
    let img = image::open(path).expect(path).to_rgba8();
    assert_eq!((img.width(), img.height()), (w, h), "{path}");
    img.into_raw()
}

fn main() -> anyhow::Result<()> {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let bin = std::fs::read(root.join("assets/rt_slim.bin"))?;
    let man = std::fs::read_to_string(root.join("assets/rt_slim.json"))?;

    // correctness at 448x256 vs the torch-generated slim reference
    let (w, h) = (448u32, 256u32);
    let rt = RifeWgpu::new(w, h, &bin, &man)?;
    println!("runtime up: channels {:?}", rt.channels());
    let a = rgba_of(root.join("demo/I0_0.png").to_str().unwrap(), w, h);
    let b = rgba_of(root.join("demo/I0_1.png").to_str().unwrap(), w, h);
    let exp = std::fs::read(root.join("demo/rt_expected_256_slim.rgb"))?;

    let out = rt.run(&a, &b, 0.5)?;
    let (mut sum, mut mx, mut n) = (0u64, 0u8, 0u64);
    for (px, ex) in out.chunks(4).zip(exp.chunks(3)) {
        for k in 0..3 {
            let d = px[k].abs_diff(ex[k]);
            sum += d as u64;
            mx = mx.max(d);
            n += 1;
        }
    }
    let mean = sum as f64 / n as f64;
    println!(
        "correctness: mean|d|={mean:.4} max|d|={mx} {}",
        if mean < 1.0 && mx <= 8 { "PASS" } else { "FAIL" }
    );

    // bench 720p
    let (w, h) = (1280u32, 720u32);
    let rt = RifeWgpu::new(w, h, &bin, &man)?;
    let a = vec![128u8; (w * h * 4) as usize];
    let b = vec![140u8; (w * h * 4) as usize];
    for _ in 0..3 {
        rt.run(&a, &b, 0.5)?;
    }
    let mut ts: Vec<f64> = (0..20)
        .map(|_| {
            let t0 = Instant::now();
            rt.run(&a, &b, 0.5).unwrap();
            t0.elapsed().as_secs_f64() * 1000.0
        })
        .collect();
    ts.sort_by(|x, y| x.partial_cmp(y).unwrap());
    println!(
        "bench 1280x720: p50={:.1}ms p10={:.1}ms fps={:.0}",
        ts[10],
        ts[2],
        1000.0 / ts[10]
    );
    Ok(())
}
