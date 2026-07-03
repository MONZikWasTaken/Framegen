// Compiles the TensorRT C++ shim and links nvinfer + CUDA runtime, but only when the
// `trt` feature is enabled (so normal/CPU builds need no C++ toolchain or CUDA).
fn main() {
    if std::env::var("CARGO_FEATURE_TRT").is_err() {
        return;
    }
    let cuda = std::env::var("CUDA_PATH")
        .unwrap_or_else(|_| r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1".to_string());
    // third_party/ lives at the workspace root, two levels above this crate.
    let root = format!("{}/../..", std::env::var("CARGO_MANIFEST_DIR").unwrap());

    cc::Build::new()
        .cpp(true)
        .std("c++17")
        .file("csrc/trt_shim.cpp")
        .include(format!("{root}/third_party/tensorrt/include"))
        .include(format!("{cuda}\\include"))
        .compile("trt_shim");

    println!("cargo:rerun-if-changed=csrc/trt_shim.cpp");
    println!("cargo:rustc-link-search=native={root}/third_party/tensorrt/lib");
    println!("cargo:rustc-link-search=native={cuda}\\lib\\x64");
    println!("cargo:rustc-link-lib=nvinfer_10");
    println!("cargo:rustc-link-lib=cudart");
}
