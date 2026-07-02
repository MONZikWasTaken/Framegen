// Minimal extern "C" shim over TensorRT 10 for in-process inference from Rust.
// Owns the runtime/engine/context and the device buffers; Rust passes host-side
// float32 CHW buffers (already preprocessed to engine size) and gets the output back.
#include "NvInfer.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <fstream>
#include <vector>
#include <iterator>

using namespace nvinfer1;

namespace {
class Logger : public ILogger {
    void log(Severity s, const char* msg) noexcept override {
        if (s <= Severity::kWARNING) std::fprintf(stderr, "[TRT] %s\n", msg);
    }
};
Logger gLogger;

struct Ctx {
    IRuntime* runtime = nullptr;
    ICudaEngine* engine = nullptr;
    IExecutionContext* context = nullptr;
    cudaStream_t stream = nullptr;
    void* d0 = nullptr;
    void* d1 = nullptr;
    void* dout = nullptr;
    int C = 0, EH = 0, EW = 0;
    int isU8 = 0; // 1 when the engine has uint8 HWC I/O (prepost fused into the graph)
    size_t inBytes = 0, outBytes = 0;
    const char* nIn0 = nullptr;
    const char* nIn1 = nullptr;
    const char* nT = nullptr; // optional scalar timestep input (variable-t engines)
    void* dT = nullptr;
    const char* nOut = nullptr;
};

size_t dtypeSize(DataType t) {
    switch (t) {
        case DataType::kFLOAT: return 4;
        case DataType::kHALF: return 2;
        case DataType::kINT8: case DataType::kUINT8: return 1;
        case DataType::kINT32: return 4;
        default: return 4;
    }
}

size_t volume(const Dims& d) {
    size_t v = 1;
    for (int i = 0; i < d.nbDims; ++i) v *= (size_t)d.d[i];
    return v;
}
} // namespace

extern "C" {

// Returns opaque handle or nullptr on failure.
void* trt_create(const char* engine_path) {
    Ctx* c = new Ctx();
    c->runtime = createInferRuntime(gLogger);
    if (!c->runtime) { delete c; return nullptr; }

    std::ifstream f(engine_path, std::ios::binary);
    if (!f) { std::fprintf(stderr, "[shim] cannot open %s\n", engine_path); delete c; return nullptr; }
    std::vector<char> buf((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());

    c->engine = c->runtime->deserializeCudaEngine(buf.data(), buf.size());
    if (!c->engine) { std::fprintf(stderr, "[shim] deserialize failed\n"); delete c; return nullptr; }
    c->context = c->engine->createExecutionContext();
    if (!c->context) { std::fprintf(stderr, "[shim] no context\n"); delete c; return nullptr; }
    cudaStreamCreate(&c->stream);

    int n = c->engine->getNbIOTensors();
    for (int i = 0; i < n; ++i) {
        const char* nm = c->engine->getIOTensorName(i);
        if (c->engine->getTensorIOMode(nm) == TensorIOMode::kINPUT) {
            // scalar/1-elem input = the optional timestep; frame inputs fill in order
            if (volume(c->engine->getTensorShape(nm)) <= 1) c->nT = nm;
            else if (!c->nIn0) c->nIn0 = nm;
            else c->nIn1 = nm;
        } else {
            c->nOut = nm;
        }
    }
    if (!c->nIn0 || !c->nIn1 || !c->nOut) { std::fprintf(stderr, "[shim] bad IO tensors\n"); delete c; return nullptr; }

    Dims di = c->engine->getTensorShape(c->nIn0);   // f32: [1,C,EH,EW]; u8: [1,H,W,3]
    Dims dov = c->engine->getTensorShape(c->nOut);
    DataType inT = c->engine->getTensorDataType(c->nIn0);
    c->isU8 = (inT == DataType::kUINT8) ? 1 : 0;
    if (c->isU8) { c->C = di.d[3]; c->EH = di.d[1]; c->EW = di.d[2]; }
    else         { c->C = di.d[1]; c->EH = di.d[2]; c->EW = di.d[3]; }
    c->inBytes = volume(di) * dtypeSize(inT);
    c->outBytes = volume(dov) * dtypeSize(c->engine->getTensorDataType(c->nOut));

    cudaMalloc(&c->d0, c->inBytes);
    cudaMalloc(&c->d1, c->inBytes);
    cudaMalloc(&c->dout, c->outBytes);
    c->context->setTensorAddress(c->nIn0, c->d0);
    c->context->setTensorAddress(c->nIn1, c->d1);
    c->context->setTensorAddress(c->nOut, c->dout);
    if (c->nT) {
        cudaMalloc(&c->dT, sizeof(float));
        c->context->setTensorAddress(c->nT, c->dT);
    }
    return c;
}

// 1 when the engine takes a variable timestep input (else it is baked at 0.5).
int trt_has_timestep(void* h) { return ((Ctx*)h)->nT != nullptr; }

// Engine (padded) input dims. For u8 engines EH/EW are the (unpadded) frame dims.
void trt_dims(void* h, int* C, int* EH, int* EW) {
    Ctx* c = (Ctx*)h;
    *C = c->C; *EH = c->EH; *EW = c->EW;
}

// 1 when the engine takes uint8 HWC frames directly (prepost fused in-graph).
int trt_is_u8(void* h) { return ((Ctx*)h)->isU8; }

// Per-tensor host buffer sizes in bytes.
void trt_io_bytes(void* h, size_t* inBytes, size_t* outBytes) {
    Ctx* c = (Ctx*)h;
    *inBytes = c->inBytes; *outBytes = c->outBytes;
}

// f32 engines: host float32 CHW (BGR, /255, zero-padded). u8 engines: raw RGB frame bytes.
// Buffer sizes must match trt_io_bytes. `timestep` is used only when the engine has a
// timestep input (trt_has_timestep); fixed engines ignore it. Returns 0 on success.
int trt_infer(void* h, const void* in0, const void* in1, void* out, float timestep) {
    Ctx* c = (Ctx*)h;
    cudaMemcpyAsync(c->d0, in0, c->inBytes, cudaMemcpyHostToDevice, c->stream);
    cudaMemcpyAsync(c->d1, in1, c->inBytes, cudaMemcpyHostToDevice, c->stream);
    if (c->dT) cudaMemcpyAsync(c->dT, &timestep, sizeof(float), cudaMemcpyHostToDevice, c->stream);
    if (!c->context->enqueueV3(c->stream)) return -1;
    cudaMemcpyAsync(out, c->dout, c->outBytes, cudaMemcpyDeviceToHost, c->stream);
    if (cudaStreamSynchronize(c->stream) != cudaSuccess) return -2;
    return 0;
}

void trt_destroy(void* h) {
    if (!h) return;
    Ctx* c = (Ctx*)h;
    if (c->d0) cudaFree(c->d0);
    if (c->d1) cudaFree(c->d1);
    if (c->dT) cudaFree(c->dT);
    if (c->dout) cudaFree(c->dout);
    if (c->stream) cudaStreamDestroy(c->stream);
    delete c->context;
    delete c->engine;
    delete c->runtime;
    delete c;
}

} // extern "C"
