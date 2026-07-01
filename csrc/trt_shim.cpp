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
    size_t inBytes = 0, outBytes = 0;
    const char* nIn0 = nullptr;
    const char* nIn1 = nullptr;
    const char* nOut = nullptr;
};
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
            if (!c->nIn0) c->nIn0 = nm; else c->nIn1 = nm;
        } else {
            c->nOut = nm;
        }
    }
    if (!c->nIn0 || !c->nIn1 || !c->nOut) { std::fprintf(stderr, "[shim] bad IO tensors\n"); delete c; return nullptr; }

    Dims di = c->engine->getTensorShape(c->nIn0);   // [1,C,EH,EW]
    Dims dov = c->engine->getTensorShape(c->nOut);
    c->C = di.d[1]; c->EH = di.d[2]; c->EW = di.d[3];
    c->inBytes = (size_t)c->C * c->EH * c->EW * sizeof(float);
    c->outBytes = (size_t)dov.d[1] * dov.d[2] * dov.d[3] * sizeof(float);

    cudaMalloc(&c->d0, c->inBytes);
    cudaMalloc(&c->d1, c->inBytes);
    cudaMalloc(&c->dout, c->outBytes);
    c->context->setTensorAddress(c->nIn0, c->d0);
    c->context->setTensorAddress(c->nIn1, c->d1);
    c->context->setTensorAddress(c->nOut, c->dout);
    return c;
}

// Engine (padded) input dims.
void trt_dims(void* h, int* C, int* EH, int* EW) {
    Ctx* c = (Ctx*)h;
    *C = c->C; *EH = c->EH; *EW = c->EW;
}

// in0/in1: host float32 [C*EH*EW] (BGR, /255, zero-padded). out: host float32 [C*EH*EW].
// Returns 0 on success.
int trt_infer(void* h, const float* in0, const float* in1, float* out) {
    Ctx* c = (Ctx*)h;
    cudaMemcpyAsync(c->d0, in0, c->inBytes, cudaMemcpyHostToDevice, c->stream);
    cudaMemcpyAsync(c->d1, in1, c->inBytes, cudaMemcpyHostToDevice, c->stream);
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
    if (c->dout) cudaFree(c->dout);
    if (c->stream) cudaStreamDestroy(c->stream);
    delete c->context;
    delete c->engine;
    delete c->runtime;
    delete c;
}

} // extern "C"
