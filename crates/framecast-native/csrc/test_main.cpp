// Standalone validation of trt_shim: load engine, run one inference on constant input.
#include <cstdio>
#include <vector>
extern "C" {
    void* trt_create(const char*);
    void trt_dims(void*, int*, int*, int*);
    int trt_infer(void*, const float*, const float*, float*);
    void trt_destroy(void*);
}
int main(int argc, char** argv) {
    if (argc < 2) { printf("usage: shim_test <engine>\n"); return 2; }
    void* h = trt_create(argv[1]);
    if (!h) { printf("create FAILED\n"); return 1; }
    int C, EH, EW; trt_dims(h, &C, &EH, &EW);
    printf("dims C=%d EH=%d EW=%d\n", C, EH, EW);
    size_t n = (size_t)C * EH * EW;
    std::vector<float> in0(n, 0.5f), in1(n, 0.5f), out(n, -1.0f);
    int r = trt_infer(h, in0.data(), in1.data(), out.data());
    double s = 0; float mn = 1e9f, mx = -1e9f;
    for (size_t i = 0; i < n; i++) { s += out[i]; if (out[i] < mn) mn = out[i]; if (out[i] > mx) mx = out[i]; }
    printf("infer=%d  out mean=%.4f min=%.4f max=%.4f\n", r, s / n, mn, mx);
    trt_destroy(h);
    return 0;
}
