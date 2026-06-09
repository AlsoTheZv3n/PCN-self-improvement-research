// Fused State-Optimization settling kernel — THE COMPILED CUDA SOURCE.
//
// This file IS the kernel. It is compiled at first use by pcn/kernels/settling_cuda.py via
// torch.utils.cpp_extension.load(sources=[this file]) (JIT, cached; name "pcn_settling_kernel_v3").
// Needs nvcc + MSVC (a vcvars64 shell on Windows) + a toolkit-matched CUDA torch (cu126).
// Design rationale, fusion strategy and benchmark matrix: docs/09, docs/12 §4d/§4g.
//
//   v1  settle_kernel        — one block per sample        (small-batch / launch-overhead win)
//   v2  settle_kernel_tiled  — one block per TB-sample tile (large-batch / weight-reuse win)
//   act codes: 0 = tanh, 1 = identity, 2 = sigmoid   (must match _act_id in settling_cuda.py)
//   correctness vs the PyTorch backend: allclose ~1e-6 on both paths (scripts/verify_kernel.py)
//   scope: 2-hidden-layer PCN (model.n == 3), fixed T, float32.
//
// The whole T-step settling loop runs in ONE launch with the states resident in shared memory
// across steps, so the weights are read from global memory once (persistent-kernel style) and
// the per-step pointwise ops (eps, phi', state update) are fused — removing the launch overhead
// that dominates the PyTorch path at MNIST widths. Mirrors pcn/settling.py::_settle_pytorch.

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

#define TB 8

// act codes: 0 = tanh, 1 = identity, 2 = sigmoid (must match _act_id in settling_cuda.py).
__device__ __forceinline__ float actf(float x, int act){
    if(act==0) return tanhf(x);
    if(act==2) return 1.0f/(1.0f+expf(-x));        // sigmoid
    return x;                                       // identity
}
__device__ __forceinline__ float actd(float x, int act){
    if(act==0){ float t=tanhf(x); return 1.0f - t*t; }
    if(act==2){ float s=1.0f/(1.0f+expf(-x)); return s*(1.0f-s); }   // sigmoid'
    return 1.0f;                                    // identity'
}

// ----- v1: one block per sample (small-batch path) ------------------------------------------
__global__ void settle_kernel(
    const float* __restrict__ s0_in, const float* __restrict__ s1_in,
    const float* __restrict__ s2_in, const float* __restrict__ s3_in,
    float* __restrict__ s1_out, float* __restrict__ s2_out, float* __restrict__ s3_out,
    const float* __restrict__ W0, const float* __restrict__ W1, const float* __restrict__ W2,
    const float* __restrict__ b0, const float* __restrict__ b1, const float* __restrict__ b2,
    int B, int n0, int n1, int n2, int n3, int T, float lr, int clamp_output, int act)
{
    int sample = blockIdx.x;
    if(sample >= B) return;
    int tid = threadIdx.x, nt = blockDim.x;
    extern __shared__ float sh[];
    float* s0=sh; float* s1=s0+n0; float* s2=s1+n1; float* s3=s2+n2;
    float* e1=s3+n3; float* e2=e1+n1; float* e3=e2+n2;
    float* p0=e3+n3; float* p1=p0+n0; float* p2=p1+n1;
    for(int i=tid;i<n0;i+=nt) s0[i]=s0_in[(size_t)sample*n0+i];
    for(int i=tid;i<n1;i+=nt) s1[i]=s1_in[(size_t)sample*n1+i];
    for(int i=tid;i<n2;i+=nt) s2[i]=s2_in[(size_t)sample*n2+i];
    for(int i=tid;i<n3;i+=nt) s3[i]=s3_in[(size_t)sample*n3+i];
    __syncthreads();
    for(int t=0;t<T;t++){
        for(int i=tid;i<n0;i+=nt) p0[i]=actf(s0[i],act);
        for(int i=tid;i<n1;i+=nt) p1[i]=actf(s1[i],act);
        for(int i=tid;i<n2;i+=nt) p2[i]=actf(s2[i],act);
        __syncthreads();
        for(int j=tid;j<n1;j+=nt){ float a=b0[j]; const float* w=W0+(size_t)j*n0;
            for(int i=0;i<n0;i++) a+=w[i]*p0[i]; e1[j]=s1[j]-a; }
        for(int j=tid;j<n2;j+=nt){ float a=b1[j]; const float* w=W1+(size_t)j*n1;
            for(int i=0;i<n1;i++) a+=w[i]*p1[i]; e2[j]=s2[j]-a; }
        for(int j=tid;j<n3;j+=nt){ float a=b2[j]; const float* w=W2+(size_t)j*n2;
            for(int i=0;i<n2;i++) a+=w[i]*p2[i]; e3[j]=s3[j]-a; }
        __syncthreads();
        for(int i=tid;i<n1;i+=nt){ float fb=0.0f;
            for(int j=0;j<n2;j++) fb+=e2[j]*W1[(size_t)j*n1+i];
            s1[i]-=lr*(e1[i]-actd(s1[i],act)*fb); }
        for(int i=tid;i<n2;i+=nt){ float fb=0.0f;
            for(int j=0;j<n3;j++) fb+=e3[j]*W2[(size_t)j*n2+i];
            s2[i]-=lr*(e2[i]-actd(s2[i],act)*fb); }
        if(!clamp_output){ for(int i=tid;i<n3;i+=nt) s3[i]-=lr*e3[i]; }
        __syncthreads();
    }
    for(int i=tid;i<n1;i+=nt) s1_out[(size_t)sample*n1+i]=s1[i];
    for(int i=tid;i<n2;i+=nt) s2_out[(size_t)sample*n2+i]=s2[i];
    for(int i=tid;i<n3;i+=nt) s3_out[(size_t)sample*n3+i]=s3[i];
}

// ----- v2: one block per TB-sample tile, weights reused across the tile (large-batch path) ---
// Bpad is a multiple of TB. P0 = phi(input) [Bpad,n0] precomputed on host. Shared holds the
// tile's hidden/output states+errors+phi (TB*(2*(n1+n2+n3)+n1+n2) floats).
__global__ void settle_kernel_tiled(
    const float* __restrict__ P0,
    const float* __restrict__ s1_in, const float* __restrict__ s2_in, const float* __restrict__ s3_in,
    float* __restrict__ s1_out, float* __restrict__ s2_out, float* __restrict__ s3_out,
    const float* __restrict__ W0, const float* __restrict__ W1, const float* __restrict__ W2,
    const float* __restrict__ b0, const float* __restrict__ b1, const float* __restrict__ b2,
    int Bpad, int n0, int n1, int n2, int n3, int T, float lr, int clamp_output, int act)
{
    int base = blockIdx.x * TB;
    int tid = threadIdx.x, nt = blockDim.x;
    extern __shared__ float sh[];
    float* s1=sh;            float* s2=s1+TB*n1;     float* s3=s2+TB*n2;
    float* e1=s3+TB*n3;      float* e2=e1+TB*n1;     float* e3=e2+TB*n2;
    float* p1=e3+TB*n3;      float* p2=p1+TB*n1;     float* P0t=p2+TB*n2;
    for(int idx=tid; idx<TB*n1; idx+=nt) s1[idx]=s1_in[(size_t)(base+idx/n1)*n1 + idx%n1];
    for(int idx=tid; idx<TB*n2; idx+=nt) s2[idx]=s2_in[(size_t)(base+idx/n2)*n2 + idx%n2];
    for(int idx=tid; idx<TB*n3; idx+=nt) s3[idx]=s3_in[(size_t)(base+idx/n3)*n3 + idx%n3];
    // v3: cache phi(input) tile once (input is clamped -> constant across all T steps)
    for(int idx=tid; idx<TB*n0; idx+=nt) P0t[idx]=P0[(size_t)(base+idx/n0)*n0 + idx%n0];
    __syncthreads();
    for(int t=0;t<T;t++){
        for(int idx=tid; idx<TB*n1; idx+=nt) p1[idx]=actf(s1[idx],act);
        for(int idx=tid; idx<TB*n2; idx+=nt) p2[idx]=actf(s2[idx],act);
        __syncthreads();
        // pred/eps: weight element read once, applied to all TB samples
        for(int j=tid;j<n1;j+=nt){ float acc[TB];
            #pragma unroll
            for(int s=0;s<TB;s++) acc[s]=b0[j];
            const float* w=W0+(size_t)j*n0;
            for(int i=0;i<n0;i++){ float wji=w[i];
                #pragma unroll
                for(int s=0;s<TB;s++) acc[s]+=wji*P0t[s*n0+i]; }
            #pragma unroll
            for(int s=0;s<TB;s++) e1[s*n1+j]=s1[s*n1+j]-acc[s]; }
        for(int j=tid;j<n2;j+=nt){ float acc[TB];
            #pragma unroll
            for(int s=0;s<TB;s++) acc[s]=b1[j];
            const float* w=W1+(size_t)j*n1;
            for(int i=0;i<n1;i++){ float wji=w[i];
                #pragma unroll
                for(int s=0;s<TB;s++) acc[s]+=wji*p1[s*n1+i]; }
            #pragma unroll
            for(int s=0;s<TB;s++) e2[s*n2+j]=s2[s*n2+j]-acc[s]; }
        for(int j=tid;j<n3;j+=nt){ float acc[TB];
            #pragma unroll
            for(int s=0;s<TB;s++) acc[s]=b2[j];
            const float* w=W2+(size_t)j*n2;
            for(int i=0;i<n2;i++){ float wji=w[i];
                #pragma unroll
                for(int s=0;s<TB;s++) acc[s]+=wji*p2[s*n2+i]; }
            #pragma unroll
            for(int s=0;s<TB;s++) e3[s*n3+j]=s3[s*n3+j]-acc[s]; }
        __syncthreads();
        for(int i=tid;i<n1;i+=nt){ float fb[TB];
            #pragma unroll
            for(int s=0;s<TB;s++) fb[s]=0.0f;
            for(int j=0;j<n2;j++){ float wji=W1[(size_t)j*n1+i];
                #pragma unroll
                for(int s=0;s<TB;s++) fb[s]+=e2[s*n2+j]*wji; }
            #pragma unroll
            for(int s=0;s<TB;s++) s1[s*n1+i]-=lr*(e1[s*n1+i]-actd(s1[s*n1+i],act)*fb[s]); }
        for(int i=tid;i<n2;i+=nt){ float fb[TB];
            #pragma unroll
            for(int s=0;s<TB;s++) fb[s]=0.0f;
            for(int j=0;j<n3;j++){ float wji=W2[(size_t)j*n2+i];
                #pragma unroll
                for(int s=0;s<TB;s++) fb[s]+=e3[s*n3+j]*wji; }
            #pragma unroll
            for(int s=0;s<TB;s++) s2[s*n2+i]-=lr*(e2[s*n2+i]-actd(s2[s*n2+i],act)*fb[s]); }
        if(!clamp_output){ for(int idx=tid; idx<TB*n3; idx+=nt) s3[idx]-=lr*e3[idx]; }
        __syncthreads();
    }
    for(int idx=tid; idx<TB*n1; idx+=nt) s1_out[(size_t)(base+idx/n1)*n1+idx%n1]=s1[idx];
    for(int idx=tid; idx<TB*n2; idx+=nt) s2_out[(size_t)(base+idx/n2)*n2+idx%n2]=s2[idx];
    for(int idx=tid; idx<TB*n3; idx+=nt) s3_out[(size_t)(base+idx/n3)*n3+idx%n3]=s3[idx];
}

std::vector<torch::Tensor> pcn_settle_so(
    torch::Tensor s0, torch::Tensor s1, torch::Tensor s2, torch::Tensor s3,
    torch::Tensor W0, torch::Tensor W1, torch::Tensor W2,
    torch::Tensor b0, torch::Tensor b1, torch::Tensor b2,
    int64_t T, double lr, int64_t clamp_output, int64_t act)
{
    int B=s0.size(0), n0=s0.size(1), n1=s1.size(1), n2=s2.size(1), n3=s3.size(1);
    auto s1o=torch::empty_like(s1), s2o=torch::empty_like(s2), s3o=torch::empty_like(s3);
    int threads = n1>n2 ? n1 : n2; if(threads>256) threads=256; if(threads<64) threads=64;
    size_t shmem=(size_t)(n0+n1+n2+n3 + n1+n2+n3 + n0+n1+n2)*sizeof(float);
    settle_kernel<<<B, threads, shmem>>>(
        s0.data_ptr<float>(),s1.data_ptr<float>(),s2.data_ptr<float>(),s3.data_ptr<float>(),
        s1o.data_ptr<float>(),s2o.data_ptr<float>(),s3o.data_ptr<float>(),
        W0.data_ptr<float>(),W1.data_ptr<float>(),W2.data_ptr<float>(),
        b0.data_ptr<float>(),b1.data_ptr<float>(),b2.data_ptr<float>(),
        B,n0,n1,n2,n3,(int)T,(float)lr,(int)clamp_output,(int)act);
    return {s1o, s2o, s3o};
}

std::vector<torch::Tensor> pcn_settle_so_tiled(
    torch::Tensor P0, torch::Tensor s1, torch::Tensor s2, torch::Tensor s3,
    torch::Tensor W0, torch::Tensor W1, torch::Tensor W2,
    torch::Tensor b0, torch::Tensor b1, torch::Tensor b2,
    int64_t T, double lr, int64_t clamp_output, int64_t act)
{
    int Bpad=s1.size(0), n0=P0.size(1), n1=s1.size(1), n2=s2.size(1), n3=s3.size(1);
    auto s1o=torch::empty_like(s1), s2o=torch::empty_like(s2), s3o=torch::empty_like(s3);
    int threads = n1>n2 ? n1 : n2; if(threads>256) threads=256; if(threads<64) threads=64;
    // v3: + TB*n0 for the cached phi(input) tile
    size_t shmem=(size_t)(TB*(2*(n1+n2+n3) + n1+n2 + n0))*sizeof(float);
    cudaFuncSetAttribute(settle_kernel_tiled, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmem);
    int blocks = Bpad / TB;
    settle_kernel_tiled<<<blocks, threads, shmem>>>(
        P0.data_ptr<float>(),s1.data_ptr<float>(),s2.data_ptr<float>(),s3.data_ptr<float>(),
        s1o.data_ptr<float>(),s2o.data_ptr<float>(),s3o.data_ptr<float>(),
        W0.data_ptr<float>(),W1.data_ptr<float>(),W2.data_ptr<float>(),
        b0.data_ptr<float>(),b1.data_ptr<float>(),b2.data_ptr<float>(),
        Bpad,n0,n1,n2,n3,(int)T,(float)lr,(int)clamp_output,(int)act);
    return {s1o, s2o, s3o};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("pcn_settle_so", &pcn_settle_so, "fused SO settling, one block per sample (v1)");
    m.def("pcn_settle_so_tiled", &pcn_settle_so_tiled, "fused SO settling, tiled over TB samples (v2)");
}
