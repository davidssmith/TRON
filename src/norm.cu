#include <stdio.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

static int threads = 32, blocks = 512;

inline void
gpuAssert (cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) { getchar(); exit(code); }
    }
}
#define cuTry(ans) { gpuAssert((ans), __FILE__, __LINE__); }



__inline__ __device__ float
warpReduceSum (float val)
{
    for (int offset = warpSize/2; offset > 0; offset /= 2)
        val += __shfl_down(val, offset);
    return val;
}


__inline__ __device__ float
blockReduceSum (float val)
{
    static __shared__ float shared[32]; // Shared mem for 32 partial sums
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    val = warpReduceSum(val);     // Each warp performs partial reduction

    if (lane == 0)
        shared[wid] = val; // Write reduced value to shared memory

    __syncthreads();              // Wait for all partial reductions

    //read from shared memory only if that warp existed
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;

    if (wid == 0)
        val = warpReduceSum(val); //Final reduce within first warp

    return val;
}

__global__ void
deviceReduceKernel (const float *in, float* out, const int  N)
{
    float sum = 0;
    //reduce multiple elements per thread
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
        sum += in[i]*in[i];
    sum = blockReduceSum(sum);
    if (threadIdx.x==0)
        out[blockIdx.x]=sum;
}

void
deviceReduce (const float *in, float* out, const int N)
{
    deviceReduceKernel<<<blocks, threads>>>(in, out, N);
    deviceReduceKernel<<<1, threads>>>(out, out, blocks);
}

__host__ float
norm2 (const float *d_x, const size_t N)
{
    float *d_y;
    float y;
    cuTry(cudaMalloc(&d_y, N*sizeof(float)));
    cuTry(cudaMemset(d_y, 0.0, N*sizeof(float)));
    deviceReduce(d_x, d_y, N);
    cuTry(cudaMemcpy(&y, d_y, sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(d_y);
    return y;
}


// __global__ void
// normkernel (const float *in, float* out, int N)
// {
//     float sum = 0.0;
//     for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
//         sum += in[i]*in[i];
//     sum = blockReduceSum(sum);
//     if (threadIdx.x == 0)
//         atomicAdd(out, sum);
// }


__global__ void
dotkernel (const float *a, const float *b, float *out, int N)
{
    float sum = 0.0;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
        sum += a[i]*b[i];
    sum = blockReduceSum(sum);
    if (threadIdx.x == 0)
        atomicAdd(out, sum);
}


__host__ float
norm (const float *d_x, const size_t N)
{
    float *d_y;
    float y;
    cuTry(cudaMalloc(&d_y, sizeof(float)));
    cuTry(cudaMemset(d_y, 0.0, sizeof(float)));
    dotkernel<<<threads,blocks>>>(d_x, d_x, d_y, N);
    cuTry(cudaMemcpy(&y, d_y, sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(d_y);
    return y;
}

__host__ float
dot (const float *d_a, const float *d_b, const size_t N)
{
    float *d_y;
    float y;
    cuTry(cudaMalloc(&d_y, sizeof(float)));
    cuTry(cudaMemset(d_y, 0.0, sizeof(float)));
    dotkernel<<<threads,blocks>>>(d_a, d_b, d_y, N);
    cuTry(cudaMemcpy(&y, d_y, sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(d_y);
    return y;
}


__host__ float
norm (const float2 *d_x, const size_t N)
{
  return norm((float*)d_x, 2*N);
}

__host__ float
dot (const float2 *d_a, const float2 *d_b, const size_t N)
{
  return dot((float*)d_a, (float*)d_b, 2*N);
}


#ifdef _TEST_NORM

__global__ void
fillarray (float2 *d_x, const int n)
{
    for (size_t id = blockIdx.x * blockDim.x + threadIdx.x; id < n; id += blockDim.x * gridDim.x) {
        d_x[id].x = 80000000.0f; //sqrtf(id+1);
        d_x[id].y = 80000000.0f;
    }
}

int
main (int argc, char *argv[])
{
    float2 *d_x;
    float y;
    const size_t N = argc > 1 ? atoi(argv[1]) : 100;
    clock_t start, end;

    cuTry(cudaMalloc(&d_x, N*sizeof(float2)));

    threads = argc > 2 ? atoi(argv[2]) : 32;
    blocks = argc > 3 ? atoi(argv[3]) : 512;
    //blocks = min((int)((N + threads - 1) / threads), blocks);
    printf("blocks = %d\nthreads = %d\n", blocks, threads);


    fillarray<<<threads,blocks>>>(d_x, N);
    start = clock();
    y = norm(d_x, N);
    cudaDeviceSynchronize();
    end = clock();
    printf("Elapsed time: %g s\n", ((float)(end - start)) / CLOCKS_PER_SEC);
    printf("norm = %f\n", y);

    cudaFree(d_x);
}

#endif
