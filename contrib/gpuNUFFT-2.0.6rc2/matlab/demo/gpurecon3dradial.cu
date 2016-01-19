
#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
#include <err.h>
#include <errno.h>
#include <math.h>
#include <complex.h>
#include <stdint.h>
#include <cufft.h>
#include <cuda_runtime.h>
#include "float_math.h"

// GLOBAL VARIABLES
#define NSTREAMS   2
#define NCHAN      8
#define MULTI_GPU  0
#define PHI        1.9416089796736116f

// TODO: use float fmaf ( float  x, float  y, float  z ) somewhere?

inline void
gpuAssert (cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) { getchar(); exit(code); }
    }
}
#define cuTry(ans) { gpuAssert((ans), __FILE__, __LINE__); }

static const char *_cudaGetErrorEnum(cufftResult error)
{
    switch (error) {
        case CUFFT_SUCCESS: return "CUFFT_SUCCESS";
        case CUFFT_INVALID_PLAN: return "CUFFT_INVALID_PLAN";
        case CUFFT_ALLOC_FAILED: return "CUFFT_ALLOC_FAILED";
        case CUFFT_INVALID_TYPE: return "CUFFT_INVALID_TYPE";
        case CUFFT_INVALID_VALUE: return "CUFFT_INVALID_VALUE";
        case CUFFT_INTERNAL_ERROR: return "CUFFT_INTERNAL_ERROR";
        case CUFFT_EXEC_FAILED: return "CUFFT_EXEC_FAILED";
        case CUFFT_SETUP_FAILED: return "CUFFT_SETUP_FAILED";
        case CUFFT_INVALID_SIZE: return "CUFFT_INVALID_SIZE";
        case CUFFT_UNALIGNED_DATA: return "CUFFT_UNALIGNED_DATA";
        default: return "<unknown>";
    }
}

#define cufftSafeCall(err)      __cufftSafeCall(err, __FILE__, __LINE__)
inline void __cufftSafeCall (cufftResult err, const char *file, const int line)
{
    if (CUFFT_SUCCESS != err) {
        fprintf(stderr, "CUFFT error in file '%s', line %d\nerror %s: %d\nterminating!\n",__FILE__, __LINE__, \
                _cudaGetErrorEnum(err), (int)err);
        cudaDeviceReset(); 
        exit(1);
    }
}

__global__ void
fftshift (float2 *d_dst, const float2* __restrict__ d_src, const int n, const int nchan)
{
    float2 tmp;
    int dn = n / 2;
    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < dn*dn; id += blockDim.x * gridDim.x)
    {
        int x = id / dn;
        int y = id % dn;
        int id1 = x*n + y;
        int id2 = (x + dn)*n + y;
        int id3 = (x + dn)*n + y + dn;
        int id4 = x*n + y + dn;
        for (int c = 0; c < nchan; ++c) {
            tmp = d_dst[id1*nchan + c]; // 1 <-> 3
            d_dst[id1*nchan + c] = d_dst[id3*nchan + c];
            d_dst[id3*nchan + c] = tmp;
            tmp = d_dst[id2*nchan + c]; // 2 <-> 4
            d_dst[id2*nchan + c] = d_dst[id4*nchan + c];
            d_dst[id4*nchan + c] = tmp;
        }
    }
}

/* Singular Value Decomposition: A = U s V'
   A is destroyed.  m must be >= n. If smaller, A should be filled with zero rows.
   Code is adapted from Collected Algorithms from ACM, Algorithm 358.
   U' is applied to the p vectors given in columns n, n+1, ..., n+p-1 of matrix A
   nu and nv specify the number of columns of U and V to calculate. Zero is ok and faster.
   See: http://www.scs.fsu.edu/~burkardt/f77_src/toms358/toms358.f */
__host__ __device__ void 
csvd (float2 *A, const int m, const int n, const int p,
        const int nu, const int nv, float *s, float2 *U, float2 *V)
{
    // U: [m x nu]   V: [n x nv]   A: [m x (n+p)]  s: [m] 
    float b[NCHAN], c[NCHAN], t[NCHAN];
    float cs, eps, eta, f, g, h, sn, tol, w, x, y, z;
    int i, j, k, k1, L, L1, nM1, np;
    float2 q, r;
    eta = 1.5e-7f;            /* eta = the relative machine precision */
    tol = 1.5e-30f;            /* tol = the smallest normalized positive number, divided by eta */
    np = n + p;
    nM1 = n - 1;
    L = 0;
    /* HOUSEHOLDER REDUCTION */
    c[0] = 0.f;
    k = 0;
    while (1)
    {
        k1 = k + 1;
        /* ELIMINATION OF A[i][k], i = k, ..., m-1 */
        z = 0.f;
        for (i = k; i < m; i++)
            z += norm(A[i*np + k]);
        b[k] = 0.f;
        if (z > tol)
        {
            z = sqrtf(z);
            b[k] = z;
            w = abs(A[k*np + k]);
            q = make_float2(1.f, 0.f);
            if (w != 0.f) q = A[k*np + k] / w;
            A[k*np + k] = q * (z + w);
            if (k != np - 1) {
                for (j = k1; j < np; j++) {
                    q = make_float2(0.f, 0.f);
                    for (i = k; i < m; i++)
                        q += conj(A[i*np + k]) * A[i*np + j];
                    q /= z * (z + w);
                    for (i = k; i < m; i++)
                        A[i*np + j] -= q * A[i*np + k];
                }
            }
            /* PHASE TRANSFORMATION */
            q = conj(A[k*np + k]) / (-abs(A[k*np + k]));
            for (j = k1; j < np; j++)
                A[k*np + j] *= q;
        }
        /* ELIMINATION OF A[k][j], j = k+2, ..., n-1 */
        if (k == nM1) break;
        z = 0.0F;
        for (j = k1; j < n; j++)
            z += norm(A[k*np + j]);
        c[k1] = 0.0F;
        if (z > tol)
        {
            z = sqrtf(z);
            c[k1] = z;
            w = abs(A[k*np + k1]);
            q = make_float2(1.f, 0.f);
            if (w != 0.f) q = A[k*np + k1] / w;
            A[k*np + k1] = q * (z + w);
            for (i = k1; i < m; i++) {
                q = make_float2(0.f, 0.f);
                for (j = k1; j < n; j++)
                    q = q + conj(A[k*np + j]) * A[i*np + j];
                q /= z * (z + w);
                for (j = k1; j < n; j++)
                    A[i*np + j] -= q * A[k*np  +j];
            }
            /* PHASE TRANSFORMATION */
            q = conj(A[k*np + k1]) / (-abs(A[k*np + k1]));
            for (i = k1; i < m; i++)
                A[i*np + k1] *= q;
        }
        k = k1;
    }
    /* TOLERANCE FOR NEGLIGIBLE ELEMENTS */
    eps = 0.f;
    for (k = 0; k < n; k++) {
        s[k] = b[k];
        t[k] = c[k];
        if (s[k] + t[k] > eps)
            eps = s[k] + t[k];
    }
    eps *= eta;
    /* INITIALIZATION OF u AND v */
    for (j = 0; j < m*nu; j++)
        U[j] = make_float2(0.f, 0.f);
    for (j = 0; j < nu; j++)
        U[j*nu + j] = make_float2(1.f, 0.f);
    for (j = 0; j < n*nv; j++)
        V[j] = make_float2(0.f, 0.f);
    for (j = 0; j < nv; j++)
        V[j*nv + j] = make_float2(1.f, 0.f);

    /* QR DIAGONALIZATION */
    for (k = nM1; k >= 0; k--)
    {
        /* TEST FOR SPLIT */
        while (1)
        {
            for (L = k; L >= 0; L--) {
                if (fabsf(t[L]) <= eps) goto Test;
                if (fabsf(s[L - 1]) <= eps) break;
            }
            /* CANCELLATION OF E(L) */
            cs = 0.0f;
            sn = 1.0f;
            L1 = L - 1;
            for (i = L; i <= k; i++)
            {
                f = sn * t[i];
                t[i] *= cs;
                if (fabsf(f) <= eps) goto Test;
                h = s[i];
                w = sqrtf(f * f + h * h);
                s[i] = w;
                cs = h / w;
                sn = -f / w;
                for (j = 0; nu > 0 && j < n; j++) {
                    x = U[j*nu + L1].x;
                    y = U[j*nu + i].x;
                    U[j*nu + L1].x = x * cs + y * sn;
                    U[j*nu + i].x = y * cs - x * sn;
                }
                if (np == n) continue;
                for (j = n; j < np; j++) {
                    q = A[L1*np + j];
                    r = A[i*np + j];
                    A[L1*np + j] = q * cs + r * sn;
                    A[i*np + j] = r * cs - q * sn;
                }
            }
            /* TEST FOR CONVERGENCE */
    Test:    w = s[k];
            if (L == k) break;
            /* ORIGIN SHIFT */
            x = s[L];
            y = s[k - 1];
            g = t[k - 1];
            h = t[k];
            f = ((y - w) * (y + w) + (g - h) * (g + h)) / (2.f * h * y);
            g = sqrtf(f * f + 1.f);
            if (f < 0.f) g = -g;
            f = ((x - w) * (x + w) + (y / (f + g) - h) * h) / x;
            /* QR STEP */
            cs = 1.f;
            sn = 1.f;
            L1 = L + 1;
            for (i = L1; i <= k; i++)
            {
                g = t[i];
                y = s[i];
                h = sn * g;
                g = cs * g;
                w = sqrtf(h * h + f * f);
                t[i - 1] = w;
                cs = f / w;
                sn = h / w;
                f = x * cs + g * sn;
                g = g * cs - x * sn;
                h = y * sn;
                y = y * cs;
                for (j = 0; nv > 0 && j < n; j++) {
                    x = V[j*nv + i - 1].x;
                    w = V[j*nv + i].x;
                    V[j*nv + i - 1].x = x * cs + w * sn;
                    V[j*nv + i].x = w * cs - x * sn;
                }
                w = sqrtf(h * h + f * f);
                s[i - 1] = w;
                cs = f / w;
                sn = h / w;
                f = cs * g + sn * y;
                x = cs * y - sn * g;
                for (j = 0; nu > 0 && j < n; j++) {
                    y = U[j*nu + i - 1].x;
                    w = U[j*nu + i].x;
                    U[j*nu + i - 1].x = y * cs + w * sn;
                    U[j*nu + i].x = w * cs - y * sn;
                }
                if (n == np) continue;
                for (j = n; j < np; j++) {
                    q = A[(i - 1)*np + j];
                    r = A[i*np + j];
                    A[(i - 1)*np + j] = q * cs + r * sn;
                    A[i*np + j] = r * cs - q * sn;
                }
            }
            t[L] = 0.f;
            t[k] = f;
            s[k] = x;
        }
        /* CONVERGENCE */
        if (w >= 0.f) continue;
        s[k] = -w;
        if (nv == 0) continue;
        for (j = 0; j < n; j++)
            V[j*nv + k] = -V[j*nv + k];
    }
    /* SORT SINGULAR VALUES */
    for (k = 0; k < n; k++)    /* sort descending */
    {
        g = -1.f;
        j = k;
        for (i = k; i < n; i++) {    /* sort descending */
            if (s[i] <= g) continue;
            g = s[i];
            j = i;
        }
        if (j == k) continue;
        s[j] = s[k];
        s[k] = g;
        for (i = 0; nv > 0 && i < n; i++) {
            q = V[i*nv + j];
            V[i*nv + j] = V[i*nv + k];
            V[i*nv + k] = q;
        }
        for (i = 0; nu > 0 && i < n; i++) {
            q = U[i*nu + j];
            U[i*nu + j] = U[i*nu + k];
            U[i*nu + k] = q;
        }
        if (n == np) continue;
        for (i = n; i < np; i++) {
            q = A[j*np + i];
            A[j*np + i] = A[k*np + i];
            A[k*np + i] = q;
        }
    }
    /* BACK TRANSFORMATION */
    for (k = nM1; nu > 0 && k >= 0; k--)
    {
        if (b[k] == 0.f) continue;
        q = -A[k*np + k] / abs(A[k*np + k]);
        for (j = 0; j < nu; j++)
            U[k*nu + j] *= q;
        for (j = 0; j < nu; j++) {
            q = make_float2(0.f, 0.f);
            for (i = k; i < m; i++)
                q = q + conj(A[i*np + k]) * U[i*nu + j];
            q /= abs(A[k*np + k]) * b[k];
            for (i = k; i < m; i++)
                U[i*nu + j] -= q * A[i*np + k];
        }
    }
    for (k = n - 2; nv > 0 && n > 1 && k >= 0; k--)
    {
        k1 = k + 1;
        if (c[k1] == 0.f) continue;
        q = conj(A[k*np + k1]) / (-abs(A[k*np + k1]));
        for (j = 0; j < nv; j++)
            V[k1*nv + j] *= q;
        for (j = 0; j < nv; j++) {
            q = make_float2(0.f, 0.f);
            for (i = k1; i < n; i++)
                q = q + A[k*np + i] * V[i*nv + j];
            q /= (abs(A[k*np + k1]) * c[k1]);
            for (i = k1; i < n; i++)
                V[i*nv + j] -= q * conj(A[k*np + i]);
        }
    }
} 


__host__ __device__ void
pinv (float2 *Ainv, float2 *A, const int n)
{
    //  A = U S V'    A^-1 = V S^-1 U'
    float2 U[NCHAN*NCHAN], V[NCHAN*NCHAN];
    float s[NCHAN];
    csvd(A, n, n, 0, n, n, s, U, V);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            U[i*n + j] /= s[j];  // U x S^-1
    for (int i = 0; i < n*n; ++i)
        Ainv[i] = make_float2(0.f,0.f);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            for (int k = 0; k < n; ++k)
                Ainv[i*n + j] += V[i*n + k] * conj(U[j*n + k]);  // V * U'
}

__global__ void
coilcombinesos (float2 *d_img, const float2 * __restrict__ d_coilimg, const int nimg, const int nchan)
{
    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < nimg*nimg; id += blockDim.x * gridDim.x) {
        float val = 0.f;
        for (int k = nchan*id; k < nchan*(id+1); ++k)
            val += norm(d_coilimg[k]);
        d_img[id].x = sqrtf(val);
        d_img[id].y = 0.f;
    }
}

__global__ void
coilcombinewalsh (float2 *d_img, float2 *d_b1, const float2 * __restrict__ d_coilimg, const int nimg, const int nchan, const int npatch)
{
    float2 A[NCHAN*(NCHAN+1)], U[NCHAN*NCHAN];
    float s[NCHAN];
    const int n = NCHAN + 1;
    for (size_t id = blockIdx.x * blockDim.x + threadIdx.x; id < nimg*nimg; id += blockDim.x * gridDim.x)
    {
        for (int c1 = 0; c1 < NCHAN; ++c1) {
            for (int c2 = 0; c2 < NCHAN; ++c2)
                A[c1*n + c2] = make_float2(0.f,0.f);
            A[c1*n + NCHAN] = d_coilimg[nchan*id+c1];
        }
        int x = id / nimg;
        int y = id % nimg;
        for (int px = max(0,x-npatch); px <= min(nimg-1,x+npatch); ++px)
            for (int py = max(0,y-npatch); py <= min(nimg-1,y+npatch); ++py)
            {
                int offset = nchan*(px*nimg + py);
                for (int c2 = 0; c2 < NCHAN; ++c2)
#pragma unroll 6
    // TODO: optimize this loop
                    for (int c1 = 0; c1 < NCHAN; ++c1)
                        A[c1*n + c2] += d_coilimg[offset+c1]*conj(d_coilimg[offset+c2]);
            }
        csvd(A, NCHAN, NCHAN, 1, NCHAN, 0, s, U, NULL);
        //csvd (float2 *A, const int m, const int n, const int p,
                //const int nu, const int nv, float *s, float2 *U, float2 *V)
        //float maxphase = cargf(d_coilimg[nchan*id]); // assume coil 0 is brightest
        d_img[id] = A[0*n + NCHAN]; // * cexpf(-maxphase);
        for (int c = 0; c < NCHAN; ++c) {
            d_b1[nchan*id + c] = sqrtf(s[0])*U[nchan*c];
        }
    }
}

__host__ __device__ float
i0f (const float x)
{
    if (x == 0.f) return 1.f;
    float z = x * x;
    float num = (z* (z* (z* (z* (z* (z* (z* (z* (z* (z* (z* (z* (z*
        (z* 0.210580722890567e-22  + 0.380715242345326e-19 ) +
        0.479440257548300e-16) + 0.435125971262668e-13 ) +
        0.300931127112960e-10) + 0.160224679395361e-7  ) +
        0.654858370096785e-5)  + 0.202591084143397e-2  ) +
        0.463076284721000e0)   + 0.754337328948189e2   ) +
        0.830792541809429e4)   + 0.571661130563785e6   ) +
        0.216415572361227e8)   + 0.356644482244025e9   ) +
        0.144048298227235e10);
    float den = (z*(z*(z-0.307646912682801e4)+
        0.347626332405882e7)-0.144048298227235e10);
    return -num/den;
}


__host__ __device__ inline float
gridkernel (const float dx, const float dy, const float dz)
{
    float r2 = dx*dx + dy*dy + dz*dz;
#ifdef KERN_KB
    const float kernwidth = 3.f;
    const float osfactor = 1.5f;
#define SQR(x) ((x)*(x))
#define BETA (M_PI*sqrtf(SQR(kernwidth/osfactor*(osfactor-0.5))-0.8))
    return r2 < kernwidth*kernwidth ? i0f(BETA * sqrtf (1.f - r2/kernwidth/kernwidth)) / i0f(BETA): 0.f;
#else
    const float sigma = 0.33f; // ballparked from Jackson et al. 1991. IEEE TMI, 10(3), 473–8
    return expf(-0.5f*r2/sigma/sigma);
#endif
}

__host__ __device__ inline float
modang (const float x)   /* rescale arbitrary angles to [0,2PI] interval */
{
    const float TWOPI = 2.f*M_PI;
    float y = fmodf(x, TWOPI);
    return y < 0.f ? y + TWOPI : y;
}

__host__ void
fillapod (float2 *d_apod, const int n, const float kernwidth)
{
    const size_t d_imgsize = n*n*sizeof(float2);
    float2 *h_apod = (float2*)malloc(d_imgsize);
    int w = int(kernwidth);

    for (int k = 0; k < n*n; ++k)
        h_apod[k] = make_float2(0.f,0.f);
    for (int x = 0; x < w; ++x) {
        for (int y = 0; y < w; ++y)
            h_apod[n*x + y].x = gridkernel(x, y);
        for (int y = n-w; y < n; ++y)
            h_apod[n*x + y].x = gridkernel(x, n-y);
    }
    for (int x = n-w; x < n; ++x) {
        for (int y = 0; y < w; ++y)
            h_apod[n*x + y].x = gridkernel(n-x, y);
        for (int y = n-w; y < n; ++y)
            h_apod[n*x + y].x = gridkernel(n-x, n-y);
    }
    cuTry(cudaMemcpy(d_apod, h_apod, d_imgsize, cudaMemcpyHostToDevice));
    cufftHandle inverse_plan_apod;
    cufftSafeCall(cufftPlan2d(&inverse_plan_apod, n, n, CUFFT_C2C));
    cufftSafeCall(cufftExecC2C(inverse_plan_apod, d_apod, d_apod, CUFFT_INVERSE));
    fftshift<<<n,n>>>(d_apod, d_apod, n, 1);
    cuTry(cudaMemcpy(h_apod, d_apod, d_imgsize, cudaMemcpyDeviceToHost));

    float maxval = 0.f;
    for (int k = 0; k < n*n; ++k) { // take magnitude and find brightest pixel at same time
        float mag = abs(h_apod[k]);
        h_apod[k] = make_float2(mag);
        maxval = mag > maxval ? mag : maxval;
    }
    for (int k = 0; k < n*n; ++k) { // normalize it
        h_apod[k].x /= maxval;
        h_apod[k].x = h_apod[k].x > 0.1f ? 1.0f / h_apod[k].x : 1.0f;
    }
    cuTry(cudaMemcpy(d_apod, h_apod, d_imgsize, cudaMemcpyHostToDevice));
    free(h_apod);
}

__global__ void
deapodize (float2 *d_img, const float2 * __restrict__ d_apod, const int nimg, const int nchan)
{
    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < nimg*nimg; id += blockDim.x * gridDim.x)
        for (int c = 0; c < nchan; ++c)
            d_img[nchan*id+c] *= d_apod[id].x; // took magnitude prior
}

__global__ void
b1map (float2 *d_b1, const float2 *__restrict__ d_coilimgs, const float2 *__restrict__ d_img, 
    const int n, const int nchan)
{
    // TODO: massively downsample this before dividing
    float t = abs(d_coilimgs[0]);
    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < n*n; id += blockDim.x * gridDim.x)
    {
        float m = abs(d_img[id]);
        for (int c = 0; c < nchan; ++c) {
            d_b1[nchan*id + c].x = m > 2.f*t ? abs(d_coilimgs[nchan*id + c]) / m : 0.f;
            d_b1[nchan*id + c].y = 0.f;
        }
    }
}


__global__ void
crop (float2* d_dst, const int ndst, const float2* __restrict__ d_src, const int nsrc, const int nchan)
{
    const int w = (nsrc - ndst) / 2;
    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < ndst*ndst; id += blockDim.x * gridDim.x)
    {
        int xdst = id / ndst;
        int ydst = id % ndst;
        int srcid = (xdst + w)*nsrc + ydst + w;
        for (int c = 0; c < nchan; ++c)
            d_dst[nchan*id + c] = d_src[nchan*srcid + c];
    }
}


// TODO: delete this?
__global__ void
zeroouter (float2* d_arr, const int n, const int nchan, const int margin)
{
    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < n*n; id += blockDim.x * gridDim.x)
    {
        int x = id / n;
        int y = id % n;
        if (x < margin || x >= n - margin || y < margin || y >= n - margin)
#pragma unroll 6
            for (int c = 0; c < nchan; ++c)
                d_arr[nchan*id + c] = make_float2(0.f, 0.f);
    }
}


__global__ void
gridradial2d (
    float2 *udata, const float2 * __restrict__ nudata, const int ngrid,
    const int nchan, const int nro, const int npe, const float kernwidth, const int peskip)
{
    // udata: [NCHAN x NGRID x NGRID], nudata: NCHAN x NRO x NPE
    float osfactor = float(ngrid) / float(nro); // oversampling factor

    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < ngrid*ngrid; id += blockDim.x * gridDim.x)
    {
        int x = id / ngrid - ngrid/2;
        int y = -(id % ngrid) + ngrid/2;
        float myradius = hypotf(x, y);
        int rmax = fminf(floorf(myradius + kernwidth)/osfactor, nro/2-1);  // Note: these lie on [-NGRID/2 ... NGRID/2-1]
        int rmin = fmaxf(ceilf(myradius - kernwidth)/osfactor, 0);  // define a circular band around the uniform point
        for (int ch = 0; ch < nchan; ++ch)
             udata[nchan*id + ch] = make_float2(0.f,0.f);
        if (rmin > nro/2-1) continue; // outside non-uniform data area

        float sdc = 0.f;

        float mytheta = modang(atan2f(float(y),float(x))); // get uniform point coordinate in non-uniform system, (r,theta) in this case
        float dtheta = atan2f(kernwidth, myradius); // narrow that band to an arc

        // TODO: replace this logic with boolean function that can be swapped out for diff acquisitions
        for (int pe = 0; pe < npe; ++pe)
        {
            float theta = modang(PHI * float(pe + peskip));
            float dtheta1 = fabsf(theta - mytheta);
            float dtheta2 = fabsf(modang(theta + M_PI) - mytheta);
            if (dtheta1 < dtheta || dtheta2 < dtheta)
            {  
                float sf, cf;
                __sincosf(theta, &sf, &cf);
                sf *= osfactor;
                cf *= osfactor;
                int rstart = dtheta1 < dtheta ? rmin : -rmax;
                int rend   = dtheta1 < dtheta ? rmax : -rmin;
                for (int r = rstart; r <= rend; ++r)  // for each POSITIVE non-uniform ro point
                {
                    float kx = r* cf; // [-NGRID/2 ... NGRID/2-1]    // TODO: compute distance in radial coordinates?
                    float ky = r* sf; // [-NGRID/2 ... NGRID/2-1]
                    float wgt = gridkernel(kx - x, ky - y);
#pragma unroll 6 
                    for (int ch = 0; ch < nchan; ++ch) {
                        float2 c = nudata[nchan*(nro*pe + r + nro/2) + ch];
                        udata[nchan*id + ch].x += wgt*c.x;
                        udata[nchan*id + ch].y += wgt*c.y;
                        sdc += wgt;
                    }
                }
            }
        }
        for (int ch = 0; sdc > 0.f && ch < nchan; ++ch) {
            udata[nchan*id + ch].x /= sdc;
            udata[nchan*id + ch].y /= sdc;
        }
    }
}


__global__ void
degridradial2d (
    float2 *nudata, const float2 * __restrict__ udata, const int ngrid,
    const int nchan, const int nro, const int npe, const float kernwidth, const int peskip)
{
    // udata: [NCHAN x NGRID x NGRID], nudata: NCHAN x NRO x NPE
    float osfactor = float(ngrid) / float(nro); // oversampling factor

    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < nro*npe; id += blockDim.x * gridDim.x)
    {
        int pe = id / nro; // find my location in the non-uniform data
        int ro = id % nro;
        float r = ro - 0.5f * nro; // convert indices to (r,theta) coordinates
#ifdef LINEAR_ANGLE
        float t = M_PI * pe / float(npe);
#else
        float t = modang(PHI*(pe + peskip)); // golden angle specific!
#endif
        float kx = r*cos(t); // Cartesian freqs of non-Cart datum  // TODO: _sincosf?
        float ky = r*sin(t);
        float x = osfactor*( kx + 0.5f * nro);  // (x,y) coordinates in grid units
        float y = osfactor*(-ky + 0.5f * nro);

        for (int ch = 0; ch < nchan; ++ch) // zero my elements
             nudata[nchan*id + ch] = make_float2(0.f,0.f);
        for (int ux = fmaxf(0.f,x-kernwidth); ux <= fminf(ngrid-1,x+kernwidth); ++ux)
        for (int uy = fmaxf(0.f,y-kernwidth); uy <= fminf(ngrid-1,y+kernwidth); ++uy)
        {
            float wgt = gridkernel(ux - x, uy - y);
#pragma unroll 6 
            for (int ch = 0; ch < nchan; ++ch) {
                float2 c = udata[nchan*(ux*ngrid + uy) + ch];
                nudata[nchan*id + ch].x += wgt*c.x;
                nudata[nchan*id + ch].y += wgt*c.y;
            }
        }
    }
}

extern "C" {  // don't mangle name, so can call from other languages

__host__ void
recongar2d (float2 *h_img, float2 *h_b1, const float2 *__restrict__ h_nudata,
    int nchan, const int nro, const int npe, const int ngrid, const int nimg,
    const int nslices, const int npe_per_slice, const int dpe, const int peskip,
    const float oversamp, const float kernwidth)
{
    float2 *d_nudata[NSTREAMS], *d_udata[NSTREAMS], *d_coilimg[NSTREAMS], 
        *d_img[NSTREAMS], *d_b1[NSTREAMS], *d_tv[NSTREAMS], *d_apodos[NSTREAMS], *d_apod[NSTREAMS];
    cudaStream_t stream[NSTREAMS];

    int ndevices;
    if (MULTI_GPU) {
    	cuTry(cudaGetDeviceCount(&ndevices));
    } else
    	ndevices = 1;
    printf("using %d CUDA devices\n", ndevices);

    int blocksize = 96;    // CUDA kernel parameters, TWEAK HERE to optimize
    int gridsize = ngrid*nchan;
    printf("kernels configured with %d blocks of %d threads\n", gridsize, blocksize);

    // array sizes
    const size_t d_nudatasize = nchan*nro*npe_per_slice*sizeof(float2);  // input data
    const size_t d_udatasize = nchan*ngrid*ngrid*sizeof(float2); // gridded data
    const size_t d_coilimgsize = nchan*nimg*nimg*sizeof(float2); // coil images
    const size_t d_imgsize = nimg*nimg*sizeof(float2); // coil-combined image
    const size_t d_gridsize = ngrid*ngrid*sizeof(float2);

    // setup FFT
    cufftHandle inverse_plan[NSTREAMS];
    const int rank = 2;
    int idist = 1, odist = 1, istride = nchan, ostride = nchan;
    int n[2] = {ngrid, ngrid};
    int inembed[]  = {ngrid, ngrid};
    int onembed[]  = {ngrid, ngrid};

    for (int j = 0; j < NSTREAMS; ++j) // allocate data and initialize apodization and kernel texture
    {
        if (MULTI_GPU) cudaSetDevice(j % ndevices);
        cuTry(cudaStreamCreate(&stream[j]));
        cufftSafeCall(cufftPlanMany(&inverse_plan[j], rank, n, onembed, ostride, odist,
            inembed, istride, idist, CUFFT_C2C, nchan));
        cufftSafeCall(cufftSetStream(inverse_plan[j], stream[j]));
        cuTry(cudaMalloc((void **)&d_nudata[j], d_nudatasize));
        cuTry(cudaMalloc((void **)&d_udata[j], d_udatasize));
        cuTry(cudaMalloc((void **)&d_coilimg[j], d_coilimgsize));
        cuTry(cudaMalloc((void **)&d_b1[j], d_coilimgsize));
        cuTry(cudaMalloc((void **)&d_img[j], d_imgsize));
        cuTry(cudaMalloc((void **)&d_tv[j], d_imgsize));
        cuTry(cudaMalloc((void **)&d_apodos[j], d_gridsize));
        cuTry(cudaMalloc((void **)&d_apod[j], d_imgsize));
        fillapod(d_apodos[j], ngrid, kernwidth);
        crop<<<nimg,nimg>>>(d_apod[j], nimg, d_apodos[j], ngrid, 1);
        cuTry(cudaFree(d_apodos[j]));
    }

    for (int s = 0; s < nslices; s += NSTREAMS) // MAIN SLICE LOOP
        for (int j = 0; j < min(NSTREAMS,nslices-s); ++j)
        {
            if (MULTI_GPU) cudaSetDevice(j % ndevices);
            int peskip = (s+j)*dpe;
            printf("[dev %d, stream %d] reconstructing slice %d/%d from PEs %d-%d\n", j%ndevices, j, s+j+1, nslices, (s+j)*dpe, (s+j+1)*dpe-1);
            cuTry(cudaMemcpyAsync(d_nudata[j], h_nudata + nchan*nro*peskip, d_nudatasize, cudaMemcpyHostToDevice, stream[j]));
            gridradial2d<<<gridsize,blocksize,0,stream[j]>>>(d_udata[j], d_nudata[j], ngrid, nchan, nro, npe_per_slice, kernwidth, peskip);
            //degridradial2d<<<gridsize,blocksize,0,stream[j]>>>(d_nudata[j], d_udata[j], ngrid, nchan, nro, npe_per_slice, kernwidth, peskip);
            //gridradial2d<<<gridsize,blocksize,0,stream[j]>>>(d_udata[j], d_nudata[j], ngrid, nchan, nro, npe_per_slice, kernwidth, peskip);
            //cuTry(cudaMemcpyAsync(d_b1map[j], d_udata[j], d_udatasize, cudaMemcpyDeviceToDevice, stream[j]));

            // uniform data -> image
            fftshift<<<gridsize,blocksize,0,stream[j]>>>(d_udata[j], d_udata[j], ngrid, nchan);
            cufftSafeCall(cufftExecC2C(inverse_plan[j], d_udata[j], d_udata[j], CUFFT_INVERSE));
            fftshift<<<gridsize,blocksize,0,stream[j]>>>(d_udata[j], d_udata[j], ngrid, nchan);
            crop<<<gridsize,blocksize,0,stream[j]>>>(d_coilimg[j], nimg, d_udata[j], ngrid, nchan);
            //coilcombinesos<<<gridsize,blocksize,0,stream[j]>>>(d_img[j], d_coilimg[j], nimg, nchan);
            coilcombinewalsh<<<gridsize,blocksize,0,stream[j]>>>(d_img[j], d_b1[j], d_coilimg[j], nimg, nchan, 1); // 0 works, 1 good, 3 optimal
            deapodize<<<gridsize,blocksize,0,stream[j]>>>(d_img[j], d_apod[j], nimg, 1);

            //tv<<<gridsize,blocksize,0,stream[j]>>>(d_tv[j], d_img[j], nimg, 1);
            //cuTry(cudaMemcpyAsync(h_img + (s+j)*nimg*nimg*nchan, d_b1map[j], d_coilimgsize, cudaMemcpyDeviceToHost, stream[j]));
            cuTry(cudaMemcpyAsync(h_img + (s+j)*nimg*nimg, d_img[j], d_imgsize, cudaMemcpyDeviceToHost, stream[j]));
            cuTry(cudaMemcpyAsync(h_b1 + (s+j)*nimg*nimg*nchan, d_b1[j], d_coilimgsize, cudaMemcpyDeviceToHost, stream[j]));
        }

    printf("freeing device memory\n");
    for (int j = 0; j < NSTREAMS; ++j) { // free allocated memory
        if (MULTI_GPU) cudaSetDevice(j % ndevices);
        cuTry(cudaFree(d_nudata[j]));
        cuTry(cudaFree(d_udata[j]));
        cuTry(cudaFree(d_coilimg[j]));
        cuTry(cudaFree(d_b1[j]));
        cuTry(cudaFree(d_img[j]));
        cuTry(cudaFree(d_apod[j]));
        cudaStreamDestroy(stream[j]);
    }
}
}

int
main ()
{
    // for testing
    float2 *h_nudata, *h_img, *h_b1;
    const int nro = 207;
    const int npe = 3182;
    int nchan = NCHAN;
    const float oversamp = 1.f;
    const float kernwidth = 2.f;
    const int ngrid = nro*oversamp;
    const int nimg = 2*nro/4;
    const int dpe = 21;
    const int npe_per_slice = nro/2;
    const int nslices = (npe - npe_per_slice) / dpe;
    const int peskip = 0;

    // allocate pinned memory, which allows async calls
#ifdef CUDA_HOST_MALLOC
    cuTry(cudaMallocHost((void**)&h_nudata, nchan*nro*npe*sizeof(float2)));
    cuTry(cudaMallocHost((void**)&h_img, nchan*nimg*nimg*sizeof(float2)));
    cuTry(cudaMallocHost((void**)&h_b1, nchan*nimg*nimg*nslices*sizeof(float2)));
#else
    h_nudata = (float2*)malloc(nchan*nro*npe*sizeof(float2));
    h_img = (float2*)malloc(nchan*nimg*nimg*sizeof(float2));
    h_b1 = (float2*)malloc(nchan*nimg*nimg*nslices*sizeof(float2));
#endif

    printf("read non-uniform data from data.fld\n");
    int infd = open("/tmp/data.fld",O_RDONLY);
    read(infd, h_nudata, sizeof(float2)*nchan*nro*npe);
    printf("sanity check: nudata[0] = %f + %f i\n", h_nudata[0].x, h_nudata[0].y);
    int imgfd = open("/tmp/img.fld",O_WRONLY|O_CREAT,0644);
    if (imgfd == -1) err(errno, "output file");
    int b1fd = open("/tmp/b1.fld",O_WRONLY|O_CREAT,0644);
    if (b1fd == -1) err(errno, "B1 file");

    // the magic happens
    // coilcompress(h_nudata, etc.)
    recongar2d(h_img, h_b1, h_nudata, nchan, nro, npe, ngrid, nimg, nslices, npe_per_slice,
        dpe, peskip, oversamp, kernwidth);

    // save results
    write(b1fd, h_b1, sizeof(float2)*nimg*nimg*nslices*nchan);
    printf("write result to img.fld\n");
    nchan = 1;  // TODO: specify number of channels to finish with
    printf("img[0]: %f + %f i\n", h_img[0].x, h_img[0].y);
    FILE *hdr = fopen("/tmp/img.hdr","w");
    fprintf(hdr, "%d %d %d %d 1\n", nchan, nimg, nimg, nslices);
    fclose(hdr);
    write(imgfd, h_img, sizeof(float2)*nimg*nimg*nslices*1);

    printf("free host memory\n");
#ifdef CUDA_HOST_MALLOC
    cudaFreeHost(&h_nudata);
    cudaFreeHost(&h_img);
    cudaFreeHost(&h_b1);
#else
    free(h_nudata);
    free(h_img);
    free(h_b1);
#endif
    cudaDeviceReset();
    close(infd);
    close(imgfd);
    close(b1fd);

    return 0;
}

