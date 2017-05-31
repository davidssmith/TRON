/*
  This file is part of the TRON package (http://github.com/davidssmith/TRON).

  The MIT License (MIT)

  Copyright (c) 2016-2017 David Smith

  Permission is hereby granted, free of charge, to any person obtaining a # copy
  of this software and associated documentation files (the "Software"), to # deal
  in the Software without restriction, including without limitation the # rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or # sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included # in all
  copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS # OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL # THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING # FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS # IN THE
  SOFTWARE.
*/

#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
#include <err.h>
#include <errno.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#include <time.h>
#include <stdint.h>
#include <cufft.h>
#include <cuda_runtime.h>

#include "float2math.h"
#include "mri.h"
#include "ra.h"
#include "tron.h"

#define MAX(a,b) ((a)>(b)?(a):(b))
#define DPRINT if(flags.verbose)printf
#define dprint(expr,fmt)  do{ if(flags.verbose)fprintf(stderr,"\e[90m%d: " #expr " = %" #fmt "\e[0m\n", __LINE__, expr); }while(0);

// MISC GLOBAL VARIABLES
static cufftHandle fft_plan[NSTREAMS], fft_plan_os[NSTREAMS];
static cudaStream_t stream[NSTREAMS];
static int ndevices;

// DEVICE ARRAYS AND SIZES
static float2 *d_nudata[NSTREAMS], *d_udata[NSTREAMS], *d_apod[NSTREAMS], *d_apodos[NSTREAMS], *d_coilimg[NSTREAMS], *d_img[NSTREAMS];
static size_t d_nudatasize; // size in bytes of non-uniform data
static size_t d_udatasize; // size in bytes of gridded data
static size_t d_coilimgsize; // multi-coil image size
static size_t d_imgsize; // coil-combined image size
static size_t h_outdatasize;

// RECON CONFIGURATION
static float grid_oversamp = 2.f;  // TODO: compute ngrid from nx, ny and oversamp
static float kernwidth = 1.f;
static float data_undersamp = 1.f;

static int prof_slide = 0;         // # of profiles to slide through the data between reconstructed images
static int skip_angles = 0;        // # of angles to skip at beginning of image stack

static int nc;  //  # of receive channels;
static int nt;  // # of repeated measurements of same trajectory
static int nro, npe1, npe2, npe1work;//, npe2work;  // radial readout and phase encodes
static int nx, ny, nz, nxos, nyos, nzos;  // Cartesian dimensions of uniform data

static struct {
    unsigned adjoint       : 1;
    unsigned postcomp      : 1;
    unsigned deapodize     : 1;
    unsigned koosh         : 1;
    unsigned verbose       : 1;
    unsigned golden_angle  : 3;   // padded to 8 bits
} flags = {0, 0, 1, 0, 0, 0};

// CONSTANTS
static const float PHI = 1.9416089796736116f;

inline void
gpuAssert (cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s in %s at L%d\n", cudaGetErrorString(code), file, line);
        if (abort) { getchar(); exit(code); }
    }
}
#define cuTry(ans) { gpuAssert((ans), __FILE__, __LINE__); }

static const char *
_cudaGetErrorEnum(cufftResult error)
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

#define cufftSafeCall(err)  __cufftSafeCall(err, __FILE__, __LINE__)
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
fftshift (float2 *dst, const int n, const int nchan)
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
            tmp = dst[id1*nchan + c]; // 1 <-> 3
            dst[id1*nchan + c] = dst[id3*nchan + c];
            dst[id3*nchan + c] = tmp;
            tmp = dst[id2*nchan + c]; // 2 <-> 4
            dst[id2*nchan + c] = dst[id4*nchan + c];
            dst[id4*nchan + c] = tmp;
        }
    }
}


__host__ void
fft_init(cufftHandle *plan, const int nx, const int ny, const int nchan)
{
  // setup FFT
  const int rank = 2;
  int idist = 1, odist = 1, istride = nchan, ostride = nchan;
  int n[2] = {nx, ny};
  int inembed[]  = {nx, ny};
  int onembed[]  = {nx, ny};
  cufftSafeCall(cufftPlanMany(plan, rank, n, onembed, ostride, odist,
      inembed, istride, idist, CUFFT_C2C, nchan));
}


__host__ void
fftwithshift (float2 *x, cufftHandle plan, const int j, const int n, const int nrep)
{
    fftshift<<<gridsize,blocksize,0,stream[j]>>>(x, n, nrep);
    cufftSafeCall(cufftExecC2C(plan, x, x, CUFFT_FORWARD));
    fftshift<<<gridsize,blocksize,0,stream[j]>>>(x, n, nrep);
}

__host__ void
ifftwithshift (float2 *x, cufftHandle plan, const int j, const int n, const int nrep)
{
    fftshift<<<gridsize,blocksize,0,stream[j]>>>(x, n, nrep);
    cufftSafeCall(cufftExecC2C(plan, x, x, CUFFT_INVERSE));
    fftshift<<<gridsize,blocksize,0,stream[j]>>>(x, n, nrep);
}

__device__ void
powit (float2 *A, const int n, const int niters)
{
    /* replace first column of square matrix A with largest eigenvector */
    float2 x[MAXCHAN], y[MAXCHAN];
    for (int k = 0; k < n; ++k)
        x[k] = make_float2(1.f, 0.f);
    for (int t = 0; t < niters; ++t) {
        for (int j = 0; j < n; ++j) {
            y[j] = make_float2(0.f,0.f);
            for (int k = 0; k < n; ++k)
               y[j] += A[j*n + k]*x[k];
        }
        // calculate the length of the resultant vector
        float norm_sq = 0.f;
        for (int k = 0; k < n; ++k)
          norm_sq += norm(y[k]);
        norm_sq = sqrtf(norm_sq);
        for (int k = 0; k < n; ++k)
            x[k] = y[k] / norm_sq;
    }
    float2 lambda = make_float2(0.f,0.f);
    for (int j = 0; j < n; ++j) {
        y[j] = make_float2(0.f,0.f);
        for (int k = 0; k < n; ++k)
           y[j] += A[j*n + k]*x[k];
        lambda += conj(x[j])*y[j];
    }
    for (int j = 0; j < n; ++j)
        A[j] = x[j];
    A[n] = lambda;  // store dominant eigenvalue in A
}

__global__ void
coilcombinesos (float2 *img, const float2 * __restrict__ coilimg, const int nimg, const int nchan)
{
    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < nimg*nimg; id += blockDim.x * gridDim.x) {
        float val = 0.f;
        for (int c = 0; c < nchan; ++c)
            val += norm(coilimg[nchan*id + c]);
        img[id].x = sqrtf(val);
        img[id].y = 0.f;
    }
}

__global__ void
coilcombinewalsh (float2 *img, const float2 * __restrict__ coilimg,
   const int nimg, const int nchan, const int nt, const int npatch)
{
    float2 A[MAXCHAN*MAXCHAN];
    for (size_t id = blockIdx.x * blockDim.x + threadIdx.x; id < nimg*nimg; id += blockDim.x * gridDim.x)
    {
        if (nchan == 1)
            img[id] = coilimg[id];
        else {
            int x = id / nimg;
            int y = id % nimg;
            for (int k = 0; k < NCHAN*NCHAN; ++k)
                A[k] = make_float2(0.f,0.f);
            for (int px = max(0,x-npatch); px <= min(nimg-1,x+npatch); ++px)
                for (int py = max(0,y-npatch); py <= min(nimg-1,y+npatch); ++py) {
                    int offset = nchan*(px*nimg + py);
                    for (int c2 = 0; c2 < nchan; ++c2)
                        for (int c1 = 0; c1 < nchan; ++c1)
                            A[c1*nchan + c2] += coilimg[offset+c1]*conj(coilimg[offset+c2]);
                }
            powit(A, nchan, 5);
            img[id] = make_float2(0.f, 0.f);
            for (int c = 0; c < nchan; ++c)
                img[id] += conj(A[c])*coilimg[nchan*id+c]; // * cexpf(-maxphase);
        }
// #ifdef CALC_B1
//         for (int c = 0; c < NCHAN; ++c) {
//             d_b1[nchan*id + c] = sqrtf(s[0])*U[nchan*c];
//         }
// #endif
    }
}

#if 0
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

#else

__host__ __device__ static float
i0f (const float x)
{
    float ax = fabsf(x);
    float ans;
    float y;

    if (ax < 3.75)
    {
        y=x/3.75,y=y*y;
        ans=1.0+y*(3.5156229+y*(3.0899424+y*(1.2067492
               +y*(0.2659732+y*(0.360768e-1+y*0.45813e-2)))));
    }
    else
    {
        y=3.75/ax;
        ans=(expf(ax)/sqrtf(ax))*(0.39894228+y*(0.1328592e-1
                +y*(0.225319e-2+y*(-0.157565e-2+y*(0.916281e-2
                +y*(-0.2057706e-1+y*(0.2635537e-1+y*(-0.1647633e-1
                +y*0.392377e-2))))))));
    }
    return ans;
}

#endif



__device__ inline float
kaiser_bessel (const float t, const float T, const int N, const float alpha)
{
   float r = 2.0f*t / float(N-1) / T;
   return i0f(M_PI*alpha*sqrtf(1.0f - r*r)) / i0f(M_PI*alpha);
}


__host__ __device__ inline float
kernel_beta (const float kernwidth, const float grid_oversamp)
{
    // from Beatty et al.
    float a = kernwidth / grid_oversamp;
    float b = grid_oversamp - 0.5f;
    return M_PI*sqrtf(a*a*b*b - 0.8);
}

#if 0

__device__ inline float
gridkernel (const float k, const int G, const float m, const float alpha)
{
    // Kaiser-Bessel from from Beatty et al.
#ifdef KERN_KB
    float W = 2.0f*m;
    float r = 2.0f*G*k / W;
    float B = kernel_beta(W, alpha);
    if (r < 1.0f)
      return (G/W)*i0f(B*sqrtf(1.0f - r*r));
    else
      return 0.f;
#else
    const float sigma = 0.33f; // ballparked from Jackson et al. 1991. IEEE TMI, 10(3), 473–8
    return expf(-0.5f*r2/sigma/sigma);
#endif
}
#else

__host__ __device__ inline float
gridkernel (const float x, const float n, const float m, const float sigma)
{
    // from Keiner, Kunis, & Potts
    // m = kernel radius, sigma = os factor
    // x e [-G/2,G/2)
#ifdef KERN_KB
    float arg = m*m - x*x;
    float b = M_PI*(2.f - 1.f/sigma);
    //float b = kernel_beta(m, sigma);
    float f = sqrtf(fabsf(arg));
    if (arg > 0.f)
        return sinhf(b*f)/f/M_PI;
    else if (arg < 0.f)
        return (sin(b*f) / f / M_PI);
    else
        return b/M_PI;
#else
    const float s = 0.33f; // ballparked from Jackson et al. 1991. IEEE TMI, 10(3), 473–8
    return expf(-0.5f*x*x/s/s);
#endif
}

#endif

__host__ __device__ inline float
gridkernelhat (const float k, const float n, const float m, const float sigma)
{
#ifdef KERN_KB
    if (fabs(k) <= n*(1.0f - 0.5f/sigma)) {
        float b = M_PI*(2.f - 1.f/sigma);
        //float b = kernel_beta(n, sigma);
        float t = 2.f*M_PI*k/n;
        float f = sqrtf(b*b - t*t);
        return i0f(m*f) / n;
    } else
        return 0.f;
#else
    return expf(-0.5f*k*k/G/G);
#endif
}


__device__ inline float
modang (const float x)   /* rescale arbitrary angles to [0,2PI] interval */
{
    const float TWOPI = 2.f*M_PI;
    float y = fmodf(x, TWOPI);
    return y < 0.f ? y + TWOPI : y;
}

__device__ inline float
minangulardist(const float a, const float b)
{
    float d1 = fabsf(modang(a - b));
    float d2 = fabsf(modang(a + M_PI) - b);
    float d3 = 2.f*M_PI - d1;
    float d4 = 2.f*M_PI - d2;
    return fminf(fminf(d1,d2),fminf(d3,d4));
}

__global__ void
deapodkernel (float2  *d_a, const int n, const int nrep, const float m, const float sigma)
{
    for (size_t id = blockIdx.x * blockDim.x + threadIdx.x; id < n*n; id += blockDim.x * gridDim.x)
    {
        float x = id / float(n) - 0.5f*n;
        float y = float(id % n) - 0.5f*n;
        // TODO: enable this
        float wgt = gridkernelhat(x, n, m, sigma) * gridkernelhat(y, n, m, sigma);
        for (int c = 0; c < nrep; ++c)
            d_a[nrep*id + c] /= wgt;
    }
}

#if 1
__host__ void
fillapod (float2 *d_apod, const int nx, const int ny, const float kernwidth, const float grid_oversamp)
{
    const size_t d_imgsize = nx*ny*sizeof(float2);
    float2 *h_apod = (float2*)malloc(d_imgsize);
    int w = int(kernwidth);
    int n = ny;  // TODO: fix this, substitute correct dims
    // TODO: replace with direct calculation from gridkernel_hat_inv

    for (int k = 0; k < n*n; ++k)
        h_apod[k] = make_float2(0.f,0.f);
    for (int x = 0; x < w; ++x) {
        for (int y = 0; y < w; ++y)
            h_apod[n*x + y].x = gridkernel(x, nx, kernwidth, grid_oversamp) * gridkernel(y, ny, kernwidth, grid_oversamp);
        for (int y = n-w; y < n; ++y)
            h_apod[n*x + y].x = gridkernel(x, nx, kernwidth, grid_oversamp) * gridkernel(n-y, ny, kernwidth, grid_oversamp);
    }
    for (int x = n-w; x < n; ++x) {
        for (int y = 0; y < w; ++y)
            h_apod[n*x + y].x = gridkernel(n-x, nx, kernwidth, grid_oversamp) * gridkernel(y, ny, kernwidth, grid_oversamp);
        for (int y = n-w; y < n; ++y)
            h_apod[n*x + y].x = gridkernel(n-x, nx, kernwidth, grid_oversamp) * gridkernel(n-y, ny, kernwidth, grid_oversamp);
    }
    cuTry(cudaMemcpy(d_apod, h_apod, d_imgsize, cudaMemcpyHostToDevice));
    cufftHandle fft_plan_apod;
    cufftSafeCall(cufftPlan2d(&fft_plan_apod, n, n, CUFFT_C2C));
    cufftSafeCall(cufftExecC2C(fft_plan_apod, d_apod, d_apod, CUFFT_INVERSE));
    fftshift<<<n,n>>>(d_apod, n, 1);
    cuTry(cudaMemcpy(h_apod, d_apod, d_imgsize, cudaMemcpyDeviceToHost));

    float maxval = 0.f;  // TODO: do this entirely on the GPU
    for (int k = 0; k < n*n; ++k) { // take magnitude and find brightest pixel at same time
        float mag = abs(h_apod[k]);
        h_apod[k] = make_float2(mag);
        maxval = mag > maxval ? mag : maxval;
    }
    for (int k = 0; k < n*n; ++k) { // normalize it   TODO: check for image artifacts
        h_apod[k].x /= maxval;
        h_apod[k].x = h_apod[k].x > 0.001f ? 1.0f / h_apod[k].x : 1.0f;
    }
    cuTry(cudaMemcpy(d_apod, h_apod, d_imgsize, cudaMemcpyHostToDevice));
    free(h_apod);
}

__global__ void
deapodize (float2 *img, const float2 * __restrict__ apod, const int nx, const int ny, const int nchan)
{
    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < nx*ny; id += blockDim.x * gridDim.x)
        for (int c = 0; c < nchan; ++c)
            img[nchan*id+c] *= apod[id].x; // took magnitude prior
}
#endif


__global__ void
precompensate (float2 *nudata, const int nchan, const int nro, const int npe1work)
{
    float a = (2.f  - 2.f / float(npe1work)) / float(nro);
    float b = 1.f / float(npe1work);
    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < npe1work; id += blockDim.x * gridDim.x)
        for (int r = 0; r < nro; ++r) {
            float sdc = a*fabsf(r - float(nro/2)) + b;
            for (int c = 0; c < nchan; ++c)
                nudata[nro*nchan*id + nchan*r + c] *= sdc;
        }
}

__global__ void
crop (float2* dst, const int nxdst, const int nydst, const float2* __restrict__ src, const int nxsrc, const int nysrc, const int nchan)
{
    const int nsrc = nxsrc, ndst = nxdst;  // TODO: eliminate this
    const int w = (nsrc - ndst) / 2;
    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < ndst*ndst; id += blockDim.x * gridDim.x)
    {
        int xdst = id / ndst;
        int ydst = id % ndst;
        int srcid = (xdst + w)*nsrc + ydst + w;
        for (int c = 0; c < nchan; ++c)
            dst[nchan*id + c] = src[nchan*srcid + c];
    }
}

// TODO: eliminate this
__global__ void
copy (float2* dst, const float2* __restrict__ src, const int n)
{
    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < n; id += blockDim.x * gridDim.x)
        dst[id] = src[id];
}

__global__ void
pad (float2* dst, const int ndst, const float2* __restrict__ src, const int nsrc, const int nchan)
{
    const int w = ndst > nsrc ? (ndst - nsrc) / 2 : 0;

    // set whole array to zero first (not most efficient!)
    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < ndst*ndst; id += blockDim.x * gridDim.x)
    {
        for (int c = 0; c < nchan; ++c)
            dst[nchan*id + c] = make_float2(0.f, 0.f);
        int xdst = id / ndst;
        int ydst = id % ndst;
        if ((xdst - w > 0) && (xdst - w < nsrc) &&
            (ydst - w > 0) && (ydst - w < nsrc))
        {
            size_t srcid = (xdst - w)*nsrc + (ydst - w);
        for (int c = 0; c < nchan; ++c)
            dst[nchan*id + c] = src[nchan*srcid + c];
        }
    }
}

extern "C" {  // don't mangle name, so can call from other languages

/*
    grid a single 2D image from input radial data
*/
__global__ void
gridradial2d (float2 *udata, const float2 * __restrict__ nudata, const int nxos,
    const int nchan, const int nro, const int npe, const float kernwidth, const float grid_oversamp,
const int skip_angles, const int flag_postcomp, const int flag_golden_angle)
{
    // udata: [NCHAN x NGRID x NGRID], nudata: NCHAN x NRO x NPE
    //float grid_oversamp = float(nxos) / float(nro); // grid_oversampling factor
    float2 utmp[MAXCHAN];
    const int blocksizex = 8; // TODO: optimize this blocking
    const int blocksizey = 4;
    const int warpsize = blocksizex*blocksizey;
    //int nblockx = nxos / blocksizex;
    int nblocky = nxos / blocksizey; // # of blocks along y dimension
    for (int tid = blockIdx.x * blockDim.x + threadIdx.x; tid < nxos*nxos; tid += blockDim.x * gridDim.x)
    {
        for (int ch = 0; ch < nchan; ch++)
          utmp[ch] = make_float2(0.f,0.f);

        //int x = id / nxos - nxos/2;
        //int y = -(id % nxos) + nxos/2;
        int z = tid / warpsize; // not a real z, just a block label
        int bx = z / nblocky;
        int by = z % nblocky;
        int zid = tid % warpsize;
        int x = zid / blocksizey + blocksizex*bx;
        int y = zid % blocksizey + blocksizey*by;
        int id = x*nxos + y; // computed linear array index for uniform data
        x = -x + nxos/2;
        y = -y + nxos/2;
        float gridpoint_radius = hypotf(float(x), float(y));
        int rmax = fminf(floorf(gridpoint_radius + kernwidth), nro/2-1);
        int rmin = fmaxf(ceilf(gridpoint_radius - kernwidth), 0);  // define a circular band around the uniform point
        for (int ch = 0; ch < nchan; ++ch)
             udata[nchan*id + ch] = make_float2(0.f,0.f);
        if (rmin > nro/2-1) continue; // outside non-uniform data area

        float sdc = 0.f;
        // get uniform point coordinate in non-uniform system, (r,theta) in this case
        float gridpoint_theta = modang(atan2f(float(y),float(x)));
        float dtheta = atan2f(kernwidth, gridpoint_radius); // narrow that band to an arc
        // profiles must line within an arc of 2*dtheta to be counted

        // TODO: replace this logic with boolean function that can be swapped out
        // for diff acquisitions
        for (int pe = 0; pe < npe; ++pe)
        {
            float profile_theta = flag_golden_angle ? modang(PHI * float(pe + skip_angles)) : float(pe) * M_PI / float(npe) + M_PI/2;
            //float dtheta1 = fabsf(modang(profile_theta - gridpoint_theta));
            //float dtheta2 = fabsf(modang(profile_theta + M_PI) - gridpoint_theta);
            //float dtheta1 = fabsf(profile_theta - gridpoint_theta);
            //float dtheta2 = fabsf(profile_theta + M_PI - gridpoint_theta);
            //float dtheta3 = 2.f*M_PI - dtheta1;
            //float dtheta4 = 2.f*M_PI - dtheta2;
            float dtheta1 = minangulardist(profile_theta, gridpoint_theta);
            if (dtheta1 <= dtheta) // || dtheta2 <= dtheta || dtheta3 <= dtheta || dtheta4 <= dtheta)
            {
                float sf, cf;
                __sincosf(profile_theta, &sf, &cf);
                //sf *= grid_oversamp;
                //cf *= grid_oversamp;
                // TODO: fix this logic, try using without dtheta1
                //int rstart = dtheta1 <= dtheta || dtheta3 <= dtheta ? rmin : -rmax;
                //int rend   = dtheta1 <= dtheta || dtheta3 <= dtheta ? rmax : -rmin;
                int rstart = fabs(profile_theta-gridpoint_theta) < 0.5f*M_PI ? rmin : -rmax;
                int rend   = fabs(profile_theta-gridpoint_theta) < 0.5f*M_PI ? rmax : -rmin;
                // TODO: add periodic BCs
                for (int r = rstart; r <= rend; ++r)  // for each POSITIVE non-uniform ro point
                for (int r = rstart; r <= rend; ++r)  // for each POSITIVE non-uniform ro point
                {
                    float kx = r*cf; // [-nxos/2 ... nxos/2-1]    // TODO: compute distance in radial coordinates?
                    float ky = r*sf; // [-nyos/2 ... nyos/2-1]
                    float wgt = gridkernel(kx - x, nxos, kernwidth, grid_oversamp) *
                            gridkernel(ky - y, nxos, kernwidth, grid_oversamp); // TODO: fix this, not nxos
                    if (flag_postcomp)
                      sdc += wgt;
                    for (int ch = 0; ch < nchan; ch++) { // unrolled by 2 'cuz faster
                        //utmp[ch] += wgt*nudata[nchan*(nro*pe + r + nro/2) + ch];
                        //utmp[ch + 1] += wgt*nudata[nchan*(nro*pe + r + nro/2) + ch + 1];
                        utmp[ch].x = __fmaf_rn(wgt,nudata[nchan*(nro*pe + r + nro/2) + ch].x, utmp[ch].x);
                        utmp[ch].y = __fmaf_rn(wgt,nudata[nchan*(nro*pe + r + nro/2) + ch].y, utmp[ch].y);
                    }
                }
            }
        }
        if (flag_postcomp && sdc > 0.f)
            for (int ch = 0; ch < nchan; ++ch)
                udata[nchan*id + ch] = utmp[ch] / sdc;
        else
            for (int ch = 0; ch < nchan; ++ch)
                udata[nchan*id + ch] = utmp[ch];
    }
}

/*  generate 2D radial data from an input 2D image */
__global__ void
degridradial2d (
    float2 *nudata, const float2 * __restrict__ udata, const int nx, const int nchan,
    const int nro, const int npe, const float kernwidth, const int skip_angles, const int flag_golden_angle)
{
    const float grid_oversamp = nro / (float)nx;
    // udata: [NCHAN x NGRID x NGRID], nudata: NCHAN x NRO x NPE
    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < nro*npe; id += blockDim.x * gridDim.x)
    {
        for (int c = 0; c < nchan; ++c) // zero my assigned unequal point
            nudata[nchan*id + c] = make_float2(0.f, 0.f);
        int pe = id / nro; // my row and column in the non-uniform data
        int ro = id % nro;
        float R = float(ro)/float(nro) - 0.5f; // [-1/2,1/2)
        float T = flag_golden_angle ? modang(PHI*(pe + skip_angles)) :
            float(pe) * M_PI / float(npe) + M_PI/2;
        // compute Cartesian coordinate of non-uniform point that we represent
        float X = nx*(R*sinf(T) + 0.5f); // TODO: use _sincosf?
        float Y = nx*(R*cosf(T) + 0.5f); // [0, nx)
        for (int xu = ceilf(X-kernwidth); xu <= (X+kernwidth); ++xu)
        for (int yu = ceilf(Y-kernwidth); yu <= (Y+kernwidth); ++yu)
        //int xu = X;
        //int yu = Y;
        {  // loop through contributing Cartesian points
            // TODO: optimize this
            float wgt = gridkernel(xu - X, nx, kernwidth, grid_oversamp) *
                        gridkernel(yu - Y, nx, kernwidth, grid_oversamp);
            int i = (xu + nx) % nx;
            int j = (yu + nx) % nx;
            wgt /= float(nro*npe*kernwidth*kernwidth);
            for (int c = 0; c < nchan; ++c) {
                float2 g = udata[nchan*(i*nx + j) + c];
                nudata[nchan*id + c].x += wgt*g.x;
                nudata[nchan*id + c].y += wgt*g.y;
            }
        }
    }
}


void
tron_init ()
{

  if (MULTI_GPU) {
    cuTry(cudaGetDeviceCount(&ndevices));
  } else
    ndevices = 1;
  DPRINT("MULTI_GPU = %d\n", MULTI_GPU);
  DPRINT("NSTREAMS = %d\n", NSTREAMS);
  DPRINT("Using %d CUDA devices\n", ndevices);
  DPRINT("Kernels configured with %d blocks of %d threads\n", gridsize, blocksize);

  // array sizes
  // TODO: this is wrong.  d_indatasize should just be the size of the work slice
  d_nudatasize = nc*nt*nro*npe1work*sizeof(float2);  // input data
  d_udatasize = nc*nt*nxos*nyos*sizeof(float2); // multi-coil gridded data
  d_coilimgsize = nc*nt*nx*ny*sizeof(float2);
  d_imgsize = nt*nx*ny*sizeof(float2);

  for (int j = 0; j < NSTREAMS; ++j) // allocate data and initialize apodization and kernel texture
  {
      DPRINT("init STREAM %d\n", j);
      if (MULTI_GPU) cudaSetDevice(j % ndevices);
      cuTry(cudaStreamCreate(&stream[j]));

      fft_init(&fft_plan[j], nx, ny, nc);
      cufftSafeCall(cufftSetStream(fft_plan[j], stream[j]));

      fft_init(&fft_plan_os[j], nxos, nyos, nc);
      cufftSafeCall(cufftSetStream(fft_plan_os[j], stream[j]));
      cuTry(cudaMalloc((void **)&d_udata[j], d_udatasize));
      cuTry(cudaMalloc((void **)&d_nudata[j],  d_nudatasize));
      cuTry(cudaMalloc((void **)&d_coilimg[j], d_coilimgsize));
      cuTry(cudaMalloc((void **)&d_img[j], d_imgsize));

      // TODO: only fill apod if depapodize is called
      // TODO: handle adjoint vs non-adjoint
      cuTry(cudaMalloc((void **)&d_apod[j], d_imgsize));
      cuTry(cudaMalloc((void **)&d_apodos[j], d_udatasize));
      fillapod(d_apodos[j], nxos, nyos, kernwidth, grid_oversamp);

      // TODO: can use only one d_apod for all streams?
      //crop<<<nx,ny>>>(d_apod[j], nx, ny, d_apodos[j], nxos, nyos, 1);


  }
}

void
tron_shutdown()
{
    DPRINT("freeing device memory\n");
    for (int j = 0; j < NSTREAMS; ++j) { // free allocated memory
        if (MULTI_GPU) cudaSetDevice(j % ndevices);
        cuTry(cudaFree(d_udata[j]));
        cuTry(cudaFree(d_nudata[j]));
        cuTry(cudaFree(d_coilimg[j]));
        cuTry(cudaFree(d_img[j]));
        cuTry(cudaFree(d_apod[j]));
        cuTry(cudaFree(d_apodos[j]));
        cudaStreamDestroy(stream[j]);
    }
    DPRINT("done freeing\n");
}


/*  Reconstruct images from 2D radial data.  This host routine calls the appropriate
    CUDA kernels in the correct order depending on the direction of recon.   */

__host__ void
recon_radial2d(float2 *h_outdata, const float2 *__restrict__ h_indata)
{
    DPRINT("recon_radial2d\n");

    tron_init();

    for (int z = 0; z < nz; ++z)
    {
        int j = z % NSTREAMS; // j is stream index
        if (MULTI_GPU) cudaSetDevice(j % ndevices);

        int peoffset = z*prof_slide;

        // address offsets into the data arrays
        size_t data_offset = nc * nt * nro * peoffset;
        size_t img_offset = nt * nx * ny * z;

        printf("[dev %d, stream %d] reconstructing slice %d/%d from PEs %d-%d (offset %ld)\n",
            j%ndevices, j, z+1, nz, z*prof_slide, (z+1)*prof_slide-1, data_offset);

        if (flags.adjoint)
        {
            cuTry(cudaMemcpyAsync(d_nudata[j], h_indata + data_offset, d_nudatasize, cudaMemcpyHostToDevice, stream[j]));
            // reverse from non-uniform data to image
            precompensate<<<gridsize,blocksize,0,stream[j]>>>(d_nudata[j], nc*nt, nro, npe1work);
            gridradial2d<<<gridsize,blocksize,0,stream[j]>>>(d_udata[j], d_nudata[j], nxos, nc*nt, nro, npe1work, kernwidth,
                grid_oversamp, skip_angles+peoffset, flags.postcomp, flags.golden_angle);
            ifftwithshift(d_udata[j], fft_plan_os[j], j, nxos, nt*nc);
            crop<<<gridsize,blocksize,0,stream[j]>>>(d_coilimg[j], nx, ny, d_udata[j], nxos, nyos, nc*nt);
            // TODO: look at indims.c vs outdims.c to decide whether to coil combine and by how much (can compress)
            //coilcombinewalsh<<<gridsize,blocksize,0,stream[j]>>>(d_img[j],d_coilimg[j], nx, nc, nt, 1); /* 0 works, 1 good, 3 better */
            coilcombinesos<<<gridsize,blocksize,0,stream[j]>>>(d_img[j], d_coilimg[j], nx, nc);
            //deapodize<<<gridsize,blocksize,0,stream[j]>>>(d_img[j], d_apod[j], nx, ny, nc*nt);
            deapodkernel<<<gridsize,blocksize,0,stream[j]>>>(d_img[j], nx, nc*nt, kernwidth, grid_oversamp);
#ifdef CUDA_HOST_MALLOC
            cuTry(cudaMemcpyAsync(h_outdata + img_offset, d_img[j], d_imgsize, cudaMemcpyDeviceToHost, stream[j]));
#else
            cuTry(cudaMemcpy(h_outdata + img_offset, d_img[j], d_imgsize, cudaMemcpyDeviceToHost));
#endif
        }
        else
        {   // forward from image to non-uniform data
            cuTry(cudaMemcpyAsync(d_img[j], h_indata + data_offset, d_imgsize, cudaMemcpyHostToDevice, stream[j]));
            deapodkernel<<<gridsize,blocksize,0,stream[j]>>>(d_img[j], nx, nc*nt, kernwidth, grid_oversamp);
            //deapodize<<<gridsize,blocksize,0,stream[j]>>>(d_img[j], d_apod[j], nx, ny, nc*nt);
            fftwithshift(d_img[j], fft_plan[j], j, nx, nc*nt);
            degridradial2d<<<gridsize,blocksize,0,stream[j]>>>(d_nudata[j], d_img[j],
                nx, nc*nt, nro, npe1work, kernwidth, skip_angles, flags.golden_angle);
#ifdef CUDA_HOST_MALLOC
            cuTry(cudaMemcpyAsync(h_outdata + nc*nt*nro*npe1work*z, d_nudata[j], d_nudatasize, cudaMemcpyDeviceToHost, stream[j]));
#else
            cuTry(cudaMemcpyAsync(h_outdata + nc*nt*nro*npe1work*z, d_nudata[j], d_nudatasize, cudaMemcpyDeviceToHost));
#endif
        }
    }

    tron_shutdown();
}




}

void
print_usage()
{
    fprintf(stderr, "Usage: tron [-3ahuv] [-r cmds] [-d prof_slide] [-k width] [-o grid_oversamp] [-s skip_angles] [-u data_undersamp] <infile.ra> [outfile.ra]\n");
    fprintf(stderr, "\t-3\t\t\t3D koosh ball trajectory\n");
    fprintf(stderr, "\t-a\t\t\tadjoint operation\n");
    fprintf(stderr, "\t-d prof_slide\t\tnumber of phase encodes to slide between slices for helical scans\n");
    fprintf(stderr, "\t-g\t\t\tgolden angle radial\n");
    fprintf(stderr, "\t-h\t\t\tshow this help\n");
    fprintf(stderr, "\t-k width\t\twidth of gridding kernel\n");
    fprintf(stderr, "\t-o grid_oversamp\tgrid oversampling factor\n");
    fprintf(stderr, "\t-r nro\t\t\tnumber of readout points\n");
    fprintf(stderr, "\t-s skip_angles\t\tnumber of initial phase encodes to skip\n");
    fprintf(stderr, "\t-u data_undersamp\tinput data undersampling factor\n");
    fprintf(stderr, "\t-v\t\t\tverbose output\n");
}


int
main (int argc, char *argv[])
{
    float2 *h_indata, *h_outdata;
    ra_t ra_in, ra_out;
    int c, index;
    char infile[1024], outfile[1024];

    opterr = 0;
    while ((c = getopt (argc, argv, "3ad:ghk:o:r:s:u:v")) != -1)
    {
        switch (c) {
            case '3':
                flags.koosh = 1;
            case 'a':
                flags.adjoint = 1;
                break;
            case 'd':
                prof_slide = atoi(optarg);
                break;
            case 'g':
                flags.golden_angle = 1;
                break;
            case 'h':
                print_usage();
                return 1;
            case 'k':
                kernwidth = atof(optarg);
                break;
            case 'o':
                grid_oversamp = atof(optarg);
                break;
            case 'u':
                data_undersamp = atof(optarg);
                break;
            case 'r':
                nro = atoi(optarg);
                break;
            case 's':
                skip_angles = atoi(optarg);
                break;
            case 'v':
                flags.verbose = 1;
                break;
            default:
                print_usage();
                return 1;
        }
    }

    // set input and output files
    snprintf(outfile, 1024, "img_tron.ra"); // default value
    if (argc == optind) {
       print_usage();
       return 1;
    }
    for (index = optind; index < argc; index++) {
      if (index == optind)
        snprintf(infile, 1024, "%s", argv[index]);
      else if (index == optind + 1)
        snprintf(outfile, 1024, "%s", argv[index]);
    }

    dprint(skip_angles, d);
    dprint(prof_slide, d);
    dprint(kernwidth, .1f);
    dprint(data_undersamp, .3f);
    dprint(grid_oversamp, .3f);
    dprint(infile, s)
    dprint(outfile, s)

    DPRINT("Reading %s\n", infile);
    ra_read(&ra_in, infile);
    h_indata = (float2*)ra_in.data;
    assert(ra_in.ndims == 5);
    DPRINT("Sanity check: indata[0] = %f + %f i\n", h_indata[0].x, h_indata[0].y);
    DPRINT("indims = {%llu, %llu, %llu, %llu, %llu}\n", ra_in.dims[0], ra_in.dims[1], ra_in.dims[2], ra_in.dims[3], ra_in.dims[4]);


    printf("WARNING: Assuming square Cartesian dimensions for now.\n");

    ra_out.ndims = 5;
    ra_out.dims = (uint64_t*) malloc(ra_out.ndims*sizeof(uint64_t));
    ra_out.dims[0] = 1;
    ra_out.flags = 0;
    ra_out.eltype = 4;
    ra_out.elbyte = 8;

    // HERE IS WHERE WE COMPUTE OUTPUT DIMENSIONS BASED ON INPUT AND OPTIONAL ARGS
    if (flags.adjoint)
    {
        nc = ra_in.dims[0];
        nt = ra_in.dims[1];
        nro = ra_in.dims[2];
        npe1 = ra_in.dims[3];
        npe2 = ra_in.dims[4];
        nx = nro / 2;
        ny = nro / 2;
        nxos = nx * grid_oversamp;
        nyos = ny * grid_oversamp;
        npe1work = data_undersamp * nro;  // TODO: fix this hack
        if (prof_slide == 0)
            prof_slide = npe1work;
        if (flags.koosh) {
            nz = nro / 2;
            nzos = nz * grid_oversamp;
        } else {
            nz = 1 + (npe1 - npe1work) / prof_slide;
            nzos = 1;
        }
        //npe2work = npe2;
        ra_out.dims[1] = nt;
        ra_out.dims[2] = nx;
        ra_out.dims[3] = ny;
        ra_out.dims[4] = nz;
        h_outdatasize = 1*nt*nx*ny*nz*sizeof(float2);
    }
    else
    {
        // TODO: this is broken probably
        nc = ra_in.dims[0];
        nt = ra_in.dims[1];
        nx = ra_in.dims[2];
        ny = ra_in.dims[3];
        nz = ra_in.dims[4];
        nxos = nx*grid_oversamp;
        nyos = ny*grid_oversamp;
        nro = grid_oversamp*nx;  // TODO: implement non-square images
        npe1work = data_undersamp * nro;  // TODO: fix this hack
        npe1 = npe1work;   // TODO: make this more customizable
        if (flags.koosh) {
            npe2 = nz;
            nzos = nz; //grid_oversamp*nz;
        } else {
            npe2 = 1;
            nzos = 1;
        }
        ra_out.dims[1] = nt;
        ra_out.dims[2] = nro;
        ra_out.dims[3] = npe1;
        ra_out.dims[4] = npe2;
        h_outdatasize = nc*nt*nro*npe1*npe2*sizeof(float2);
    }
    ra_out.size = h_outdatasize;
    dprint(h_outdatasize,ld);
    assert(nc % 2 == 0 || nc == 1); // only single or even dimensions implemented for now

    dprint(grid_oversamp,f);
    dprint(nc,d);
    dprint(nt,d);
    dprint(nro,d);
    dprint(npe1,d);
    dprint(npe2,d);
    dprint(nx,d);
    dprint(ny,d);
    dprint(nz,d);
    dprint(nxos,d);
    dprint(nyos,d);
    dprint(nzos,d);
    dprint(npe1work,d);


#ifdef CUDA_HOST_MALLOC
    // allocate pinned memory, which allows async calls
    cuTry(cudaMallocHost((void**)&h_outdata, h_outdatasize));
#else
    h_outdata = (float2*)malloc(h_outdatasize);
#endif

    DPRINT("Running reconstruction ...\n ");
    clock_t start = clock();

    // TODO: put more setup inside recon routine. main() should just load data and call recon
    recon_radial2d(h_outdata, h_indata);

    clock_t end = clock();
    DPRINT("Elapsed time: %.2f s\n", ((float)(end - start)) / CLOCKS_PER_SEC);

    DPRINT("Saving result to %s\n", outfile);
    ra_out.data = (uint8_t*)h_outdata;
    ra_write(&ra_out, outfile);


    DPRINT("Cleaning up.\n");
    ra_free(&ra_in);
#ifdef CUDA_HOST_MALLOC
    cudaFreeHost(&h_outdata);
#else
    free(h_outdata);
#endif
    cudaDeviceReset();

    return 0;
}
