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
#include "cublas_v2.h"

#include "float2math.h"
#include "mri.h"
#include "ra.h"
#include "tron.h"
#include "norm.h"

#define MAX(a,b) ((a)>(b)?(a):(b))
#define MIN(a,b) ((a)<(b)?(a):(b))
#define DPRINT if(flags.verbose)printf
#define dprint(expr,fmt)  do{ if(flags.verbose)fprintf(stderr,"\e[90m%d: " #expr " = %" #fmt "\e[0m\n", __LINE__, expr); }while(0);

// MISC GLOBAL VARIABLES
static cufftHandle fft_plan[NSTREAMS], fft_plan_os[NSTREAMS];
static cudaStream_t stream[NSTREAMS];
static int ndevices;

// DEVICE ARRAYS AND SIZES
static float2 *d_u[NSTREAMS], *d_v[NSTREAMS];
static size_t d_datasize; // size in bytes of non-uniform data
static size_t h_outdatasize;

// DEFAULT RECON CONFIGURATION
static float gridos = 2.f;  // TODO: compute ngrid from nx, ny and oversamp
static float kernwidth = 2.f;
static float data_undersamp = 1.f;

static int prof_slide = 0;         // # of profiles to slide through the data between reconstructed images
static int skip_angles = 0;        // # of angles to skip at beginning of image stack
static int peoffset = 0;

static int nc;  //  # of receive channels;
static int nt;  // # of repeated measurements of same trajectory
static int nro, npe1, npe2, npe1work;//, npe2work;  // radial readout and phase encodes
static int nx, ny, nz, nxos, nyos, nzos;  // Cartesian dimensions of uniform data

static struct {
    unsigned adjoint       : 1;
    unsigned deapodize     : 1;
    unsigned koosh         : 1;
    unsigned verbose       : 1;
    unsigned golden_angle  : 4;   // padded to 8 bits
} flags = {0, 1, 0, 0, 0};

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
_cufftGetErrorEnum(cufftResult error)
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

static const char *
_cublasGetErrorEnum (cublasStatus_t error)
{
    switch (error) {
        case CUBLAS_STATUS_SUCCESS: return "Success";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "Not Initiaziled";
        case CUBLAS_STATUS_ALLOC_FAILED: return "Alloc Failed";
        case CUBLAS_STATUS_INVALID_VALUE: return "Invalid Value";
        case CUBLAS_STATUS_ARCH_MISMATCH: return "Arch Mismatch";
        case CUBLAS_STATUS_MAPPING_ERROR: return "Mapping Error";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "Exec Failed";
        case CUBLAS_STATUS_INTERNAL_ERROR: return "Internal Error";
        case CUBLAS_STATUS_NOT_SUPPORTED: return "Not Supported";
        case CUBLAS_STATUS_LICENSE_ERROR: return "License Error";
        default: return "<unknown>";
    }
}

#define cufftSafeCall(err)  __cufftSafeCall(err, __FILE__, __LINE__)
inline void __cufftSafeCall (cufftResult err, const char *file, const int line)
{
    if (CUFFT_SUCCESS != err) {
        fprintf(stderr, "CUFFT error in file '%s', line %d\nerror %s: %d\nterminating!\n",__FILE__, __LINE__, \
                _cufftGetErrorEnum(err), (int)err);
        cudaDeviceReset();
        exit(1);
    }
}

#define cublasSafeCall(err)  __cublasSafeCall(err, __FILE__, __LINE__)
inline void __cublasSafeCall (cublasStatus_t err, const char *file, const int line)
{
    if (CUBLAS_STATUS_SUCCESS != err) {
        fprintf(stderr, "CUBLAS error in file '%s', line %d\nerror %s: %d\nterminating!\n",__FILE__, __LINE__, \
                _cublasGetErrorEnum(err), (int)err);
        exit(1);
    }
}

enum fftshift_direction { FFT_SHIFT_FORWARD, FFT_SHIFT_INVERSE };

__global__ void
fftshift (float2 *dst, float2 *src, const int n, const int nchan, int direction)
{
    int offset = direction == FFT_SHIFT_FORWARD ? n/2 : n - n/2;

    for (int idsrc = blockIdx.x * blockDim.x + threadIdx.x; idsrc < n*n; idsrc += blockDim.x * gridDim.x)
    {
        int xsrc = idsrc / n;
        int ysrc = idsrc % n;
        int xdst = (xsrc + offset) % n;
        int ydst = (ysrc + offset) % n;
        int iddst = n*xdst + ydst;
        for (int c = 0; c < nchan; ++c) {
          dst[iddst*nchan + c].x = src[idsrc*nchan + c].x;
          dst[iddst*nchan + c].y = src[idsrc*nchan + c].y;
        }
    }
}


__global__ void
fftshift_inplace (float2 *dst, const int n, const int nchan)
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
  if (nchan == 1)
      cufftSafeCall(cufftPlan2d(plan, nx, ny, CUFFT_C2C));
  else {
      const int rank = 2;
      int idist = 1, odist = 1, istride = nchan, ostride = nchan;
      int n[2] = {nx, ny};
      int inembed[]  = {nx, ny};
      int onembed[]  = {nx, ny};
      cufftSafeCall(cufftPlanMany(plan, rank, n, onembed, ostride, odist,
          inembed, istride, idist, CUFFT_C2C, nchan));
  }
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
        if (nchan > 1) {
          float val = 0.f;
          for (int c = 0; c < nchan; ++c)
              val += norm(coilimg[nchan*id + c]);
          img[id].x = sqrtf(val);
          img[id].y = 0.f;
        } else
          img[id] = coilimg[id];
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

#if 1
__device__ float
besseli0 (const float x)
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
besseli0 (float x)
{
    if (x < 0)  // TODO: is this correct?
       x = -x;
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
kernel_shape (const float kernwidth, const float gridos)
{
//#define BEATTY_BETA
#ifdef BEATTY_BETA
    float a = kernwidth / gridos;
    float b = gridos - 0.5f;
    return M_PI*sqrtf(a*a*b*b - 0.8f);
#else
    //return M_PI*(2.f - 1.f/gridos);
    return 2.34f*2.0f*kernwidth;
#endif
}


__device__ inline float
gridkernel (const float x, const float kernwidth, const float sigma)
{
  // x in [-kernwidth,kernwidth]
  // from FesslerA
  //const float J = 2.0f*kernwidth;  // TODO: substitute kernwidth for J
  //const float alpha = 2.34f*J;
  float alpha = kernel_shape(kernwidth, sigma);
  if (fabsf(x) < kernwidth) {
      float r = x/kernwidth;
      float f = sqrtf(1.0f - r*r);
      return besseli0(alpha*f) / besseli0(alpha);
  } else
      return 0.0f;
}

__device__ inline float
gridkernelhat (const float u, const float kernwidth, const float sigma)
{
    // u in [-n/2,n/2]
    // from Fessler
    const float J = 2.0f*kernwidth;
    //const float alpha = 2.34f*J;
    float alpha = kernel_shape(kernwidth, sigma);
    //const int d = 1;
    float r = M_PI*J*u;
    float q = r*r - alpha*alpha;  // TODO: fix DomainError
    float y, z;
    if (q > 0) {
        z = sqrtf(q);
        y = J * sinf(z) / z / besseli0(alpha);
    } else if (q < 0) {
        z = sqrtf(-q);
        y = J * sinhf(z) / z / besseli0(alpha);
    } else
        y = 0.0f;
    // identity: J_1/2(z) = sin(z) * sqrt(2/pi/z)
    return y;
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
        float x = id / float(n) - 0.5f*n;  // TODO: simplify this
        float y = float(id % n) - 0.5f*n;
        float scale = 1.f / (n * sigma);
        float r = sqrtf(x*x + y*y);
        //float wgt = gridkernelhat(x*scale, m, sigma) * gridkernelhat(y*scale, m, sigma);
        float wgt = gridkernelhat(r*scale, m, sigma);
        for (int c = 0; c < nrep; ++c)
            d_a[nrep*id + c] /= (wgt > 0.f ? wgt : 1.f);
    }
}


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


extern "C" {  // don't mangle name, so can call from other languages

/*
    grid a single 2D image from input radial data
*/
__global__ void
gridradial2d (float2 *udata, const float2 * __restrict__ nudata, const int nxos,
    const int nchan, const int nro, const int npe, const float kernwidth, const float gridos,
const int skip_angles, const int flag_golden_angle)
{
    // udata: [NCHAN x NGRID x NGRID], nudata: NCHAN x NRO x NPE
    float2 utmp[MAXCHAN];
    //const int blocksx = 8; // TODO: optimize this blocking
    //const int blocksy = 4;
    //const int warpsize = blocksx*blocksy;
    //int nblockx = nxos / blocksx;
    //int nblocky = nxos / blocksy; // # of blocks along y dimension

    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < nxos*nxos; id += blockDim.x * gridDim.x)
    {
        for (int ch = 0; ch < nchan; ch++)
            utmp[ch] = make_float2(0.f,0.f);

        // figure out this thread's Cartesian and radial coordinates
        int Y = id / nxos - nxos/2;
        int X = (id % nxos) - nxos/2;
        float R = hypotf(float(X), float(Y));

        // more complicated, but faster ... can probably optimize better by sorting, though
        // int z = tid / warpsize; // not a real z, just a block label
        // int by = z / nblocky;
        // int bx = z % nblocky;
        // int zid = tid % warpsize;
        // int y = zid / blocksy + blocksx*by;
        // int x = zid % blocksy + blocksy*bx;
        // int id = y*nxos + x; // computed linear array index for uniform data
        // x -= nxos/2;
        // y -= nxos/2;

        // define a circular band around the uniform point
        int Rmax = fminf(floorf(R + kernwidth), nxos/2-1);
        int Rmin = fmaxf(ceilf(R - kernwidth), 0);

        // zero the temporary work array
        // for (int ch = 0; ch < nchan; ++ch)
        //      udata[nchan*id + ch] = make_float2(0.f, 0.f);
        //if (Rmin > nxos/2-1) continue; // skip gridding if outside non-uniform data area

        // get uniform point coordinate in non-uniform system, (r,theta) in this case
        float T = modang(atan2f(float(Y),float(X)));
        float dT = atan2f(kernwidth, R); // narrow that band to an arc
        // profiles must lie within an arc of 2*dT to be counted

        // TODO: replace this logic with boolean function that can be swapped out
        // for diff acquisitions
        for (int pe = 0; pe < npe; ++pe)
        {
            float t = flag_golden_angle ? modang(PHI * float(pe + skip_angles)) : pe*M_PI / float(npe);
            float dt1 = minangulardist(t, T);
            if (dt1 <= dT)
            {
                float sf, cf;
                __sincosf(t, &sf, &cf);
                // TODO: fix this logic, try using without dt1
                int rstart = fabs(t-T) < 0.5f*M_PI ? Rmin : -Rmax;
                int rend   = fabs(t-T) < 0.5f*M_PI ? Rmax : -Rmin;
                for (int r = rstart; r <= rend; ++r)  // for each POSITIVE non-uniform ro point
                {
                    float kx = r*cf; // [-nxos/2 ... nxos/2-1]    // TODO: compute distance in radial coordinates?
                    float ky = r*sf; // [-nyos/2 ... nyos/2-1]
                    float dx = sqrtf((kx-X)*(kx-X) + (ky-Y)*(ky-Y));
                    float wgt = gridkernel(dx, kernwidth, gridos);
                    //float wgt = gridkernel(kx-X, kernwidth, gridos) * gridkernel(ky-Y, kernwidth, gridos);
                    int ridx = (r * nro) / nxos;
                    for (int ch = 0; ch < nchan && wgt > 0.f; ch++) {
                        utmp[ch] += wgt*nudata[nchan*(nro*pe + ridx + nro/2) + ch];
                        //utmp[ch].x = __fmaf_rn(wgt, nudata[nchan*(nro*pe + ridx + nro/2) + ch].x, utmp[ch].x);
                        //utmp[ch].y = __fmaf_rn(wgt, nudata[nchan*(nro*pe + ridx + nro/2) + ch].y, utmp[ch].y);
                    }
                }
            }
        }
        for (int ch = 0; ch < nchan; ++ch)
            udata[nchan*id + ch] = utmp[ch] / nxos / sqrtf(gridos);
    }
}


/*  generate 2D radial data from an input 2D image */
__global__ void
degridradial2d (
    /* udata: [NCHAN x NGRID x NGRID], nudata: NCHAN x NRO x NPE */
    float2 *nudata, const float2 * __restrict__ udata, const int n, const int nrep,
    const int nro, const int npe, const float W, const float gridos, const int skip_angles,
    const int flag_golden_angle)
{
    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < nro*npe; id += blockDim.x * gridDim.x)
    {
        for (int c = 0; c < nrep; ++c) // zero my assigned unequal point
            nudata[nrep*id + c] = make_float2(0.f, 0.f);
        int pe = id / nro; // my row and column in the non-uniform data
        int ro = id % nro;
        // thread's polar coordinates
        // TODO: is this R correct?
        float R = float(ro)/float(nro) - 0.5f; // [-1/2,1/2)
        float T = flag_golden_angle ? modang(PHI*(pe + skip_angles)) : pe*M_PI/float(npe);

        // thread's Cartesian coordinates
        float X = n*R*sinf(T) + (n + 1)/2; // TODO: use _sincosf?
        float Y = n*R*cosf(T) + (n + 1)/2; // [0, n)
        for (int xu = ceilf(X-W); xu <= (X+W); ++xu)
         {
            //float wgtx = gridkernel(xu-X, W, gridos) / n;
            for (int yu = ceilf(Y-W); yu <= (Y+W); ++yu)
            {   // loop through contributing Cartesian points
                float dx = sqrtf((xu-X)*(xu-X) + (yu-Y)*(yu-Y));
                float wgt = gridkernel(dx, W, gridos) / sqrtf(npe*nro);
                //float wgt = wgtx * gridkernel(yu-Y, W, gridos);
                int i = (xu + n) % n; // periodic domain
                int j = (yu + n) % n;
                int offset = nrep*(i*n + j);
                for (int c = 0; c < nrep; ++c) {  // TODO: use nutmp temp array here for better cache usage
                    nudata[nrep*id + c] += wgt*udata[offset + c];
                    //utmp[ch].x = __fmaf_rn(wgt,nudata[nchan*(nro*pe + ridx + nro/2) + ch].x, utmp[ch].x);
                    //nudata[nrep*id + c].x = __fmaf_rn(wgt, nudata[nrep*id + c].x, udata[offset + c].x);
                    //nudata[nrep*id + c].y = __fmaf_rn(wgt, nudata[nrep*id + c].y, udata[offset + c].y);
                }
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
    DPRINT("Kernels configured with %d blocks of %d threads\n", threads, blocks);

    d_datasize = nc*nt*max(nro*npe1work, nxos*nyos)*sizeof(float2);  // input data

    for (int j = 0; j < NSTREAMS; ++j) // allocate data and initialize apodization and kernel texture
    {
        DPRINT("init STREAM %d\n", j);
        if (MULTI_GPU)
            cudaSetDevice(j % ndevices);
        cuTry(cudaStreamCreate(&stream[j]));
        fft_init(&fft_plan[j], nx, ny, nc);
        cufftSafeCall(cufftSetStream(fft_plan[j], stream[j]));
        fft_init(&fft_plan_os[j], nxos, nyos, nc);
        cufftSafeCall(cufftSetStream(fft_plan_os[j], stream[j]));
        cuTry(cudaMalloc((void **)&d_u[j], d_datasize));
        cuTry(cudaMalloc((void **)&d_v[j], d_datasize));
    }
}

void
tron_shutdown()
{
    DPRINT("freeing device memory ... ");
    for (int j = 0; j < NSTREAMS; ++j) { // free allocated memory
        if (MULTI_GPU)
            cudaSetDevice(j % ndevices);
        cuTry(cudaFree(d_u[j]));
        cuTry(cudaFree(d_v[j]));
        cudaStreamDestroy(stream[j]);
    }
    DPRINT("done.\n");
}


void
tron_nufft_adj_radial2d (float2 *d_out, float2 *d_in, const int j)
{
    // NUFFT adjoint begin
    precompensate<<<threads,blocks,0,stream[j]>>>(d_in, nc*nt, nro, npe1work);
    gridradial2d<<<threads,blocks,0,stream[j]>>>(d_out, d_in, nxos, nc*nt, nro, npe1work, kernwidth,
        gridos, skip_angles+peoffset, flags.golden_angle);
    fftshift<<<threads,blocks,0,stream[j]>>>(d_in, d_out, nxos, nt*nc, FFT_SHIFT_INVERSE);
    cufftSafeCall(cufftExecC2C(fft_plan_os[j], d_in, d_out, CUFFT_INVERSE));
    fftshift<<<threads,blocks,0,stream[j]>>>(d_in, d_out, nxos, nc*nt, FFT_SHIFT_FORWARD);
    crop<<<threads,blocks,0,stream[j]>>>(d_out, nx, ny, d_in, nxos, nyos, nc*nt);
    deapodkernel<<<threads,blocks,0,stream[j]>>>(d_out, nx, nc*nt, kernwidth, gridos);
}

void
tron_nufft_radial2d (float2 *d_out, float2 *d_in, const int j)
{
    deapodkernel<<<threads,blocks,0,stream[j]>>>(d_in, nx, nc*nt, kernwidth, gridos);
    fftshift<<<threads,blocks,0,stream[j]>>>(d_out, d_in, nx, nc*nt, FFT_SHIFT_FORWARD);
    cufftSafeCall(cufftExecC2C(fft_plan[j], d_out, d_out, CUFFT_FORWARD));
    fftshift<<<threads,blocks,0,stream[j]>>>(d_in, d_out, nx, nc*nt, FFT_SHIFT_INVERSE);
    degridradial2d<<<threads,blocks,0,stream[j]>>>(d_out, d_in, nx, nc*nt,
         nro, npe1work, kernwidth, gridos, skip_angles, flags.golden_angle);
}

void
copy (float2* d_dst, float2* d_src, const size_t N, const int j)
{
  cuTry(cudaMemcpyAsync(d_dst, d_src, N*sizeof(float2), cudaMemcpyDeviceToDevice, stream[j]));
}


__global__ void
Caxpy (float2 *d_z, float2 *d_y, float2 *d_x, float alpha, const size_t N)
{
    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < N; id += blockDim.x * gridDim.x)
        d_z[id] = d_y[id] +  alpha*d_x[id];
}

void
tron_cgnr_radial2d (float2* d_out, float2 *d_in, const int j, const int niter)
{
    // Based on Algorithm 1 from Knopp et al. 2007, Intl J of Biomed Imag
    // TODO: split into one pointer per stream?
    float2 *d_ztilde, *d_p, *d_ptilde, *d_r;
    float alpha, beta, res1, res2;
    float2 zres;
    const int inc = 1;
    cublasHandle_t handle;
    cublasStatus_t stat;
    stat = cublasCreate(&handle);


    const size_t N = nx*ny*nc*nt;
    const size_t n = nro*npe1*nc*nt;
    dprint(N,d);
    dprint(n,d);
    cuTry(cudaMalloc((void **)&d_ztilde, d_datasize));
    cuTry(cudaMalloc((void **)&d_p, d_datasize));
    cuTry(cudaMemset((float*)d_p, 0.f, N*2));
    cuTry(cudaMalloc((void **)&d_ptilde, d_datasize));
    cuTry(cudaMalloc((void **)&d_r, d_datasize));
    tron_nufft_adj_radial2d(d_ztilde, d_in, j); // ztilde = A^H W r
    copy(d_ptilde, d_ztilde, N, j); // ptilde = ztilde
    for (int t = 0; t < niter; ++t)
    {
        copy(d_u[j], d_ptilde, N, j);
        tron_nufft_radial2d(d_v[j], d_u[j], j); // v = A*ptilde
        copy(d_u[j], d_v[j], n, j);
        precompensate<<<threads,blocks,0,stream[j]>>>(d_u[j], nc*nt, nro, npe1work); // W*v
        cublasScnrm2(handle, N, (cuComplex*)d_ztilde, inc, &res1);
        cublasCdotc(handle, n, (cuComplex*)d_v[j], inc, (cuComplex*)d_u[j], inc, &zres);
        alpha = res1 / zres.x;
        //alpha = norm(d_ztilde, N) / dot(d_v[j], d_u[j], n);
        dprint(alpha,f);
        Caxpy<<<threads,blocks,0,stream[j]>>>(d_p, d_p, d_ptilde, alpha, N);
        Caxpy<<<threads,blocks,0,stream[j]>>>(d_r, d_r, d_v[j],  -alpha, n);

        cublasScnrm2(handle, N, (cuComplex*)d_ztilde, inc, &res1);
        copy(d_u[j], d_r, n, j); // ptilde = ztilde
        tron_nufft_adj_radial2d(d_ztilde, d_u[j], j); // ztilde = A^H W r
        cublasScnrm2(handle, N, (cuComplex*)d_ztilde, inc, &res2);
        beta = res2 / res1;
        dprint(beta,f);
        Caxpy<<<threads,blocks,0,stream[j]>>>(d_ptilde, d_ztilde, d_ptilde, beta, N);
    }

    copy(d_out, d_p, N, j);

    cudaFree(d_p);
    cudaFree(d_ptilde);
    cudaFree(d_r);
    cudaFree(d_ztilde);
    cublasDestroy(handle);
}


// void tron_set_grid_oversampling (const float g) { gridos = g; }
// void tron_set_data_undersampling (const float u) { data_undersamp = u; }
// void tron_set_golden_angle () { flags.golden_angle = 1; }
// void tron_set_profile_slide (const int s ) { prof_slide = s; }
// void tron_set_skip_angles (const int s) { skip_angles = s; }

/*  Reconstruct images from 2D radial data.  This host routine calls the appropriate
    CUDA kernels in the correct order depending on the direction of recon.   */

__host__ void
recon_radial2d (float2 *h_outdata, const float2 *__restrict__ h_indata)
{
    DPRINT("recon_radial2d\n");
    tron_init();

    for (int z = 0; z < nz; ++z)
    {
        int j = z % NSTREAMS; // j is the stream index
        if (MULTI_GPU) cudaSetDevice(j % ndevices);

        peoffset = z*prof_slide;
        size_t data_offset = nc*nt*nro*peoffset;  // address offsets into the data arrays
        size_t img_offset = nt*nx*ny*z;

        printf("[dev %d, stream %d] reconstructing slice %d/%d from PEs %d-%d\n",
            j%ndevices, j, z+1, nz, z*prof_slide, (z+1)*prof_slide-1);

        if (flags.adjoint) { // copy working data to GPU
            cuTry(cudaMemcpyAsync(d_u[j], h_indata + data_offset,
                nc*nt*nro*npe1work*sizeof(float2), cudaMemcpyHostToDevice, stream[j]));
        } else {
            cuTry(cudaMemcpyAsync(d_u[j], h_indata + data_offset,
                nc*nt*nx*ny*sizeof(float2), cudaMemcpyHostToDevice, stream[j]));
        }
        int niter = 0;
        if (flags.adjoint) {  // process data resident on GPU
            for (int t = 0; t < niter; ++t) {
              tron_nufft_adj_radial2d(d_v[j], d_u[j], j);
              tron_nufft_radial2d(d_u[j], d_v[j], j);
            }
            tron_nufft_adj_radial2d(d_v[j], d_u[j], j);

            //tron_cgnr_radial2d (d_v[j], d_u[j], j, niter);
        } else
            tron_nufft_radial2d(d_v[j], d_u[j], j);


        if (flags.adjoint)  // send result back to CPU
        {
            coilcombinesos<<<threads,blocks,0,stream[j]>>>(d_u[j], d_v[j], nx, nc); // TODO: should this have nt in it?
            // TODO: look at nc to decide whether to coil combine and by how much (can compress)
            //coilcombinewalsh<<<threads,blocks,0,stream[j]>>>(d_img[j],d_coilimg[j], nx, nc, nt, 1); /* 0 works, 1 good, 3 better */
#ifdef CUDA_HOST_MALLOC
            cuTry(cudaMemcpyAsync(h_outdata + img_offset, d_u[j],
                nx*ny*nt*sizeof(float2), cudaMemcpyDeviceToHost, stream[j]));
#else
            cuTry(cudaMemcpy(h_outdata + img_offset, d_u[j],
                nx*ny*nt*sizeof(float2), cudaMemcpyDeviceToHost));
#endif
        } else {
#ifdef CUDA_HOST_MALLOC
            cuTry(cudaMemcpyAsync(h_outdata + nc*nt*nro*npe1work*z, d_v[j],
                nc*nt*nro*npe1work*sizeof(float2), cudaMemcpyDeviceToHost, stream[j]));
#else
            cuTry(cudaMemcpyAsync(h_outdata + nc*nt*nro*npe1work*z, d_v[j],
                nc*nt*nro*npe1work*sizeof(float2), cudaMemcpyDeviceToHost));
#endif
        }
    }

    tron_shutdown();
}




}

void
print_usage()
{
    fprintf(stderr, "Trajectory-optimized Non-uniform Fast Fourier Transform\n");
    fprintf(stderr, "Usage: tron [-3ahuv] [-r cmds] [-d prof_slide] [-k width] [-o gridos] [-s skip_angles] [-u data_undersamp] <infile.ra> [outfile.ra]\n");
    fprintf(stderr, "\t-3\t\t\t3D koosh ball trajectory\n");
    fprintf(stderr, "\t-a\t\t\tadjoint operation\n");
    fprintf(stderr, "\t-d prof_slide\t\tnumber of phase encodes to slide between slices for helical scans\n");
    fprintf(stderr, "\t-g\t\t\tgolden angle radial\n");
    fprintf(stderr, "\t-h\t\t\tshow this help\n");
    fprintf(stderr, "\t-k width\t\twidth of gridding kernel\n");
    fprintf(stderr, "\t-o gridos\t\tgrid oversampling factor\n");
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
                gridos = atof(optarg);
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
    dprint(gridos, .3f);
    dprint(infile, s)
    dprint(outfile, s)

    DPRINT("Reading %s\n", infile);
    ra_read(&ra_in, infile);
    h_indata = (float2*)ra_in.data;
    assert(ra_in.ndims == 5);
    DPRINT("Sanity check: indata[0] = %f + %f i\n", h_indata[0].x, h_indata[0].y);
    DPRINT("indims = {%llu, %llu, %llu, %llu, %llu}\n", ra_in.dims[0], ra_in.dims[1], ra_in.dims[2], ra_in.dims[3], ra_in.dims[4]);
    DPRINT("WARNING: Assuming square Cartesian dimensions for now.\n");

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
        nxos = nx * gridos;
        nyos = ny * gridos;
        if (npe1 <= nro * data_undersamp)  /* must be implicitly undersampled */
            npe1work = npe1;
        else
            npe1work = nro * data_undersamp;
        if (prof_slide == 0)
            prof_slide = npe1work;
        if (flags.koosh) {
            nz = nro / 2;
            nzos = nz * gridos;
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
        nc = ra_in.dims[0];
        nt = ra_in.dims[1];
        nx = ra_in.dims[2];
        ny = ra_in.dims[3];
        nz = ra_in.dims[4];
        nxos = nx*gridos;
        nyos = ny*gridos;
        nro = gridos*nx;  // TODO: implement non-square images
        npe1work = data_undersamp * nro;
        npe1 = npe1work;
        if (flags.koosh) {
            npe2 = nz;
            nzos = nz; //gridos*nz ?;
        } else {
            npe2 = 1;
            nzos = 1;
        }
        ra_out.dims[1] = nt;
        ra_out.dims[2] = nro;
        ra_out.dims[3] = npe1;
        ra_out.dims[4] = npe2;
        gridos = 1.f;
        h_outdatasize = nc*nt*nro*npe1*npe2*sizeof(float2);
    }
    ra_out.size = h_outdatasize;
    dprint(h_outdatasize,ld);
    assert(nc % 2 == 0 || nc == 1); // only single or even dimensions implemented for now

    dprint(data_undersamp,f);
    dprint(gridos,f);
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
