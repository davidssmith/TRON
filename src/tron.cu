/*
  This file is part of the TRON package (http://github.com/davidssmith/tron).

  The MIT License (MIT)

  Copyright (c) 2016 David Smith

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


// CONFIGURATION PARAMETERS
// TODO: softcode as many as possible
#define NSTREAMS        2
#define MULTI_GPU       0
#define NCHAN           6
#define MAXCHAN         6
#define MAX_RECON_CMDS  20
static const int blocksize = 96;    // TWEAK: CUDA kernel parameters, optimize for your device
static const int gridsize = 2048;
static char flag_verbose = 0;

// GLOBAL VARIABLES
static float2 *d_indata[NSTREAMS], *d_outdata[NSTREAMS], *d_apodos[NSTREAMS], *d_apod[NSTREAMS], *d_tmp[NSTREAMS];
static cufftHandle fft_plan[NSTREAMS], fft_plan_os[NSTREAMS];
static cudaStream_t stream[NSTREAMS];
static int ndevices;

static size_t d_indatasize; // size in bytes of non-uniform data
static size_t d_outdatasize; // size in bytes of gridded data
static size_t d_coilimgsize; // multi-coil image size
static size_t d_imgsize; // coil-combined image size
static size_t d_gridsize;
static size_t h_outdatasize;


#define DPRINT if(flag_verbose)printf
#define dprint(expr,fmt)  do{ if(flag_verbose)printf(#expr " = %" #fmt "\n", expr); }while(0);


// non-uniform data shape: nchan x nrep x nro x npe
// uniform data shape:     nchan x nrep x ngrid x ngrid x nz
// image shape:            nchan x nrep x nimg x nimg x nz
// coil-combined image:            nrep x nimg x nimg x nz



void
TRON_set_default_plan (TRON_plan *p)
{
    DPRINT("TRON_set_default_plan\n");

    // TODO: REMOVE THESE
    p->nchan = 0;
    p->nrep = 0;  // # of repeated measurements of same trajectory
    p->nro = 0;
    p->npe = 0;
    p->ngrid = 0;
    p->nx = 0;
    p->ny = 0;
    p->nz = 0;

    for (int i = 0; i < 5; ++i) {
        p->in_dims.n[i] = 0;
        p->out_dims.n[i] = 0;
    }

    // STYLE PARAMETERS
    p->prof_slide = 0;
    p->skip_angles = 0;
    p->prof_per_image = 0;
    p->grid_oversamp = 2.f;
    p->data_undersamp = 1.f;
    p->kernwidth = 2.f;
    p->nchunks = 1;;    // # of chunks to divide input data into for recon

    // BOOLEAN OPTIONS
    p->flags.adjoint = 0;
    p->flags.postcomp = 0;
    p->flags.deapodize = 1;
    p->flags.golden_angle = 0;
}

// CONSTANTS
static const float PHI = 1.9416089796736116f;

inline void
gpuAssert (cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
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
  DPRINT("fft_init\n");
  dprint(nx,d);
  dprint(ny,d);
  dprint(nchan,d);
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
   const int nimg, const int nchan, const int npatch)
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
gridkernel (const float dx, const float dy, const float kernwidth, const float grid_oversamp)
{
    float r2 = dx*dx + dy*dy;
#ifdef KERN_KB
    //const float kernwidth = 2.f;
#define SQR(x) ((x)*(x))
#define BETA (M_PI*sqrtf(SQR(kernwidth/grid_oversamp*(grid_oversamp-0.5))-0.8))
    return r2 < kernwidth*kernwidth ? i0f(BETA * sqrtf (1.f - r2/kernwidth/kernwidth)) / i0f(BETA): 0.f;
#else
    const float sigma = 0.33f; // ballparked from Jackson et al. 1991. IEEE TMI, 10(3), 473–8
    return expf(-0.5f*r2/sigma/sigma);
#endif
}

// TODO: eliminate or combine the kernels
__host__ __device__ inline float
degridkernel (const float dx, const float dy, const float kernwidth, const float grid_oversamp)
{
    float r2 = dx*dx + dy*dy;
#ifdef KERN_KB
    //const float kernwidth = 2.f;
#define SQR(x) ((x)*(x))
#define BETA (M_PI*sqrtf(SQR(kernwidth/grid_oversamp*(grid_oversamp-0.5))-0.8))
    return r2 < kernwidth*kernwidth ? i0f(BETA * sqrtf (1.f - r2/kernwidth/kernwidth)) / i0f(BETA): 0.f;
#else
    const float sigma = 0.33f; // ballparked from Jackson et al. 1991. IEEE TMI, 10(3), 473–8
    return expf(-0.5f*r2/sigma/sigma);
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

__host__ void
fillapod (float2 *d_apod, const int nx, const int ny, const float kernwidth, const float grid_oversamp)
{
    DPRINT("fillapod\n");
    const size_t d_imgsize = nx*ny*sizeof(float2);
    float2 *h_apod = (float2*)malloc(d_imgsize);
    int w = int(kernwidth);
    int n = ny;  // TODO: fix this, substitute correct dims

    for (int k = 0; k < n*n; ++k)
        h_apod[k] = make_float2(0.f,0.f);
    for (int x = 0; x < w; ++x) {
        for (int y = 0; y < w; ++y)
            h_apod[n*x + y].x = gridkernel(x, y, kernwidth, grid_oversamp);
        for (int y = n-w; y < n; ++y)
            h_apod[n*x + y].x = gridkernel(x, n-y, kernwidth, grid_oversamp);
    }
    for (int x = n-w; x < n; ++x) {
        for (int y = 0; y < w; ++y)
            h_apod[n*x + y].x = gridkernel(n-x, y, kernwidth, grid_oversamp);
        for (int y = n-w; y < n; ++y)
            h_apod[n*x + y].x = gridkernel(n-x, n-y, kernwidth, grid_oversamp);
    }
    cuTry(cudaMemcpy(d_apod, h_apod, d_imgsize, cudaMemcpyHostToDevice));
    cufftHandle fft_plan_apod;
    cufftSafeCall(cufftPlan2d(&fft_plan_apod, n, n, CUFFT_C2C));
    cufftSafeCall(cufftExecC2C(fft_plan_apod, d_apod, d_apod, CUFFT_INVERSE));
    fftshift<<<n,n>>>(d_apod, n, 1);
    cuTry(cudaMemcpy(h_apod, d_apod, d_imgsize, cudaMemcpyDeviceToHost));

    float maxval = 0.f;
    for (int k = 0; k < n*n; ++k) { // take magnitude and find brightest pixel at same time
        float mag = abs(h_apod[k]);
        h_apod[k] = make_float2(mag);
        maxval = mag > maxval ? mag : maxval;
    }
    for (int k = 0; k < n*n; ++k) { // normalize it   TODO: check for image artifacts
        h_apod[k].x /= maxval;
        h_apod[k].x = h_apod[k].x > 0.1f ? 1.0f / h_apod[k].x : 1.0f;
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

__global__ void  // TODO: fix this
degrid_deapodize (float2 *img, const int nimg, const int nchan,
    float kernwidth, float grid_oversamp)
{
    grid_oversamp = 1.f;
    kernwidth = 1.f;
    float beta = kernwidth*(grid_oversamp-0.5)/grid_oversamp;
    beta = M_PI*sqrtf(beta*beta - 0.8);
    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < nimg*nimg; id += blockDim.x * gridDim.x)
    {
        int y = id % nimg - nimg/2;
        int x = id / nimg - nimg/2;
        float r = sqrtf(x*x + y*y);
        float d = M_PI*kernwidth*r/nimg;
        float s = d > beta ? sqrtf(d*d - beta*beta) : 1.f;
        float f = s != 0.f ? sinf(s) / s : 1.f;
        for (int c = 0; c < nchan; ++c)
            img[nchan*id+c] /= f;
    }
}


//__device__ float
//degrid_deapodize (const float r, const int ngrid, const float kernwidth, const float grid_oversamp)
//{
//#define SQR(x) ((x)*(x))
//#define BETA (M_PI*sqrtf(SQR(kernwidth/grid_oversamp*(grid_oversamp-0.5))-0.8))
    //float a = M_PI*kernwidth*r/float(ngrid);
    //float y = sqrtf(a*a - BETA*BETA);
    //float w = sinf(y) / y;
    //return w == 0.f ? 1.f : w;
//}


__global__ void
precompensate (float2 *nudata, const int nchan, const int nro, const int npe_per_image, const int nrest)
{
    float a = (2.f  - 2.f / float(npe_per_image)) / float(nro);
    float b = 1.f / float(npe_per_image);
    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < nrest; id += blockDim.x * gridDim.x)
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
    // set whole array to zero first (not most efficient!)
    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < ndst*ndst; id += blockDim.x * gridDim.x)
        for (int c = 0; c < nchan; ++c)
            dst[nchan*id + c] = make_float2(0.f, 0.f);
    // insert src into center of dst
    const int w = (ndst - nsrc) / 2;
    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < nsrc*nsrc; id += blockDim.x * gridDim.x)
    {
        int xdst = id / nsrc;
        int ydst = id % nsrc;
        int dstid = (xdst + w)*nsrc + ydst + w;
        for (int c = 0; c < nchan; ++c)
            dst[nchan*dstid + c] = src[nchan*id + c];
    }
}

extern "C" {  // don't mangle name, so can call from other languages

/*
    grid a single 2D image from input radial data
*/
__global__ void
gridradial2d (float2 *udata, const float2 * __restrict__ nudata, const int ngrid,
    const int nchan, const int nro, const int npe, const float kernwidth, const float grid_oversamp,
const int skip_angles, const int flag_postcomp, const int flag_golden_angle)
{
    // udata: [NCHAN x NGRID x NGRID], nudata: NCHAN x NRO x NPE
    //float grid_oversamp = float(ngrid) / float(nro); // grid_oversampling factor
    float2 utmp[MAXCHAN];
    const int blocksizex = 8; // TODO: optimize this blocking
    const int blocksizey = 4;
    const int warpsize = blocksizex*blocksizey;
    //int nblockx = ngrid / blocksizex;
    int nblocky = ngrid / blocksizey; // # of blocks along y dimension
    for (int tid = blockIdx.x * blockDim.x + threadIdx.x; tid < ngrid*ngrid; tid += blockDim.x * gridDim.x)
    {
        for (int ch = 0; ch < nchan; ch++)
          utmp[ch] = make_float2(0.f,0.f);

        //int x = id / ngrid - ngrid/2;
        //int y = -(id % ngrid) + ngrid/2;
        int z = tid / warpsize; // not a real z, just a block label
        int bx = z / nblocky;
        int by = z % nblocky;
        int zid = tid % warpsize;
        int x = zid / blocksizey + blocksizex*bx;
        int y = zid % blocksizey + blocksizey*by;
        int id = x*ngrid + y; // computed linear array index for uniform data
        x = -x + ngrid/2;
        y -= ngrid/2;
        float gridpoint_radius = hypotf(float(x), float(y));
        int rmax = fminf(floorf(gridpoint_radius + kernwidth)/grid_oversamp, nro/2-1);
        int rmin = fmaxf(ceilf(gridpoint_radius - kernwidth)/grid_oversamp, 0);  // define a circular band around the uniform point
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
                sf *= grid_oversamp;
                cf *= grid_oversamp;
                // TODO: fix this logic, try using without dtheta1
                //int rstart = dtheta1 <= dtheta || dtheta3 <= dtheta ? rmin : -rmax;
                //int rend   = dtheta1 <= dtheta || dtheta3 <= dtheta ? rmax : -rmin;
                int rstart = fabs(profile_theta-gridpoint_theta) < 0.5f*M_PI ? rmin : -rmax;
                int rend   = fabs(profile_theta-gridpoint_theta) < 0.5f*M_PI ? rmax : -rmin;
                for (int r = rstart; r <= rend; ++r)  // for each POSITIVE non-uniform ro point
                for (int r = rstart; r <= rend; ++r)  // for each POSITIVE non-uniform ro point
                {
                    float kx = r*cf; // [-NGRID/2 ... NGRID/2-1]    // TODO: compute distance in radial coordinates?
                    float ky = r*sf; // [-NGRID/2 ... NGRID/2-1]
                    float wgt = gridkernel(kx - x, ky - y, kernwidth, grid_oversamp);
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
    float2 *nudata, const float2 * __restrict__ udata, const int nimg,
    const int nchan, const int nro, const int npe, const float kernwidth,
    const float grid_oversamp, const int skip_angles, const int flag_golden_angle)
{
    // udata: [NCHAN x NGRID x NGRID], nudata: NCHAN x NRO x NPE
    //float grid_oversamp = float(ngrid) / float(nro); // grid_oversampling factor
    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < nro*npe; id += blockDim.x * gridDim.x)
    {
        int pe = id / nro; // find my location in the non-uniform data
        int ro = id % nro;
        float r = (ro - 0.5f * nro )/ (float)(nro); // [-0.5,0.5-1/nro] convert indices to (r,theta) coordinates
        float t = flag_golden_angle ? modang(PHI*(pe + skip_angles)) : float(pe) * M_PI / float(npe)+ M_PI/2;
        float kx = r*cos(t); // [-0.5,0.5-1/nro] Cartesian freqs of non-Cart datum  // TODO: _sincosf?
        float ky = r*sin(t); // [-0.5,0.5-1/nro]
        float x = nimg*(0.5 - kx);  // [0,ngrid] (x,y) coordinates in grid units
        float y = nimg*(ky + 0.5);

        for (int ch = 0; ch < nchan; ++ch) // zero my elements
             nudata[nchan*id + ch] = make_float2(0.f, 0.f);
        for (int ux = fmaxf(0.f,x-kernwidth); ux <= fminf(nimg-1,x+kernwidth); ++ux)
        for (int uy = fmaxf(0.f,y-kernwidth); uy <= fminf(nimg-1,y+kernwidth); ++uy)
        {
            float wgt = degridkernel(ux - x, uy - y, kernwidth, grid_oversamp);
            for (int ch = 0; ch < nchan; ++ch) {
                float2 c = udata[nchan*(ux*nimg + uy) + ch] / (nro*npe*kernwidth*kernwidth); // TODO: check this
                nudata[nchan*id + ch].x += wgt*c.x;
                nudata[nchan*id + ch].y += wgt*c.y;
            }
        }
    }
}


void
tron_init (TRON_plan *p)
{

  uint64_t ngridx = p->out_dims.x*p->grid_oversamp;
  uint64_t ngridy = p->out_dims.y*p->grid_oversamp;

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
  d_indatasize = dim_prod(&p->in_dims)*sizeof(float2) / p->nchunks;  // input data
  d_outdatasize = dim_prod(&p->out_dims)*sizeof(float2) / p->nchunks; // multi-coil gridded data

  // TODO: eliminate these by simplifying data storage
  d_gridsize = ngridx*ngridy*sizeof(float2);  // single channel grid size
  d_coilimgsize = p->in_dims.c*p->out_dims.x*p->out_dims.y*sizeof(float2); // coil images
  d_imgsize = p->out_dims.x*p->out_dims.y*sizeof(float2); // coil-combined image

  dprint(d_indatasize,ld);
  dprint(d_outdatasize,ld);
  dprint(d_gridsize,ld);
  dprint(d_coilimgsize,ld);
  dprint(d_imgsize,ld);

  for (int j = 0; j < NSTREAMS; ++j) // allocate data and initialize apodization and kernel texture
  {
      DPRINT("init STREAM %d\n", j);
      if (MULTI_GPU) cudaSetDevice(j % ndevices);
      cuTry(cudaStreamCreate(&stream[j]));

      fft_init(&fft_plan[j], p->out_dims.x, p->out_dims.y, p->out_dims.c);
      cufftSafeCall(cufftSetStream(fft_plan[j], stream[j]));

      fft_init(&fft_plan_os[j], ngridx, ngridy, p->out_dims.c);
      cufftSafeCall(cufftSetStream(fft_plan_os[j], stream[j]));

      cuTry(cudaMalloc((void **)&d_indata[j],  d_indatasize));
      cuTry(cudaMalloc((void **)&d_outdata[j], d_outdatasize));
      cuTry(cudaMalloc((void **)&d_tmp[j],     d_outdatasize));

      // // cuTry(cudaMemset(d_outdata[j], 0, p->d_outdatasize));
      // cuTry(cudaMalloc((void **)&d_coilimg[j], d_coilimgsize));
      // //cuTry(cudaMalloc((void **)&d_b1[j], d_coilimgsize));
      // cuTry(cudaMalloc((void **)&d_img[j], d_imgsize));

      // TODO: only fill apod if depapodize is called
      // TODO: handle adjoint vs non-adjoint
      cuTry(cudaMalloc((void **)&d_apodos[j], d_gridsize));
      cuTry(cudaMalloc((void **)&d_apod[j], d_imgsize));
      fillapod(d_apodos[j], p->out_dims.x, p->out_dims.y, p->kernwidth, p->grid_oversamp);
      crop<<<p->nimg,p->nimg>>>(d_apod[j], p->out_dims.x, p->out_dims.y, d_apodos[j], ngridx, ngridy, 1);
      cuTry(cudaFree(d_apodos[j]));

  }
}

void
tron_shutdown()
{
    DPRINT("freeing device memory\n");
    for (int j = 0; j < NSTREAMS; ++j) { // free allocated memory
        if (MULTI_GPU) cudaSetDevice(j % ndevices);
        cuTry(cudaFree(d_indata[j]));
        cuTry(cudaFree(d_outdata[j]));
        cuTry(cudaFree(d_tmp[j]));

        // cuTry(cudaFree(d_coilimg[j]));
        // //cuTry(cudaFree(d_b1[j]));
        // cuTry(cudaFree(d_img[j]));
        cuTry(cudaFree(d_apod[j]));
        cudaStreamDestroy(stream[j]);
    }
}


/*  Reconstruct images from 2D radial data.  This host routine calls the appropriate
    CUDA kernels in the correct order depending on the direction of recon.   */

__host__ void
recon_radial_2d(float2 *h_outdata, const float2 *__restrict__ h_indata, TRON_plan *p)
{
    DPRINT("recon_radial_2d\n");
    tron_init(p);
    int ngridx = p->out_dims.x * p->grid_oversamp;
    int ngridy = p->out_dims.y * p->grid_oversamp;
    dprint(p->out_dims.z, d);

    for (int z = 0; z < p->out_dims.z; ++z)
    {
        int j = z % NSTREAMS; // j is stream index
        if (MULTI_GPU) cudaSetDevice(j % ndevices);

        int peoffset = z*p->prof_slide;

        // address offsets into the data arrays
        size_t data_offset = p->in_dims.c * p->in_dims.r * peoffset;
        size_t img_offset = p->out_dims.x * p->out_dims.y * z;

        printf("[dev %d, stream %d] reconstructing slice %d/%d from PEs %d-%d (offset %ld)\n",
            j%ndevices, j, z+1, p->out_dims.z, z*p->prof_slide, (z+1)*p->prof_slide-1, data_offset);

        dprint(img_offset,ld);
        dprint(data_offset,ld);
        dprint(d_indatasize,ld);
        dprint(stream[j], ld);
        cuTry(cudaMemcpyAsync(d_indata[j], h_indata + data_offset, d_indatasize, cudaMemcpyHostToDevice, stream[j]));
        DPRINT("input data copied\n");

        if (p->flags.adjoint)
        {
            DPRINT("performing ADJOINT\n");
            // reverse from non-uniform data to image
            precompensate<<<gridsize,blocksize,0,stream[j]>>>(d_indata[j], p->in_dims.c, p->in_dims.r, p->prof_per_image, p->in_dims.theta);
            gridradial2d<<<gridsize,blocksize,0,stream[j]>>>(d_outdata[j], d_indata[j],
                ngridx, p->in_dims.c, p->in_dims.r, p->prof_per_image, p->kernwidth,
                p->grid_oversamp, p->skip_angles+peoffset, p->flags.postcomp, p->flags.golden_angle);
            ifftwithshift(d_outdata[j], fft_plan_os[j], j, ngridx, p->out_dims.t*p->out_dims.c);
            crop<<<gridsize,blocksize,0,stream[j]>>>(d_tmp[j], p->out_dims.x, p->out_dims.y, d_outdata[j], ngridx, ngridy, p->in_dims.c);
            // TODO: look at in_dims.c vs out_dims.c to decide whether to coil combine and by how much (can compress)
            coilcombinewalsh<<<gridsize,blocksize,0,stream[j]>>>(d_outdata[j], d_tmp[j], p->out_dims.x, p->in_dims.c, 1); /* 0 works, 1 good, 3 better */
            //coilcombinesos<<<gridsize,blocksize,0,stream[j]>>>(d_outdata[j], d_tmp[j], nimg, nchan);
            deapodize<<<gridsize,blocksize,0,stream[j]>>>(d_outdata[j], d_apod[j], p->out_dims.x, p->out_dims.y, p->out_dims.c);
            dprint(img_offset,ld);
            cuTry(cudaMemcpyAsync(h_outdata + img_offset, d_outdata[j], d_imgsize, cudaMemcpyDeviceToHost, stream[j]));
        }
        else
        {   // forward from image to non-uniform data
            DPRINT("performing FORWARD\n");
            degrid_deapodize<<<gridsize,blocksize,0,stream[j]>>>(d_indata[j], p->in_dims.x, 1, p->kernwidth, p->grid_oversamp);
            fftwithshift(d_indata[j], fft_plan_os[j], j, p->in_dims.x, p->in_dims.t*p->in_dims.c);
            //copy<<<gridsize,blocksize,0,stream[j]>>>(d_indata[j], d_img[j], nimg*nimg);
            degridradial2d<<<gridsize,blocksize,0,stream[j]>>>(d_outdata[j], d_indata[j],
                p->in_dims.x, p->in_dims.c, p->out_dims.r, p->out_dims.theta, p->kernwidth, p->grid_oversamp, p->skip_angles, p->flags.golden_angle);
            cuTry(cudaMemcpyAsync(h_outdata + p->out_dims.c*p->out_dims.r*p->out_dims.theta*z, d_indata[j], d_indatasize, cudaMemcpyDeviceToHost, stream[j]));
        }

    }

    tron_shutdown();
}




}

void
print_usage()
{
    fprintf(stderr, "Usage: tron [-3ahuv] [-r cmds] [-d prof_slide] [-k width] [-o grid_oversamp] [-s skip_angles] [-u data_undersamp] <infile.ra> [outfile.ra]\n");
    fprintf(stderr, "\t-3\t\t\3D koosh ball trajectory\n");
    fprintf(stderr, "\t-a\t\t\tadjoint operation\n");
    fprintf(stderr, "\t-d prof_slide\t\t\tnumber of phase encodes to slide between slices for helical scans\n");
    fprintf(stderr, "\t-g\t\t\tgolden angle radial\n");
    fprintf(stderr, "\t-h\t\t\tshow this help\n");
    fprintf(stderr, "\t-k width\t\twidth of gridding kernel\n");
    fprintf(stderr, "\t-o grid_oversamp\t\tgrid grid oversampling factor\n");
    fprintf(stderr, "\t-r nro\t\t\tnumber of readout points\n");
    fprintf(stderr, "\t-s skip_angles\t\tnumber of initial phase encodes to skip\n");
    fprintf(stderr, "\t-u data_undersamp\t\tinput data undersampling factor\n");
    fprintf(stderr, "\t-v\t\t\tverbose output\n");
}


int
main (int argc, char *argv[])
{
    float2 *h_indata, *h_outdata;
    ra_t ra_in, ra_out;
    int c, index;
    char infile[1024], outfile[1024];

    TRON_plan p;
    TRON_set_default_plan(&p);

    opterr = 0;
    while ((c = getopt (argc, argv, "3ad:ghk:o:r:s:u:v")) != -1)
    {
        switch (c) {
            case '3':
                p.flags.koosh = 1;
            case 'a':
                p.flags.adjoint = 1;
                break;
            case 'd':
                p.prof_slide = atoi(optarg);
                break;
            case 'g':
                p.flags.golden_angle = 1;
                break;
            case 'h':
                print_usage();
                return 1;
            case 'k':
                p.kernwidth = atof(optarg);
                break;
            case 'o':
                p.grid_oversamp = atof(optarg);
                break;
            case 'u':
                p.data_undersamp = atof(optarg);
                break;
            case 'r':
                p.nro = atoi(optarg);
                break;
            case 's':
                p.skip_angles = atoi(optarg);
                break;
            case 'v':
                flag_verbose = 1;
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

    dprint(p.skip_angles, d);
    dprint(p.prof_slide, d);
    dprint(p.kernwidth, .1f);
    dprint(p.data_undersamp, .3f);
    dprint(p.grid_oversamp, .3f);
    dprint(infile, s)
    dprint(outfile, s)

    DPRINT("Reading %s\n", infile);
    ra_read(&ra_in, infile);
    h_indata = (float2*)ra_in.data;
    assert(ra_in.ndims == 5);
    memcpy(p.in_dims.n, ra_in.dims, 5*sizeof(uint64_t));
    DPRINT("Sanity check: indata[0] = %f + %f i\n", h_indata[0].x, h_indata[0].y);
    DPRINT("in_dims = {%lu, %lu, %lu, %lu, %lu}\n", p.in_dims.c, p.in_dims.t, p.in_dims.x, p.in_dims.y, p.in_dims.z);
    assert(p.in_dims.c % 2 == 0 || p.in_dims.c == 1); // only single or even dimensions implemented for now


    printf("WARNING: Assuming square Cartesian dimensions for now.\n");

    // HERE IS WHERE WE COMPUTE OUTPUT DIMENSIONS BASED ON INPUT AND OPTIONAL ARGS
    if (p.flags.adjoint)
    {
        p.out_dims.c = 1;
        p.out_dims.t = p.in_dims.t;
        p.out_dims.x = p.in_dims.r / 2;
        p.out_dims.y = p.in_dims.r / 2;
        p.prof_per_image = p.data_undersamp * p.in_dims.r;  // TODO: fix this hack
        if (p.flags.koosh)
            p.out_dims.z = p.in_dims.r / 2;
        else
            p.out_dims.z = (p.in_dims.y - p.prof_per_image) / p.prof_slide;
    }
    else
    {
        p.prof_per_image = p.data_undersamp * p.in_dims.x;  // TODO: fix this hack
        p.out_dims.c = 1;
        p.out_dims.t = p.in_dims.t;
        p.out_dims.r = p.in_dims.x * 2;  // TODO: implement non-square images
        p.out_dims.theta = p.out_dims.r;   // TODO: make this more customizable
        p.out_dims.phi = p.flags.koosh ? p.out_dims.z : 1;
        p.prof_per_image = p.out_dims.theta;
    }
    p.nchunks = p.in_dims.n[3]*p.in_dims.n[4] / p.prof_per_image;
    dprint(p.nchunks,d);
    h_outdatasize = sizeof(float2);
    for (int k = 0; k < 5; ++k)
        h_outdatasize *= p.out_dims.n[k];


    // allocate pinned memory, which allows async calls
#ifdef CUDA_HOST_MALLOC
    cuTry(cudaMallocHost((void**)&h_outdata, h_outdatasize));
#else
    h_outdata = (float2*)malloc(h_outdatasize);
#endif


    DPRINT("Running reconstruction ...\n ");
    clock_t start = clock();

    recon_radial_2d(h_outdata, h_indata, &p);

    clock_t end = clock();
    DPRINT("Elapsed time: %.2f s\n", ((float)(end - start)) / CLOCKS_PER_SEC);

    DPRINT("Saving result to %s\n", outfile);
    ra_out.flags = 0;
    ra_out.eltype = 4;
    ra_out.elbyte = 8;
    ra_out.size = h_outdatasize;
    ra_out.ndims = 5;
    memcpy(ra_out.dims, p.out_dims.n, 5*sizeof(uint64_t));
    ra_out.data = (uint8_t*)h_outdata;
    ra_write(&ra_out, outfile);

    DPRINT("Cleaning up.\n");
    ra_free(&ra_in);
#ifdef CUDA_HOST_MALLOC
    //cudaFreeHost(&h_indata);
    cudaFreeHost(&h_outdata);
#else
    //free(h_indata);
    free(h_outdata);
#endif
    cudaDeviceReset();

    return 0;
}
