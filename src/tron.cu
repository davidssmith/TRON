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
#include <math.h>
#include <complex.h>
#include <time.h>
#include <stdint.h>
#include <cufft.h>
#include <cuda_runtime.h>
#include "float2math.h"
#include "ra.h"

// GLOBAL VARIABLES
#define NSTREAMS   2
#define NCHAN      8
#define MAXCHAN    8
#define MULTI_GPU  1
#define PHI        1.9416089796736116f

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



__host__ __device__ void
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
coilcombinewalsh (float2 *d_img, const float2 * __restrict__ d_coilimg, 
   const int nimg, const int nchan, const int npatch)
{
    float2 A[MAXCHAN*MAXCHAN];
    for (size_t id = blockIdx.x * blockDim.x + threadIdx.x; id < nimg*nimg; id += blockDim.x * gridDim.x)
    {
        int x = id / nimg;
        int y = id % nimg;
        for (int k = 0; k < NCHAN*NCHAN; ++k)
            A[k] = make_float2(0.f,0.f);
        for (int px = max(0,x-npatch); px <= min(nimg-1,x+npatch); ++px)
            for (int py = max(0,y-npatch); py <= min(nimg-1,y+npatch); ++py) {
                int offset = nchan*(px*nimg + py);
                for (int c2 = 0; c2 < nchan; ++c2)
                    for (int c1 = 0; c1 < nchan; ++c1)
                        A[c1*nchan + c2] += d_coilimg[offset+c1]*conj(d_coilimg[offset+c2]);
            }
        powit(A, nchan, 5);
        d_img[id] = make_float2(0.f, 0.f);
        for (int c = 0; c < nchan; ++c)
            d_img[id] += conj(A[c])*d_coilimg[nchan*id+c]; // * cexpf(-maxphase);
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
gridkernel (const float dx, const float dy)
{
    float r2 = dx*dx + dy*dy;
#ifdef KERN_KB
    const float kernwidth = 2.f;
    const float osfactor = 2.f;
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


extern "C" {  // don't mangle name, so can call from other languages


__global__ void
gridradial2d (
    float2 *udata, const float2 * __restrict__ nudata, const int ngrid,
    const int nchan, const int nro, const int npe, const float kernwidth,
const int peskip)
{
    // udata: [NCHAN x NGRID x NGRID], nudata: NCHAN x NRO x NPE
    float osfactor = float(ngrid) / float(nro); // oversampling factor
    float2 utmp[NCHAN];
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
        x -= ngrid/2;
        y = -y +  ngrid/2;
        float myradius = hypotf(x, y);
        int rmax = fminf(floorf(myradius + kernwidth)/osfactor, nro/2-1);
        int rmin = fmaxf(ceilf(myradius - kernwidth)/osfactor, 0);  // define a circular band around the uniform point
        for (int ch = 0; ch < nchan; ++ch)
             udata[nchan*id + ch] = make_float2(0.f,0.f);
        if (rmin > nro/2-1) continue; // outside non-uniform data area

        float sdc = 0.f;

        float mytheta = modang(atan2f(float(y),float(x))); // get uniform point coordinate in non-uniform system, (r,theta) in this case
        float dtheta = atan2f(kernwidth, myradius); // narrow that band to an arc

        // TODO: replace this logic with boolean function that can be
        // swapped out for diff acquisitions
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
                    sdc += wgt;
                    for (int ch = 0; ch < nchan; ch += 2) {
                        utmp[ch] += wgt*nudata[nchan*(nro*pe + r + nro/2) + ch];
                        utmp[ch + 1] += wgt*nudata[nchan*(nro*pe + r + nro/2) + ch + 1];
                    }
                }
            }
        }
        for (int ch = 0; sdc > 0.f && ch < nchan; ++ch)
            udata[nchan*id + ch] = utmp[ch] / sdc;
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
        float t = modang(PHI*(pe + peskip)); // golden angle specific!
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
            for (int ch = 0; ch < nchan; ++ch) {
                float2 c = udata[nchan*(ux*ngrid + uy) + ch];
                nudata[nchan*id + ch].x += wgt*c.x;
                nudata[nchan*id + ch].y += wgt*c.y;
            }
        }
    }
}


__host__ void
recongar2d (float2 *h_img, const float2 *__restrict__ h_nudata,
    int nchan, const int nro, const int npe, const int nslices, const int ndyn,
    const int ngrid, const int nimg, const int npe_per_dyn, const int dpe, const int peskip,
    const float oversamp, const float kernwidth)
{
    float2 *d_nudata[NSTREAMS], *d_udata[NSTREAMS], *d_coilimg[NSTREAMS],
        *d_img[NSTREAMS], *d_b1[NSTREAMS], *d_apodos[NSTREAMS], *d_apod[NSTREAMS];
    cudaStream_t stream[NSTREAMS];

    int ndevices;
    if (MULTI_GPU) {
    	cuTry(cudaGetDeviceCount(&ndevices));
    } else
    	ndevices = 1;
    printf("MULTI_GPU = %d\n", MULTI_GPU);
    printf("NSTREAMS = %d\n", NSTREAMS);
    printf("using %d CUDA devices\n", ndevices);

    int blocksize = 96;    // CUDA kernel parameters, TWEAK HERE to optimize
    int gridsize = 2048;
    printf("kernels configured with %d blocks of %d threads\n", gridsize, blocksize);

    // array sizes
    const size_t d_nudatasize = nchan*nro*npe_per_dyn*sizeof(float2);  // input data
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
        cuTry(cudaMemset(d_udata[j], 0, d_udatasize));
        cuTry(cudaMalloc((void **)&d_coilimg[j], d_coilimgsize));
        cuTry(cudaMalloc((void **)&d_b1[j], d_coilimgsize));
        cuTry(cudaMalloc((void **)&d_img[j], d_imgsize));
        cuTry(cudaMalloc((void **)&d_apodos[j], d_gridsize));
        cuTry(cudaMalloc((void **)&d_apod[j], d_imgsize));
        fillapod(d_apodos[j], ngrid, kernwidth);
        crop<<<nimg,nimg>>>(d_apod[j], nimg, d_apodos[j], ngrid, 1);
        cuTry(cudaFree(d_apodos[j]));
    }

    printf("iterating over %d slices\n", nslices);
    for (int s = 0; s < nslices; ++s) // MAIN LOOP
        for (int t = 0; t < ndyn; ++t)
        {
            int j = t % NSTREAMS; // stream
            if (MULTI_GPU) cudaSetDevice(j % ndevices);
            int peoffset = t*dpe;
            size_t data_offset = nchan*nro*(npe*s + peoffset);
            size_t img_offset = nimg*nimg*(ndyn*s + t);
            printf("[dev %d, stream %d] reconstructing slice %d/%d, dyn %d/%d from PEs %d-%d (offset %ld)\n", j%ndevices, j, s+1,
                nslices, t+1, ndyn, t*dpe, (t+1)*dpe-1, data_offset);
            cuTry(cudaMemcpyAsync(d_nudata[j], h_nudata + data_offset, d_nudatasize, cudaMemcpyHostToDevice, stream[j]));
            gridradial2d<<<gridsize,blocksize,0,stream[j]>>>(d_udata[j], d_nudata[j], ngrid, nchan, nro, npe_per_dyn, kernwidth, peskip+peoffset);
            fftshift<<<gridsize,blocksize,0,stream[j]>>>(d_udata[j], d_udata[j], ngrid, nchan);
            cufftSafeCall(cufftExecC2C(inverse_plan[j], d_udata[j], d_udata[j], CUFFT_INVERSE));
            fftshift<<<gridsize,blocksize,0,stream[j]>>>(d_udata[j], d_udata[j], ngrid, nchan);
            crop<<<gridsize,blocksize,0,stream[j]>>>(d_coilimg[j], nimg, d_udata[j], ngrid, nchan);
            if (nchan > 1) {
#ifdef WALSH_COMB
                coilcombinewalsh<<<gridsize,blocksize,0,stream[j]>>>(d_img[j], d_b1[j], d_coilimg[j], nimg, nchan, 1); // 0 works, 1 good, 3 optimal
#else
                coilcombinesos<<<gridsize,blocksize,0,stream[j]>>>(d_img[j], d_coilimg[j], nimg, nchan);
#endif
            }
            deapodize<<<gridsize,blocksize,0,stream[j]>>>(d_img[j], d_apod[j], nimg, 1);
            cuTry(cudaMemcpyAsync(h_img + img_offset, d_img[j], d_imgsize, cudaMemcpyDeviceToHost, stream[j]));
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


void 
print_usage()
{ 
    fprintf(stderr, "Usage: rr2d [-a] [-d dpe] [-o oversamp] [-s peskip] [-w kernwidth] <infile.ra> [outfile.ra]\n");
    fprintf(stderr, "\t-a\t\t\tuse adjoint transform\n");
    fprintf(stderr, "\t-d dpe\t\t\tnumber of phase encodes to skip between slices\n");
    fprintf(stderr, "\t-o oversamp\t\tgrid oversampling factor\n");
    fprintf(stderr, "\t-s peskip\t\tnumber of phase encodes to skip at beginning\n");
    fprintf(stderr, "\t-w kernwidth\t\twidth of gridding kernel\n");
}


int
main (int argc, char *argv[])
{
    // for testing
    float2 *h_nudata, *h_img;
    float oversamp = 2.f;   // option o
    float kernwidth = 2.f;  // option w
    int dpe = 21;  // option d
    int peskip = 0; //7999;  // option s  
    int adjoint = 0; // option a
    ra_t ra_in, ra_out;
    int c, index;
    char infile[1024], outfile[1024];
  

    opterr = 0;
    while ((c = getopt (argc, argv, "ad:ho:s:w:")) != -1)
    {
        switch (c) {
            case 'a':
                printf("Using adjoint transformation.\n");
                adjoint = 1;  // perform adjoint operation
                break;
            case 'd':
                dpe = atoi(optarg);
                break;
            case 'o':
                oversamp = atof(optarg);
                break;      
            case 's':
                peskip = atoi(optarg);
                break;
            case 'w':
                kernwidth = atof(optarg);
                break;
            case 'h':
                print_usage();
                break;
            default:
                print_usage();
                return 1;
        }
    }
    

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
    printf("Skipping first %d PEs.\n", peskip);  
    printf("PE spacing set to %d.\n", dpe);
    printf("Kernel width set to %f.\n", kernwidth);
    printf("Oversampling factor set to %f.\n", oversamp);
    printf("Infile: %s\n", infile);
    printf("Outfile: %s\n", outfile);

    printf("reading %s\n", infile);
    ra_read(&ra_in, infile);
    printf("dims = {%lld, %lld, %lld, %lld}\n", ra_in.dims[0],
        ra_in.dims[1], ra_in.dims[2], ra_in.dims[3]);
    int nchan = ra_in.dims[0];
    assert(nchan % 2 == 0);
    //int necho = ra_in.dims[1];
    int nro = ra_in.dims[2];
    int npe = ra_in.dims[3];
    int ngrid = nro*oversamp;
    int npe_per_slice = nro/2;
    int nimg = 3*nro/4;
    int ndyn = (npe - npe_per_slice) / dpe;
    int nslices = 1;
    h_nudata = (float2*)ra_in.data;

    printf("sanity check: nudata[0] = %f + %f i\n", h_nudata[0].x, h_nudata[0].y);


    // allocate pinned memory, which allows async calls
#ifdef CUDA_HOST_MALLOC
    //cuTry(cudaMallocHost((void**)&h_nudata, nchan*nro*npe*sizeof(float2)));
    cuTry(cudaMallocHost((void**)&h_img, nimg*nimg*ndyn*nslices*sizeof(float2)));
#else
    //h_nudata = (float2*)malloc(nchan*nro*npe*sizeof(float2));
    h_img = (float2*)malloc(nimg*nimg*ndyn*nslices*sizeof(float2));
#endif



    clock_t start = clock();
    // the magic happens
    recongar2d(h_img, h_nudata, nchan, nro, npe, nslices, ndyn, ngrid, nimg, npe_per_slice,
        dpe, peskip, oversamp, kernwidth);
    clock_t end = clock();
    printf("elapsed time: %.2f s\n", ((float)(end - start)) / CLOCKS_PER_SEC);

    // save results
    ra_out.flags = 0;
    ra_out.eltype = 4;
    ra_out.elbyte = 8;
    ra_out.size = sizeof(float2)*nimg*nimg*nslices*ndyn;
    ra_out.ndims = 4;
    ra_out.dims = (uint64_t*)malloc(4*sizeof(uint64_t));
    ra_out.dims[0] = 1;
    ra_out.dims[1] = nimg;
    ra_out.dims[2] = nimg;
    ra_out.dims[3] = ndyn;
    ra_out.data = (uint8_t*)h_img;
    printf("write result to %s\n", outfile);
    ra_write(&ra_out, outfile);

    
    printf("free host memory\n");

    ra_free(&ra_in);
#ifdef CUDA_HOST_MALLOC
    //cudaFreeHost(&h_nudata);
    cudaFreeHost(&h_img);
#else
    //free(h_nudata);
    free(h_img);
#endif
    cudaDeviceReset();

    return 0;
}