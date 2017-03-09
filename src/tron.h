#ifndef _TRON_H
#define _TRON_H

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



// non-uniform data shape: nchan x nrep x nro x npe
// uniform data shape:     nchan x nrep x ngrid x ngrid x nz
// image shape:            nchan x nrep x nimg x nimg x nz
// coil-combined image:            nrep x nimg x nimg x nz


typedef struct {
    float grid_oversamp;  // TODO: compute ngrid from nx, ny and oversamp
    float kernwidth;
    float data_undersamp;

    int prof_per_image;     // # of profiles to use for each reconstructed image; default: data_undersamp * nro
    int prof_slide;         // # of profiles to slide through the data between reconstructed images
    int skip_angles;        // # of angles to skip at beginning of image stack

    dim_t in_dims;
    dim_t out_dims;

    int nchan;  // p->dims.c;
    int nrep;  // p->dims.t; # of repeated measurements of same trajectory
    int nro;
    int npe;
    int ngrid;   // p.out_dims.x * grid_oversamp;
    int nx, ny, nz;
    int nimg;

    struct {
        unsigned adjoint       : 1;
        unsigned postcomp      : 1;
        unsigned deapodize     : 1;
        unsigned koosh         : 1;
        unsigned golden_angle  : 5;   // padded to 8 bits
    } flags;

} TRON_plan;

extern "C" {  // don't mangle name, so can call from other languages

/* grid a single 2D image from input radial data */
__global__ void
gridradial2d (float2 *udata, const float2 * __restrict__ nudata, const int ngrid,
    const int nchan, const int nro, const int npe, const float kernwidth, const float grid_oversamp,
const int skip_angles, const int flag_postcomp, const int flag_golden_angle);

/*  generate 2D radial data from an input 2D image */
__global__ void degridradial2d (
    float2 *nudata, const float2 * __restrict__ udata, const int nimg,
    const int nchan, const int nro, const int npe, const float kernwidth,
    const float grid_oversamp, const int skip_angles, const int flag_golden_angle);

/*  Reconstruct images from 2D radial data.  This host routine calls the appropriate
    CUDA kernels in the correct order depending on the direction of recon.   */
__host__ void
recon_radial_2d (float2 *h_outdata, const float2 *__restrict__ h_indata, TRON_plan *p);

}

#endif /* _TRON_H */
