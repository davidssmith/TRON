/*
  This file is part of the RA package (http://github.com/davidssmith/ra).

  The MIT License (MIT)

  Copyright (c) 2015-2017 David Smith

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in all
  copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
  SOFTWARE.
*/
#include <assert.h>
#include <err.h>
#include <errno.h>
#include <fcntl.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sysexits.h>
#include <unistd.h>


#include "ra.h"
#include "float16.h"

#ifdef USE_CUDA
#  include "cuda_runtime.h"
inline void
gpuAssert (cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) { getchar(); exit(code); }
    }
}
#  define cuTry(ans) { gpuAssert((ans), __FILE__, __LINE__); }
#endif


void
validate_magic (const uint64_t magic)
{
   if (magic != RA_MAGIC_NUMBER) {
        fprintf(stderr, "Invalid RA file.\n");
        exit(EX_DATAERR);
   }
}

void
valid_read(int fd, void *buf, const size_t count)
{
    size_t nread = read(fd, buf, count);
    if (nread != count) {
        fprintf(stderr, "Read %lu B instead of %lu B.\n", nread, count);
        exit(EX_IOERR);
    }
}


void
valid_write(int fd, const void *buf, const size_t count)
{
    size_t nwrote = write(fd, buf, count);
    if (nwrote != count) {
        fprintf(stderr, "Wrote %lu B instead of %lu B.\n", nwrote, count);
        exit(EX_IOERR);
    }
}


// TODO: extend validation checks to internal consistency

void
ra_query (const char *path)
{
    ra_t a;
    int j, fd;
    uint64_t magic;
    fd = open(path, O_RDONLY);
    if (fd == -1)
        err(errno, "unable to open output file for writing");
    valid_read(fd, &magic, sizeof(uint64_t));
    validate_magic(magic);
    valid_read(fd, &(a.flags), sizeof(uint64_t));
    valid_read(fd, &(a.eltype), sizeof(uint64_t));
    valid_read(fd, &(a.elbyte), sizeof(uint64_t));
    valid_read(fd, &(a.size), sizeof(uint64_t));
    valid_read(fd, &(a.ndims), sizeof(uint64_t));
    printf("---\nname: %s\n", path);
    printf("endian: %s\n", a.flags  & RA_FLAG_BIG_ENDIAN ? "big" : "little");
    printf("type: %c%u\n", RA_TYPE_CODES[a.eltype], a.elbyte*8);
    printf("eltype: %u\n", a.eltype);
    printf("elbyte: %u\n", a.elbyte);
    printf("size: %u\n", a.size);
    printf("dimension: %u\n", a.ndims);
    a.dims = (uint64_t*)malloc(a.ndims*sizeof(uint64_t));
    valid_read(fd, a.dims, a.ndims*sizeof(uint64_t));
    printf("shape:\n");
    for (j = 0; j < a.ndims; ++j)
        printf("  - %lu\n", a.dims[j]);
    printf("...\n");
    close(fd);
    free(a.dims);
}


void
ra_dims (const char *path)
{
    ra_t a;
    int fd;
    uint64_t magic;
    fd = open(path, O_RDONLY);
    if (fd == -1)
        err(errno, "unable to open output file for writing");
    valid_read(fd, &magic, sizeof(uint64_t));
    validate_magic(magic);
    lseek(fd, 4*sizeof(uint64_t), SEEK_CUR);
    valid_read(fd, &(a.ndims), sizeof(uint64_t));
    a.dims = (uint64_t*)malloc(a.ndims*sizeof(uint64_t));
    valid_read(fd, a.dims, a.ndims*sizeof(uint64_t));
    for (uint64_t i = 0; i < a.ndims; ++i)
        printf("%lu ", a.dims[i]);
    printf("\n");
}


int
ra_read (ra_t *a, const char *path)
{
    int fd;
    uint64_t bytestoread, bytesleft, magic;
    fd = open(path, O_RDONLY);
    if (fd == -1)
        err(errno, "unable to open output file for writing");
    valid_read(fd, &magic, sizeof(uint64_t));
    validate_magic(magic);
    valid_read(fd, &(a->flags), sizeof(uint64_t));
    if ((a->flags & RA_UNKNOWN_FLAGS) != 0) {
        fprintf(stderr, "Warning: This RA file must have been written by a newer version of this\n");
        fprintf(stderr, "code. Correctness of input is not guaranteed. Update your version of the\n");
        fprintf(stderr, "RawArray package to stop this warning.\n");
    }
    valid_read(fd, &(a->eltype), sizeof(uint64_t));
    valid_read(fd, &(a->elbyte), sizeof(uint64_t));
    valid_read(fd, &(a->size), sizeof(uint64_t));
    valid_read(fd, &(a->ndims), sizeof(uint64_t));
    a->dims = (uint64_t*)malloc(a->ndims*sizeof(uint64_t));
    valid_read(fd, a->dims, a->ndims*sizeof(uint64_t));
    bytesleft = a->size;
    // if (a->flags & RA_FLAG_COMPRESSED)
    //     bytesleft += 16;
#ifdef USE_CUDA
    cuTry(cudaMallocHost((void**)&(a->data), a->size))
#else
    a->data = (uint8_t*)malloc(bytesleft);
#endif
    if (a->data == NULL)
        err(errno, "unable to allocate memory for data");
    uint8_t *data_cursor = a->data;
    while (bytesleft > 0) {
        bytestoread = bytesleft < RA_MAX_BYTES ? bytesleft : RA_MAX_BYTES;
        valid_read(fd, data_cursor, bytestoread);
        data_cursor += bytestoread;
        bytesleft -= bytestoread;
    }
    close(fd);
    return 0;
}



int
ra_write (ra_t *a, const char *path)
{
    int fd;
    uint64_t bytesleft, bufsize;
    uint8_t *data_in_cursor;
    fd = open(path, O_WRONLY|O_TRUNC|O_CREAT,0644);
    if (fd == -1)
        err(errno, "unable to open output file for writing");
    /* write the easy stuff */
    valid_write(fd, &RA_MAGIC_NUMBER, sizeof(uint64_t));
    valid_write(fd, &(a->flags), sizeof(uint64_t));
    valid_write(fd, &(a->eltype), sizeof(uint64_t));
    valid_write(fd, &(a->elbyte), sizeof(uint64_t));
    valid_write(fd, &(a->size), sizeof(uint64_t));
    valid_write(fd, &(a->ndims), sizeof(uint64_t));
    valid_write(fd, a->dims, a->ndims*sizeof(uint64_t));

    bytesleft = a->size;
    // if (a->flags & RA_FLAG_COMPRESSED) bytesleft += 16;
    data_in_cursor = a->data;

    bufsize = bytesleft < RA_MAX_BYTES ? bytesleft : RA_MAX_BYTES;
    while (bytesleft > 0) {
        valid_write(fd, data_in_cursor, bufsize);
        data_in_cursor += bufsize / sizeof(uint8_t);
        bytesleft -= bufsize;
    }

    close(fd);
    return 0;
}


void
ra_free (ra_t *a)
{
#ifdef USE_CUDA
    cudaFreeHost(a->data);
#else
    free(a->dims);
#endif
    free(a->data);
}


int
ra_reshape (ra_t *r, const uint64_t newdims[], const uint64_t ndimsnew)
{
    uint64_t newsize = 1;
    for (uint64_t k = 0; k < ndimsnew; ++k)
            newsize *= newdims[k];
    assert(r->size == newsize*r->elbyte);
    // if new dims preserve total number of elements, then change the dims
    r->ndims = ndimsnew;
    size_t newdimsize = ndimsnew*sizeof(uint64_t);
    free(r->dims);
    r->dims = (uint64_t*)malloc(newdimsize);
    memcpy(r->dims, newdims, newdimsize);
    return 0;
}


void
validate_conversion (const ra_t* r, const uint64_t neweltype, const uint64_t newelbyte)
{
    if (neweltype == RA_TYPE_USER) {
        printf("USER-defined type must be handled by the USER. :-)\n");
        exit(EX_USAGE);
    } //else if (r->eltype == RA_TYPE_COMPLEX && neweltype != RA_TYPE_COMPLEX) {
      //  printf("Warning: converting complex to non-complex types may discard information.\n");
    //} else if (newelbyte == r->elbyte && neweltype == r->eltype) {
    //    printf("Specified type is already the type of the source. Nothing to be done.\n");
    //    exit(EX_OK);
   //else if (newelbyte < r->elbyte && r->eltype != RA_TYPE_INT && r->eltype != RA_TYPE_UINT)
      //  printf("Warning: reducing type size may cause loss of precision.\n");
}


/*
union {
    double f;
    int64_t i;
    uint64_t u;
} t8;

union {
    float f;
    int32_t i;
    uint32_t u;
} t4;

union {
    float16 f;
    int16_t i;
    uint16_t u;
} t2;

union {
    int8_t i;
    uint8_t u;
    char c;
} t1;

*/

// float x = -3.14f;
// memcpy(&t4, &x, sizeof(float));

#undef CASE
#define CASE(TYPE1,BYTE1,TYPE2,BYTE2) \
    (r->eltype == RA_TYPE_##TYPE1 && r->elbyte == BYTE1 && \
     eltype == RA_TYPE_##TYPE2 && elbyte == BYTE2)

#define DECLARE_TYPED_CPTR(TYPE,BYTE,NAME)  RA_CTYPE_##TYPE##_##BYTE *NAME;

#define CONVERT(TYPE1,TYPE2) { \
    TYPE1 *tmp_src; tmp_src = (TYPE1 *)r->data; \
    TYPE2 *tmp_dst; tmp_dst = (TYPE2 *)tmp_data; \
    for (size_t i = 0; i < nelem; ++i) tmp_dst[i] = tmp_src[i]; }

#define CONVERT_TO_F16(TYPE1,TYPE2) { \
    TYPE1 *tmp_src; tmp_src = (TYPE1 *)r->data; \
    TYPE2 *tmp_dst; tmp_dst = (TYPE2 *)tmp_data; \
    for (size_t i = 0; i < nelem; ++i) tmp_dst[i] = float_to_float16((float)tmp_src[i]); }

#define CONVERT_FROM_F16(TYPE1,TYPE2) { \
    TYPE1 *tmp_src; tmp_src = (TYPE1 *)r->data; \
    TYPE2 *tmp_dst; tmp_dst = (TYPE2 *)tmp_data; \
    for (size_t i = 0; i < nelem; ++i) tmp_dst[i] = float16_to_float(tmp_src[i]); }

void
ra_convert (ra_t *r, const uint64_t eltype, const uint64_t elbyte)
{
    uint64_t j, nelem;
    uint8_t *tmp_data;

    if (r->eltype == eltype && r->elbyte == elbyte) // nothing to do
        return;

    // make sure this conversion will work
    validate_conversion(r, eltype, elbyte);

    // set new properties
    nelem = 1;
    for (j = 0; j < r->ndims; ++j)
        nelem *= r->dims[j];

    // convert the data type
    uint64_t newsize = elbyte * r->size / r->elbyte;
    tmp_data = (uint8_t*)malloc(newsize);

    if CASE(INT,1,INT,2)             // INT -> INT
        CONVERT(int8_t,int16_t)
    else if CASE(INT,1,INT,4)
        CONVERT(int8_t,int32_t)
    else if CASE(INT,1,INT,8)
        CONVERT(int8_t,int64_t)

    else if CASE(INT,2,INT,1)
        CONVERT(int16_t,int8_t)
    else if CASE(INT,2,INT,4)
        CONVERT(int16_t,int32_t)
    else if CASE(INT,2,INT,8)
        CONVERT(int16_t,int64_t)

    else if CASE(INT,4,INT,1)
        CONVERT(int32_t,int8_t)
    else if CASE(INT,4,INT,2)
        CONVERT(int32_t,int16_t)
    else if CASE(INT,4,INT,8)
        CONVERT(int32_t,int64_t)

    else if CASE(INT,8,INT,1)
        CONVERT(int64_t,int8_t)
    else if CASE(INT,8,INT,2)
        CONVERT(int64_t,int16_t)
    else if CASE(INT,8,INT,4)
        CONVERT(int64_t,int32_t)

    else if CASE(UINT,1,INT,1)        // UINT -> INT
        CONVERT(uint8_t,int8_t)
    else if CASE(UINT,1,INT,2)
        CONVERT(uint8_t,int16_t)
    else if CASE(UINT,1,INT,4)
        CONVERT(uint8_t,int32_t)
    else if CASE(UINT,1,INT,8)
        CONVERT(uint8_t,int64_t)

    else if CASE(UINT,2,INT,1)
        CONVERT(uint16_t,int8_t)
    else if CASE(UINT,2,INT,2)
        CONVERT(uint16_t,int16_t)
    else if CASE(UINT,2,INT,4)
        CONVERT(uint16_t,int32_t)
    else if CASE(UINT,2,INT,8)
        CONVERT(uint16_t,int64_t)

    else if CASE(UINT,4,INT,1)
        CONVERT(uint32_t,int8_t)
    else if CASE(UINT,4,INT,2)
        CONVERT(uint32_t,int16_t)
    else if CASE(UINT,4,INT,4)
        CONVERT(uint32_t,int32_t)
    else if CASE(UINT,4,INT,8)
        CONVERT(uint32_t,int64_t)

    else if CASE(UINT,8,INT,1)
        CONVERT(uint64_t,int8_t)
    else if CASE(UINT,8,INT,2)
        CONVERT(uint64_t,int16_t)
    else if CASE(UINT,8,INT,4)
        CONVERT(uint64_t,int32_t)
    else if CASE(UINT,8,INT,8)
        CONVERT(uint64_t,int64_t)

    else if CASE(UINT,1,UINT,2)        // UINT -> UINT
        CONVERT(uint8_t,uint16_t)
    else if CASE(UINT,1,UINT,4)
        CONVERT(uint8_t,uint32_t)
    else if CASE(UINT,1,UINT,8)
        CONVERT(uint8_t,uint64_t)

    else if CASE(UINT,2,UINT,1)
        CONVERT(uint16_t,uint8_t)
    else if CASE(UINT,2,UINT,4)
        CONVERT(uint16_t,uint32_t)
    else if CASE(UINT,2,UINT,8)
        CONVERT(uint16_t,uint64_t)

    else if CASE(UINT,4,UINT,1)
        CONVERT(uint32_t,uint8_t)
    else if CASE(UINT,4,UINT,2)
        CONVERT(uint32_t,uint16_t)
    else if CASE(UINT,4,UINT,8)
        CONVERT(uint32_t,uint64_t)

    else if CASE(UINT,8,UINT,1)
        CONVERT(uint64_t,uint8_t)
    else if CASE(UINT,8,UINT,2)
        CONVERT(uint64_t,uint16_t)
    else if CASE(UINT,8,UINT,4)
        CONVERT(uint64_t,uint32_t)

    else if CASE(INT,1,FLOAT,4)
        CONVERT(int8_t,float)
    else if CASE(INT,2,FLOAT,4)
        CONVERT(int16_t,float)
    else if CASE(INT,4,FLOAT,4)
        CONVERT(int32_t,float)
    else if CASE(INT,8,FLOAT,4)
        CONVERT(int64_t,float)
    else if CASE(UINT,1,FLOAT,4)
        CONVERT(uint8_t,float)
    else if CASE(UINT,2,FLOAT,4)
        CONVERT(uint16_t,float)
    else if CASE(UINT,4,FLOAT,4)
        CONVERT(uint32_t,float)
    else if CASE(UINT,8,FLOAT,4)
        CONVERT(uint64_t,float)

    else if CASE(INT,1,FLOAT,2)
        CONVERT_TO_F16(int8_t,float16)
    else if CASE(UINT,1,FLOAT,2)
        CONVERT_TO_F16(uint8_t,float16)
    else if CASE(INT,2,FLOAT,2)
        CONVERT_TO_F16(int16_t,float16)
    else if CASE(UINT,2,FLOAT,2)
        CONVERT_TO_F16(uint16_t,float16)
    else if CASE(FLOAT,2,INT,1)
        CONVERT_FROM_F16(float16,int8_t)
    else if CASE(FLOAT,2,UINT,1)
        CONVERT_FROM_F16(float16,uint8_t)
    else if CASE(FLOAT,2,INT,2)
        CONVERT_FROM_F16(float16,int16_t)
    else if CASE(FLOAT,2,UINT,2)
        CONVERT_FROM_F16(float16,uint16_t)

    else if CASE(FLOAT,4,UINT,1)
        CONVERT(float,uint8_t)
    else if CASE(FLOAT,4,UINT,2)
        CONVERT(float,uint16_t)
    else if CASE(FLOAT,4,UINT,4)
        CONVERT(float,uint32_t)
    else if CASE(FLOAT,4,UINT,8)
        CONVERT(float,uint64_t)
    else if CASE(FLOAT,4,INT,1)
        CONVERT(float,int8_t)
    else if CASE(FLOAT,4,INT,2)
        CONVERT(float,int16_t)
    else if CASE(FLOAT,4,INT,4)
        CONVERT(float,int32_t)
    else if CASE(FLOAT,4,INT,8)
        CONVERT(float,int64_t)

    else if CASE(INT,1,FLOAT,8)
        CONVERT(int8_t,double)
    else if CASE(INT,2,FLOAT,8)
        CONVERT(int16_t,double)
    else if CASE(INT,4,FLOAT,8)
        CONVERT(int32_t,double)
    else if CASE(INT,8,FLOAT,8)
        CONVERT(int64_t,double)
    else if CASE(UINT,1,FLOAT,8)
        CONVERT(uint8_t,double)
    else if CASE(UINT,2,FLOAT,8)
        CONVERT(uint16_t,double)
    else if CASE(UINT,4,FLOAT,8)
        CONVERT(uint32_t,double)
    else if CASE(UINT,8,FLOAT,8)
        CONVERT(uint64_t,double)

    else if CASE(FLOAT,8,UINT,1)
        CONVERT(double,uint8_t)
    else if CASE(FLOAT,8,UINT,2)
        CONVERT(double,uint16_t)
    else if CASE(FLOAT,8,UINT,4)
        CONVERT(double,uint32_t)
    else if CASE(FLOAT,8,UINT,8)
        CONVERT(double,uint64_t)
    else if CASE(FLOAT,8,INT,1)
        CONVERT(double,int8_t)
    else if CASE(FLOAT,8,INT,2)
        CONVERT(double,int16_t)
    else if CASE(FLOAT,8,INT,4)
        CONVERT(double,int32_t)
    else if CASE(FLOAT,8,INT,8)
        CONVERT(double,int64_t)

    else if CASE(FLOAT,4,FLOAT,8)     // FLOATS AND COMPLEX
        CONVERT(float,double)
    else if CASE(FLOAT,8,FLOAT,4)
        CONVERT(double,float)
    else if CASE(FLOAT,4,FLOAT,2)
        CONVERT_TO_F16(float,uint16_t)
    else if CASE(FLOAT,8,FLOAT,2)
        CONVERT_TO_F16(double,uint16_t)
    else if CASE(FLOAT,2,FLOAT,4)
        CONVERT_FROM_F16(uint16_t,float)
    else if CASE(FLOAT,2,FLOAT,8)
        CONVERT_FROM_F16(uint16_t,double)
    else if CASE(COMPLEX,16,COMPLEX,8) {
        nelem *= 2;
        CONVERT(double,float)
    } else if CASE(COMPLEX,8,COMPLEX,16) {
        nelem *= 2;
        CONVERT(float,double)
    } else if CASE(COMPLEX,8,COMPLEX,4) {
        nelem *= 2;
        CONVERT_TO_F16(float,float16)
    } else if CASE(FLOAT,4,COMPLEX,8) {
        float *tmp_src = (float *)r->data;
        float *tmp_dst = (float *)tmp_data;
        for (size_t i = 0; i < nelem; ++i) {
            tmp_dst[2*i] = tmp_src[i];
            tmp_dst[2*i+1] = 0.f;
        }
    } else if CASE(FLOAT,8,COMPLEX,16) {
        double *tmp_src = (double *)r->data;
        double *tmp_dst = (double *)tmp_data;
        for (size_t i = 0; i < nelem; ++i) {
            tmp_dst[2*i] = tmp_src[i];
            tmp_dst[2*i+1] = 0.;
        }
    } else if CASE(COMPLEX,8,FLOAT,4) {       // complex -> float using real part
        float *tmp_src = (float *)r->data;
        float *tmp_dst = (float *)tmp_data;
        for (size_t i = 0; i < nelem; ++i) {
            tmp_dst[i] = tmp_src[2*i];
        }
    } else if CASE(COMPLEX,16,FLOAT,8) {
        double *tmp_src = (double *)r->data;
        double *tmp_dst = (double *)tmp_data;
        for (size_t i = 0; i < nelem; ++i) {
            tmp_dst[i] = tmp_src[2*i];
        }
    } else {
        printf("Specified type and size did not conform to any supported combinations.\n");
        exit(EX_USAGE);
    }

    r->eltype = eltype;
    r->elbyte = elbyte;
    r->size = newsize;
    free(r->data);
    r->data = (uint8_t*)tmp_data;
}


uint64_t
calc_min_elbyte_int (const int64_t max, const int64_t min)
{
    //printf("min: %d, max: %d\n", min, max);
    int minbits_reqd = log(max)/log(2);
    //printf("minbits_reqd: %d\n", minbits_reqd);
    uint64_t m = 8;
    while (m < minbits_reqd)
        m *= 2;
    return m / 8;
}

uint64_t
calc_min_elbyte_float (const double max, const double min)
{
    //printf("min: %g, max: %g\n", min, max);
    double dynamic_range = fabs(min/max);
    //printf("dynamic_range: %g\n", dynamic_range);
    // if (minbits < 16)
    //     return 2;
    // else if (minbits < 32)
    //     return 4;
    // else
    return 8;
}


#define MINMAX_AND_RESCALE_INT(NAME,TYPE) \
    { TYPE* rdr; rdr = (TYPE*) r->data; \
      int64_t min = rdr[0]; int64_t max = rdr[0]; \
      for (size_t i = 1; i < nelem; ++i) { \
          if (rdr[i] < min) min = rdr[i]; \
          if (rdr[i] > max) max = rdr[i]; }\
      min_elbyte = calc_min_elbyte_int(max, min); \
     }

 #define MINMAX_AND_RESCALE_FLOAT(NAME,TYPE) \
     { TYPE* rdr; rdr = (TYPE*) r->data; \
       double min = rdr[0]; double max = rdr[0]; \
       for (size_t i = 1; i < nelem; ++i) { \
           if (rdr[i] < min) min = rdr[i]; \
           if (rdr[i] > max) max = rdr[i]; }\
       min_elbyte = calc_min_elbyte_float(max, min); \
      }

#undef CASE
#define CASE(TYPE1,BYTE1) \
    (r->eltype == RA_TYPE_##TYPE1 && r->elbyte == BYTE1)


int
ra_squash (ra_t *r)
{
    uint64_t nelem = 1, min_elbyte = 8, orig_elbyte = r->elbyte;
    for (uint64_t j = 0; j < r->ndims; ++j)
        nelem *= r->dims[j];

    if CASE(INT,2)
        MINMAX_AND_RESCALE_INT(r,int16_t)
    else if CASE(INT,4)
        MINMAX_AND_RESCALE_INT(r,int32_t)
    else if CASE(INT,8)
        MINMAX_AND_RESCALE_INT(r,int64_t)
    else if CASE(UINT,2)
        MINMAX_AND_RESCALE_INT(r,uint16_t)
    else if CASE(UINT,4)
        MINMAX_AND_RESCALE_INT(r,uint32_t)
    else if CASE(UINT,8)
        MINMAX_AND_RESCALE_INT(r,uint64_t)
    else if CASE(FLOAT,4)
        MINMAX_AND_RESCALE_FLOAT(r,float)   // TODO: implement float16
    else if CASE(FLOAT,8)
        MINMAX_AND_RESCALE_FLOAT(r,double)
    else if CASE(COMPLEX,8) {
        nelem *= 2;
        MINMAX_AND_RESCALE_FLOAT(r,float)
    } else if CASE(COMPLEX,16) {
        nelem *= 2;
        MINMAX_AND_RESCALE_FLOAT(r,double)
    }

    if (min_elbyte < r->elbyte) {
        orig_elbyte = r->elbyte;
        ra_convert(r, r->eltype, min_elbyte);
    }

    return min_elbyte != orig_elbyte;
}


int
ra_diff (const ra_t *a, const ra_t *b)
{
    if (a->flags  != b->flags)  return 1;
    if (a->eltype != b->eltype) return 2;
    if (a->elbyte != b->elbyte) return 3;
    if (a->size   != b->size)   return 4;
    if (a->ndims  != b->ndims)  return 5;
    for (size_t i = 0; i < a->ndims; ++i)
        if (a->dims[i] != b->dims[i]) return 6;
    for (size_t i = 0; i < a->size; ++i)
        if (a->data[i] != b->data[i]) { printf(" <<%u %u>> ", a->data[i], b->data[i]); return 7; }
    return 0;
}

/*
void
ra_export_pgm (const ra_t *a)
{
    double datamax, datamin, tmp;
    assert(a->ndims == 2);
    printf("P2\n%d %d\n", a->dims[0], a->dims[1]);
    int maxpix = 255;
    printf("%d\n", maxval);
    for (int j = 0; j < a->dims[0]*a->dims[1])
    {
        if (a->eltype == RA_TYPE_COMPLEX)
            tmp = sqrtf(a->data[2*j]a->data[2*j+1]);
        else
            tmp = a->data[j];
        if (j == 0) {
            datamin = tmp;
            datamax = tmp;
        }
        if (tmp > datamax)
            datamax = tmp;
        if (tmp < datamin)
            datamin = tmp;
    }
    for (int j = 0; j < a->dims[0]*a->dims[1])
    {
        printf("%d ", (int)((maxpix*(a->data[j] - datamin)) / (datamax - datamin)));
        if ((j+1) % a->dims[1] == 0) printf("\n");
    }

}
*/
