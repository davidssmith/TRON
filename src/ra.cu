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
