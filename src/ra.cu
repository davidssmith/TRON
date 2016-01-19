
/* This is a slight modification to the RA file format provided at
   http://github.com/davidssmith/ra
   that includes CUDA support for allocating the RA data area as pinned
   memory which in turn allows asynchronous kernel execution.
*/

#include "ra.h"
#include "cuda_runtime.h"

inline void
gpuAssert (cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) { getchar(); exit(code); }
    }
}
#define cuTry(ans) { gpuAssert((ans), __FILE__, __LINE__); }

void
ra_query (const char *path)
{
    ra_t a;
    int j, fd;
    uint64_t magic;
    printf("---\nname: %s\n", path);
    fd = open(path, O_RDONLY);
    if (fd == -1)
        err(errno, "unable to open output file for writing");
    read(fd, &magic, sizeof(uint64_t));
    read(fd, &(a.flags), sizeof(uint64_t));
    read(fd, &(a.eltype), sizeof(uint64_t));
    read(fd, &(a.elbyte), sizeof(uint64_t));
    read(fd, &(a.size), sizeof(uint64_t));
    read(fd, &(a.ndims), sizeof(uint64_t));
    printf("endian: %s\n", a.flags  & RA_FLAG_BIG_ENDIAN ? "big" : "little");
    printf("type: %s%lld\n", RA_TYPE_NAMES[a.eltype], a.elbyte*8);
    printf("size: %lld\n", a.size);
    printf("dimension: %lld\n", a.ndims);
    a.dims = (uint64_t*)malloc(a.ndims*sizeof(uint64_t));
    read(fd, a.dims, a.ndims*sizeof(uint64_t));
    printf("shape:\n");
    for (j = 0; j < a.ndims; ++j)
        printf("  - %lld\n", a.dims[j]);
    printf("...\n");
    close(fd);
}

int
ra_read (ra_t *a, const char *path)
{
    int fd;
    uint64_t bytestoread, bytesleft, magic;
    fd = open(path, O_RDONLY);
    if (fd == -1)
        err(errno, "unable to open output file for writing");
    read(fd, &magic, sizeof(uint64_t));
    read(fd, &(a->flags), sizeof(uint64_t));
    read(fd, &(a->eltype), sizeof(uint64_t));
    read(fd, &(a->elbyte), sizeof(uint64_t));
    read(fd, &(a->size), sizeof(uint64_t));
    read(fd, &(a->ndims), sizeof(uint64_t));
    a->dims = (uint64_t*)malloc(a->ndims*sizeof(uint64_t));
    read(fd, a->dims, a->ndims*sizeof(uint64_t));
#ifdef CUDA_HOST_MALLOC
    cuTry(cudaMallocHost((void**)&(a->data), a->size))
    printf("RA using CUDA malloc\n");
#else
    a->data = (uint8_t*)malloc(a->size);
#endif
    if (a->data == NULL)
        err(errno, "unable to allocate memory for data");
    uint8_t *data_cursor = a->data;

    bytesleft = a->size;
    while (bytesleft > 0) {
        bytestoread = bytesleft < RA_MAX_BYTES ? bytesleft : RA_MAX_BYTES;
        read(fd, data_cursor, bytestoread);
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
    write(fd, &RA_MAGIC_NUMBER, sizeof(uint64_t));
    write(fd, &(a->flags), sizeof(uint64_t));
    write(fd, &(a->eltype), sizeof(uint64_t));
    write(fd, &(a->elbyte), sizeof(uint64_t));
    write(fd, &(a->size), sizeof(uint64_t));
    write(fd, &(a->ndims), sizeof(uint64_t));
    write(fd, a->dims, a->ndims*sizeof(uint64_t));

    bytesleft = a->size;
    data_in_cursor = a->data;

    bufsize = bytesleft < RA_MAX_BYTES ? bytesleft : RA_MAX_BYTES;
    while (bytesleft > 0) {
        write(fd, data_in_cursor, bufsize);
        data_in_cursor += bufsize / sizeof(uint8_t);
        bytesleft -= bufsize;
    }

    close(fd);
    return 0;
}


#ifdef TEST

#ifndef float2
    typedef struct { float x,y; } float2;
#endif

int
main ()
{
    float2 *r, *s;
    ra_t a,b;
    int k;
    uint64_t N = 12*sizeof(float2);
    printf("test data is %llu floats\n", N/sizeof(float2));
    r = (float2*)malloc(N);
    if (r == NULL)
        printf("could not allocate memory for test data\n");
    for (k = 0; k < N/sizeof(float2); ++k) {
        r[k].x = k;
        r[k].y = -1/(float)k;
    }
    a.flags = 0;
    a.eltype = RA_TYPE_COMPLEX;
    a.elbyte = sizeof(float2);
    a.size = N;
    a.ndims = 2;
    a.dims = (uint64_t*)malloc(a.ndims*sizeof(uint64_t));
    a.dims[0] = 3;
    a.dims[1] = 4;
    a.data = (void*)r;
    ra_write(&a, "test.ra");
    ra_read(&b, "test.ra");
    s = (float2*)b.data;
    for (k = 0; k < b.size/sizeof(float2); ++k) {
        if (r[k].x != s[k].x)
            printf("%f != %f\n",r[k].x, s[k].x);
    }
    for (k = 0; k < 10; ++k)
        printf("%f+%fim\n", s[k].x, s[k].y);
    printf("TESTS PASSED!\n");
    ra_free(&a);
    ra_free(&b);
    ra_query("test.ra");
    return 0;
}
#endif


void
ra_free (ra_t *a)
{
#ifdef CUDA_HOST_MALLOC
    cudaFreeHost(a->data);
#else
    free(a->data);
#endif
    free(a->dims);
}

