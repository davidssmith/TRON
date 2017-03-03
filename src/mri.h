#ifndef _MRI_H
#define _MRI_H

#include <assert.h>
#include <string.h>
#include "ra.h"

// #ifndef float2
//     typedef struct { float x, y; } float2;
// #endif

typedef union {
    struct { uint64_t c,  t,  x,     y,   z; };
    struct { uint64_t c_, t_, r, theta, phi; };
    uint64_t n[5];
} dim_t;

typedef struct {
    dim_t dims;    // channels x repeats x space1 x space2 x space3
    float2 *data;  // single precision complex
} MR_data;

void
MR_data_read (MR_data *d, char *rafile)
{
    ra_t r;
    ra_read(&r, rafile);
    d->data = (float2*)malloc(r.size);
    memcpy(d->data, r.data, r.size);
    memcpy(d->dims.n, r.dims, r.ndims*sizeof(uint64_t));
    ra_free(&r);
}

void
MR_data_free (MR_data *d)
{
    free(d->data);
}




#endif /* _MRI_H */
