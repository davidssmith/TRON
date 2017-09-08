#ifndef _NORM_H
#define _NORM_H

__host__ float norm (const float *d_x, const size_t N);
__host__ float norm (const float2 *d_x, const size_t N);
__host__ float dot (const float *d_a, const float *d_b, const size_t N);
__host__ float dot (const float2 *d_a, const float2 *d_b, const size_t N);

#endif
