
#ifndef _FLOAT_MATH_H
#define _FLOAT_MATH_H

#include <math.h>
#include <cuda_runtime.h>

//inline float fminf (const float a, const float b) { return a < b ? a : b; }
//inline float fmaxf (const float a, const float b) { return a > b ? a : b; }

inline __device__ __host__ float clamp(const float f, const float a, const float b) { return fmaxf(a, fminf(f, b)); }

inline __host__ __device__ float2 make_float2(const float s) { return make_float2(s, 0.f); }

inline __host__ __device__ float2 operator-(const float2 a) { return make_float2(-a.x, -a.y); }
inline __host__ __device__ float2 operator+(const float2 a, const float2 b) { return make_float2(a.x + b.x, a.y + b.y); }
inline __host__ __device__ void operator+=(float2 &a, const float2 b) { a.x += b.x; a.y += b.y; }
inline __host__ __device__ float2 operator-(const float2 a, const float2 b) { return make_float2(a.x - b.x, a.y - b.y); }
inline __host__ __device__ void operator-=(float2 &a, const float2 b) { a.x -= b.x; a.y -= b.y; }
inline __host__ __device__ float2 operator*(const float2 a, const float s) { return make_float2(a.x * s, a.y * s); }
inline __host__ __device__ float2 operator*(const float s, const float2 a) { return make_float2(a.x * s, a.y * s); }
inline __host__ __device__ void operator*=(float2 &a, const float s) { a.x *= s; a.y *= s; }
inline __host__ __device__ void operator/=(float2 &a, const float s) { float inv = 1.0f / s; a *= inv; }
inline __host__ __device__ float2 operator/(const float2 a, const float2 b) { return make_float2(a.x / b.x, a.y / b.y); }
inline __host__ __device__ float2 operator/(const float2 a, const float s)
{
    float inv = 1.0f / s;
    return a * inv;
}
inline __host__ __device__ void operator*=(float2 &a, const float2 b) 
{ 
    float s = a.x;
    float t = a.y;
    a.x = s*b.x - t*b.y; 
    a.y = t*b.x + s*b.y;
}
inline __host__ __device__ float2 operator*(const float2 a, const float2 b) 
{ 
    // (a.x + i a.y) * (b.x + i b.y) = a.x*b.x - a.y*b.y + i(a.y*b.x + a.x*b.y)
    return make_float2(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x);
}

inline __host__ __device__ float dot (const float2 a, const float2 b)
{
    float2 res;
    res.x =  a.x * b.x + a.y * b.y;
    res.y = -a.y * b.x + b.y * a.x;
    return res;
}
inline __host__ __device__ float norm (const float2 a) { return a.x * a.x + a.y * a.y; }
inline __host__ __device__ float abs (const float2 a) { return hypotf(a.x,a.y); }
inline __host__ __device__ float2 conj (const float2 a) { return make_float2(a.x,-a.y); }

inline __host__ __device__ float2 operator/(const float s, const float2 b)
{
    return s * conj(b) / norm(b);
}
//inline __host__ __device__ float2 operator/(const float2 a, const float2 b)
//{
//    return a * conj(b) / norm(b);
//}

inline __host__ __device__ float cargf (const float2 a) { return atan2f(a.x, a.y); }

inline __device__ float2 cexpf (const float a)
{ 
    float c, s;
    __sincosf(a, &s, &c);
    return make_float2(c, s);
}

#endif /* _FLOAT_MATH_H */
