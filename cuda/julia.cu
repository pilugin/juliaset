#include "julia.cu.h"

namespace {

struct cuComplex
{
    real r;
    real i;

    __device__ cuComplex(real rr = 0.0, real ii = 0.0) : r{rr}, i{ii} { ; }

    __device__ real magnitude2() const
    {
        return r*r + i*i;
    }

    __device__ cuComplex operator+(const cuComplex& other) const
    {
        return cuComplex{r + other.r, i + other.i};
    }

    __device__ cuComplex operator*(const cuComplex& other) const
    {
        return cuComplex{r*other.r - i*other.i, i*other.r + r*other.i};
    }
};

__device__ int julia(real seedR, real seedI, real x, real y, size_t maxIter)
{
    cuComplex c{seedR, seedI}; // -0.8; 0.156
    cuComplex a{x, y};
    int i=0;
    for (; i<maxIter; ++i)
    {
        a = a*a + c;
        if (a.magnitude2() > 1000)
        {
            break;
        }
    }
    return i;
}

__global__ void juliaKernel(uchar4* ptr,
                            real seedR, real seedI,
                            int w, int h,
                            real x0, real x1, real y0, real y1,
                            const uchar4* gradient, size_t gradientSize)
{
    const int px = threadIdx.x + blockIdx.x * blockDim.x;
    const int py = threadIdx.y + blockIdx.y * blockDim.y;
    const int offset = px + py*w;

    const real x = x0 + (x1 - x0)*static_cast<real>(px)/w;
    const real y = y0 + (y1 - y0)*static_cast<real>(py)/h;

    if (px >= w || py >= h)
    {
        return;
    }

    uchar4& pixel = ptr[offset];

    const size_t maxIter = gradientSize;

    const int juliaValue = julia(seedR, seedI, x, y, maxIter);

    const real c = static_cast<real>(juliaValue) / static_cast<real>(maxIter);

    pixel = gradient[ static_cast<int>(static_cast<real>(gradientSize - 1) * c) ];
}

} // namespace

void renderJuliaSet(void* devPtr,
                    real seedR, real seedI,
                    int w, int h,
                    real x0, real x1, real y0, real y1,
                    const void* gradient, size_t gradientSize)
{
    const unsigned int blockSize = 32;

    dim3 grid{(w + blockSize-1)/blockSize, (h + blockSize - 1)/blockSize};
    dim3 block{blockSize, blockSize};

    juliaKernel<<<grid, block>>>
                            (static_cast<uchar4*>(devPtr),
                             seedR, seedI,
                             w, h,
                             x0, x1, y0, y1,
                             static_cast<const uchar4*>(gradient), gradientSize);
}
