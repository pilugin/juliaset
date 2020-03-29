#include "julia.cu.h"

namespace {

struct cuComplex
{
    float r;
    float i;

    __device__ cuComplex(float rr = 0.0, float ii = 0.0) : r{rr}, i{ii} { ; }

    __device__ float magnitude2() const
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

__device__ int julia(float x, float y, size_t maxIter)
{
    cuComplex c{-0.8, 0.156}; // -0.8; 0.156
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
                            int w, int h,
                            float x0, float x1, float y0, float y1,
                            const uchar4* gradient, size_t gradientSize)
{
    const int px = threadIdx.x + blockIdx.x * blockDim.x;
    const int py = threadIdx.y + blockIdx.y * blockDim.y;
    const int offset = px + py*w;

    const float x = x0 + (x1 - x0)*static_cast<float>(px)/w;
    const float y = y0 + (y1 - y0)*static_cast<float>(py)/h;

    if (px >= w || py >= h)
    {
        return;
    }

    uchar4& pixel = ptr[offset];

    const size_t maxIter = gradientSize;

    const int juliaValue = julia(x, y, maxIter);

    const float c = static_cast<float>(juliaValue) / static_cast<float>(maxIter);

    pixel = gradient[ static_cast<int>(static_cast<float>(gradientSize - 1) * c) ];
}

} // namespace

void renderJuliaSet(void* devPtr,
                    int w, int h,
                    float x0, float x1, float y0, float y1,
                    const void* gradient, size_t gradientSize)
{
    const unsigned int blockSize = 32;

    dim3 grid{(w + blockSize-1)/blockSize, (h + blockSize - 1)/blockSize};
    dim3 block{blockSize, blockSize};

    juliaKernel<<<grid, block>>>
                            (static_cast<uchar4*>(devPtr),
                             w, h,
                             x0, x1, y0, y1,
                             static_cast<const uchar4*>(gradient), gradientSize);
}
