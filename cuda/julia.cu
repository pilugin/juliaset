#include "julia.cu.h"

namespace {

struct cuComplex
{
    double r;
    double i;

    __device__ cuComplex(double rr = 0.0, double ii = 0.0) : r{rr}, i{ii} { ; }

    __device__ double magnitude2() const
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

constexpr double DIM = 1000.;

__device__ int julia(int x, int y, double scale, size_t maxIter)
{
    double jx = scale * (DIM/2. - x)/(DIM/2.);
    double jy = scale * (DIM/2. - y)/(DIM/2.);

    cuComplex c{-0.8, 0.156}; // -0.8; 0.156
    cuComplex a{jx, jy};
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

__global__ void juliaKernel(uchar4* ptr, int w, int h, double scale, const uchar4* gradient, size_t gradientSize)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y*w;

    if (x >= w || y >= h)
    {
        return;
    }

    uchar4& pixel = ptr[offset];

    const size_t maxIter = gradientSize * 2;

    int juliaValue = julia(x, y, scale, maxIter);

    const float c = static_cast<float>(juliaValue) / static_cast<float>(maxIter);

    pixel = gradient[ static_cast<int>(static_cast<float>(gradientSize - 1) * c) ];
}

} // namespace

void renderJuliaSet(void* devPtr, int w, int h, double scaleFactor, const void* gradient, size_t gradientSize)
{
    const unsigned int blockSize = 32;

    dim3 grid{(w + blockSize-1)/blockSize, (h + blockSize - 1)/blockSize};
    dim3 block{blockSize, blockSize};

    juliaKernel<<<grid, block>>>
                            (static_cast<uchar4*>(devPtr), w, h, scaleFactor,
                             static_cast<const uchar4*>(gradient), gradientSize);
}
