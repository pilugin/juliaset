#include "cuda_helpers.h"

#include <stdexcept>
#include <cstring>

#include <cuda_gl_interop.h>

namespace cuda {

void init()
{
    cudaDeviceProp prop;
    int dev;
    memset(&prop, 0, sizeof(cudaDeviceProp));
    prop.major = 3;
    prop.minor = 2;

    cudaChooseDevice(&dev, &prop) || Err{"Failed to choose device"};
    cudaGLSetGLDevice(dev) || Err{"Failed to set gl device"};
}

void *registerBuffer(GLuint buf)
{
    cudaGraphicsResource *res = 0;
    cudaGraphicsGLRegisterBuffer(&res, buf, cudaGraphicsRegisterFlagsNone) || Err{"Failed to register buffer"};
    return res;
}

void unregisterBuffer(void *res)
{
    cudaGraphicsUnregisterResource((cudaGraphicsResource *) res) || Err{"Failed to unregister resource for buffer"};
}

void *map(void *res)
{
    cudaGraphicsMapResources(1, (cudaGraphicsResource **) &res) || Err{"Failed to map resource"};
    void *devPtr = 0;
    size_t size;
    cudaGraphicsResourceGetMappedPointer(&devPtr, &size, (cudaGraphicsResource *) res) || Err{"Failed to get device pointer"};
    return devPtr;
}

void unmap(void *res)
{
    cudaGraphicsUnmapResources(1,(cudaGraphicsResource **) &res) || Err{"Failed to unmap resource"};
}

}
