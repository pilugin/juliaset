#pragma once

#include "cuda_err.h"

#include <GL/gl.h>
#include <cuda.h>

namespace cuda {

void init();
void *registerBuffer(GLuint buf);
void unregisterBuffer(void *res);
void *map(void *res);
void unmap(void *res);

}