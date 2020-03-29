#pragma once

//#define real double
#define real float

void renderJuliaSet(void* devPtr,
                    real seedR, real seedI,
                    int w, int h,
                    real x0, real x1, real y0, real y1,
                    const void* gradient, size_t gradientSize);
