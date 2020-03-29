#pragma once

void renderJuliaSet(void* devPtr,
                    int w, int h,
                    float x0, float x1, float y0, float y1,
                    const void* gradient, size_t gradientSize);
