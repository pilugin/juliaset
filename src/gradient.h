#pragma once

#include <vector>
#include <cuda.h>

struct GradientStop
{
    float pos;
    uint32_t color;
};

std::vector<GradientStop> generateDefaultGradientStops();

std::vector<uint32_t> generateGradient(const std::vector<GradientStop>& stops, size_t size);
