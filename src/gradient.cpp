#include "gradient.h"

#include <cassert>

std::vector<GradientStop> generateDefaultGradientStops()
{
    return {
        { 0.0,    0xFF640700 },
        { 0.16,   0xFFcb6b20 },
        { 0.42,   0xFFffffed },
        { 0.6425, 0xFF00aaff },
        { 0.6575, 0xFF000200 },
        { 1.0,    0xFF000000 },
    };
}

namespace {

inline uint32_t interpolate(uint32_t a, uint32_t b, float c)
{
    assert(c >= 0.0);
    assert(c <= 1.0);
    auto da = static_cast<double>(a);
    auto db = static_cast<double>(b);
    auto dc = static_cast<double>(c);
    return static_cast<uint32_t>( da + (db - da)*dc );
}

}

std::vector<uint32_t> generateGradient(const std::vector<GradientStop>& stops, size_t size)
{
    assert(stops.size() > 1);
    assert(stops.front().pos == 0.0);
    assert(stops.back().pos == 1.0);

    const float step = 1.0 / size;
    float v = 0.0;

    std::vector<uint32_t> rv;
    rv.reserve(size);

    auto stop0 = stops.begin();
    auto stop1 = stop0 + 1;
    assert(stop0->pos < stop1->pos);

    for (size_t i=0; i<size; ++i, v += step)
    {
        if (v > stop1->pos)
        {
            ++stop0;
            ++stop1;
            assert(stop0->pos < stop1->pos);
            assert(stop1 != stops.end());
        }

        const float dpos = stop1->pos - stop0->pos;
        const float curV = v - stop0->pos;
        const float c = curV / dpos;

        rv.push_back(interpolate(stop0->color, stop1->color, c));
    }

    return rv;
}