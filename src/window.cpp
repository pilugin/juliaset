#include "window.h"

#include "cuda_helpers.h"
#include "gradient.h"

#include "cuda/julia.cu.h"

#include <iostream>
#include <iomanip>
#include <algorithm>

using cuda::Err;

namespace {

void printColor(size_t i, size_t size, uint32_t rgba)
{
    float a = ((rgba >> 24)&0xFF) / 255.;
    float r = ((rgba >> 16)&0xFF) / 255.;
    float g = ((rgba >>  8)&0xFF) / 255.;
    float b = ((rgba >>  0)&0xFF) / 255.;

    std::cout << "#" << std::fixed << std::setw(6) << (static_cast<float>(i) / size) << " :" << a << " " << r << " " << g << " " << b << " hex:" << std::hex << rgba << "\n";
}

QPointF pointToModel(const QPoint& point, const QRectF& modelRect, int w, int h)
{
    const auto sw = static_cast<qreal>(w);
    const auto sh = static_cast<qreal>(h);

    const auto x0 = modelRect.left();
    const auto x1 = modelRect.right();
    const auto y0 = modelRect.top();
    const auto y1 = modelRect.bottom();

    const auto sx = static_cast<qreal>(point.x());
    const auto sy = static_cast<qreal>(point.y());

    return QPointF{
        (x1 - x0)*(sw - sx)/sw + x0,
        (y1 - y0)*(sh - sy)/sh + y0,
    };
}

QRectF scaleModelRect(const QPointF& zoomPoint, const QRectF& modelRect, qreal c)
{
    const auto cx = zoomPoint.x();
    const auto cy = zoomPoint.y();

    const auto x0 = modelRect.left();
    const auto x1 = modelRect.right();
    const auto y0 = modelRect.top();
    const auto y1 = modelRect.bottom();

    return QRectF{
        QPointF{cx - (cx - x0)*c, cy - (cy - y0)*c},
        QPointF{cx + (x1 - cx)*c, cy + (y1 - cy)*c},
    };
}

QRectF resizeModelRect(int oldw, int oldh, int neww, int newh, const QRectF& modelRect)
{
    const auto x0 = modelRect.left();
    const auto x1 = modelRect.right();
    const auto y0 = modelRect.top();
    const auto y1 = modelRect.bottom();

    const auto cx = (x1 + x0)/2.;
    const auto cy = (y1 + y0)/2.;

    const auto s = (x1 - x0)/static_cast<qreal>(oldw);
    static_cast<void>(oldh);
    const auto w = static_cast<qreal>(neww);
    const auto h = static_cast<qreal>(newh);

    return QRectF{
        QPointF{cx - (w/2.)*s, cy - (h/2.)*s},
        QPointF{cx + (w/2.)*s, cy + (h/2.)*s},
    };
}

}

Window::Window(QWidget* parent)
    : QOpenGLWidget{parent}
{
    setMouseTracking(true);
}

Window::~Window()
{
    if (devGradient_)
    {
        cudaFree(devGradient_) || Err{"Failed to free device memory"};
        devGradient_ = nullptr;
        devGradientSize_ = 0;
    }
}

void Window::initializeGL()
{
    initializeOpenGLFunctions();

    cuda::init();

    constexpr size_t gradientSize = 128;

    // TODO use constant memory here
    cudaMalloc(&devGradient_, gradientSize * sizeof(uint32_t)) || Err{"Failed to allocate device memory"};
    const auto hostGradient = generateGradient(generateDefaultGradientStops(), gradientSize);
    cudaMemcpy(devGradient_, hostGradient.data(), hostGradient.size()*sizeof(uint32_t), cudaMemcpyHostToDevice) || Err{"Failed to memcpy to device"};
    devGradientSize_ = gradientSize;

    // check interpolation
#if 0
    for (size_t i=0; i<hostGradient.size(); ++i)
    {
        printColor(i, hostGradient.size()-1, hostGradient[i]);
    }
#endif

    timerId_ = startTimer(30);

    glGenBuffers(1, &buf_);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, buf_);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, viewportSize_.width() * viewportSize_.height() * 4, NULL, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0); //< unbinding the pixel_unpack buffer

    cudaBufHandle_ = cuda::registerBuffer(buf_);

    glGenTextures(1, &texture_);
    glBindTexture(GL_TEXTURE_2D, texture_);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, viewportSize_.width(), viewportSize_.height(), 0, GL_BGRA, GL_UNSIGNED_BYTE, 0);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glEnable(GL_TEXTURE_2D);

    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
}

void Window::resizeGL(int w, int h)
{
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, w,
            0.0, h,
            -1.0, 1.0);

    glViewport(0, 0, w, h);

    // update size of PBO and Texture
    glBindTexture(GL_TEXTURE_2D, texture_);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_BGRA, GL_UNSIGNED_BYTE, 0);

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, buf_);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, w * h * 4, NULL, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0); //< unbinding the pixel_unpack buffer

    modelRect_ = resizeModelRect(viewportSize_.width(), viewportSize_.height(), w, h, modelRect_);
    viewportSize_.setWidth(w);
    viewportSize_.setHeight(h);
}

void Window::paintGL()
{
    glClear(GL_COLOR_BUFFER_BIT);

    void *devPtr = cuda::map(cudaBufHandle_);
    renderJuliaSet(devPtr,
                   viewportSize_.width(), viewportSize_.height(),
                   modelRect_.left(), modelRect_.right(), modelRect_.top(), modelRect_.bottom(),
                   devGradient_, devGradientSize_);
    cuda::unmap(cudaBufHandle_);

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, buf_);
    glBindTexture(GL_TEXTURE_2D, texture_);
    // Fast path due to BGRA
    // If the buffer is bound to PIXEL_UNPACK_BUFFER - use it as a texture source
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0,
                    viewportSize_.width(), viewportSize_.height(),
                    GL_BGRA, GL_UNSIGNED_BYTE, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    // do something with the texture
    glBegin(GL_QUADS);
        glTexCoord2f(0, 0);
        glVertex2f(0, 0);
        glTexCoord2f(1, 0);
        glVertex2f(viewportSize_.width(), 0);
        glTexCoord2f(1, 1);
        glVertex2f(viewportSize_.width(), viewportSize_.height());
        glTexCoord2f(0, 1);
        glVertex2f(0, viewportSize_.height());
    glEnd();
}

void Window::timerEvent(QTimerEvent* e)
{
    if (e->timerId() == timerId_)
    {
        const auto p = pointToModel(mousePoint_, modelRect_, viewportSize_.width(), viewportSize_.height());
        modelRect_ = scaleModelRect(p, modelRect_, scale_);
        update();
    }
}

void Window::mousePressEvent(QMouseEvent* e)
{
    constexpr qreal scaleStep = 0.975;


    scale_ = e->button() == Qt::LeftButton  ? scaleStep :
             e->button() == Qt::RightButton ? 1./scaleStep :
                                            1.;
    if (scale_ != 1.)
    {
        const auto p = pointToModel(mousePoint_, modelRect_, viewportSize_.width(), viewportSize_.height());
        modelRect_ = scaleModelRect(p, modelRect_, scale_);
        update();
        startZoomTimer();
    }
}

void Window::mouseMoveEvent(QMouseEvent* e)
{
    mousePoint_ = e->pos();
}

void Window::mouseReleaseEvent(QMouseEvent* e)
{
    stopZoomTimer();
}

void Window::startZoomTimer()
{
    constexpr int zoomInterval = 30;

    if (!timerId_)
    {
        timerId_ = startTimer(zoomInterval);
    }
}

void Window::stopZoomTimer()
{
    if (timerId_)
    {
        killTimer(timerId_);
        timerId_ = 0;
    }
}
