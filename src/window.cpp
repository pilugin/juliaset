#include "window.h"

#include "cuda_helpers.h"
#include "gradient.h"

#include "cuda/julia.cu.h"

#include <QImage>

using cuda::Err;

Window::Window(QWidget* parent)
    : QOpenGLWidget{parent}
{
    ;
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

    cudaMalloc(&devGradient_, gradientSize * sizeof(uint32_t)) || Err{"Failed to allocate device memory"};
    const auto hostGradient = generateGradient(generateDefaultGradientStops(), gradientSize);
    cudaMemcpy(devGradient_, hostGradient.data(), hostGradient.size()*sizeof(uint32_t), cudaMemcpyHostToDevice) || Err{"Failed to memcpy to device"};
    devGradientSize_ = gradientSize;

    timerId_ = startTimer(30);

    QImage img("some_image.png");
    imgSize_ = img.size();
    img = img.convertToFormat(QImage::Format_RGB32); // BGRA on little endian

    glGenBuffers(1, &buf_);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, buf_);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, imgSize_.width() * imgSize_.height() * 4, 0 /*img.constBits()*/, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0); //< unbinding the pixel_unpack buffer

    cudaBufHandle_ = cuda::registerBuffer(buf_);

    glGenTextures(1, &texture_);
    glBindTexture(GL_TEXTURE_2D, texture_);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, imgSize_.width(), imgSize_.height(), 0, GL_BGRA, GL_UNSIGNED_BYTE, 0);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

//    glEnable(GL_BLEND);
    glEnable(GL_TEXTURE_2D);

//    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
}

void Window::resizeGL(int w, int h)
{
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, w, h, 1.0, -1.0, 1.0);

    glViewport(0, 0, w, h);
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
}

void Window::paintGL()
{
    glClear(GL_COLOR_BUFFER_BIT);

    void *devPtr = cuda::map(cudaBufHandle_);
    renderJuliaSet(devPtr, imgSize_.width(), imgSize_.height(), scaleFactor_, devGradient_, devGradientSize_);
    cuda::unmap(cudaBufHandle_);

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, buf_);
    glBindTexture(GL_TEXTURE_2D, texture_);
    // Fast path due to BGRA
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, imgSize_.width(), imgSize_.height(), GL_BGRA, GL_UNSIGNED_BYTE, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    // do something with the texture
    glBegin(GL_QUADS);
        glTexCoord2f(0, 0);
        glVertex2f(0, 0);
        glTexCoord2f(1, 0);
        glVertex2f(width(), 0);
        glTexCoord2f(1, 1);
        glVertex2f(width(), height());
        glTexCoord2f(0,1);
        glVertex2f(0, height());
    glEnd();
}

void Window::timerEvent(QTimerEvent* e)
{
    if (e->timerId() == timerId_)
    {
        scaleFactor_ *= 0.985;
        update();
    }
}
