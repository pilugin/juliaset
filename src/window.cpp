#include "window.h"

#include "cuda_helpers.h"
#include "gradient.h"

#include "cuda/julia.cu.h"

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

    // TODO use constant memory here
    cudaMalloc(&devGradient_, gradientSize * sizeof(uint32_t)) || Err{"Failed to allocate device memory"};
    const auto hostGradient = generateGradient(generateDefaultGradientStops(), gradientSize);
    cudaMemcpy(devGradient_, hostGradient.data(), hostGradient.size()*sizeof(uint32_t), cudaMemcpyHostToDevice) || Err{"Failed to memcpy to device"};
    devGradientSize_ = gradientSize;

    timerId_ = startTimer(30);

    glGenBuffers(1, &buf_);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, buf_);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, geometry().width() * geometry().height() * 4, NULL, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0); //< unbinding the pixel_unpack buffer

    cudaBufHandle_ = cuda::registerBuffer(buf_);

    glGenTextures(1, &texture_);
    glBindTexture(GL_TEXTURE_2D, texture_);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, geometry().width(), geometry().height(), 0, GL_BGRA, GL_UNSIGNED_BYTE, 0);

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
            h, 0.0,
            -1.0, 1.0);

    glViewport(0, 0, w, h);

    // update size of PBO and Texture
    glBindTexture(GL_TEXTURE_2D, texture_);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_BGRA, GL_UNSIGNED_BYTE, 0);

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, buf_);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, w * h * 4, NULL, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0); //< unbinding the pixel_unpack buffer

}

void Window::paintGL()
{
    glClear(GL_COLOR_BUFFER_BIT);

    void *devPtr = cuda::map(cudaBufHandle_);
    renderJuliaSet(devPtr, geometry().width(), geometry().height(), scaleFactor_, devGradient_, devGradientSize_);
    cuda::unmap(cudaBufHandle_);

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, buf_);
    glBindTexture(GL_TEXTURE_2D, texture_);
    // Fast path due to BGRA
    // If the buffer is bound to PIXEL_UNPACK_BUFFER - use it as a texture source
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, geometry().width(), geometry().height(), GL_BGRA, GL_UNSIGNED_BYTE, 0);
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
