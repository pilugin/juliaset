#pragma once

#include <QtOpenGL>
#include <QSize>

class Window: public QOpenGLWidget, protected QOpenGLFunctions
{
    Q_OBJECT

  public:
    Window(QWidget* parent = nullptr);
    ~Window();

  protected:
    void initializeGL() override;
    void resizeGL(int w, int h) override;
    void paintGL() override;

    void timerEvent(QTimerEvent* e) override;

  private:
    GLuint buf_;
    GLuint texture_;
    void* cudaBufHandle_ {nullptr};

    void* devGradient_ {nullptr};
    size_t devGradientSize_ {0};

    int timerId_ {0};
    double scaleFactor_ {1.0};
};
