#pragma once

#include <QtOpenGL>
#include <QRectF>

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

    void mousePressEvent(QMouseEvent* e) override;
    void mouseMoveEvent(QMouseEvent* e) override;

  private:
    GLuint buf_;
    GLuint texture_;
    void* cudaBufHandle_ {nullptr};

    void* devGradient_ {nullptr};
    size_t devGradientSize_ {0};

    QRectF modelRect_ {-1., -1., 2., 2.};
    QSize viewportSize_ {100, 100};
    QPoint prevMousePos_ {};
    QPoint mousePressPos_ {};

    float scale_ {1.};
};
