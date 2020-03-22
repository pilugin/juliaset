#include <QApplication>
#include "window.h"

int main(int argc, char** argv)
{
    QApplication app{argc, argv};

    Window w;
    w.setGeometry(0, 0, 200, 100);
    w.show();

    return app.exec();
}
