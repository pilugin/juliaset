TEMPLATE = app
TARGET = juliaset-qt

INCLUDEPATH += . src

QT += opengl
CONFIG += c++14

OBJECTS_DIR=.obj
MOC_DIR=.obj
DESTDIR=bin

# The following define makes your compiler warn you if you use any
# feature of Qt which has been marked as deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if you use deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

# Input
HEADERS += src/window.h \
           src/cuda_helpers.h \
           src/cuda_err.h \
           src/gradient.h

SOURCES += src/main.cpp \
           src/window.cpp \
           src/cuda_helpers.cpp \
           src/gradient.cpp

#
### CUDA ###
#
CUDA_DIR = /home/piliuhko/.cache/bazel/_bazel_piliuhko/ws/external/cuda9_pc_linux_amd64_gcc

INCLUDEPATH += $$CUDA_DIR/include
LIBS += -L$$CUDA_DIR/lib64 -lcuda -lcudart

CUDA_SOURCES = cuda/julia.cu

cuda.input = CUDA_SOURCES
cuda.output = ${OBJECTS_DIR}/${QMAKE_FILE_BASE}.cu.o
cuda.commands = $$CUDA_DIR/bin/nvcc -std=c++14 -c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
cuda.dependency_type = TYPE_C
cuda.depend_command = $$CUDA_DIR/bin/nvcc -M ${QMAKE_FILE_NAME}

QMAKE_EXTRA_COMPILERS += cuda
