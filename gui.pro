######################################################################
# Automatically generated by qmake (3.0) Sat May 23 06:52:26 2015
######################################################################
FORMS    += ui/mainapp.ui

QMAKE_CFLAGS += -std=c++11 -Wno-unused-result -Wno-unused-parameter -Wno-unused-variable
QMAKE_CXXFLAGS += -std=c++11 -Wno-unused-result -Wno-unused-parameter -Wno-unused-variable
QMAKE_CLEAN += gui

UI_DIR = ./ui
MOC_DIR = ./ui

CXXFLAGS += -L/opt/lib64 -lcudart -lcufft -std=c++11 -O3 -Wall -Wno-unused-result -lboost_system -lboost_thread -lsfml-audio
LFLAGS   += -lboost_system -lboost_thread -lsfml-audio
greaterThan(QT_MAJOR_VERSION, 4): QT += widgets
TEMPLATE = app
TARGET = gui
INCLUDEPATH += . ./ui

OTHER_FILES += ./parametric_eq_cuda.cu

# CUDA settings <-- may change depending on your system
CUDA_SOURCES += ./parametric_eq_cuda.cu
CUDA_SDK = "/opt/cuda"   # Path to cuda SDK install
CUDA_DIR = "/opt/cuda"            # Path to cuda toolkit install

# Input
HEADERS += parametric_eq.hh parametric_eq_cuda.cuh WavData.hh ui/mainapp.hh
SOURCES += main.cc WavData.cc ui/mainapp.cc

# DO NOT EDIT BEYOND THIS UNLESS YOU KNOW WHAT YOU ARE DOING....

SYSTEM_NAME = unix         # Depending on your system either 'Win32', 'x64', or 'Win64'
SYSTEM_TYPE = 64            # '32' or '64', depending on your system
CUDA_ARCH = sm_20
NVCC_OPTIONS =--use_fast_math


# include paths
INCLUDEPATH += $$CUDA_DIR/include

# library directories
QMAKE_LIBDIR += $$CUDA_DIR/lib64/

CUDA_OBJECTS_DIR = ./


# Add the necessary libraries
CUDA_LIBS = -lcuda -lcudart

# The following makes sure all path names (which often include spaces) are put between quotation marks
CUDA_INC = $$join(INCLUDEPATH,'" -I"','-I"','"')
#LIBS += $$join(CUDA_LIBS,'.so ', '', '.so')
LIBS += $$CUDA_LIBS -lcudart -lcufft -std=c++11 -O3 -Wall -Wno-unused-result -lboost_system -lboost_thread -lsfml-audio

# Configuration of the Cuda compiler
CONFIG(debug, debug|release) {
    # Debug mode
    cuda_d.input = CUDA_SOURCES
    cuda_d.output = $$CUDA_OBJECTS_DIR/${QMAKE_FILE_BASE}_cuda.o
    cuda_d.commands = $$CUDA_DIR/bin/nvcc -D_DEBUG $$NVCC_OPTIONS $$CUDA_INC $$NVCC_LIBS --machine=$$SYSTEM_TYPE -arch=$$CUDA_ARCH -c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
    cuda_d.dependency_type = TYPE_C
    QMAKE_EXTRA_COMPILERS += cuda_d
}
else {
    # Release mode
    cuda.input = CUDA_SOURCES
    cuda.output = $$CUDA_OBJECTS_DIR/${QMAKE_FILE_BASE}_cuda.o
    cuda.commands = $$CUDA_DIR/bin/nvcc $$NVCC_OPTIONS $$CUDA_INC $$NVCC_LIBS --machine=$$SYSTEM_TYPE -arch=$$CUDA_ARCH -c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
    cuda.dependency_type = TYPE_C
    QMAKE_EXTRA_COMPILERS += cuda
}
