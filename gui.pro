FORMS    += ui/mainapp.ui

QMAKE_CXXFLAGS  += -std=c++11 -Wall -Wno-unused-result \
                   -Wno-unused-parameter -Wno-unused-variable
QMAKE_LFLAGS    += -lboost_system -lboost_thread -lsfml-audio \
                   -lsfml-system -lQt5Widgets -lQt5Gui -lQt5Core \
                   -lGL -lpthread
QMAKE_CLEAN     += gui 

UI_DIR = ./ui
MOC_DIR = ./ui

# Specific flags for release configuration
QMAKE_CXXFLAGS_RELEASE  -= -O2
QMAKE_CXXFLAGS_RELEASE  += -O3 #-DNDEBUG

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets printsupport
TEMPLATE = app
TARGET = gui

INCLUDEPATH     += . ./ui 

OTHER_FILES += ./parametric_eq_cuda.cu

# CUDA settings <-- may change depending on your system
CUDA_SOURCES += ./parametric_eq_cuda.cu
CUDA_SDK = "/opt/cuda"            # Path to cuda SDK install
CUDA_DIR = "/opt/cuda"            # Path to cuda toolkit install

# Input
HEADERS += parametric_eq.hh parametric_eq_cuda.cuh eq_stream.hh \
           WavData.hh ui/mainapp.hh ui/qcustomplot.hh
SOURCES += main.cc parametric_eq.cc eq_stream.cc WavData.cc \
           ui/mainapp.cc ui/qcustomplot.cc

# DO NOT EDIT BEYOND THIS UNLESS YOU KNOW WHAT YOU ARE DOING....

SYSTEM_NAME = unix
SYSTEM_TYPE = 64
CUDA_ARCH = sm_20
NVCC_OPTIONS = --use_fast_math

# Include paths
INCLUDEPATH += $$CUDA_DIR/include

# Library directories
QMAKE_LIBDIR += $$CUDA_DIR/lib64/
CUDA_OBJECTS_DIR = ./

# Add the necessary libraries
CUDA_LIBS = -lcuda -lcudart -lcufft

# The following makes sure all path names (which often include spaces) are put between quotation marks
CUDA_INC = $$join(INCLUDEPATH,'" -I"','-I"','"')
LIBS += $$CUDA_LIBS 

# Configuration of the Cuda compiler

CONFIG(debug, debug|release) {
    # Debug mode
    cuda_d.input = CUDA_SOURCES
    cuda_d.output = $$CUDA_OBJECTS_DIR/${QMAKE_FILE_BASE}_cuda.o
    cuda_d.commands = $$CUDA_DIR/bin/nvcc -D_DEBUG $$NVCC_OPTIONS $$CUDA_INC \
                      $$NVCC_LIBS --machine=$$SYSTEM_TYPE -arch=$$CUDA_ARCH \
                      -c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
    cuda_d.dependency_type = TYPE_C
    QMAKE_EXTRA_COMPILERS += cuda_d
}
else {
    # Release mode
    cuda.input = CUDA_SOURCES
    cuda.output = $$CUDA_OBJECTS_DIR/${QMAKE_FILE_BASE}_cuda.o
    cuda.commands = $$CUDA_DIR/bin/nvcc $$NVCC_OPTIONS $$CUDA_INC $$NVCC_LIBS \
                    --machine=$$SYSTEM_TYPE -arch=$$CUDA_ARCH -c -o \
                    ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
    cuda.dependency_type = TYPE_C
    QMAKE_EXTRA_COMPILERS += cuda
}
