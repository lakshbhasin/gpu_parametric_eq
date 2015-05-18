CXX 	?= /usr/bin/g++
NVCC	?= $(CUDA_BIN_PATH)/nvcc

# CUDA path flags
CUDA_PATH	?= /opt/cuda
CUDA_INC_PATH   ?= $(CUDA_PATH)/include
CUDA_BIN_PATH   ?= $(CUDA_PATH)/bin
CUDA_LIB_PATH   ?= $(CUDA_PATH)/lib64

# CUDA code generation flags
GENCODE_FLAGS   := -gencode arch=compute_20,code=sm_20 \
		   -gencode arch=compute_30,code=sm_30 \
		   -gencode arch=compute_35,code=sm_35

# g++ flags (not including CUDA flags)
CXXFLAGS 	= -std=c++11 -O3 -Wall
LDFLAGS 	= -lboost_system -lboost_thread -lsfml-audio

# Comment this out if debugging.
# CXXFLAGS 	+= -DNDEBUG

# OS-specific CUDA build flags
ifneq ($(DARWIN),)
      LDFLAGS   += -Xlinker -rpath $(CUDA_LIB_PATH) -L$(CUDA_LIB_PATH) -lcudart -lcufft
      CXXFLAGS  += -arch $(OS_ARCH) -I$(CUDA_INC_PATH)
else
  ifeq ($(OS_SIZE),32)
      LDFLAGS   += -L$(CUDA_LIB_PATH) -lcudart -lcufft
      CXXFLAGS  += -m32 -I$(CUDA_INC_PATH)
  else
      LDFLAGS   += -L$(CUDA_LIB_PATH) -lcudart -lcufft
      CXXFLAGS  += -m64 -I$(CUDA_INC_PATH)
  endif
endif

# OS-architecture specific flags
ifeq ($(OS_SIZE),32)
      NVCCFLAGS := -m32 -lcufft
else
      NVCCFLAGS := -m64 -lcufft
endif

NVCCFLAGS += -O3 -I$(CUDA_INC_PATH)


TARGETS = parametric_eq wav_test

all: $(TARGETS)

wav_test: wav_test.o WavData.o
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

parametric_eq: parametric_eq.o parametric_eq_cuda.o WavData.o
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

parametric_eq_cuda.o: parametric_eq_cuda.cu parametric_eq_cuda.cuh
	$(NVCC) $(NVCCFLAGS) $(EXTRA_NVCCFLAGS) $(GENCODE_FLAGS) -c $< -o $@

%.o: %.cc
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f *.o $(TARGETS)

again: clean $(TARGETS)
