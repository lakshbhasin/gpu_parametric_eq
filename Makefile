CXX = g++
FLAGS = -c

EXE = wav_test
SRCS = WavData.cc wav_test.cc
OBJS = $(subst .cc,.o, $(SRCS))

all: $(EXE)

$(EXE): $(OBJS)
	$(CXX) -o $(EXE) $(OBJS)

clean:
	$(RM) $(OBJS) $(EXE)
