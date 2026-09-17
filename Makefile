# ============================================================================
# C++ Software TPU Engine Makefile
# ============================================================================

CXX      = g++
CXXFLAGS = -std=c++17 -O3 -mavx2 -mfma -fopenmp -Wall -Wextra
TARGET   = software_tpu.exe
SOURCES  = main.cpp
HEADERS  = MemoryArena.cpp Tensor.h GEMM.h

all: $(TARGET)

$(TARGET): $(SOURCES) $(HEADERS)
	$(CXX) $(CXXFLAGS) $(SOURCES) -o $(TARGET)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET) *.o *.obj

.PHONY: all run clean
