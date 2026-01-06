CXX = g++
NVCC = nvcc

CXXFLAGS = -O3 -std=c++17
OMPFLAGS = -fopenmp
NVCCFLAGS = -O3 -std=c++17

INCLUDES = -Iinclude

all: conv_cpu conv_omp conv_gpu_naive

conv_cpu: src/main_cpu.cpp src/conv_cpu.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $^

conv_omp: src/main_omp.cpp src/conv_cpu.cpp src/conv_omp.cpp
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) $(INCLUDES) -o $@ $^

conv_gpu_naive: src/main_gpu.cpp src/conv_cpu.cpp src/conv_cuda.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $@ $^

clean:
	rm -f conv_cpu conv_omp conv_gpu_naive
