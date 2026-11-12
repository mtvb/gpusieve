# gpusieve and its tools (see README.txt).
#
#   make            gpusieve, absort and ab2keys
#   make gpusieve   only the GPU siever, which needs the CUDA Toolkit (nvcc)
#   make tools      only absort and ab2keys, which need no CUDA
#   make clean
#
# Settings, on the command line (make CUDA_ARCH="70 80" NVCC_CCBIN=g++-13):
#   CUDA_ARCH    compute capabilities to build gpusieve for: "native" (the GPUs of this machine, the default; without a
#                visible GPU nvcc warns and builds for its oldest default instead), "all" (every one this nvcc supports,
#                for a binary to run on other machines), "all-major" (only the X.0 ones, which also
#                run on the X.y GPUs) or a list such as "75 80 86 89 90 120", which also adds PTX for the last one, so that
#                newer GPUs can run it.
#   NVCC         the CUDA compiler (default nvcc)
#   NVCC_THREADS how many compute capabilities nvcc compiles at once (default 0: as many as there are CPUs)
#   NVCC_CCBIN   the host compiler for nvcc, if the default one is newer than the CUDA version supports (CUDA 12.4: GCC 13 at
#                most), for example NVCC_CCBIN=g++-13
#   OPT          optimization flags of the host code (default -O3 -march=native: the programs run on the machine they are
#                built on)

CC = gcc
CXX = g++
NVCC ?= nvcc
NVCC_CCBIN ?=
NVCC_THREADS ?= 0
CUDA_ARCH ?= native
OPT ?= -O3 -march=native

ifneq ($(filter $(CUDA_ARCH),native all all-major),)
NVCC_ARCH = -arch=$(CUDA_ARCH)
else
NVCC_ARCH = $(foreach a,$(CUDA_ARCH),-gencode arch=compute_$(a),code=sm_$(a)) \
            -gencode arch=compute_$(lastword $(CUDA_ARCH)),code=compute_$(lastword $(CUDA_ARCH))
endif
NVCCFLAGS = -std=c++17 -O3 $(NVCC_ARCH) -t $(NVCC_THREADS) $(if $(NVCC_CCBIN),-ccbin $(NVCC_CCBIN)) -Xcompiler "$(OPT)"

.PHONY: all tools clean
all: gpusieve tools
tools: absort ab2keys

gpusieve: gpusieve.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

absort: absort.cpp
	$(CXX) -std=c++17 $(OPT) -fopenmp -o $@ $<

ab2keys: ab2keys.c
	$(CC) $(OPT) -o $@ $<

clean:
	rm -f gpusieve absort ab2keys
