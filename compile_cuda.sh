#!/usr/bin/env bash

set -euxo pipefail

nvcc -shared --compiler-options -fPIC -gencode=arch=compute_86,code=sm_86 -O2 cbits/cuda_kernels.cu -o cbits/libcuda_kernels.so
