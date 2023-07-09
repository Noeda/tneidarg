#pragma once

#include "cuda_fp16.h"

extern "C" {
    void sigmoid_kernel_half(void* raw_dst, int pitch, int rows, int cols, cudaStream_t* stream);
    void sigmoid_tanh_kernel_half(void* raw_dst, int pitch, int rows, int cols, cudaStream_t* stream);
    void add_mat(void* raw_dst, int dst_pitch,
                 void* mat1, int mat1_pitch,
                 void* mat2, int mat2_pitch,
                 int nrows, int ncols,
                 cudaStream_t* stream);
    void sub_mat(void* raw_dst, int dst_pitch,
                 void* mat1, int mat1_pitch,
                 void* mat2, int mat2_pitch,
                 int nrows, int ncols,
                 cudaStream_t* stream);
    void scale_mat(void* raw_matrix, void* scale, int pitch, int rows, int cols, cudaStream_t* stream);

    void write_gaussian_randoms_2d(void* raw_dst, int seed, int pitch, int rows, int cols, cudaStream_t* stream, double mean, double stdev);
}
