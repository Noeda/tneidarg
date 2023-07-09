#include "cuda_kernels.h"
#include "curand_kernel.h"
#include <stdio.h>

template<class T>
__global__ void sigmoid_kernel (T* dst, int pitch, int rows, int cols)
{
    // x = row
    // y = col
    const int global_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int global_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (global_x >= rows || global_y >= cols) {
        return;
    }
    const int offset = global_y * pitch + global_x;
    dst[offset] = 1.0 / (1.0 + exp((float) -dst[offset]));
}

template<class T>
__global__ void sigmoid_tanh_kernel (T* dst, int pitch, int rows, int cols)
{
    // x = row
    // y = col
    const int global_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int global_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (global_x >= rows || global_y >= cols) {
        return;
    }
    const int offset = global_y * pitch + global_x;
    dst[offset] = tanh((float) dst[offset]);
}

template<class T>
__global__ void add_mat_kernel (T* dst, int dst_pitch,
                                T* mat1, int mat1_pitch,
                                T* mat2, int mat2_pitch,
                                int rows, int cols) {
    // x = row
    // y = col
    const int global_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int global_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (global_x >= rows || global_y >= cols) {
        return;
    }
    const int dst_offset = global_y * dst_pitch + global_x;
    const int mat1_offset = global_y * mat1_pitch + global_x;
    const int mat2_offset = global_y * mat2_pitch + global_x;
    dst[dst_offset] = mat1[mat1_offset] + mat2[mat2_offset];
}

template<class T>
__global__ void sub_mat_kernel (T* dst, int dst_pitch,
                                T* mat1, int mat1_pitch,
                                T* mat2, int mat2_pitch,
                                int rows, int cols) {
    // x = row
    // y = col
    const int global_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int global_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (global_x >= rows || global_y >= cols) {
        return;
    }
    const int dst_offset = global_y * dst_pitch + global_x;
    const int mat1_offset = global_y * mat1_pitch + global_x;
    const int mat2_offset = global_y * mat2_pitch + global_x;
    dst[dst_offset] = mat1[mat1_offset] - mat2[mat2_offset];
}

template<class T>
__global__ void scale_mat_kernel(T* matrix, T* scale, int pitch, int rows, int cols)
{
    // x = row
    // y = col
    const int global_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int global_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (global_x >= rows || global_y >= cols) {
        return;
    }
    const int offset = global_y * pitch + global_x;
    matrix[offset] *= scale[0];
}

template<class T>
__global__ void run_philox_rng_kernel_gaussian(T* matrix, int seed, int pitch, int rows, int cols, float mean, float stdev)
{
    const int global_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int global_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (global_x >= rows || global_y >= cols) {
        return;
    }
    const int offset = global_y * pitch + global_x;
    // curand_init is expensive according to CUDA documentation.
    // But I'm going to prioritize simplicity over performance for now.
    curandStatePhilox4_32_10_t state;
    curand_init(seed, offset, 0, &state);
    float rand = curand_normal(&state);
    matrix[offset] = (T) (rand * stdev + mean);
}

extern "C" {
    void sigmoid_kernel_half(void* raw_dst, int pitch, int rows, int cols, cudaStream_t* stream)
    {
        pitch /= 2;
        __half* dst = (__half*) raw_dst;
        if (cols == 1) {
            const int block_size = 256;
            const int grid_size = (rows + block_size - 1) / block_size;
            sigmoid_kernel<<<grid_size, block_size, 0, *stream>>>(dst, pitch, rows, cols);
        } else {
            const int block_size_x = 32;
            const int block_size_y = 32;
            const int grid_size_x = (rows + block_size_x - 1) / block_size_x;
            const int grid_size_y = (cols + block_size_y - 1) / block_size_y;
            dim3 grid_size(grid_size_x, grid_size_y);
            dim3 block_size(block_size_x, block_size_y);
            sigmoid_kernel<<<grid_size, block_size, 0, *stream>>>(dst, pitch, rows, cols);
        }
    };

    void sigmoid_tanh_kernel_half(void* raw_dst, int pitch, int rows, int cols, cudaStream_t* stream)
    {
        pitch /= 2;
        __half* dst = (__half*) raw_dst;
        if (cols == 1) {
            const int block_size = 256;
            const int grid_size = (rows + block_size - 1) / block_size;
            sigmoid_tanh_kernel<<<grid_size, block_size, 0, *stream>>>(dst, pitch, rows, cols);
        } else {
            const int block_size_x = 32;
            const int block_size_y = 32;
            const int grid_size_x = (rows + block_size_x - 1) / block_size_x;
            const int grid_size_y = (cols + block_size_y - 1) / block_size_y;
            dim3 grid_size(grid_size_x, grid_size_y);
            dim3 block_size(block_size_x, block_size_y);
            sigmoid_tanh_kernel<<<grid_size, block_size, 0, *stream>>>(dst, pitch, rows, cols);
        }
    };

    void add_mat(void* raw_dst, int dst_pitch,
                 void* raw_mat1, int mat1_pitch,
                 void* raw_mat2, int mat2_pitch,
                 int nrows, int ncols,
                 cudaStream_t* stream) {
        __half* dst = (__half*) raw_dst;
        __half* mat1 = (__half*) raw_mat1;
        __half* mat2 = (__half*) raw_mat2;
        dst_pitch /= 2;
        mat1_pitch /= 2;
        mat2_pitch /= 2;
        if (ncols == 1) {
            const int block_size = 256;
            const int grid_size = (nrows + block_size - 1) / block_size;
            add_mat_kernel<<<grid_size, block_size, 0, *stream>>>(dst, dst_pitch, mat1, mat1_pitch, mat2, mat2_pitch, nrows, ncols);
        } else {
            const int block_size_x = 32;
            const int block_size_y = 32;
            const int grid_size_x = (nrows + block_size_x - 1) / block_size_x;
            const int grid_size_y = (ncols + block_size_y - 1) / block_size_y;
            dim3 grid_size(grid_size_x, grid_size_y);
            dim3 block_size(block_size_x, block_size_y);
            add_mat_kernel<<<grid_size, block_size, 0, *stream>>>(dst, dst_pitch, mat1, mat1_pitch, mat2, mat2_pitch, nrows, ncols);
        }
    }

    void sub_mat(void* raw_dst, int dst_pitch,
                 void* raw_mat1, int mat1_pitch,
                 void* raw_mat2, int mat2_pitch,
                 int nrows, int ncols,
                 cudaStream_t* stream) {
        __half* dst = (__half*) raw_dst;
        __half* mat1 = (__half*) raw_mat1;
        __half* mat2 = (__half*) raw_mat2;
        dst_pitch /= 2;
        mat1_pitch /= 2;
        mat2_pitch /= 2;
        if (ncols == 1) {
            const int block_size = 256;
            const int grid_size = (nrows + block_size - 1) / block_size;
            sub_mat_kernel<<<grid_size, block_size, 0, *stream>>>(dst, dst_pitch, mat1, mat1_pitch, mat2, mat2_pitch, nrows, ncols);
        } else {
            const int block_size_x = 32;
            const int block_size_y = 32;
            const int grid_size_x = (nrows + block_size_x - 1) / block_size_x;
            const int grid_size_y = (ncols + block_size_y - 1) / block_size_y;
            dim3 grid_size(grid_size_x, grid_size_y);
            dim3 block_size(block_size_x, block_size_y);
            sub_mat_kernel<<<grid_size, block_size, 0, *stream>>>(dst, dst_pitch, mat1, mat1_pitch, mat2, mat2_pitch, nrows, ncols);
        }
    }

    void scale_mat(void* raw_matrix, void* raw_scale, int pitch, int rows, int cols, cudaStream_t* stream) {
        __half* matrix = (__half*) raw_matrix;
        __half* scale = (__half*) raw_scale;
        pitch /= 2;
        if (cols == 1) {
            const int block_size = 256;
            const int grid_size = (rows + block_size - 1) / block_size;
            scale_mat_kernel<<<grid_size, block_size, 0, *stream>>>(matrix, scale, pitch, rows, cols);
        } else {
            const int block_size_x = 32;
            const int block_size_y = 32;
            const int grid_size_x = (rows + block_size_x - 1) / block_size_x;
            const int grid_size_y = (cols + block_size_y - 1) / block_size_y;
            dim3 grid_size(grid_size_x, grid_size_y);
            dim3 block_size(block_size_x, block_size_y);
            scale_mat_kernel<<<grid_size, block_size, 0, *stream>>>(matrix, scale, pitch, rows, cols);
        }
    }

    void write_gaussian_randoms_2d(void* raw_dst, int seed, int pitch, int rows, int cols, cudaStream_t* stream, double mean, double stdev) {
        __half* matrix = (__half*) raw_dst;
        pitch /= 2;

        if (cols == 1) {
            const int block_size = 256;
            const int grid_size = (rows + block_size - 1) / block_size;
            run_philox_rng_kernel_gaussian<<<grid_size, block_size, 0, *stream>>>(matrix, seed, pitch, rows, cols, (float) mean, (float) stdev);
        } else {
            const int block_size_x = 32;
            const int block_size_y = 32;
            const int grid_size_x = (rows + block_size_x - 1) / block_size_x;
            const int grid_size_y = (cols + block_size_y - 1) / block_size_y;
            dim3 grid_size(grid_size_x, grid_size_y);
            dim3 block_size(block_size_x, block_size_y);
            run_philox_rng_kernel_gaussian<<<grid_size, block_size, 0, *stream>>>(matrix, seed, pitch, rows, cols, (float) mean, (float) stdev);
        }
    }
}
