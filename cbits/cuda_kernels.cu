#include "cuda_kernels.h"
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
__global__  void lstm_memory_kernel(T* new_memory, T* memory, T* forget_gate, T* input_gate, T* input, int rows)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= rows) {
        return;
    }
    new_memory[x] = forget_gate[x] * memory[x] + input_gate[x] * input[x];
}

template<class T>
__global__  void lstm_output_kernel(T* out, T* x, T* y, int rows)
{
    const int offset = blockIdx.x * blockDim.x + threadIdx.x;
    if (offset >= rows) {
        return;
    }
    out[offset] = tanh((float) x[offset]) * (1.0 / (1.0 + exp((float) -y[offset])));
}

template<class T>
__global__  void lstm_bias_last_act(T* out, T* bias, T* weight, T* act, int rows)
{
    const int offset = blockIdx.x * blockDim.x + threadIdx.x;
    if (offset >= rows) {
        return;
    }
    out[offset] = bias[offset] + weight[offset] * act[offset];
}

extern "C" {
    void sigmoid_kernel_half(void* raw_dst, int pitch, int rows, int cols)
    {
        __half* dst = (__half*) raw_dst;
        if (cols == 1) {
            const int block_size = 256;
            const int grid_size = (rows + block_size - 1) / block_size;
            sigmoid_kernel<<<grid_size, block_size>>>(dst, pitch, rows, cols);
        } else {
            const int block_size_x = 32;
            const int block_size_y = 32;
            const int grid_size_x = (rows + block_size_x - 1) / block_size_x;
            const int grid_size_y = (cols + block_size_y - 1) / block_size_y;
            dim3 grid_size(grid_size_x, grid_size_y);
            dim3 block_size(block_size_x, block_size_y);
            sigmoid_kernel<<<grid_size, block_size>>>(dst, pitch, rows, cols);
        }
    };

    void sigmoid_tanh_kernel_half(void* raw_dst, int pitch, int rows, int cols)
    {
        __half* dst = (__half*) raw_dst;
        if (cols == 1) {
            const int block_size = 256;
            const int grid_size = (rows + block_size - 1) / block_size;
            sigmoid_tanh_kernel<<<grid_size, block_size>>>(dst, pitch, rows, cols);
        } else {
            const int block_size_x = 32;
            const int block_size_y = 32;
            const int grid_size_x = (rows + block_size_x - 1) / block_size_x;
            const int grid_size_y = (cols + block_size_y - 1) / block_size_y;
            dim3 grid_size(grid_size_x, grid_size_y);
            dim3 block_size(block_size_x, block_size_y);
            sigmoid_tanh_kernel<<<grid_size, block_size>>>(dst, pitch, rows, cols);
        }
    };

    void lstm_memory_half(void* new_memory, void* memory, void* forget_gate, void* input_gate, void* input, int rows) {
        const int block_size = 256;
        const int grid_size = (rows + block_size - 1) / block_size;
        lstm_memory_kernel<<<grid_size, block_size>>>((__half*) new_memory, (__half*) memory, (__half*) forget_gate, (__half*) input_gate, (__half*) input, rows);
    }

    void lstm_output_half(void* out, void* x, void* y, int rows) {
        const int block_size = 256;
        const int grid_size = (rows + block_size - 1) / block_size;
        lstm_output_kernel<<<grid_size, block_size>>>((__half*) out, (__half*) x, (__half*) y, rows);
    }

    void lstm_bias_last_act_half(void* out, void* bias, void* weights, void* act, int rows) {
        const int block_size = 256;
        const int grid_size = (rows + block_size - 1) / block_size;
        lstm_bias_last_act<<<grid_size, block_size>>>((__half*) out, (__half*) bias, (__half*) weights, (__half*) act, rows);
    }
}
