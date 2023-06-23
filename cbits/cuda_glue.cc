#include "cuda_runtime_api.h"
#include "cublas_v2.h"
#include "cuda_kernels.h"
#include <stdlib.h>
#include <stdio.h>

extern "C" {

/* There's no error checking other than printing to stderr and calling
 * abort() if anything goes wrong.
 *
 * The only exception is tensor allocation where return value indicates
 * memory allocation failure so that Haskell can run garbage collection and
 * try again. */

#define PANIC(msg) { fprintf(stderr, "%s\n", msg); abort(); }

cublasHandle_t init_cuda(void) {
    // CUDA
    int device_count;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess) PANIC("cudaGetDeviceCount failed");
    if (device_count == 0) PANIC("No CUDA devices found");
    err = cudaSetDevice(0);
    if (err != cudaSuccess) PANIC("cudaSetDevice failed");

    // CUBLAS
    cublasHandle_t handle;
    cublasStatus_t status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) PANIC("cublasCreate failed");
    return handle;
}

void shutdown_cuda(cublasHandle_t handle) {
    cublasStatus_t status = cublasDestroy(handle);
    if (status != CUBLAS_STATUS_SUCCESS) PANIC("cublasDestroy failed");
    cudaError_t err = cudaDeviceReset();
    if (err != cudaSuccess) PANIC("cudaDeviceReset failed");
}

int cuda_alloc_2d(size_t width, size_t height, size_t elem_size, void** ptr, size_t* pitch) {
    cudaError_t err = cudaMallocPitch(ptr, pitch, width * elem_size, height);
    if (err != cudaSuccess) {
        return -1;
    }
    return 0;
}

void cuda_dealloc(void* ptr) {
    cudaError_t err = cudaFree(ptr);
    if (err != cudaSuccess) PANIC("cudaFree failed")
}

void cuda_memset_2d(void* ptr, size_t pitch, size_t width, size_t height, int value) {
    cudaError_t err = cudaMemset2D(ptr, pitch, value, width, height);
    if (err != cudaSuccess) PANIC("cudaMemset2D failed");
}

void cuda_copy_from_host_to_device_2d(void* dst, size_t dst_pitch, const void* src, size_t src_pitch, size_t width, size_t height) {
    cudaError_t err = cudaMemcpy2D(dst, dst_pitch, src, src_pitch, width, height, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) PANIC("cudaMemcpy2D failed");
}

void cuda_copy_from_device_to_host_2d(void* dst, size_t dst_pitch, const void* src, size_t src_pitch, size_t width, size_t height) {
    cudaError_t err = cudaMemcpy2D(dst, dst_pitch, src, src_pitch, width, height, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) PANIC("cudaMemcpy2D failed");
}

void cuda_copy_from_device_to_device_2d(void* dst, size_t dst_pitch, const void* src, size_t src_pitch, size_t width, size_t height) {
    cudaError_t err = cudaMemcpy2D(dst, dst_pitch, src, src_pitch, width, height, cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess) PANIC("cudaMemcpy2D failed");
}

void cuda_matmul(cublasHandle_t handle,
                 void* dst, size_t dst_pitch,
                 void* mat1, size_t mat1_pitch,
                 void* mat2, size_t mat2_pitch,
                 size_t final_rows, size_t final_cols, size_t common_dim) {
    __half alpha = __float2half(1.0f);
    __half beta = __float2half(0.0f);
    cublasStatus_t status = cublasHgemm(handle,
                                        CUBLAS_OP_N,
                                        CUBLAS_OP_N,
                                        final_rows,
                                        final_cols,
                                        common_dim,
                                        &alpha,
                                        (__half*) mat1, mat1_pitch / sizeof(__half),
                                        (__half*) mat2, mat2_pitch / sizeof(__half),
                                        &beta,
                                        (__half*) dst, dst_pitch / sizeof(__half));
    if (status != CUBLAS_STATUS_SUCCESS) PANIC("cublasHgemm failed");
}

void cuda_matmul_vec(cublasHandle_t handle,
                 void* dst, size_t dst_pitch,
                 void* mat1, size_t mat1_pitch,
                 void* mat2, size_t mat2_pitch,
                 size_t final_rows, size_t common_dim) {
    __half alpha = __float2half(1.0f);
    __half beta = __float2half(0.0f);
    cublasStatus_t status = cublasHgemm(handle,
                                        CUBLAS_OP_N,
                                        CUBLAS_OP_N,
                                        final_rows,
                                        1,
                                        common_dim,
                                        &alpha,
                                        (__half*) mat1, mat1_pitch / sizeof(__half),
                                        (__half*) mat2, mat2_pitch / sizeof(__half),
                                        &beta,
                                        (__half*) dst, dst_pitch / sizeof(__half));
    if (status != CUBLAS_STATUS_SUCCESS) PANIC("cublasHgemm failed");
}

void cuda_matmul_batched(cublasHandle_t handle,
                         void** dsts, size_t dst_pitch,
                         void** mat1s, size_t mat1_pitch,
                         void** mat2s, size_t mat2_pitch,
                         size_t final_rows, size_t final_cols, size_t common_dim,
                         int nbatches,
                         double multiplier) {
    __half alpha = __float2half(1.0f);
    __half beta = __double2half(multiplier);
    __half** mat1s_gpu;
    __half** mat2s_gpu;
    __half** dsts_gpu;

    // zero out destination if beta is not 0
    if (multiplier != 0.0) {
        for (int i = 0; i < nbatches; i++) {
            cuda_memset_2d(dsts[i], dst_pitch, final_rows * sizeof(__half), final_cols, 0);
        }
    }

    cudaError_t err = cudaMalloc((void**) &mat1s_gpu, nbatches * sizeof(__half*));
    if (err != cudaSuccess) PANIC("cudaMalloc failed");
    err = cudaMalloc((void**) &mat2s_gpu, nbatches * sizeof(__half*));
    if (err != cudaSuccess) PANIC("cudaMalloc failed");
    err = cudaMalloc((void**) &dsts_gpu, nbatches * sizeof(__half*));
    if (err != cudaSuccess) PANIC("cudaMalloc failed");

    err = cudaMemcpy(mat1s_gpu, mat1s, nbatches * sizeof(__half*), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) PANIC("cudaMemcpy failed");
    err = cudaMemcpy(mat2s_gpu, mat2s, nbatches * sizeof(__half*), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) PANIC("cudaMemcpy failed");
    err = cudaMemcpy(dsts_gpu, dsts, nbatches * sizeof(__half*), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) PANIC("cudaMemcpy failed");

    cublasStatus_t status = cublasHgemmBatched(handle,
                                               CUBLAS_OP_N,
                                               CUBLAS_OP_N,
                                               final_rows,
                                               final_cols,
                                               common_dim,
                                               &alpha,
                                               (__half**) mat1s_gpu, mat1_pitch / sizeof(__half),
                                               (__half**) mat2s_gpu, mat2_pitch / sizeof(__half),
                                               &beta,
                                               (__half**) dsts_gpu, dst_pitch / sizeof(__half),
                                               nbatches);
    if (status != CUBLAS_STATUS_SUCCESS) PANIC("cublasHgemmBatched failed");

    err = cudaFree(mat1s_gpu);
    if (err != cudaSuccess) PANIC("cudaFree failed");
    err = cudaFree(mat2s_gpu);
    if (err != cudaSuccess) PANIC("cudaFree failed");
    err = cudaFree(dsts_gpu);
    if (err != cudaSuccess) PANIC("cudaFree failed");
}

void cuda_sigmoid(void* dst, size_t dst_pitch, size_t rows, size_t cols)
{
    sigmoid_kernel_half(dst, (int) dst_pitch, (int) rows, (int) cols);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) PANIC("sigmoid_kernel_half failed");
}

void cuda_sigmoid_tanh(void* dst, size_t dst_pitch, size_t rows, size_t cols)
{
    sigmoid_tanh_kernel_half(dst, (int) dst_pitch, (int) rows, (int) cols);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) PANIC("sigmoid_tanh_kernel_half failed");
}

void cuda_lstm_memory(void* new_memory, void* memory, void* forget_gate, void* input_gate, void* input, size_t rows)
{
    lstm_memory_half(new_memory, memory, forget_gate, input_gate, input, (int) rows);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) PANIC("sigmoid_tanh_kernel_half failed");
}

void cuda_lstm_output(void* out, void* x, void* y, size_t rows)
{
    lstm_output_half(out, x, y, (int) rows);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) PANIC("sigmoid_tanh_kernel_half failed");
}

void cuda_lstm_bias_last_act(void* out, void* bias, void* weight, void* act, size_t rows)
{
    lstm_bias_last_act_half(out, bias, weight, act, (int) rows);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) PANIC("sigmoid_tanh_kernel_half failed");
}

} // extern "C"
