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

void cuda_memset_2d(void* ptr, size_t pitch, size_t width, size_t height, int value, cudaStream_t* stream) {
    cudaError_t err = cudaMemset2DAsync(ptr, pitch, value, width, height, *stream);
    if (err != cudaSuccess) PANIC("cudaMemset2D failed");
}

void cuda_copy_from_host_to_device_2d(void* dst, size_t dst_pitch, const void* src, size_t src_pitch, size_t width, size_t height) {
    cudaError_t err = cudaMemcpy2D(dst, dst_pitch, src, src_pitch, width, height, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy2D failed: %s\n", cudaGetErrorString(err));
        PANIC("cudaMemcpy2D failed");
    }
}

void cuda_copy_from_device_to_host_2d(void* dst, size_t dst_pitch, const void* src, size_t src_pitch, size_t width, size_t height) {
    cudaError_t err = cudaMemcpy2D(dst, dst_pitch, src, src_pitch, width, height, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy2D failed: %s\n", cudaGetErrorString(err));
        PANIC("cudaMemcpy2D failed");
    }
}

void cuda_copy_from_device_to_device_2d(void* dst, size_t dst_pitch, const void* src, size_t src_pitch, size_t width, size_t height, cudaStream_t* stream) {
    cudaError_t err = cudaMemcpy2DAsync(dst, dst_pitch, src, src_pitch, width, height, cudaMemcpyDeviceToDevice, *stream);
    if (err != cudaSuccess) PANIC("cudaMemcpy2DAsync failed");
}

void cuda_matmul(cublasHandle_t handle,
                 void* dst, size_t dst_pitch,
                 void* mat1, size_t mat1_pitch,
                 void* mat2, size_t mat2_pitch,
                 size_t final_rows, size_t final_cols, size_t common_dim,
                 cudaStream_t* stream) {
    __half alpha = __float2half(1.0f);
    __half beta = __float2half(0.0f);
    cublasSetStream(handle, *stream);
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
                 size_t final_rows, size_t common_dim,
                 cudaStream_t* stream) {
    __half alpha = __float2half(1.0f);
    __half beta = __float2half(0.0f);
    cublasSetStream(handle, *stream);
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
                         double multiplier,
                         cudaStream_t* stream) {
    __half alpha = __float2half(1.0f);
    __half beta = __double2half(multiplier);
    __half** mat1s_gpu;
    __half** mat2s_gpu;
    __half** dsts_gpu;

    cublasSetStream(handle, *stream);
    // zero out destination if beta is not 0
    if (multiplier != 0.0) {
        for (int i = 0; i < nbatches; i++) {
            cuda_memset_2d(dsts[i], dst_pitch, final_rows * sizeof(__half), final_cols, 0, stream);
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

void cuda_outer_product(cublasHandle_t handle,
                        void* dst, size_t dst_pitch,
                        void* vec1, size_t vec1_pitch,
                        void* vec2, size_t vec2_pitch,
                        size_t rows, size_t cols,
                        cudaStream_t* stream) {
    __half alpha = __float2half(1.0f);
    __half beta = __float2half(0.0f);
    cublasSetStream(handle, *stream);
    cublasStatus_t status = cublasHgemm(handle,
                                        CUBLAS_OP_N,
                                        CUBLAS_OP_T,
                                        rows,
                                        cols,
                                        1,
                                        &alpha,
                                        (__half*) vec1, vec1_pitch / sizeof(__half),
                                        (__half*) vec2, vec2_pitch / sizeof(__half),
                                        &beta,
                                        (__half*) dst, dst_pitch / sizeof(__half));
    if (status != CUBLAS_STATUS_SUCCESS) PANIC("cublasHgemm failed");
}

void cuda_sigmoid(void* dst, size_t dst_pitch, size_t rows, size_t cols, cudaStream_t* stream)
{
    sigmoid_kernel_half(dst, (int) dst_pitch, (int) rows, (int) cols, stream);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) PANIC("sigmoid_kernel_half failed");
}

void cuda_sigmoid_tanh(void* dst, size_t dst_pitch, size_t rows, size_t cols, cudaStream_t* stream)
{
    sigmoid_tanh_kernel_half(dst, (int) dst_pitch, (int) rows, (int) cols, stream);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) PANIC("sigmoid_tanh_kernel_half failed");
}

void cuda_add(void* dst, size_t dst_pitch,
              void* mat1, size_t mat1_pitch,
              void* mat2, size_t mat2_pitch,
              size_t rows, size_t cols, cudaStream_t* stream) {
    add_mat(dst, (int) dst_pitch,
            mat1, (int) mat1_pitch,
            mat2, (int) mat2_pitch,
            (int) rows, (int) cols,
            stream);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) PANIC("add_mat failed");
}

void cuda_sub(void* dst, size_t dst_pitch,
              void* mat1, size_t mat1_pitch,
              void* mat2, size_t mat2_pitch,
              size_t rows, size_t cols, cudaStream_t* stream) {
    sub_mat(dst, (int) dst_pitch,
            mat1, (int) mat1_pitch,
            mat2, (int) mat2_pitch,
            (int) rows, (int) cols,
            stream);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) PANIC("add_mat failed");
}

void cuda_scale(void* matrix, void* scale, size_t pitch, size_t rows, size_t cols, cudaStream_t* stream)
{
    scale_mat(matrix, scale, (int) pitch, (int) rows, (int) cols, stream);
}

size_t cuda_size_of_cuda_event_t()
{
    return sizeof(cudaEvent_t);
}

void cuda_create_event(cudaEvent_t* event, cudaStream_t* stream) {
    cudaError_t err = cudaEventCreate(event);
    if (err != cudaSuccess) PANIC("cudaEventCreate failed");
    cudaEventRecord(*event, *stream);
}

void cuda_destroy_event(cudaEvent_t* event) {
    cudaError_t err = cudaEventDestroy(*event);
    if (err != cudaSuccess) PANIC("cudaEventDestroy failed");
}

size_t cuda_size_of_cuda_stream_t()
{
    return sizeof(cudaStream_t);
}

void cuda_create_stream(cudaStream_t* stream) {
    cudaError_t err = cudaStreamCreate(stream);
    if (err != cudaSuccess) PANIC("cudaStreamCreate failed");
}

void cuda_destroy_stream(cudaStream_t* stream) {
    cudaError_t err = cudaStreamDestroy(*stream);
    if (err != cudaSuccess) PANIC("cudaStreamDestroy failed");
}

void cuda_make_stream_wait_for_event(cudaStream_t* stream, cudaEvent_t* event) {
    cudaError_t err = cudaStreamWaitEvent(*stream, *event, 0);
    if (err != cudaSuccess) PANIC("cudaStreamWaitEvent failed");
}

} // extern "C"
