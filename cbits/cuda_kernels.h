#pragma once

#include "cuda_fp16.h"

extern "C" {
    void sigmoid_kernel_half(void* raw_dst, int pitch, int rows, int cols);
    void sigmoid_tanh_kernel_half(void* raw_dst, int pitch, int rows, int cols);
    void lstm_memory_half(void* new_memory, void* memory, void* forget_gate, void* input_gate, void* input, int rows);
    void lstm_output_half(void* out, void* x, void* y, int rows);
    void lstm_bias_last_act_half(void* out, void* bias, void* weight, void* act, int rows);
}
