/* 
 * kernels.h
 *
 *  Created on: Nov 9, 2025
 *  
 *  CUDA kernel function declarations and implementations
*/

#ifndef KERNELS_H
#define KERNELS_H

#include <cuda.h>
#include "config.h"

// Matrix-vector multiplication kernel
__global__ void matvec_mult_kernel(float *W, float *x, float *b, float *out, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < cols) {
        float sum = b[idx];
        for (int i = 0; i < rows; i++) {
            sum += W[i * cols + idx] * x[i];
        }
        out[idx] = sum;
    }
}

// Activation function kernels
__global__ void relu_kernel(float *in, float *out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = (in[idx] > 0) ? in[idx] : 0.0f;
    }
}

__global__ void drelu_kernel(float *in, float *out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = (in[idx] > 0) ? 1.0f : 0.0f;
    }
}

__global__ void softmax_kernel(float *in, float *out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = expf(in[idx]);
    }
}

// Loss calculation kernel
__global__ void loss_kernel(float *pred, float *label, float *loss, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        loss[idx] = -label[idx] * logf(pred[idx] + 1e-8f);
    }
}

// Backpropagation kernels
__global__ void delta_output_kernel(float *label, float *output, float *delta, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        delta[idx] = label[idx] - output[idx];
    }
}

__global__ void delta_hidden_kernel(float *delta_next, float *W, float *activation, float *delta, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows) {
        float err = 0.0f;
        // Compute error: sum of (delta_next[k] * W[idx][k]) for all k
        for (int k = 0; k < cols; k++) {
            err += delta_next[k] * W[idx * cols + k];
        }
        // Multiply by ReLU derivative
        delta[idx] = err * ((activation[idx] > 0) ? 1.0f : 0.0f);
    }
}

// Weight update kernels
__global__ void update_weights_kernel(float *W, float *delta, float *activation, float lr, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;
    if (idx < total) {
        int row = idx / cols;
        int col = idx % cols;
        W[idx] += lr * delta[col] * activation[row];
    }
}

__global__ void update_bias_kernel(float *b, float *delta, float lr, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        b[idx] += lr * delta[idx];
    }
}

#endif