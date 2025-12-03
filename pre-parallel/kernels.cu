/* kernels.cu
 *
 *  Created on: Nov 9, 2025
 *  
 *  Location for CUDA kernels  kernels should be defined here, and prototypes placed in kernels.h
 *
 *  Example:
 *     __global__ void test_kernel(){}
 */
/* kernels.cu */
#include <cuda.h>
#include "kernels.h"
#include "config.h"

// Matrix-vector multiplication: W * x + b = out
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

// ReLU activation
__global__ void relu_kernel(float *in, float *out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = (in[idx] > 0) ? in[idx] : 0.0f;
    }
}

// ReLU derivative
__global__ void drelu_kernel(float *in, float *out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = (in[idx] > 0) ? 1.0f : 0.0f;
    }
}

// Softmax activation (simplified version - requires max and sum to be computed separately)
// This kernel computes exp(x - max) for each element
// Note: For full softmax, you'll need reduction kernels for max and sum
__global__ void softmax_kernel(float *in, float *out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // This assumes max has already been subtracted and sum has been computed
        // For now, we'll do a simple version that requires host-side reduction
        // A full GPU softmax would need shared memory reduction
        out[idx] = expf(in[idx]);
    }
}

// Loss calculation (cross-entropy)
__global__ void loss_kernel(float *pred, float *label, float *loss, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        loss[idx] = -label[idx] * logf(pred[idx] + 1e-8f);
    }
}

// Output layer delta: delta = label - output
__global__ void delta_output_kernel(float *label, float *output, float *delta, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        delta[idx] = label[idx] - output[idx];
    }
}

// Hidden layer delta: delta = (delta_next * W) * drelu(activation)
// delta_next is [cols], W is [rows x cols], activation is [rows], delta is [rows]
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

// Weight update: W += lr * delta * activation
// W is [rows x cols], delta is [cols], activation is [rows]
__global__ void update_weights_kernel(float *W, float *delta, float *activation, float lr, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;
    if (idx < total) {
        int row = idx / cols;
        int col = idx % cols;
        W[idx] += lr * delta[col] * activation[row];
    }
}

// Bias update: b += lr * delta
__global__ void update_bias_kernel(float *b, float *delta, float lr, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        b[idx] += lr * delta[idx];
    }
}