/*
    nnp.cu

    Created on: Nov 9, 2025
    Serial implementation of a simple feedforward neural network for MNIST digit classification.

    Network architecture:
    - Input layer: 784 neurons (28x28 pixels)
    - Hidden layer 1: 128 neurons, ReLU activation
    - Hidden layer 2: 64 neurons, ReLU activation
    - Output layer: 10 neurons, Softmax activation

    Training:
    - Loss function: Categorical Cross-Entropy
    - Optimizer: Stochastic Gradient Descent (SGD)
*/
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>
#include "config.h"
#include "loader.h"
#include "nnp.h"
#include "kernels.h"
#include "kernels.cu"
/* Helper function to check CUDA errors */
void checkCudaError(cudaError_t error, const char *file, int line) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d: %s\n", file, line, cudaGetErrorString(error));
        exit(1);
    }
}
#define CHECK_CUDA_ERROR(err) checkCudaError(err, __FILE__, __LINE__)

/* Activation functions for relu layers
* Arguments:
*   x: input value
* Returns:
*   activated value based on ReLU function 
*/
float relu(float x) { return x > 0 ? x : 0; }

/* Derivative of ReLU activation function
* Arguments:
*   y: output value from ReLU function
* Returns:
*   derivative value
*/
float drelu(float y) { return y > 0 ? 1 : 0; }

/* Softmax activation function
* Arguments:
*   z: input array
*   out: output array to store softmax results
*   len: length of the input/output arrays
*/ 
void softmax(float *z, float *out, int len) {
    float max = z[0];
    for (int i=1;i<len;i++) if (z[i]>max) max=z[i];
    float sum=0;
    for (int i=0;i<len;i++){ out[i]=expf(z[i]-max); sum+=out[i]; }
    for (int i=0;i<len;i++) out[i]/=sum;
}

/* Initialize weights with small random values
* Arguments:
*   w: weight array to initialize
*   size: number of weights
*/
void init_weights(float *w, int size) {
    for (int i=0;i<size;i++)
        w[i] = ((float)rand()/RAND_MAX - 0.5f) * 0.1f;
}

/* Train the model using stochastic gradient descent 
* Arguments:
*   model (out): pointer to the MODEL structure which holds network parameters. It is populated by this function.
* Returns:
*   None
*/
void train_model(MODEL* model){
    // Initialize weights on host
    init_weights(model->W1, SIZE*H1); init_weights(model->b1, H1);
    init_weights(model->W2, H1*H2); init_weights(model->b2, H2);
    init_weights(model->W3, H2*CLASSES); init_weights(model->b3, CLASSES);

    // ===== Allocate device memory for model =====
    float *d_W1, *d_b1, *d_W2, *d_b2, *d_W3, *d_b3;
    CHECK_CUDA_ERROR(cudaMalloc(&d_W1, SIZE * H1 * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_b1, H1 * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_W2, H1 * H2 * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_b2, H2 * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_W3, H2 * CLASSES * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_b3, CLASSES * sizeof(float)));

    // ===== Allocate device memory for activations and deltas =====
    float *d_h1, *d_h1a, *d_h2, *d_h2a, *d_out, *d_outa;
    float *d_delta1, *d_delta2, *d_delta3;
    CHECK_CUDA_ERROR(cudaMalloc(&d_h1, H1 * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_h1a, H1 * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_h2, H2 * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_h2a, H2 * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_out, CLASSES * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_outa, CLASSES * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_delta1, H1 * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_delta2, H2 * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_delta3, CLASSES * sizeof(float)));

    // ===== Allocate device memory for input and label =====
    float *d_input, *d_label;
    CHECK_CUDA_ERROR(cudaMalloc(&d_input, SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_label, CLASSES * sizeof(float)));

    // ===== Copy model to device =====
    CHECK_CUDA_ERROR(cudaMemcpy(d_W1, model->W1, SIZE * H1 * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_b1, model->b1, H1 * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_W2, model->W2, H1 * H2 * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_b2, model->b2, H2 * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_W3, model->W3, H2 * CLASSES * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_b3, model->b3, CLASSES * sizeof(float), cudaMemcpyHostToDevice));

    // ===== Kernel launch configuration =====
    int threadsPerBlock = 256;
    int blocksH1 = (H1 + threadsPerBlock - 1) / threadsPerBlock;
    int blocksH2 = (H2 + threadsPerBlock - 1) / threadsPerBlock;
    int blocksCLASSES = (CLASSES + threadsPerBlock - 1) / threadsPerBlock;

    // Host arrays for softmax and loss (small arrays, keep on host)
    float out_host[CLASSES], outa_host[CLASSES];

    for (int epoch=0; epoch<EPOCHS; epoch++) {
        float loss=0;
        for (int n=0; n<NUM_TRAIN; n++) {
            // ===== Copy input and label to device =====
            CHECK_CUDA_ERROR(cudaMemcpy(d_input, train_data[n], SIZE * sizeof(float), cudaMemcpyHostToDevice));
            CHECK_CUDA_ERROR(cudaMemcpy(d_label, train_label[n], CLASSES * sizeof(float), cudaMemcpyHostToDevice));

            // ---------- Forward Pass ----------
            // Layer 1: input -> h1
            matvec_mult_kernel<<<blocksH1, threadsPerBlock>>>(d_W1, d_input, d_b1, d_h1, SIZE, H1);
            CHECK_CUDA_ERROR(cudaGetLastError());
            cudaDeviceSynchronize();
            // Apply ReLU: h1 -> h1a
            relu_kernel<<<blocksH1, threadsPerBlock>>>(d_h1, d_h1a, H1);
            CHECK_CUDA_ERROR(cudaGetLastError());
            cudaDeviceSynchronize();

            // Layer 2: h1a -> h2
            matvec_mult_kernel<<<blocksH2, threadsPerBlock>>>(d_W2, d_h1a, d_b2, d_h2, H1, H2);
            CHECK_CUDA_ERROR(cudaGetLastError());
            cudaDeviceSynchronize();
            // Apply ReLU: h2 -> h2a
            relu_kernel<<<blocksH2, threadsPerBlock>>>(d_h2, d_h2a, H2);
            CHECK_CUDA_ERROR(cudaGetLastError());
            cudaDeviceSynchronize();

            // Layer 3: h2a -> out
            matvec_mult_kernel<<<blocksCLASSES, threadsPerBlock>>>(d_W3, d_h2a, d_b3, d_out, H2, CLASSES);
            CHECK_CUDA_ERROR(cudaGetLastError());
            cudaDeviceSynchronize();

            // Softmax: Copy to host, compute softmax, copy back
            CHECK_CUDA_ERROR(cudaMemcpy(out_host, d_out, CLASSES * sizeof(float), cudaMemcpyDeviceToHost));
            softmax(out_host, outa_host, CLASSES);
            CHECK_CUDA_ERROR(cudaMemcpy(d_outa, outa_host, CLASSES * sizeof(float), cudaMemcpyHostToDevice));

            // ---------- Loss ----------
            for (int k=0;k<CLASSES;k++)
                loss -= train_label[n][k]*logf(outa_host[k]+1e-8f);

            // ---------- Backpropagation ----------
            // Delta 3: output layer
            delta_output_kernel<<<blocksCLASSES, threadsPerBlock>>>(d_label, d_outa, d_delta3, CLASSES);
            CHECK_CUDA_ERROR(cudaGetLastError());
            cudaDeviceSynchronize();

            // Delta 2: hidden layer 2
            delta_hidden_kernel<<<blocksH2, threadsPerBlock>>>(d_delta3, d_W3, d_h2a, d_delta2, H2, CLASSES);
            CHECK_CUDA_ERROR(cudaGetLastError());
            cudaDeviceSynchronize();

            // Delta 1: hidden layer 1
            delta_hidden_kernel<<<blocksH1, threadsPerBlock>>>(d_delta2, d_W2, d_h1a, d_delta1, H1, H2);
            CHECK_CUDA_ERROR(cudaGetLastError());
            cudaDeviceSynchronize();

            // ---------- Weight Updates ----------
            // Update W3 and b3
            int blocksW3 = ((H2 * CLASSES) + threadsPerBlock - 1) / threadsPerBlock;
            update_weights_kernel<<<blocksW3, threadsPerBlock>>>(d_W3, d_delta3, d_h2a, LR, H2, CLASSES);
            CHECK_CUDA_ERROR(cudaGetLastError());
            cudaDeviceSynchronize();
            update_bias_kernel<<<blocksCLASSES, threadsPerBlock>>>(d_b3, d_delta3, LR, CLASSES);
            CHECK_CUDA_ERROR(cudaGetLastError());
            cudaDeviceSynchronize();

            // Update W2 and b2
            int blocksW2 = ((H1 * H2) + threadsPerBlock - 1) / threadsPerBlock;
            update_weights_kernel<<<blocksW2, threadsPerBlock>>>(d_W2, d_delta2, d_h1a, LR, H1, H2);
            CHECK_CUDA_ERROR(cudaGetLastError());
            cudaDeviceSynchronize();
            update_bias_kernel<<<blocksH2, threadsPerBlock>>>(d_b2, d_delta2, LR, H2);
            CHECK_CUDA_ERROR(cudaGetLastError());
            cudaDeviceSynchronize();

            // Update W1 and b1
            int blocksW1 = ((SIZE * H1) + threadsPerBlock - 1) / threadsPerBlock;
            update_weights_kernel<<<blocksW1, threadsPerBlock>>>(d_W1, d_delta1, d_input, LR, SIZE, H1);
            CHECK_CUDA_ERROR(cudaGetLastError());
            cudaDeviceSynchronize();
            update_bias_kernel<<<blocksH1, threadsPerBlock>>>(d_b1, d_delta1, LR, H1);
            CHECK_CUDA_ERROR(cudaGetLastError());
            cudaDeviceSynchronize();
        }
        printf("Epoch %d, Loss=%.4f\n", epoch, loss/NUM_TRAIN);
    }

    // ===== Copy model back to host =====
    CHECK_CUDA_ERROR(cudaMemcpy(model->W1, d_W1, SIZE * H1 * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(model->b1, d_b1, H1 * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(model->W2, d_W2, H1 * H2 * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(model->b2, d_b2, H2 * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(model->W3, d_W3, H2 * CLASSES * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(model->b3, d_b3, CLASSES * sizeof(float), cudaMemcpyDeviceToHost));

    // ===== Free device memory =====
    cudaFree(d_W1); cudaFree(d_b1);
    cudaFree(d_W2); cudaFree(d_b2);
    cudaFree(d_W3); cudaFree(d_b3);
    cudaFree(d_h1); cudaFree(d_h1a);
    cudaFree(d_h2); cudaFree(d_h2a);
    cudaFree(d_out); cudaFree(d_outa);
    cudaFree(d_delta1); cudaFree(d_delta2); cudaFree(d_delta3);
    cudaFree(d_input); cudaFree(d_label);
}

/* Save the trained model to a binary file
* Arguments:
*   model: pointer to the MODEL structure containing trained weights and biases
* Returns:
*   None
*/
void save_model(MODEL* model){
	FILE *f = fopen("model.bin", "wb");
	fwrite(model->W1, sizeof(float), SIZE*H1, f);
	fwrite(model->b1, sizeof(float), H1, f);
	fwrite(model->W2, sizeof(float), H1*H2, f);
	fwrite(model->b2, sizeof(float), H2, f);
	fwrite(model->W3, sizeof(float), H2*CLASSES, f);
	fwrite(model->b3, sizeof(float), CLASSES,f);
	fclose(f);
}

/* Load the trained model from a binary file
* Arguments:
*   model (out): pointer to the MODEL structure to populate with loaded weights and biases
* Returns:
*   None
*/
void load_model(MODEL* model){
	FILE *f = fopen("model.bin", "rb");
	fread(model->W1, sizeof(float), SIZE*H1, f);
	fread(model->b1, sizeof(float), H1, f);
	fread(model->W2, sizeof(float), H1*H2, f);
	fread(model->b2, sizeof(float), H2, f);
	fread(model->W3, sizeof(float), H2*CLASSES, f);
	fread(model->b3, sizeof(float), CLASSES, f);
	fclose(f);
}

/* Predict the class of a given input image
* Arguments:
*   x: input image array (flattened 28x28 pixels)
*   model: pointer to the MODEL structure containing trained weights and biases
* Returns:
*   None (prints predicted class and confidence)
*/
void predict(float *x, MODEL* model){
    float h1[H1], h1a[H1], h2[H2], h2a[H2], out[CLASSES], outa[CLASSES];

    // forward pass
    for (int j=0;j<H1;j++){ h1[j]=model->b1[j]; for(int i=0;i<SIZE;i++) h1[j]+=x[i]*model->W1[i*H1+j]; h1a[j]=relu(h1[j]); }
    for (int j=0;j<H2;j++){ h2[j]=model->b2[j]; for(int i=0;i<H1;i++) h2[j]+=h1a[i]*model->W2[i*H2+j]; h2a[j]=relu(h2[j]); }
    for (int k=0;k<CLASSES;k++){ out[k]=model->b3[k]; for(int j=0;j<H2;j++) out[k]+=h2a[j]*model->W3[j*CLASSES+k]; }
    softmax(out,outa,CLASSES);

    // print predicted class
    int pred=0; float max=outa[0];
    for(int k=1;k<CLASSES;k++) if(outa[k]>max){ max=outa[k]; pred=k; }
    printf("Predicted digit: %d (confidence %.2f)\n", pred, max);
}


