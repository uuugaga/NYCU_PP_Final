#include <cuda.h>
#include <cuComplex.h>
#include "kernel.h"

__global__ void DFTKernel(cuDoubleComplex* input, cuDoubleComplex* output, int M, int N) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    int v = blockIdx.y * blockDim.y + threadIdx.y;

    if (u < M && v < N) {
        cuDoubleComplex sum = make_cuDoubleComplex(0, 0);
        for (int x = 0; x < M; ++x) {
            for (int y = 0; y < N; ++y) {
                double angle = 2 * M_PI * ((u * x / (double)M) + (v * y / (double)N));
                cuDoubleComplex expVal = make_cuDoubleComplex(cos(angle), -sin(angle));
                sum = cuCadd(sum, cuCmul(input[x * N + y], expVal));
            }
        }
        output[u * N + v] = sum;
    }
}

__global__ void IDFTKernel(cuDoubleComplex* input, cuDoubleComplex* output, int M, int N) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < M && y < N) {
        cuDoubleComplex sum = make_cuDoubleComplex(0, 0);
        for (int u = 0; u < M; ++u) {
            for (int v = 0; v < N; ++v) {
                double angle = 2 * M_PI * ((u * x / (double)M) + (v * y / (double)N));
                cuDoubleComplex expVal = make_cuDoubleComplex(cos(angle), sin(angle));
                sum = cuCadd(sum, cuCmul(input[u * N + v], expVal));
            }
        }
        output[x * N + y] = cuCdiv(sum, make_cuDoubleComplex(M * N, 0));
    }
}

void RunDFT(cuDoubleComplex* input, cuDoubleComplex* output, int M, int N) {
    cuDoubleComplex *d_input, *d_output;
    size_t size = M * N * sizeof(cuDoubleComplex);

    // Allocate device memory
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);

    // Copy input data to GPU
    cudaMemcpy(d_input, input, size, cudaMemcpyHostToDevice);

    // Kernel launch configuration
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((M + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (N + threadsPerBlock.y - 1) / threadsPerBlock.y);
    DFTKernel<<<numBlocks, threadsPerBlock>>>(d_input, d_output, M, N);

    // Copy result back to host
    cudaMemcpy(output, d_output, size, cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_input);
    cudaFree(d_output);
}

// Function to run IDFT on the GPU
void RunIDFT(cuDoubleComplex* input, cuDoubleComplex* output, int M, int N) {
    cuDoubleComplex *d_input, *d_output;
    size_t size = M * N * sizeof(cuDoubleComplex);

    // Allocate device memory
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);

    // Copy input data to GPU
    cudaMemcpy(d_input, input, size, cudaMemcpyHostToDevice);

    // Kernel launch configuration
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((M + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (N + threadsPerBlock.y - 1) / threadsPerBlock.y);
    IDFTKernel<<<numBlocks, threadsPerBlock>>>(d_input, d_output, M, N);

    // Copy result back to host
    cudaMemcpy(output, d_output, size, cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_input);
    cudaFree(d_output);
}