#include <iostream>
#include <complex>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "cuda_runtime.h"
#include <cuComplex.h>
#include "kernel.h"
#include <omp.h>

using namespace std;

typedef complex<double> Complex;
typedef vector<vector<Complex>> ComplexMatrix;

// Forward declaration of CUDA functions
extern "C" void RunDFT(cuDoubleComplex* input, cuDoubleComplex* output, int M, int N);
extern "C" void RunIDFT(cuDoubleComplex* input, cuDoubleComplex* output, int M, int N);

ComplexMatrix RunCUDADFT(const ComplexMatrix &complexInput) {
    int M = complexInput.size();
    int N = complexInput[0].size();

    // Allocate memory for input and output data
    cuDoubleComplex *input = new cuDoubleComplex[M * N];
    cuDoubleComplex *output = new cuDoubleComplex[M * N];

    // Convert ComplexMatrix to cuDoubleComplex array
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            input[i * N + j] = make_cuDoubleComplex(complexInput[i][j].real(), complexInput[i][j].imag());
        }
    }

    // Run DFT using CUDA
    RunDFT(input, output, M, N);

    // Convert cuDoubleComplex array back to ComplexMatrix
    ComplexMatrix resultDFT(M, vector<Complex>(N));
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            resultDFT[i][j] = Complex(output[i * N + j].x, output[i * N + j].y);
        }
    }

    // Free dynamically allocated memory
    delete[] input;
    delete[] output;

    return resultDFT;
}

ComplexMatrix RunCUDAIDFT(const ComplexMatrix &complexInput) {
    int M = complexInput.size();
    int N = complexInput[0].size();

    // Allocate memory for input and output data
    cuDoubleComplex *input = new cuDoubleComplex[M * N];
    cuDoubleComplex *output = new cuDoubleComplex[M * N];

    // Convert ComplexMatrix to cuDoubleComplex array
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            input[i * N + j] = make_cuDoubleComplex(complexInput[i][j].real(), complexInput[i][j].imag());
        }
    }

    // Run IDFT using CUDA
    RunIDFT(input, output, M, N);

    // Convert cuDoubleComplex array back to ComplexMatrix
    ComplexMatrix resultIDFT(M, vector<Complex>(N));
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            resultIDFT[i][j] = Complex(output[i * N + j].x, output[i * N + j].y);
        }
    }

    // Free dynamically allocated memory
    delete[] input;
    delete[] output;

    return resultIDFT;
}


// Convert OpenCV Mat to ComplexMatrix
ComplexMatrix MatToComplexMatrix(const cv::Mat &mat) {
    ComplexMatrix output(mat.rows, vector<Complex>(mat.cols));

    for (int i = 0; i < mat.rows; ++i) {
        for (int j = 0; j < mat.cols; ++j) {
            output[i][j] = Complex(mat.at<float>(i, j), 0);
        }
    }

    return output;
}

// Convert ComplexMatrix to OpenCV Mat
cv::Mat ComplexMatrixToMat(const ComplexMatrix &matrix) {
    cv::Mat output(matrix.size(), matrix[0].size(), CV_32F);

    for (size_t i = 0; i < matrix.size(); ++i) {
        for (size_t j = 0; j < matrix[i].size(); ++j) {
            output.at<float>(i, j) = abs(matrix[i][j]);
        }
    }

    return output;
}

int main() {
    double startTotal = omp_get_wtime();

    // Load, resize, and normalize the image and watermark as before
    cv::Mat inputImage = cv::imread("./img/input_image.png", cv::IMREAD_GRAYSCALE);
    cv::Mat watermark = cv::imread("./img/watermark.png", cv::IMREAD_GRAYSCALE);

    // Resize images
    cv::resize(inputImage, inputImage, cv::Size(), 0.2, 0.2, cv::INTER_LINEAR);
    cv::resize(watermark, watermark, cv::Size(), 0.05, 0.05, cv::INTER_LINEAR);

    // Normalize the images
    inputImage.convertTo(inputImage, CV_32F, 1.0 / 255.0);
    watermark.convertTo(watermark, CV_32F, 1.0 / 255.0);

    // repeat watermark image to match the input image size
    cv::Mat watermarkRepeated;
    cv::repeat(watermark, inputImage.rows / watermark.rows + 1, inputImage.cols / watermark.cols + 1, watermarkRepeated);

    // Convert Mat to ComplexMatrix
    ComplexMatrix complexInput = MatToComplexMatrix(inputImage);
    ComplexMatrix complexWatermark = MatToComplexMatrix(watermarkRepeated);

    double startDFT = omp_get_wtime();
    ComplexMatrix inputDFT = RunCUDADFT(complexInput);
    ComplexMatrix watermarkDFT = RunCUDADFT(complexWatermark);
    double endDFT = omp_get_wtime();
    cout << "DFT execution time: " << endDFT - startDFT << " seconds" << endl;


    double alpha = 0.8;
    ComplexMatrix watermarkedDFT(inputDFT.size(), vector<Complex>(inputDFT[0].size()));
    for (size_t i = 0; i < inputDFT.size(); ++i) {
        for (size_t j = 0; j < inputDFT[i].size(); ++j) {
            watermarkedDFT[i][j] = inputDFT[i][j] + alpha * watermarkDFT[i][j];
        }
    }

    double startIDFT = omp_get_wtime();
    ComplexMatrix watermarkedImage = RunCUDAIDFT(watermarkedDFT);
    double endIDFT = omp_get_wtime();
    cout << "IDFT execution time: " << endIDFT - startIDFT << " seconds" << endl;
    

    // Convert ComplexMatrix back to Mat and normalize
    cv::Mat resultImage = ComplexMatrixToMat(watermarkedImage);
    resultImage.convertTo(resultImage, CV_8U, 255.0);

    // Save the image
    cv::imwrite("./img/watermarked_image.png", resultImage);

    double endTotal = omp_get_wtime();
    cout << "Total execution time: " << endTotal - startTotal << " seconds" << endl;
    return 0;
}
