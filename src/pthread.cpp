#include <iostream>
#include <complex>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <omp.h>
#include <pthread.h>

using namespace std;
typedef complex<double> Complex;
typedef vector<vector<Complex>> ComplexMatrix;

struct ThreadData {
    int startRow, endRow;
    ComplexMatrix *input, *output;
    int M, N;
};

// Thread function for DFT computation
void *DFTThread(void *args) {
    ThreadData *data = (ThreadData *)args;
    int M = data->M;
    int N = data->N;
    ComplexMatrix *input = data->input;
    ComplexMatrix *output = data->output;

    for (int u = data->startRow; u < data->endRow; ++u) {
        for (int v = 0; v < N; ++v) {
            for (int x = 0; x < M; ++x) {
                for (int y = 0; y < N; ++y) {
                    double angle = 2 * M_PI * ((u * x / (double)M) + (v * y / (double)N));
                    (*output)[u][v] += (*input)[x][y] * exp(-Complex(0, 1) * angle);
                }
            }
        }
        // printf("u = %d\n", u);
    }

    pthread_exit(NULL);
}

// Modified DFT function using pthreads
ComplexMatrix DFT(ComplexMatrix &input, int numThreads) {
    int M = input.size();
    int N = input[0].size();
    ComplexMatrix output(M, vector<Complex>(N, 0));

    pthread_t threads[numThreads];
    ThreadData threadData[numThreads];

    int rowsPerThread = M / numThreads;
    for (int i = 0; i < numThreads; ++i) {
        threadData[i].startRow = i * rowsPerThread;
        threadData[i].endRow = (i == numThreads - 1) ? M : (i + 1) * rowsPerThread;
        threadData[i].input = &input;
        threadData[i].output = &output;
        threadData[i].M = M;
        threadData[i].N = N;

        pthread_create(&threads[i], NULL, DFTThread, (void *)&threadData[i]);
    }

    // Join threads
    for (int i = 0; i < numThreads; ++i) {
        pthread_join(threads[i], NULL);
    }

    return output;
}

// Thread function for IDFT computation
void *IDFTThread(void *args) {
    ThreadData *data = (ThreadData *)args;
    int M = data->M;
    int N = data->N;
    ComplexMatrix *input = data->input;
    ComplexMatrix *output = data->output;

    for (int x = data->startRow; x < data->endRow; ++x) {
        for (int y = 0; y < N; ++y) {
            for (int u = 0; u < M; ++u) {
                for (int v = 0; v < N; ++v) {
                    double angle = 2 * M_PI * ((u * x / (double)M) + (v * y / (double)N));
                    (*output)[x][y] += (*input)[u][v] * exp(Complex(0, 1) * angle);
                }
            }
            (*output)[x][y] /= Complex(M * N, 0);
        }
    }

    pthread_exit(NULL);
}

// Modified IDFT function using pthreads
ComplexMatrix IDFT(ComplexMatrix &input, int numThreads) {
    int M = input.size();
    int N = input[0].size();
    ComplexMatrix output(M, vector<Complex>(N, 0));

    pthread_t threads[numThreads];
    ThreadData threadData[numThreads];

    int rowsPerThread = M / numThreads;
    for (int i = 0; i < numThreads; ++i) {
        threadData[i].startRow = i * rowsPerThread;
        threadData[i].endRow = (i == numThreads - 1) ? M : (i + 1) * rowsPerThread;
        threadData[i].input = &input;
        threadData[i].output = &output;
        threadData[i].M = M;
        threadData[i].N = N;

        pthread_create(&threads[i], NULL, IDFTThread, (void *)&threadData[i]);
    }

    // Join threads
    for (int i = 0; i < numThreads; ++i) {
        pthread_join(threads[i], NULL);
    }

    return output;
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


// Main function (for demonstration)
int main(int argc, char** argv) {
    if (argc != 2) {
        cerr << "Usage: " << argv[0] << " <num_threads>" << endl;
        return 1;
    }

    double start, end;
    
    int numThreads = stoi(argv[1]);

    double startTotal = omp_get_wtime();

    // Load the image and watermark
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

    // Perform DFT
    double startDFT = omp_get_wtime();
    ComplexMatrix inputDFT = DFT(complexInput, numThreads);
    ComplexMatrix watermarkDFT = DFT(complexWatermark, numThreads);
    double endDFT = omp_get_wtime();
    cout << "DFT execution time: " << endDFT - startDFT << " seconds" << endl;

    // Add a watermark in the frequency domain
    double alpha = 0.8;
    ComplexMatrix watermarkedDFT(inputDFT.size(), vector<Complex>(inputDFT[0].size()));
    for (size_t i = 0; i < inputDFT.size(); ++i) {
        for (size_t j = 0; j < inputDFT[i].size(); ++j) {
            watermarkedDFT[i][j] = inputDFT[i][j] + alpha * watermarkDFT[i][j];
        }
    }

    // Perform IDFT
    double startIDFT = omp_get_wtime();
    ComplexMatrix watermarkedImage = IDFT(watermarkedDFT, numThreads);
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