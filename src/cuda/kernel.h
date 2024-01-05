// kernel.h
#ifndef KERNEL_H
#define KERNEL_H

#include <cuComplex.h>

#ifdef __cplusplus
extern "C" {
#endif

void RunDFT(cuDoubleComplex* input, cuDoubleComplex* output, int M, int N);
void RunIDFT(cuDoubleComplex* input, cuDoubleComplex* output, int M, int N);

#ifdef __cplusplus
}
#endif

#endif // KERNEL_H
