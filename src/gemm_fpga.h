#ifndef GEMM_FPGA_H
#define GEMM_FPGA_H

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cstring>
#include <string.h>
#include "CL/opencl.h"
#include "aocl_utils.h"
#include "darknet.h"

using namespace aocl_utils;


bool gemm_init();
void gemm_run(int M, int N, int K, float ALPHA,
  float *A, int lda,
  float *B, int ldb,
  float *C, int ldc);
void gemm_cleanup();




#endif
