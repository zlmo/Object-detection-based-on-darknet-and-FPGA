#ifndef FPGA_H
#define FPGA_H

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

bool init();
void run();
void cleanup();




#endif
