
#ifndef OPENCLGEMM_H_INCLUDED
#define OPENCLGEMM_H_INCLUDED

//#define CL_HPP_MINIMUM_OPENCL_VERSION   110
//#define CL_HPP_TARGET_OPENCL_VERSION    120
//#define CL_HPP_ENABLE_EXCEPTIONS

#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdio>
#include <cstdlib>

#define CL_USE_DEPRECATED_OPENCL_1_1_APIS
#include <CL/cl.hpp>


#include "OpenCLDemo.h"

void openCLGEMM(unsigned char *bufIn, unsigned char *bufOut, int *info);

#endif