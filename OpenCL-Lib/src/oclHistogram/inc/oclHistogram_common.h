/*
 * Copyright 1993-2009 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and 
 * proprietary rights in and to this software and related documentation. 
 * Any use, reproduction, disclosure, or distribution of this software 
 * and related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 *
 * Please refer to the applicable NVIDIA end user license agreement (EULA) 
 * associated with this source code for terms and conditions that govern 
 * your use of this NVIDIA software.
 * 
 */

#ifndef OCLHISTOGRAM_COMMON_H
#define OCLHISTOGRAM_COMMON_H

#include <oclUtils.h>

////////////////////////////////////////////////////////////////////////////////
// Common definitions
////////////////////////////////////////////////////////////////////////////////
#define HISTOGRAM64_BIN_COUNT 64U
#define HISTOGRAM256_BIN_COUNT 256U
typedef cl_uint uint;
typedef cl_uchar uchar;

////////////////////////////////////////////////////////////////////////////////
// Reference CPU histogram
////////////////////////////////////////////////////////////////////////////////
extern "C" void histogram64CPU(
    uint *h_Histogram,
    void *h_Data,
    uint byteCount
);

extern "C" void histogram256CPU(
    uint *h_Histogram,
    void *h_Data,
    uint byteCount
);

////////////////////////////////////////////////////////////////////////////////
// GPU histogram
////////////////////////////////////////////////////////////////////////////////
extern "C" void initHistogram64(cl_context cxGPUContext, cl_command_queue cqParamCommandQue, const char **argv);
extern "C" void initHistogram256(cl_context cxGPUContext, cl_command_queue cqParamCommandQue, const char **argv);
extern "C" void closeHistogram64(void);
extern "C" void closeHistogram256(void);
extern "C" size_t histogram64(cl_command_queue cqCommandQueue, cl_mem d_Histogram, cl_mem d_Data, uint byteCount);
extern "C" size_t histogram256(cl_command_queue cqCommandQueue, cl_mem d_Histogram, cl_mem d_Data, uint byteCount);

#endif
