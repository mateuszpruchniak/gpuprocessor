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

#include <oclUtils.h>
#include "oclConvolutionSeparable_common.h"

////////////////////////////////////////////////////////////////////////////////
// OpenCL launcher for convolutionRows / convolutionColumns kernels
////////////////////////////////////////////////////////////////////////////////
//OpenCL convolutionSeparable program
static cl_program
    cpConvolutionSeparable;

//OpenCL convolutionSeparable kernels
static cl_kernel
    ckConvolutionRows, ckConvolutionColumns;

extern "C" void initConvolutionSeparable(cl_context cxGPUContext, cl_command_queue cqParamCommandQue, const char **argv){
    cl_int ciErrNum;
    size_t kernelLength;

    shrLog(LOGBOTH, 0, "Loading ConvolutionSeparable.cl...\n");
        char *cPathAndName = shrFindFilePath("ConvolutionSeparable.cl", argv[0]);
        shrCheckError(cPathAndName != NULL, shrTRUE);
        char *cConvolutionSeparable = oclLoadProgSource(cPathAndName, "// My comment\n", &kernelLength);
        shrCheckError(cConvolutionSeparable != NULL, shrTRUE);

    shrLog(LOGBOTH, 0, "Creating convolutionSeparable program...\n");
        cpConvolutionSeparable = clCreateProgramWithSource(cxGPUContext, 1, (const char **)&cConvolutionSeparable, &kernelLength, &ciErrNum);
        shrCheckError(ciErrNum, CL_SUCCESS);

    shrLog(LOGBOTH, 0, "Building convolutionSeparable program...\n");
        ciErrNum = clBuildProgram(cpConvolutionSeparable, 0, NULL, "-cl-mad-enable", NULL, NULL);
        shrCheckError(ciErrNum, CL_SUCCESS);

    shrLog(LOGBOTH, 0, "Creating convolutionSeparable kernels...\n");
        ckConvolutionRows = clCreateKernel(cpConvolutionSeparable, "convolutionRows", &ciErrNum);
        shrCheckError(ciErrNum, CL_SUCCESS);
        ckConvolutionColumns = clCreateKernel(cpConvolutionSeparable, "convolutionColumns", &ciErrNum);
        shrCheckError(ciErrNum, CL_SUCCESS);

    free(cConvolutionSeparable);
}

extern "C" void closeConvolutionSeparable(void){
    cl_int ciErrNum;

    ciErrNum  = clReleaseKernel(ckConvolutionColumns);
    ciErrNum |= clReleaseKernel(ckConvolutionRows);
    ciErrNum |= clReleaseProgram(cpConvolutionSeparable);
}

extern "C" void convolutionRows(
    cl_command_queue cqCommandQueue,
    cl_mem d_Dst,
    cl_mem d_Src,
    cl_mem c_Kernel,
    cl_uint imageW,
    cl_uint imageH
){
    cl_int ciErrNum;
    size_t localWorkSize[2], globalWorkSize[2];

    const int   ROWS_BLOCKDIM_X = 16;
    const int   ROWS_BLOCKDIM_Y = 4;
    const int ROWS_RESULT_STEPS = 4;
    const int   ROWS_HALO_STEPS = 1;

    assert( ROWS_BLOCKDIM_X * ROWS_HALO_STEPS >= KERNEL_RADIUS );
    assert( imageW % (ROWS_RESULT_STEPS * ROWS_BLOCKDIM_X) == 0 );
    assert( imageH % ROWS_BLOCKDIM_Y == 0 );

    ciErrNum  = clSetKernelArg(ckConvolutionRows, 0, sizeof(cl_mem),       (void*)&d_Dst);
    ciErrNum |= clSetKernelArg(ckConvolutionRows, 1, sizeof(cl_mem),       (void*)&d_Src);
    ciErrNum |= clSetKernelArg(ckConvolutionRows, 2, sizeof(cl_mem),       (void*)&c_Kernel);
    ciErrNum |= clSetKernelArg(ckConvolutionRows, 3, sizeof(unsigned int), (void*)&imageW);
    ciErrNum |= clSetKernelArg(ckConvolutionRows, 4, sizeof(unsigned int), (void*)&imageH);
    ciErrNum |= clSetKernelArg(ckConvolutionRows, 5, sizeof(unsigned int), (void*)&imageW);
    shrCheckError(ciErrNum, CL_SUCCESS);

    localWorkSize[0] = ROWS_BLOCKDIM_X;
    localWorkSize[1] = ROWS_BLOCKDIM_Y;
    globalWorkSize[0] = imageW / ROWS_RESULT_STEPS;
    globalWorkSize[1] = imageH;

    ciErrNum = clEnqueueNDRangeKernel(cqCommandQueue, ckConvolutionRows, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
    shrCheckError(ciErrNum, CL_SUCCESS);
}

extern "C" void convolutionColumns(
    cl_command_queue cqCommandQueue,
    cl_mem d_Dst,
    cl_mem d_Src,
    cl_mem c_Kernel,
    cl_uint imageW,
    cl_uint imageH
){
    cl_int ciErrNum;
    size_t localWorkSize[2], globalWorkSize[2];

    const int   COLUMNS_BLOCKDIM_X = 16;
    const int   COLUMNS_BLOCKDIM_Y = 8;
    const int COLUMNS_RESULT_STEPS = 4;
    const int   COLUMNS_HALO_STEPS = 1;

    assert( COLUMNS_BLOCKDIM_Y * COLUMNS_HALO_STEPS >= KERNEL_RADIUS );
    assert( imageW % COLUMNS_BLOCKDIM_X == 0 );
    assert( imageH % (COLUMNS_RESULT_STEPS * COLUMNS_BLOCKDIM_Y) == 0 );

    ciErrNum  = clSetKernelArg(ckConvolutionColumns, 0, sizeof(cl_mem),       (void*)&d_Dst);
    ciErrNum |= clSetKernelArg(ckConvolutionColumns, 1, sizeof(cl_mem),       (void*)&d_Src);
    ciErrNum |= clSetKernelArg(ckConvolutionColumns, 2, sizeof(cl_mem),       (void*)&c_Kernel);
    ciErrNum |= clSetKernelArg(ckConvolutionColumns, 3, sizeof(unsigned int), (void*)&imageW);
    ciErrNum |= clSetKernelArg(ckConvolutionColumns, 4, sizeof(unsigned int), (void*)&imageH);
    ciErrNum |= clSetKernelArg(ckConvolutionColumns, 5, sizeof(unsigned int), (void*)&imageW);
    shrCheckError(ciErrNum, CL_SUCCESS);

    localWorkSize[0] = COLUMNS_BLOCKDIM_X;
    localWorkSize[1] = COLUMNS_BLOCKDIM_Y;
    globalWorkSize[0] = imageW;
    globalWorkSize[1] = imageH / COLUMNS_RESULT_STEPS;

    ciErrNum = clEnqueueNDRangeKernel(cqCommandQueue, ckConvolutionColumns, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
    shrCheckError(ciErrNum, CL_SUCCESS);
}
