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
#include "oclBlackScholes_common.h"



static cl_program cpBlackScholes;   //OpenCL program
static cl_kernel  ckBlackScholes;   //OpenCL kernel
static cl_command_queue cqDefaultCommandQueue;

extern "C" void initBlackScholes(cl_context cxGPUContext, cl_command_queue cqParamCommandQueue, const char **argv){
    cl_int ciErrNum;
    size_t kernelLength;

    shrLog(LOGBOTH, 0, "...loading BlackScholes.cl\n");
        char *cBlackScholes = oclLoadProgSource(shrFindFilePath("BlackScholes.cl", argv[0]), "// My comment\n", &kernelLength);
        shrCheckError(cBlackScholes != NULL, shrTRUE);

    shrLog(LOGBOTH, 0, "...creating BlackScholes program\n");
        cpBlackScholes = clCreateProgramWithSource(cxGPUContext, 1, (const char **)&cBlackScholes, &kernelLength, &ciErrNum);
        shrCheckError(ciErrNum, CL_SUCCESS);

    shrLog(LOGBOTH, 0, "...building BlackScholes program\n");
        ciErrNum = clBuildProgram(cpBlackScholes, 0, NULL, NULL, NULL, NULL);
        shrCheckError(ciErrNum, CL_SUCCESS);

    shrLog(LOGBOTH, 0, "...creating BlackScholes kernels\n");
        ckBlackScholes = clCreateKernel(cpBlackScholes, "BlackScholes", &ciErrNum);
        shrCheckError(ciErrNum, CL_SUCCESS);

    cqDefaultCommandQueue = cqParamCommandQueue;
    free(cBlackScholes);
}

extern "C" void closeBlackScholes(void){
    cl_int ciErrNum;
    ciErrNum  = clReleaseKernel(ckBlackScholes);
    ciErrNum |= clReleaseProgram(cpBlackScholes);
    shrCheckError(ciErrNum, CL_SUCCESS);
}


////////////////////////////////////////////////////////////////////////////////
// OpenCL Black-Scholes kernel launcher
////////////////////////////////////////////////////////////////////////////////
extern "C" void BlackScholes(
    cl_command_queue cqCommandQueue,
    cl_mem d_Call, //Call option price
    cl_mem d_Put,  //Put option price
    cl_mem d_S,    //Current stock price
    cl_mem d_X,    //Option strike price
    cl_mem d_T,    //Option years
    cl_float R,    //Riskless rate of return
    cl_float V,    //Stock volatility
    cl_uint optionCount
){
    cl_int ciErrNum;

    if(!cqCommandQueue)
        cqCommandQueue = cqDefaultCommandQueue;

    ciErrNum  = clSetKernelArg(ckBlackScholes, 0, sizeof(cl_mem),   (void *)&d_Call);
    ciErrNum |= clSetKernelArg(ckBlackScholes, 1, sizeof(cl_mem),   (void *)&d_Put);
    ciErrNum |= clSetKernelArg(ckBlackScholes, 2, sizeof(cl_mem),   (void *)&d_S);
    ciErrNum |= clSetKernelArg(ckBlackScholes, 3, sizeof(cl_mem),   (void *)&d_X);
    ciErrNum |= clSetKernelArg(ckBlackScholes, 4, sizeof(cl_mem),   (void *)&d_T);
    ciErrNum |= clSetKernelArg(ckBlackScholes, 5, sizeof(cl_float), (void *)&R);
    ciErrNum |= clSetKernelArg(ckBlackScholes, 6, sizeof(cl_float), (void *)&V);
    ciErrNum |= clSetKernelArg(ckBlackScholes, 7, sizeof(cl_uint),  (void *)&optionCount);
    shrCheckError(ciErrNum, CL_SUCCESS);

    //Run the kernel
    size_t globalWorkSize = 30 * 1024;
    ciErrNum = clEnqueueNDRangeKernel(cqCommandQueue, ckBlackScholes, 1, NULL, &globalWorkSize, NULL, 0, NULL, NULL);
    shrCheckError(ciErrNum, CL_SUCCESS);
}
