/*
 * Copyright 1993-2009 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and 
 * proprietary rights in and to this software and related documentation. 
 * Any use, reproduction, disclosure, or distribution of this software 
 * and related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 * 
 */
 
/* Matrix transpose with Cuda 
 * Host code.

 * This example transposes arbitrary-size matrices.  It compares a naive
 * transpose kernel that suffers from non-coalesced writes, to an optimized
 * transpose with fully coalesced memory access and no bank conflicts.  On 
 * a G80 GPU, the optimized transpose can be more than 10x faster for large
 * matrices.
 */

// standard utility and system includes
#include <oclUtils.h>

#define BLOCK_DIM 16

// forward declarations
// *********************************************************************
int runTest( int argc, const char** argv);
extern "C" void computeGold( float* reference, float* idata, 
                         const unsigned int size_x, const unsigned int size_y );

// Main Program
// *********************************************************************
int main( int argc, const char** argv) 
{    
    // set logfile name and start logs
    shrSetLogFileName ("oclTranspose.txt");
    shrLog(LOGBOTH, 0, "%s Starting...\n\n", argv[0]); 

    // run the main test
    int result = runTest(argc, argv);
    shrCheckError(result, 0);

    // finish
    shrEXIT(argc, argv);
}

//! Run a simple test for CUDA
// *********************************************************************
int runTest( int argc, const char** argv) 
{
    cl_context cxGPUContext;
    cl_command_queue cqCommandQueue;
    cl_program cpProgram;
    cl_kernel ckNaiveKernel;
    cl_kernel ckKernel;
    size_t szGlobalWorkSize[2];
    size_t szLocalWorkSize[2];
    cl_int ciErrNum;

    const unsigned int size_x = 256;
    const unsigned int size_y = 4096;

    // size of memory required to store the matrix
    const unsigned int mem_size = sizeof(float) * size_x * size_y;

    // create the OpenCL context on a GPU device
    cxGPUContext = clCreateContextFromType(0, CL_DEVICE_TYPE_GPU, NULL, NULL, &ciErrNum);
    shrCheckError(ciErrNum, CL_SUCCESS);
  
    // get devices
    cl_device_id device;
    if( shrCheckCmdLineFlag(argc, argv, "device") ) {
      int device_nr = 0;
      shrGetCmdLineArgumenti(argc, argv, "device", &device_nr);
      device = oclGetDev(cxGPUContext, device_nr);
    } else {
      device = oclGetMaxFlopsDev(cxGPUContext);
    }

    // create a command-queue
    cqCommandQueue = clCreateCommandQueue(cxGPUContext, device, 0, &ciErrNum);
    shrCheckError(ciErrNum, CL_SUCCESS);

    // allocate and initalize host memory
    float* h_idata = (float*)malloc(mem_size);
    srand(15235911);
    shrFillArray(h_idata, (size_x * size_y));

    // allocate device memory and copy host to device memory
    cl_mem d_idata = clCreateBuffer(cxGPUContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
				                    mem_size, h_idata, &ciErrNum);
    shrCheckError(ciErrNum, CL_SUCCESS);
    cl_mem d_odata = clCreateBuffer(cxGPUContext, CL_MEM_WRITE_ONLY ,
    			                    mem_size, NULL, &ciErrNum);
    shrCheckError(ciErrNum, CL_SUCCESS);

    // Program Setup
    size_t program_length;
    char* source_path = shrFindFilePath("transpose.cl", argv[0]);
    shrCheckError(source_path != NULL, shrTRUE);
    char *source = oclLoadProgSource(source_path, "", &program_length);
    shrCheckError(source != NULL, shrTRUE);

    // create the program
    cpProgram = clCreateProgramWithSource(cxGPUContext, 1,
					  (const char **)&source, &program_length, &ciErrNum);
    shrCheckError(ciErrNum, CL_SUCCESS);
    
    // build the program
    ciErrNum = clBuildProgram(cpProgram, 0, NULL, "-cl-mad-enable", NULL, NULL);
    if (ciErrNum != CL_SUCCESS)
    {
        // write out standard error, Build Log and PTX, then return error
        shrLog(LOGBOTH | ERRORMSG, ciErrNum, STDERROR);
        oclLogBuildInfo(cpProgram, oclGetFirstDev(cxGPUContext));
        oclLogPtx(cpProgram, oclGetFirstDev(cxGPUContext), "oclTranspose.ptx");
        return(EXIT_FAILURE); 
    }

    // create the naive transpose kernel
    ckNaiveKernel = clCreateKernel(cpProgram, "transpose_naive", &ciErrNum);
    shrCheckError(ciErrNum, CL_SUCCESS);

    // set the args values for the naive kernel
    ciErrNum  = clSetKernelArg(ckNaiveKernel, 0, sizeof(cl_mem), (void *) &d_odata);
    ciErrNum |= clSetKernelArg(ckNaiveKernel, 1, sizeof(cl_mem), (void *) &d_idata);
    ciErrNum |= clSetKernelArg(ckNaiveKernel, 2, sizeof(int), &size_x );
    ciErrNum |= clSetKernelArg(ckNaiveKernel, 3, sizeof(int), &size_y );
    shrCheckError(ciErrNum, CL_SUCCESS);

    // create the more optimized transpose kernel
    ckKernel = clCreateKernel(cpProgram, "transpose", &ciErrNum);
    shrCheckError(ciErrNum, CL_SUCCESS);

    // set the args values for the more optimized transpose kernel
    ciErrNum  = clSetKernelArg(ckKernel, 0, sizeof(cl_mem), (void *) &d_odata);
    ciErrNum |= clSetKernelArg(ckKernel, 1, sizeof(cl_mem), (void *) &d_idata);
    ciErrNum |= clSetKernelArg(ckKernel, 2, sizeof(int), &size_x );
    ciErrNum |= clSetKernelArg(ckKernel, 3, sizeof(int), &size_y );
    ciErrNum |= clSetKernelArg(ckKernel, 4, sizeof(float) * BLOCK_DIM * (BLOCK_DIM+1), NULL );
    shrCheckError(ciErrNum, CL_SUCCESS);

    // setup execution paramete
    szLocalWorkSize[0] = BLOCK_DIM;
    szLocalWorkSize[1] = BLOCK_DIM;
    szGlobalWorkSize[0] = shrRoundUp(size_x, BLOCK_DIM);
    szGlobalWorkSize[1] = shrRoundUp(size_y, BLOCK_DIM);
                                   
    // warmup so we don't time driver startup
    ciErrNum = clEnqueueNDRangeKernel(cqCommandQueue, ckNaiveKernel, 2, NULL,
                                      szGlobalWorkSize, szLocalWorkSize, 
                                      0, NULL, NULL);
    ciErrNum |= clEnqueueNDRangeKernel(cqCommandQueue, ckKernel, 2, NULL,
                                      szGlobalWorkSize, szLocalWorkSize, 
                                      0, NULL, NULL);
    shrCheckError(ciErrNum, CL_SUCCESS);

    // execute the naive kernel 1000 times
    int numIterations = 1000;
    oclPrintDevName(LOGBOTH, device);
    shrLog(LOGBOTH, 0, "\nTransposing a %d by %d matrix of floats...\n\n", size_x, size_y);
    shrDeltaT(0);
    for (int i = 0; i < numIterations; ++i)
    {
        ciErrNum |= clEnqueueNDRangeKernel(cqCommandQueue, ckNaiveKernel, 2, NULL,
                               szGlobalWorkSize, szLocalWorkSize, 
                               0, NULL, NULL);
        clFinish(cqCommandQueue);
    }
    shrCheckError(ciErrNum, CL_SUCCESS);
    double naiveTime = shrDeltaT(0)/(double)numIterations;

    // execute the more optimized kernel 1000 times
    shrDeltaT(0);
    for (int i = 0; i < numIterations; ++i)
    {
        ciErrNum |= clEnqueueNDRangeKernel(cqCommandQueue, ckKernel, 2, NULL,
                               szGlobalWorkSize, szLocalWorkSize, 
                               0, NULL, NULL);
        clFinish(cqCommandQueue);
    }
    shrCheckError(ciErrNum, CL_SUCCESS);
    double optimizedTime = shrDeltaT(0)/(double)numIterations;

#ifdef GPU_PROFILING
    // log times
    shrLog(LOGBOTH | MASTER, 0, "oclTranspose-naive, Throughput = %.4f, Time = %.5f, Size = %u, NumDevsUsed = %u, Workgroup = %u\n", 
          (1.0e-9 * double(size_x * size_y)/naiveTime), naiveTime, (size_x * size_y), 1, (szLocalWorkSize[0] * szLocalWorkSize[1])); 
    shrLog(LOGBOTH | MASTER, 0, "oclTranspose-optimized, Throughput = %.4f, Time = %.5f, Size = %u, NumDevsUsed = %u, Workgroup = %u\n", 
          (1.0e-9 * double(size_x * size_y)/optimizedTime), optimizedTime, (size_x * size_y), 1, (szLocalWorkSize[0] * szLocalWorkSize[1])); 
#endif

    // copy result from device to host
    float* h_odata = (float*) malloc(mem_size);
    ciErrNum = clEnqueueReadBuffer(cqCommandQueue, d_odata, CL_TRUE, 0,
                                   mem_size, h_odata, 0, NULL, NULL);
    shrCheckError(ciErrNum, CL_SUCCESS);
   
    // compute reference solution and cross check results
    float* reference = (float*) malloc( mem_size);
    computeGold( reference, h_idata, size_x, size_y);
    shrLog(LOGBOTH, 0, "\nComparing results with CPU computation... \n\n");
    shrBOOL res = shrComparef( reference, h_odata, size_x * size_y);
    shrLog(LOGBOTH, 0,  "TEST %s\n\n", (1 == res) ? "PASSED" : "FAILED !!!");

    // cleanup memory
    free(h_idata);
    free(h_odata);
    free(reference);
    free(source);
    free(source_path);

    // cleanup OpenCL
    clReleaseMemObject(d_idata);
    clReleaseMemObject(d_odata);
    clReleaseKernel(ckKernel);
    clReleaseKernel(ckNaiveKernel);
    clReleaseProgram(cpProgram);
    clReleaseCommandQueue(cqCommandQueue);
    clReleaseContext(cxGPUContext);
 
    return 0;
}
