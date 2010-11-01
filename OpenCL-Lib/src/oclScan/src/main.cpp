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
#include "oclScan_common.h"

////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, const char **argv){
    cl_context      cxGPUContext;    //OpenCL context
    cl_command_queue cqCommandQueue; //OpenCL command que
    cl_mem      d_Input, d_Output;   //OpenCL memory buffer objects

    size_t dataBytes;
    cl_int ciErrNum;

    uint *h_Input, *h_OutputCPU, *h_OutputGPU;
    const uint N = 13 * 1048576 / 2;

    shrSetLogFileName ("oclScan.txt");
    shrLog(LOGBOTH, 0, "%s Starting...\n\n", argv[0]); 

    shrLog(LOGBOTH, 0, "Initializing data...\n");
        h_Input     = (uint *)malloc(N * sizeof(uint));
        h_OutputCPU = (uint *)malloc(N * sizeof(uint));
        h_OutputGPU = (uint *)malloc(N * sizeof(uint));
        srand(2009);
        for(uint i = 0; i < N; i++)
            h_Input[i] = rand();

    shrLog(LOGBOTH, 0, "Initializing OpenCL...\n");
        cxGPUContext = clCreateContextFromType(0, CL_DEVICE_TYPE_GPU, NULL, NULL, &ciErrNum);
        shrCheckError(ciErrNum, CL_SUCCESS);

        //Get the list of GPU devices associated with context
        ciErrNum = clGetContextInfo(cxGPUContext, CL_CONTEXT_DEVICES, 0, NULL, &dataBytes);
        cl_device_id *cdDevices = (cl_device_id *)malloc(dataBytes);
        ciErrNum |= clGetContextInfo(cxGPUContext, CL_CONTEXT_DEVICES, dataBytes, cdDevices, NULL);
        shrCheckError(ciErrNum, CL_SUCCESS);

        //Create a command-queue
        cqCommandQueue = clCreateCommandQueue(cxGPUContext, cdDevices[0], 0, &ciErrNum);
        shrCheckError(ciErrNum, CL_SUCCESS);

        //Discard temp storage
        free(cdDevices);

    shrLog(LOGBOTH, 0, "Initializing OpenCL scan...\n");
        initScan(cxGPUContext, cqCommandQueue, argv);

    shrLog(LOGBOTH, 0, "Creating OpenCL memory objects...\n\n");
        d_Input = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, N * sizeof(uint), h_Input, &ciErrNum);
        shrCheckError(ciErrNum, CL_SUCCESS);
        d_Output = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, N * sizeof(uint), NULL, &ciErrNum);
        shrCheckError(ciErrNum, CL_SUCCESS);

    int globalFlag = 1; // init pass/fail flag to pass
    size_t szWorkgroup;
    int iCycles = 100;
    for(uint arrayLength = MIN_SHORT_ARRAY_SIZE; arrayLength <= MAX_SHORT_ARRAY_SIZE; arrayLength *= 2){
        shrLog(LOGBOTH, 0, "Testing array length %u (%u arrays in the batch)...\n", arrayLength, N / arrayLength);
            clFinish(cqCommandQueue);
            shrDeltaT(0);
            for (int i = 0; i<iCycles; i++)
            {
                szWorkgroup = scanExclusiveShort(
                    cqCommandQueue,
                    d_Output,
                    d_Input,
                    N / arrayLength,
                    arrayLength
                );
                clFinish(cqCommandQueue);
            }

#ifdef GPU_PROFILING
            if (arrayLength == MAX_SHORT_ARRAY_SIZE)
            {
                double timerValue = shrDeltaT(0)/(double)iCycles;
                shrLog(LOGBOTH | MASTER, 0, "oclScan-Short, Throughput = %.4f, Time = %.5f, Size = %u, NumDevsUsed = %u, Workgroup = %u\n", 
                       (1.0e-6 * (double)arrayLength/timerValue), timerValue, arrayLength, 1, szWorkgroup);
            }
#endif

        shrLog(LOGBOTH, 0, "Validating the results...\n"); 
            shrLog(LOGBOTH, 0, " ...reading back OpenCL memory\n");
                ciErrNum = clEnqueueReadBuffer(cqCommandQueue, d_Output, CL_TRUE, 0, N * sizeof(uint), h_OutputGPU, 0, NULL, NULL);
                shrCheckError(ciErrNum, CL_SUCCESS);

            shrLog(LOGBOTH, 0, " ...scanExclusiveHost()\n");
                scanExclusiveHost(
                    h_OutputCPU,
                    h_Input,
                    N / arrayLength,
                    arrayLength
                );

            // Compare GPU results with CPU results and accumulate error for this test
            shrLog(LOGBOTH, 0, " ...comparing the results:  ");
                int localFlag = 1;
                for(uint i = 0; i < N; i++)
                    if(h_OutputCPU[i] != h_OutputGPU[i]) localFlag = 0;

            // Log message on individual test result, then accumulate to global flag
            shrLog(LOGBOTH, 0, " Results %s\n\n\n", (localFlag == 1) ? "Match" : "Don't Match !!!");
            globalFlag = globalFlag && localFlag;
    }

    for(uint arrayLength = MIN_LARGE_ARRAY_SIZE; arrayLength <= MAX_LARGE_ARRAY_SIZE; arrayLength *= 2){
        shrLog(LOGBOTH, 0, "Testing array length %u (%u arrays in the batch)...\n", arrayLength, N / arrayLength);
            clFinish(cqCommandQueue);
            shrDeltaT(0);
            for (int i = 0; i<iCycles; i++)
            {
                szWorkgroup = scanExclusiveLarge(
                    cqCommandQueue,
                    d_Output,
                    d_Input,
                    N / arrayLength,
                    arrayLength
                );
                clFinish(cqCommandQueue);
            }

#ifdef GPU_PROFILING
            if (arrayLength == MAX_LARGE_ARRAY_SIZE)
            {
                double timerValue = shrDeltaT(0)/(double)iCycles;
                 shrLog(LOGBOTH | MASTER, 0, "oclScan-Large, Throughput = %.4f, Time = %.5f, Size = %u, NumDevsUsed = %u, Workgroup = %u\n", 
                       (1.0e-6 * (double)arrayLength/timerValue), timerValue, arrayLength, 1, szWorkgroup);
            }
#endif

        shrLog(LOGBOTH, 0, "Validating the results...\n"); 
            shrLog(LOGBOTH, 0, " ...reading back OpenCL memory\n");
                ciErrNum = clEnqueueReadBuffer(cqCommandQueue, d_Output, CL_TRUE, 0, N * sizeof(uint), h_OutputGPU, 0, NULL, NULL);
                shrCheckError(ciErrNum, CL_SUCCESS);

            shrLog(LOGBOTH, 0, " ...scanExclusiveHost()\n");
                scanExclusiveHost(
                    h_OutputCPU,
                    h_Input,
                    N / arrayLength,
                    arrayLength
                );

            // Compare GPU results with CPU results and accumulate error for this test
            shrLog(LOGBOTH, 0, " ...comparing the results:  ");
                int localFlag = 1;
                for(uint i = 0; i < N; i++)
                    if(h_OutputCPU[i] != h_OutputGPU[i]) localFlag = 0;

            // Log message on individual test result, then accumulate to global flag
            shrLog(LOGBOTH, 0, " Results %s\n\n", (localFlag == 1) ? "Match" : "Don't Match !!!");
            globalFlag = globalFlag && localFlag;
    }

    // pass or fail (cumulative... all tests in the loop)
    shrLog(LOGBOTH, 0, "TEST %s\n\n", globalFlag ? "PASSED" : "FAILED !!!");

    shrLog(LOGBOTH, 0, "Shutting down...\n");
        //Release kernels and program
        closeScan();

        //Release other OpenCL Objects
        ciErrNum  = clReleaseMemObject(d_Output);
        ciErrNum |= clReleaseMemObject(d_Input);
        ciErrNum |= clReleaseCommandQueue(cqCommandQueue);
        ciErrNum |= clReleaseContext(cxGPUContext);
        shrCheckError(ciErrNum, CL_SUCCESS);

        //Release host buffers
        free(h_OutputGPU);
        free(h_OutputCPU);
        free(h_Input);

        //Finish
        shrEXIT(argc, argv);
}
