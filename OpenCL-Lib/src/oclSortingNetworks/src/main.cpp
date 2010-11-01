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

//Standard utilities and systems includes
#include <oclUtils.h>
#include "oclSortingNetworks_common.h"

////////////////////////////////////////////////////////////////////////////////
//Test driver
////////////////////////////////////////////////////////////////////////////////
int main(int argc, const char **argv){
    cl_context cxGPUContext;
    cl_command_queue cqCommandQueue;
    cl_mem d_InputKey, d_InputVal, d_OutputKey, d_OutputVal;

    size_t dataBytes;
    cl_int ciErrNum;
    uint *h_InputKey, *h_InputVal, *h_OutputKeyGPU, *h_OutputValGPU;

    const uint dir = 1;
    const uint N = 1048576;
    const uint numValues = 65536;

    // set logfile name and start logs
    shrSetLogFileName ("oclSortingNetworks.txt");
    shrLog(LOGBOTH, 0, "%s Starting...\n\n", argv[0]); 

    shrLog(LOGBOTH, 0, "Initializing data...\n");
        h_InputKey      = (uint *)malloc(N * sizeof(uint));
        h_InputVal      = (uint *)malloc(N * sizeof(uint));
        h_OutputKeyGPU  = (uint *)malloc(N * sizeof(uint));
        h_OutputValGPU  = (uint *)malloc(N * sizeof(uint));
        srand(2009);
        for(uint i = 0; i < N; i++)
            h_InputKey[i] = rand() % numValues;
        fillValues(h_InputVal, N);

    shrLog(LOGBOTH, 0, "Initializing OpenCL...\n");
        cxGPUContext = clCreateContextFromType(0, CL_DEVICE_TYPE_GPU, NULL, NULL, &ciErrNum);
        shrCheckError(ciErrNum, CL_SUCCESS);

        // get the list of GPU devices associated with context
        ciErrNum = clGetContextInfo(cxGPUContext, CL_CONTEXT_DEVICES, 0, NULL, &dataBytes);
        cl_device_id *cdDevices = (cl_device_id *)malloc(dataBytes);
        ciErrNum |= clGetContextInfo(cxGPUContext, CL_CONTEXT_DEVICES, dataBytes, cdDevices, NULL);
        shrCheckError(ciErrNum, CL_SUCCESS);

        //Create a command-queue
        cqCommandQueue = clCreateCommandQueue(cxGPUContext, cdDevices[0], 0, &ciErrNum);
        shrCheckError(ciErrNum, CL_SUCCESS);

    shrLog(LOGBOTH, 0, "Initializing OpenCL bitonic sorter...\n");
        initBitonicSort(cxGPUContext, cqCommandQueue, argv);

    shrLog(LOGBOTH, 0, "Creating OpenCL memory objects...\n\n");
        d_InputKey = clCreateBuffer(cxGPUContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, N * sizeof(cl_uint), h_InputKey, &ciErrNum);
        shrCheckError(ciErrNum, CL_SUCCESS);
        d_InputVal = clCreateBuffer(cxGPUContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, N * sizeof(cl_uint), h_InputVal, &ciErrNum);
        shrCheckError(ciErrNum, CL_SUCCESS);
        d_OutputKey = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, N * sizeof(cl_uint), NULL, &ciErrNum);
        shrCheckError(ciErrNum, CL_SUCCESS);
        d_OutputVal = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, N * sizeof(cl_uint), NULL, &ciErrNum);
        shrCheckError(ciErrNum, CL_SUCCESS);

    //Temp storage for key array validation routine
    uint *srcHist = (uint *)malloc(numValues * sizeof(uint));
    uint *resHist = (uint *)malloc(numValues * sizeof(uint));

#ifdef GPU_PROFILING
    cl_event startTime, endTime;
    ciErrNum = clSetCommandQueueProperty(cqCommandQueue, CL_QUEUE_PROFILING_ENABLE, CL_TRUE, NULL);
    shrCheckError(ciErrNum, CL_SUCCESS);
#endif

    int globalFlag = 1;// init pass/fail flag to pass
    for(uint arrayLength = 64; arrayLength <= N; arrayLength *= 2){
        shrLog(LOGBOTH, 0, "Test array length %u (%u arrays in the batch)...\n", arrayLength, N / arrayLength);

#ifdef GPU_PROFILING
            clFinish(cqCommandQueue);
            ciErrNum = clEnqueueMarker(cqCommandQueue, &startTime);
            shrCheckError(ciErrNum, CL_SUCCESS);
            shrDeltaT(0);
#endif

            size_t szWorkgroup = bitonicSort(
                NULL,
                d_OutputKey,
                d_OutputVal,
                d_InputKey,
                d_InputVal,
                N / arrayLength,
                arrayLength,
                dir
            );
            shrCheckError(szWorkgroup > 0, true); 

#ifdef GPU_PROFILING
            if (arrayLength == N)
            {
                ciErrNum = clEnqueueMarker(cqCommandQueue, &endTime);
                shrCheckError(ciErrNum, CL_SUCCESS);
                clFinish(cqCommandQueue);
                double timerValue = shrDeltaT(0);
                shrLog(LOGBOTH | MASTER, 0, "oclSortingNetworks, Throughput = %.4f, Time = %.5f, Size = %u, NumDevsUsed = %u, Workgroup = %u\n", 
                       (1.0e-6 * (double)arrayLength/timerValue), timerValue, arrayLength, 1, szWorkgroup);

                cl_ulong startTimeVal = 0, endTimeVal = 0;
                ciErrNum = clGetEventProfilingInfo(
                    startTime, 
                    CL_PROFILING_COMMAND_END, 
                    sizeof(cl_ulong),
                    &startTimeVal,
                    NULL
                );

                ciErrNum = clGetEventProfilingInfo(
                    endTime, 
                    CL_PROFILING_COMMAND_END, 
                    sizeof(cl_ulong),
                    &endTimeVal,
                    NULL
                );

                shrLog(LOGBOTH, 0, "OpenCL time: %.5f s\n", 1.0e-9 * (double)(endTimeVal - startTimeVal));
            }
#endif

        //Reading back results from device to host
        ciErrNum = clEnqueueReadBuffer(cqCommandQueue, d_OutputKey, CL_TRUE, 0, N * sizeof(cl_uint), h_OutputKeyGPU, 0, NULL, NULL);
        shrCheckError(ciErrNum, CL_SUCCESS);
        ciErrNum = clEnqueueReadBuffer(cqCommandQueue, d_OutputVal, CL_TRUE, 0, N * sizeof(cl_uint), h_OutputValGPU, 0, NULL, NULL);
        shrCheckError(ciErrNum, CL_SUCCESS);

        //Check if keys array is not corrupted and properly ordered
        int keysFlag = validateSortedKeys(h_OutputKeyGPU, h_InputKey, N / arrayLength, arrayLength, numValues, dir, srcHist, resHist);

        //Check if values array is not corrupted
        int valuesFlag = validateSortedValues(h_OutputKeyGPU, h_OutputValGPU, h_InputKey, N / arrayLength, arrayLength);

        // accumulate any error or failure
        globalFlag = globalFlag && keysFlag && valuesFlag;
    }

    // pass or fail (cumulative... all tests in the loop)
    shrLog(LOGBOTH, 0, "TEST %s\n\n", globalFlag ? "PASSED" : "FAILED !!!");

    // Start Cleanup
    shrLog(LOGBOTH, 0, "Shutting down...\n");
        //Discard temp storage for key validation routine
        free(srcHist);
        free(resHist);

        //Release kernels and program
        closeBitonicSort();

        //Release other OpenCL Objects
        ciErrNum  = clReleaseMemObject(d_OutputVal);
        ciErrNum |= clReleaseMemObject(d_OutputKey);
        ciErrNum |= clReleaseMemObject(d_InputVal);
        ciErrNum |= clReleaseMemObject(d_InputKey);
        ciErrNum |= clReleaseCommandQueue(cqCommandQueue);
        ciErrNum |= clReleaseContext(cxGPUContext);
        shrCheckError(ciErrNum, CL_SUCCESS);

        //Release host buffers
        free(cdDevices);
        free(h_OutputValGPU);
        free(h_OutputKeyGPU);
        free(h_InputVal);
        free(h_InputKey);

        //Finish
        shrEXIT(argc, argv);
}
