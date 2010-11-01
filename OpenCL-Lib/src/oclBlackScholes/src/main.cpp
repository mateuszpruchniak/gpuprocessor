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



// standard utilities and systems includes
#include <oclUtils.h>
#include "oclBlackScholes_common.h"



////////////////////////////////////////////////////////////////////////////////
// Helper functions
////////////////////////////////////////////////////////////////////////////////
double executionTime(cl_event &event){
    cl_ulong start, end;

    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);

    return (double)1.0e-9 * (end - start); // convert nanoseconds to seconds on return
}

////////////////////////////////////////////////////////////////////////////////
// Random float helper
////////////////////////////////////////////////////////////////////////////////
float randFloat(float low, float high){
    float t = (float)rand() / (float)RAND_MAX;
    return (1.0f - t) * low + t * high;
}

////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, const char **argv){
    cl_context      cxGPUContext;    //OpenCL context
    cl_command_queue cqCommandQueue; //OpenCL command que
    cl_mem                           //OpenCL memory buffer objects
        d_Call,
        d_Put,
        d_S,
        d_X,
        d_T;

    size_t dataBytes;
    cl_int ciErrNum;

    float
        *h_CallCPU,
        *h_PutCPU,
        *h_CallGPU,
        *h_PutGPU,
        *h_S,
        *h_X,
        *h_T;

    const unsigned int   optionCount = 4000000;
    const float                    R = 0.02f;
    const float                    V = 0.30f;

    // set logfile name and start logs
    shrSetLogFileName ("oclBlackScholes.txt");
    shrLog(LOGBOTH, 0, "%s Starting...\n\n", argv[0]); 

    shrLog(LOGBOTH, 0, "Allocating and initializing host memory...\n");
        h_CallCPU = (float *)malloc(optionCount * sizeof(float));
        h_PutCPU  = (float *)malloc(optionCount * sizeof(float));
        h_CallGPU = (float *)malloc(optionCount * sizeof(float));
        h_PutGPU  = (float *)malloc(optionCount * sizeof(float));
        h_S       = (float *)malloc(optionCount * sizeof(float));
        h_X       = (float *)malloc(optionCount * sizeof(float));
        h_T       = (float *)malloc(optionCount * sizeof(float));

        srand(2009);
        for(unsigned int i = 0; i < optionCount; i++){
            h_CallCPU[i] = -1.0f;
            h_PutCPU[i]  = -1.0f;
            h_S[i]       = randFloat(5.0f, 30.0f);
            h_X[i]       = randFloat(1.0f, 100.0f);
            h_T[i]       = randFloat(0.25f, 10.0f);
        }

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

    shrLog(LOGBOTH, 0, "Creating OpenCL memory objects...\n");
        d_Call = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, optionCount * sizeof(float), NULL, &ciErrNum);
        shrCheckError(ciErrNum, CL_SUCCESS);
        d_Put  = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, optionCount * sizeof(float), NULL, &ciErrNum);
        shrCheckError(ciErrNum, CL_SUCCESS);
        d_S    = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, optionCount * sizeof(float), h_S, &ciErrNum);
        shrCheckError(ciErrNum, CL_SUCCESS);
        d_X    = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, optionCount * sizeof(float), h_X, &ciErrNum);
        shrCheckError(ciErrNum, CL_SUCCESS);
        d_T    = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, optionCount * sizeof(float), h_T, &ciErrNum);
        shrCheckError(ciErrNum, CL_SUCCESS);

    shrLog(LOGBOTH, 0, "Starting up BlackScholes...\n");
        initBlackScholes(cxGPUContext, cqCommandQueue, argv);

    shrLog(LOGBOTH, 0, "Running OpenCL BlackScholes...\n");
#ifdef GPU_PROFILING
    int numIterations = 15;
    for(int i = -1; i < numIterations; i++){
        //Start timing only after the first warmup iteration
        if(i == 0){
            clFinish(cqCommandQueue);
            shrDeltaT(0);
        }
#endif

        BlackScholes(
            cqCommandQueue,
            d_Call,
            d_Put,
            d_S,
            d_X,
            d_T,
            R,
            V,
            optionCount
        );

#ifdef GPU_PROFILING
    }
    clFinish(cqCommandQueue);
    double dElapsedTime = shrDeltaT(0) / numIterations;
    shrLog(LOGBOTH | MASTER, 0, "oclBlackScholes, Throughput = %.4f, Time = %.5f, Size = %u, NumDevsUsed = %i, Workgroup = %u\n", 
        (double)optionCount * 1.0e-9/dElapsedTime, dElapsedTime, optionCount, 1, 0); 
#endif


    shrLog(LOGBOTH, 0, "Reading back OpenCL BlackScholes results...\n");
        ciErrNum = clEnqueueReadBuffer(cqCommandQueue, d_Call, CL_TRUE, 0, optionCount * sizeof(float), h_CallGPU, 0, NULL, NULL);
        shrCheckError(ciErrNum, CL_SUCCESS);
        ciErrNum = clEnqueueReadBuffer(cqCommandQueue, d_Put, CL_TRUE, 0, optionCount * sizeof(float), h_PutGPU, 0, NULL, NULL);
        shrCheckError(ciErrNum, CL_SUCCESS);

    shrLog(LOGBOTH, 0, "Comparing against Host/C++ computation...\n"); 
        BlackScholesCPU(h_CallCPU, h_PutCPU, h_S, h_X, h_T, R, V, optionCount);
        double deltaCall = 0, deltaPut = 0, sumCall = 0, sumPut = 0;
        double L1call, L1put;
        for(unsigned int i = 0; i < optionCount; i++){
            sumCall += fabs(h_CallCPU[i]);
            sumPut  += fabs(h_PutCPU[i]);
            deltaCall += fabs(h_CallCPU[i] - h_CallGPU[i]);
            deltaPut  += fabs(h_PutCPU[i] - h_PutGPU[i]);
        }
        L1call = deltaCall / sumCall; L1put = deltaPut / sumPut;
        shrLog(LOGBOTH, 0, "Relative L1 (call, put) = (%.3e, %.3e)\n\n", L1call, L1put);

    shrLog(LOGBOTH, 0, "TEST %s\n\n", ((L1call < 1E-6) && (L1put < 1E-6)) ? "PASSED" : "FAILED !!!");

    shrLog(LOGBOTH, 0, "Shutting down...\n");
        closeBlackScholes();
        ciErrNum  = clReleaseMemObject(d_T);
        ciErrNum |= clReleaseMemObject(d_X);
        ciErrNum |= clReleaseMemObject(d_S);
        ciErrNum |= clReleaseMemObject(d_Put);
        ciErrNum |= clReleaseMemObject(d_Call);
        ciErrNum |= clReleaseCommandQueue(cqCommandQueue);
        ciErrNum |= clReleaseContext(cxGPUContext);
        shrCheckError(ciErrNum, CL_SUCCESS);

        free(h_T);
        free(h_X);
        free(h_S);
        free(h_PutGPU);
        free(h_CallGPU);
        free(h_PutCPU);
        free(h_CallCPU);

        shrEXIT(argc, argv);
}
