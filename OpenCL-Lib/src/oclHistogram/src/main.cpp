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
#include "oclHistogram_common.h"

////////////////////////////////////////////////////////////////////////////////
//Test driver
////////////////////////////////////////////////////////////////////////////////
int main(int argc, const char **argv){
    cl_context       cxGPUContext;      //OpenCL context
    cl_command_queue cqCommandQueue;    //OpenCL command que
    cl_mem    d_Data, d_Histogram;      //OpenCL memory buffer objects

    size_t dataBytes;
    cl_int ciErrNum;
    int PassFailFlag = 1;

    uchar *h_Data;
    uint *h_HistogramCPU, *h_HistogramGPU;
    const uint byteCount = 128 * 1048576;


#ifdef GPU_PROFILING
    const uint   numRuns = 10;
#else
    const uint   numRuns = 1;
#endif

    // set logfile name and start logs
    shrSetLogFileName ("oclHistogram.txt");
    shrLog(LOGBOTH, 0, "%s Starting...\n\n", argv[0]); 

    shrLog(LOGBOTH, 0, "Initializing data...\n");
        h_Data         = (uchar *)malloc(byteCount              * sizeof(uchar));
        h_HistogramCPU = (uint  *)malloc(HISTOGRAM256_BIN_COUNT * sizeof(uint));
        h_HistogramGPU = (uint  *)malloc(HISTOGRAM256_BIN_COUNT * sizeof(uint));
        srand(2009);
        for(uint i = 0; i < byteCount; i++)
            h_Data[i] = rand() & 0xFFU;

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

        //Discard temp storage
        free(cdDevices);

    shrLog(LOGBOTH, 0, "Allocating OpenCL memory...\n\n\n");
        d_Data = clCreateBuffer(cxGPUContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, byteCount * sizeof(cl_char), h_Data, &ciErrNum);
        shrCheckError(ciErrNum, CL_SUCCESS);
        d_Histogram = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, HISTOGRAM256_BIN_COUNT * sizeof(uint), NULL, &ciErrNum);
        shrCheckError(ciErrNum, CL_SUCCESS);

#ifdef GPU_PROFILING
    cl_event startTime, endTime;
    ciErrNum = clSetCommandQueueProperty(cqCommandQueue, CL_QUEUE_PROFILING_ENABLE, CL_TRUE, NULL);
    shrCheckError(ciErrNum, CL_SUCCESS);
#endif

    {
        shrLog(LOGBOTH, 0, "Initializing 64-bin OpenCL histogram...\n");
            size_t szWorkgroup;
            initHistogram64(cxGPUContext, cqCommandQueue, argv);

        shrLog(LOGBOTH, 0, "Running 64-bin OpenCL histogram for %u bytes (%u runs)...\n\n", byteCount, numRuns);

#ifdef GPU_PROFILING
            clFinish(cqCommandQueue);
            ciErrNum = clEnqueueMarker(cqCommandQueue, &startTime);
            shrCheckError(ciErrNum, CL_SUCCESS);
            shrDeltaT(0);
#endif
            // Run the histogram iter numRuns times
            for(uint iter = 0; iter < numRuns; iter++)
                szWorkgroup = histogram64(NULL, d_Histogram, d_Data, byteCount);

#ifdef GPU_PROFILING
            ciErrNum = clEnqueueMarker(cqCommandQueue, &endTime);
            shrCheckError(ciErrNum, CL_SUCCESS);
            clFinish(cqCommandQueue);
            double dAvgTime = shrDeltaT(0) / (double)numRuns;
            shrLog(LOGBOTH | MASTER, 0, "oclHistogram64, Throughput = %.4f, Time = %.5f, Size = %u, NumDevsUsed = %u, Workgroup = %u\n", 
                (1.0e-6 * (double)byteCount / dAvgTime), dAvgTime, byteCount, 1, szWorkgroup); 

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

            shrLog(LOGBOTH, 0, "\nOpenCL time: %.5f s\n\n", 1.0e-9 * (double)(endTimeVal - startTimeVal)/(double)numRuns);
#endif

        shrLog(LOGBOTH, 0, "Validating OpenCL results...\n");
            shrLog(LOGBOTH, 0, " ...reading back OpenCL results\n");
                ciErrNum = clEnqueueReadBuffer(cqCommandQueue, d_Histogram, CL_TRUE, 0, HISTOGRAM64_BIN_COUNT * sizeof(uint), h_HistogramGPU, 0, NULL, NULL);
                shrCheckError(ciErrNum, CL_SUCCESS);

            shrLog(LOGBOTH, 0, " ...histogram64CPU()\n");
                histogram64CPU(h_HistogramCPU, h_Data, byteCount);

            shrLog(LOGBOTH, 0, " ...comparing the results\n");
                for(uint i = 0; i < HISTOGRAM64_BIN_COUNT; i++)
                {
                    if(h_HistogramGPU[i] != h_HistogramCPU[i])
                    {
                        PassFailFlag = 0;
                    }
                }
            shrLog(LOGBOTH, 0, PassFailFlag ? " ...64-bin histograms match\n\n" : " ***64-bin histograms do not match!!!***\n\n" );

        shrLog(LOGBOTH, 0, "Shutting down 64-bin OpenCL histogram\n\n\n"); 
            //Release kernels and program
            closeHistogram64();
    }

    {
        shrLog(LOGBOTH, 0, "Initializing 256-bin OpenCL histogram...\n");
            size_t szWorkgroup;
            initHistogram256(cxGPUContext, cqCommandQueue, argv);

        shrLog(LOGBOTH, 0, "Running 256-bin OpenCL histogram for %u bytes (%u runs)...\n\n", byteCount, numRuns);

#ifdef GPU_PROFILING
            clFinish(cqCommandQueue);
            ciErrNum = clEnqueueMarker(cqCommandQueue, &startTime);
            shrCheckError(ciErrNum, CL_SUCCESS);
            shrDeltaT(0);
#endif

            // Run the histogram iter numRuns times
            for(uint iter = 0; iter < numRuns; iter++)
                szWorkgroup = histogram256(NULL, d_Histogram, d_Data, byteCount);

#ifdef GPU_PROFILING
            ciErrNum = clEnqueueMarker(cqCommandQueue, &endTime);
            shrCheckError(ciErrNum, CL_SUCCESS);
            clFinish(cqCommandQueue);
            double dAvgTime = shrDeltaT(0) / (double)numRuns;
            shrLog(LOGBOTH | MASTER, 0, "oclHistogram256, Throughput = %.4f, Time = %.5f, Size = %u, NumDevsUsed = %u, Workgroup = %u\n",  
                (1.0e-6 * (double)byteCount / dAvgTime), dAvgTime, byteCount, 1, szWorkgroup); 

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

            shrLog(LOGBOTH, 0, "\nOpenCL time: %.5f s\n\n", 1.0e-9 * (double)(endTimeVal - startTimeVal)/(double)numRuns);
#endif

        shrLog(LOGBOTH, 0, "Validating OpenCL results...\n");
            shrLog(LOGBOTH, 0, " ...reading back OpenCL results\n");
                ciErrNum = clEnqueueReadBuffer(cqCommandQueue, d_Histogram, CL_TRUE, 0, HISTOGRAM256_BIN_COUNT * sizeof(uint), h_HistogramGPU, 0, NULL, NULL);
                shrCheckError(ciErrNum, CL_SUCCESS);

            shrLog(LOGBOTH, 0, " ...histogram256CPU()\n");
                histogram256CPU(h_HistogramCPU, h_Data, byteCount);

            shrLog(LOGBOTH, 0, " ...comparing the results\n");
                for(uint i = 0; i < HISTOGRAM256_BIN_COUNT; i++)
                {
                    if(h_HistogramGPU[i] != h_HistogramCPU[i])
                    {
                        PassFailFlag = 0;
                    }
                }
            shrLog(LOGBOTH, 0, PassFailFlag ? " ...256-bin histograms match\n\n" : " ***256-bin histograms do not match!!!***\n\n" );

        shrLog(LOGBOTH, 0, "Shutting down 256-bin OpenCL histogram...\n\n\n"); 
            //Release kernels and program
            closeHistogram256();
    }

    // pass or fail (for both 64 bit and 256 bit histograms)
    shrLog(LOGBOTH, 0, "TEST %s\n\n", PassFailFlag ? "PASSED" : "FAILED !!!");

    shrLog(LOGBOTH, 0, "Shutting down...\n");
        //Release other OpenCL Objects
        ciErrNum  = clReleaseMemObject(d_Histogram);
        ciErrNum |= clReleaseMemObject(d_Data);
        ciErrNum |= clReleaseCommandQueue(cqCommandQueue);
        ciErrNum |= clReleaseContext(cxGPUContext);
        shrCheckError(ciErrNum, CL_SUCCESS);

        //Release host buffers
        free(h_HistogramGPU);
        free(h_HistogramCPU);
        free(h_Data);

        //Finish
        shrEXIT(argc, argv);
}
