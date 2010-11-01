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

#include "oclConvolutionSeparable_common.h"

////////////////////////////////////////////////////////////////////////////////
// Test driver
////////////////////////////////////////////////////////////////////////////////
int main(int argc, const char **argv){
    cl_context                             cxGPUContext; //OpenCL context
    cl_command_queue                        cqCommandQueue; //OpenCL command que
    cl_mem         c_Kernel, d_Input, d_Buffer, d_Output; //OpenCL memory buffer objects
    cl_float *h_Kernel, *h_Input, *h_Buffer, *h_OutputCPU, *h_OutputGPU;

    size_t dataBytes;
    cl_int ciErrNum;

    const unsigned int imageW = 2048;
    const unsigned int imageH = 1024;

#ifdef GPU_PROFILING
    const unsigned int numIterations = 100;
#else
    const unsigned int numIterations = 1;
#endif

    // set logfile name and start logs
    shrSetLogFileName ("oclConvolutionSeparable.txt");
    shrLog(LOGBOTH, 0, "%s Starting...\n\n", argv[0]); 

    shrLog(LOGBOTH, 0, "Allocating and initializing host memory...\n");
        h_Kernel    = (cl_float *)malloc(KERNEL_LENGTH * sizeof(cl_float));
        h_Input     = (cl_float *)malloc(imageW * imageH * sizeof(cl_float));
        h_Buffer    = (cl_float *)malloc(imageW * imageH * sizeof(cl_float));
        h_OutputCPU = (cl_float *)malloc(imageW * imageH * sizeof(cl_float));
        h_OutputGPU = (cl_float *)malloc(imageW * imageH * sizeof(cl_float));

        srand(2009);
        for(unsigned int i = 0; i < KERNEL_LENGTH; i++)
            h_Kernel[i] = (cl_float)(rand() % 16);

        for(unsigned int i = 0; i < imageW * imageH; i++)
            h_Input[i] = (cl_float)(rand() % 16);

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

    shrLog(LOGBOTH, 0, "Initializing OpenCL separable convolution...\n");
        initConvolutionSeparable(cxGPUContext, cqCommandQueue, argv);

    shrLog(LOGBOTH, 0, "Creating OpenCL memory objects...\n");
        c_Kernel = clCreateBuffer(cxGPUContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, KERNEL_LENGTH * sizeof(cl_float), h_Kernel, &ciErrNum);
        shrCheckError(ciErrNum, CL_SUCCESS);
        d_Input = clCreateBuffer(cxGPUContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, imageW * imageH * sizeof(cl_float), h_Input, &ciErrNum);
        shrCheckError(ciErrNum, CL_SUCCESS);
        d_Buffer = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, imageW * imageH * sizeof(cl_float), NULL, &ciErrNum);
        shrCheckError(ciErrNum, CL_SUCCESS);
        d_Output = clCreateBuffer(cxGPUContext, CL_MEM_WRITE_ONLY, imageW * imageH * sizeof(cl_float), NULL, &ciErrNum);
        shrCheckError(ciErrNum, CL_SUCCESS);

    shrLog(LOGBOTH, 0, "Applying separable convolution to %u x %u image...\n\n", imageW, imageH);
    
#ifdef GPU_PROFILING
    cl_event startTime, endTime;
    ciErrNum = clSetCommandQueueProperty(cqCommandQueue, CL_QUEUE_PROFILING_ENABLE, CL_TRUE, NULL);
    shrCheckError(ciErrNum, CL_SUCCESS);

    // Warmup, make sure queue is finished, start GPU and host timers
    convolutionRows(cqCommandQueue, d_Buffer, d_Input, c_Kernel,imageW,imageH );
    clFinish(cqCommandQueue);
    ciErrNum = clEnqueueMarker(cqCommandQueue, &startTime);
    shrCheckError(ciErrNum, CL_SUCCESS);
    shrDeltaT(0);
#endif

        for(unsigned int iter = 0; iter < numIterations; iter++){
            convolutionRows(
                cqCommandQueue,
                d_Buffer,
                d_Input,
                c_Kernel,
                imageW,
                imageH
            );

            convolutionColumns(
                cqCommandQueue,
                d_Output,
                d_Buffer,
                c_Kernel,
                imageW,
                imageH
            );

        }
        
#ifdef GPU_PROFILING
        // stop the timers on GPU and host
        ciErrNum = clEnqueueMarker(cqCommandQueue, &endTime);
        shrCheckError(ciErrNum, CL_SUCCESS);
        clFinish(cqCommandQueue);
        double dAvgTime = shrDeltaT(0)/(double)numIterations;
        shrLog(LOGBOTH | MASTER, 0, "oclConvolutionSeparable, Throughput = %.4f, Time = %.5f, Size = %u, NumDevsUsed = %i\n", 
            (1.0e-6 * (double)(imageW * imageH)/ dAvgTime), dAvgTime, (imageW * imageH), 1); 

        // get profiling info 
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
        shrLog(LOGBOTH, 0, "\nOpenCL time: %.5f s\n\n", 1.0e-9 * ((double)endTimeVal - (double)startTimeVal)/ (double)numIterations);

#endif

    shrLog(LOGBOTH, 0, "Reading back OpenCL results...\n\n");
        ciErrNum = clEnqueueReadBuffer(cqCommandQueue, d_Output, CL_TRUE, 0, imageW * imageH * sizeof(cl_float), h_OutputGPU, 0, NULL, NULL);
        shrCheckError(ciErrNum, CL_SUCCESS);

    shrLog(LOGBOTH, 0, "Comparing against Host/C++ computation...\n"); 
        convolutionRowHost(h_Buffer, h_Input, h_Kernel, imageW, imageH, KERNEL_RADIUS);
        convolutionColumnHost(h_OutputCPU, h_Buffer, h_Kernel, imageW, imageH, KERNEL_RADIUS);
        double sum = 0, delta = 0;
        double L2norm;
        for(unsigned int i = 0; i < imageW * imageH; i++){
            delta += (h_OutputCPU[i] - h_OutputGPU[i]) * (h_OutputCPU[i] - h_OutputGPU[i]);
            sum += h_OutputCPU[i] * h_OutputCPU[i];
        }
        L2norm = sqrt(delta / sum);
        shrLog(LOGBOTH, 0, "Relative L2 norm: %.3e\n\n", L2norm);

    shrLog(LOGBOTH, 0, "TEST %s\n\n", (L2norm < 1e-6)  ? "PASSED" : "FAILED !!!");

    // cleanup
    closeConvolutionSeparable();
    ciErrNum  = clReleaseMemObject(d_Output);
    ciErrNum |= clReleaseMemObject(d_Buffer);
    ciErrNum |= clReleaseMemObject(d_Input);
    ciErrNum |= clReleaseMemObject(c_Kernel);
    ciErrNum |= clReleaseCommandQueue(cqCommandQueue);
    ciErrNum |= clReleaseContext(cxGPUContext);
    shrCheckError(ciErrNum, CL_SUCCESS);

    free(cdDevices);
    free(h_OutputGPU);
    free(h_OutputCPU);
    free(h_Buffer);
    free(h_Input);
    free(h_Kernel);

    // finish
    shrEXIT(argc, argv);
}
