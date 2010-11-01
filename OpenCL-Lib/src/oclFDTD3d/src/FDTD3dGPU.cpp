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

#include "FDTD3dGPU.h"

#include <oclUtils.h>
#include <iostream>
#include <algorithm>

bool getTargetDeviceGlobalMemSize(memsize_t *result, const int argc, const char **argv)
{
    bool ok = true;
    cl_context        context      = 0;
    cl_device_id     *devices      = 0;
    cl_uint           deviceCount  = 0;
    cl_uint           targetDevice = 0;
    cl_ulong          memsize      = 0;
    cl_int            errnum       = 0;

    // Create the OpenCL context
    if (ok)
    {
        shrLog(LOGBOTH, 0, " clCreateContextFromType\n");
        context = clCreateContextFromType(0, CL_DEVICE_TYPE_GPU, NULL, NULL, &errnum);
        if (context == (cl_context)0) 
        {
            shrLog(LOGBOTH | ERRORMSG, 0.0, "clCreateContextFromType (returned %d).\n", errnum);
            ok = false;
        }
    }

    // Get the list of GPU devices associated with the context
    if (ok)
    {
        shrLog(LOGBOTH, 0, " clGetContextInfo\n"); 
        size_t szParmDataBytes;
        clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &szParmDataBytes);
        devices = (cl_device_id *)malloc(szParmDataBytes);
        clGetContextInfo(context, CL_CONTEXT_DEVICES, szParmDataBytes, devices, NULL);
        deviceCount = (cl_uint)(szParmDataBytes / sizeof(cl_device_id));
        if (deviceCount == 0)
        {
            shrLog(LOGBOTH | ERRORMSG, 0.0, "clGetContextInfo (no devices found).\n");
            ok = false;
        }
    }

    // Select target device (device 0 by default)
    if (ok)
    {
        char *device = 0;
        if (shrGetCmdLineArgumentstr(argc, argv, "device", &device))
        {
            targetDevice = (cl_uint)atoi(device);
            if (targetDevice >= deviceCount)
            {
                shrLog(LOGBOTH | ERRORMSG, 0.0, "invalid target device specified on command line (device %d does not exist).\n", targetDevice);
                ok = false;
            }
        }
        else
        {
            targetDevice = 0;
        }
        if (device)
            free(device);
    }

    // Query target device for maximum memory allocation
    if (ok)
    {
        shrLog(LOGBOTH, 0, " clGetDeviceInfo\n"); 
        errnum = clGetDeviceInfo(devices[targetDevice], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &memsize, NULL);
        if (errnum != CL_SUCCESS)
        {
            shrLog(LOGBOTH | ERRORMSG, 0.0, "clGetDeviceInfo (returned %d).\n", errnum);
            ok = false;
        }
    }

    // Save the result
    if (ok)
    {
        *result = (memsize_t)memsize;
    }

    // Cleanup
    if (devices)
        free(devices);
    if (context)
        clReleaseContext(context);
    return ok;
}

bool fdtdGPU(float *output, const float *input, const float *coeff, const int dimx, const int dimy, const int dimz, const int radius, const int timesteps, const int argc, const char **argv)
{
    bool ok = true;
    const size_t      volumeSize = (dimx + 2 * radius) * (dimy + 2 * radius) * (dimz + 2 * radius);
    cl_context        context      = 0;
    cl_device_id     *devices      = 0;
    cl_command_queue  commandQueue = 0;
    cl_mem            bufferOut    = 0;
    cl_mem            bufferIn     = 0;
    cl_mem            bufferCoeff  = 0;
    cl_program        program      = 0;
    cl_kernel         kernel       = 0;
    cl_event         *kernelEvents = 0;
#ifdef GPU_PROFILING
    cl_ulong          kernelEventStart;
    cl_ulong          kernelEventEnd;
#endif
    double            hostElapsedTimeS;
    char             *cPathAndName = 0;
    char             *cSourceCL = 0;
    size_t            szKernelLength;
    size_t            globalWorkSize[2];
    size_t            localWorkSize[2];
    cl_uint           deviceCount  = 0;
    cl_uint           targetDevice = 0;
    cl_int            errnum       = 0;
    char              buildOptions[128];

    // Create the OpenCL context
    if (ok)
    {
        shrLog(LOGBOTH, 0, " clCreateContextFromType\n");
        context = clCreateContextFromType(0, CL_DEVICE_TYPE_GPU, NULL, NULL, &errnum);
        if (context == (cl_context)0) 
        {
            shrLog(LOGBOTH | ERRORMSG, 0.0, "clCreateContextFromType (returned %d).\n", errnum);
            ok = false;
        }
    }

    // Get the list of GPU devices associated with the context
    if (ok)
    {
        shrLog(LOGBOTH, 0, " clGetContextInfo\n"); 
        size_t szParmDataBytes;
        clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &szParmDataBytes);
        devices = (cl_device_id *)malloc(szParmDataBytes);
        clGetContextInfo(context, CL_CONTEXT_DEVICES, szParmDataBytes, devices, NULL);
        deviceCount = (cl_uint)(szParmDataBytes / sizeof(cl_device_id));
        if (deviceCount == 0)
        {
            shrLog(LOGBOTH | ERRORMSG, 0.0, "clGetContextInfo (no devices found).\n");
            ok = false;
        }
    }

    // Select target device (device 0 by default)
    if (ok)
    {
        char *device = 0;
        if (shrGetCmdLineArgumentstr(argc, argv, "device", &device))
        {
            targetDevice = (cl_uint)atoi(device);
            if (targetDevice >= deviceCount)
            {
                shrLog(LOGBOTH | ERRORMSG, 0.0, "invalid target device specified on command line (device %d does not exist).\n", targetDevice);
                ok = false;
            }
        }
        else
        {
            targetDevice = 0;
        }
        if (device)
            free(device);
    }

    // Create a command-queue
    if (ok)
    {
        shrLog(LOGBOTH, 0, " clCreateCommandQueue\n"); 
        commandQueue = clCreateCommandQueue(context, devices[targetDevice], 0, &errnum);
        if (commandQueue == (cl_command_queue)0)
        {
            shrLog(LOGBOTH | ERRORMSG, 0.0, "clCreateCommandQueue (returned %d).\n", errnum);
            ok = false;
        }
    }
    
#ifdef GPU_PROFILING
    if (ok)
    {
        shrLog(LOGBOTH, 0, " clSetCommandQueueProperty\n"); 
        errnum = clSetCommandQueueProperty(commandQueue, CL_QUEUE_PROFILING_ENABLE, CL_TRUE, NULL);
        if (errnum != CL_SUCCESS)
        {
            shrLog(LOGBOTH | ERRORMSG, 0.0, "clSetCommandQueueProperty (returned %d).\n", errnum);
            ok = false;
        }
    }
#endif

    // Create memory buffer objects
    if (ok)
    {
        shrLog(LOGBOTH, 0, " clCreateBuffer bufferOut\n"); 
        bufferOut = clCreateBuffer(context, CL_MEM_READ_WRITE, volumeSize * sizeof(float), NULL, &errnum);
        if (errnum != CL_SUCCESS)
        {
            shrLog(LOGBOTH | ERRORMSG, 0.0, "clCreateBuffer (returned %d).\n", errnum);
            ok = false;
        }
    }
    if (ok)
    {
        shrLog(LOGBOTH, 0, " clCreateBuffer bufferIn\n"); 
        bufferIn = clCreateBuffer(context, CL_MEM_READ_WRITE, volumeSize * sizeof(float), NULL, &errnum);
        if (errnum != CL_SUCCESS)
        {
            shrLog(LOGBOTH | ERRORMSG, 0.0, "clCreateBuffer (returned %d).\n", errnum);
            ok = false;
        }
    }
    if (ok)
    {
        shrLog(LOGBOTH, 0, " clCreateBuffer bufferCoeff\n"); 
        bufferCoeff = clCreateBuffer(context, CL_MEM_READ_ONLY, (radius + 1) * sizeof(float), NULL, &errnum);
        if (errnum != CL_SUCCESS)
        {
            shrLog(LOGBOTH | ERRORMSG, 0.0, "clCreateBuffer (returned %d).\n", errnum);
            ok = false;
        }
    }

    // Load the kernel from file
    if (ok)
    {
        shrLog(LOGBOTH, 0, " shrFindFilePath\n"); 
        cPathAndName = shrFindFilePath(clSourceFile, argv[0]);
        if (cPathAndName == NULL)
        {
            shrLog(LOGBOTH | ERRORMSG, 0.0, "shrFindFilePath.\n");
            ok = false;
        }
    }
    if (ok)
    {
        shrLog(LOGBOTH, 0, " oclLoadProgSource\n"); 
        cSourceCL = oclLoadProgSource(cPathAndName, "// Preamble\n", &szKernelLength);
        if (cSourceCL == NULL)
        {
            shrLog(LOGBOTH | ERRORMSG, 0.0, "oclLoadProgSource.\n");
            ok = false;
        }
    }

    // Create the program
    if (ok)
    {
        shrLog(LOGBOTH, 0, " clCreateProgramWithSource\n");
        program = clCreateProgramWithSource(context, 1, (const char **)&cSourceCL, &szKernelLength, &errnum);
        if (program == (cl_program)NULL)
        {
            shrLog(LOGBOTH | ERRORMSG, 0.0, "clCreateProgramWithSource (returned %d).\n", errnum);
            ok = false;
        }
    }

    // Build the program
    if (ok)
    {
#ifdef WIN32
        if (sprintf_s(buildOptions, sizeof(buildOptions), "-DRADIUS=%d -cl-mad-enable", radius) < 0)
        {
            shrLog(LOGBOTH | ERRORMSG, 0.0, "sprintf_s.\n");
            ok = false;
        }
#else
        if (snprintf(buildOptions, sizeof(buildOptions), "-DRADIUS=%d -cl-mad-enable", radius) < 0)
        {
            shrLog(LOGBOTH | ERRORMSG, 0.0, "snprintf.\n");
            ok = false;
        }
#endif
    }
    if (ok)
    {
        shrLog(LOGBOTH, 0, " clBuildProgram (%s)\n", buildOptions);
        errnum = clBuildProgram(program, 0, NULL, buildOptions, NULL, NULL);
        if (errnum != CL_SUCCESS)
        {
            char buildLog[10240];
            clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, sizeof(buildLog), buildLog, NULL);
            shrLog(LOGBOTH | ERRORMSG, 0.0, "clBuildProgram (returned %d).\n", errnum);
            shrLog(LOGBOTH, 0, "Log:\n%s\n", buildLog);
            ok = false;
        }
    }

    // Create the kernel
    if (ok)
    {
        shrLog(LOGBOTH, 0, " clCreateKernel\n");
        kernel = clCreateKernel(program, "FiniteDifferences", &errnum);
        if (kernel == (cl_kernel)NULL || errnum != CL_SUCCESS)
        {
            shrLog(LOGBOTH | ERRORMSG, 0.0, "clCreateKernel (returned %d).\n", errnum);
            ok = false;
        }
    }

    // Get the maximum work group size
    size_t maxWorkSize;
    if (ok)
    {
        shrLog(LOGBOTH, 0, " clGetKernelWorkGroupInfo\n");
        errnum = clGetKernelWorkGroupInfo(kernel, devices[0], CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &maxWorkSize, NULL);
        if (errnum != CL_SUCCESS)
        {
            shrLog(LOGBOTH | ERRORMSG, 0.0, "clGetKernelWorkGroupInfo (returned %d).\n", errnum);
            ok = false;
        }
    }

    // Check for a command-line specified work group size
    size_t userWorkSize;
    int userWorkSizeInt;
    if (ok)
    {
        if (shrGetCmdLineArgumenti(argc, argv, "work-group-size", &userWorkSizeInt))
        {
            // Constrain to a multiple of k_localWorkX
            userWorkSize = (userWorkSizeInt / k_localWorkX * k_localWorkX);
            // Constrain within allowed bounds
            userWorkSize = CLAMP(userWorkSize, k_localWorkMin, maxWorkSize);
        }
        else
        {
            userWorkSize = maxWorkSize;
        }
    }

    // Set the work group size
    if (ok)
    {
        localWorkSize[0] = k_localWorkX;
        localWorkSize[1] = std::min<size_t>(userWorkSize / k_localWorkX, (size_t)k_localWorkX);
        globalWorkSize[0] = localWorkSize[0] * ceil((float)dimx / localWorkSize[0]);
        globalWorkSize[1] = localWorkSize[1] * ceil((float)dimy / localWorkSize[1]);
        shrLog(LOGBOTH, 0, " set work group size to %dx%d\n", localWorkSize[0], localWorkSize[1]);
    }
    if (ok)
    {
        if (globalWorkSize[0] != (size_t)dimx || globalWorkSize[1] != (size_t)dimy)
        {
            shrLog(LOGBOTH | ERRORMSG, 0.0, "problem size (x and y) must be a multiple of the work size (%dx%d).\n", localWorkSize[0], localWorkSize[1]);
            ok = false;
        }
    }

    // Copy the input to the device input buffer
    if (ok)
    {
        shrLog(LOGBOTH, 0, " clEnqueueWriteBuffer bufferIn\n");
        errnum = clEnqueueWriteBuffer(commandQueue, bufferIn, CL_TRUE, 0, volumeSize * sizeof(float), input, 0, NULL, NULL);
        if (errnum != CL_SUCCESS)
        {
            shrLog(LOGBOTH | ERRORMSG, 0.0, "clEnqueueWriteBuffer bufferIn (returned %d).\n", errnum);
            ok = false;
        }
    }
    // Copy the input to the device output buffer (actually only need the halo)
    if (ok)
    {
        shrLog(LOGBOTH, 0, " clEnqueueWriteBuffer bufferOut\n");
        errnum = clEnqueueWriteBuffer(commandQueue, bufferOut, CL_TRUE, 0, volumeSize * sizeof(float), input, 0, NULL, NULL);
        if (errnum != CL_SUCCESS)
        {
            shrLog(LOGBOTH | ERRORMSG, 0.0, "clEnqueueWriteBuffer bufferOut (returned %d).\n", errnum);
            ok = false;
        }
    }
    // Copy the coefficients to the device coefficient buffer
    if (ok)
    {
        shrLog(LOGBOTH, 0, " clEnqueueWriteBuffer bufferCoeff\n");
        errnum = clEnqueueWriteBuffer(commandQueue, bufferCoeff, CL_TRUE, 0, (radius + 1) * sizeof(float), coeff, 0, NULL, NULL);
        if (errnum != CL_SUCCESS)
        {
            shrLog(LOGBOTH | ERRORMSG, 0.0, "clEnqueueWriteBuffer bufferCoeff (returned %d).\n", errnum);
            ok = false;
        }
    }

    // Start the clock
    shrDeltaT(0);

    // Set the constant arguments
    if (ok)
    {
        shrLog(LOGBOTH, 0, " clSetKernelArg 2-6\n");
        errnum  = clSetKernelArg(kernel, 2, (localWorkSize[0] + 2 * radius) * (localWorkSize[1] + 2 * radius) * sizeof(float), NULL);
        errnum |= clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&bufferCoeff);
        errnum |= clSetKernelArg(kernel, 4, sizeof(int), &dimx);
        errnum |= clSetKernelArg(kernel, 5, sizeof(int), &dimy);
        errnum |= clSetKernelArg(kernel, 6, sizeof(int), &dimz);
        if (errnum != CL_SUCCESS)
        {
            shrLog(LOGBOTH | ERRORMSG, 0.0, "clSetKernelArg 2-5 (returned %d).\n", errnum);
            ok = false;
        }
    }

    // Allocate the events
    if (ok)
    {
        shrLog(LOGBOTH, 0, " malloc events\n");
        if ((kernelEvents = (cl_event *)calloc(timesteps, sizeof(cl_event))) == NULL)
        {
            shrLog(LOGBOTH | ERRORMSG, 0.0, "malloc.\n");
            shrLog(LOGBOTH, 0, "Insufficient memory, please try a smaller volume (use --help for syntax).\n");
            ok = false;
        }
    }

    // Execute the FDTD
    cl_mem bufferSrc = bufferIn;
    cl_mem bufferDst = bufferOut;
    shrLog(LOGBOTH, 0, " GPU FDTD loop\n");
    for (int it = 0 ; ok && it < timesteps ; it++)
    {
        shrLog(LOGBOTH, 0, "\tt = %d ", it);

        // Set the dynamic arguments
        if (ok)
        {
            shrLog(LOGBOTH, 0, ", clSetKernelArg 0-1, ");
            errnum = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&bufferDst);
            errnum |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&bufferSrc);
            if (errnum != CL_SUCCESS)
            {
                shrLog(LOGBOTH | ERRORMSG, 0.0, "clSetKernelArg 0-1 (returned %d).\n", errnum);
                ok = false;
            }
        }

        // Launch the kernel
        if (ok)
        {
            shrLog(LOGBOTH, 0, "clEnqueueNDRangeKernel\n");
            errnum = clEnqueueNDRangeKernel(commandQueue, kernel, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, &kernelEvents[it]);
            if (errnum != CL_SUCCESS)
            {
                shrLog(LOGBOTH | ERRORMSG, 0.0, "clEnqueueNDRangeKernel (returned %d).\n", errnum);
                ok = false;
            }
        }
        // Toggle the buffers
        cl_mem tmp = bufferSrc;
        bufferSrc = bufferDst;
        bufferDst = tmp;
    }
    shrLog(LOGBOTH, 0, "\n");

    // Wait for the kernel to complete
    if (ok)
    {
        shrLog(LOGBOTH, 0, " clWaitForEvents\n");
        errnum = clWaitForEvents(1, &kernelEvents[timesteps-1]);
        if (errnum != CL_SUCCESS)
        {
            shrLog(LOGBOTH | ERRORMSG, 0.0, " clWaitForEvents (returned %d).\n", errnum);
            ok = false;
        }
    }

    // Stop the clock
    hostElapsedTimeS = shrDeltaT(0);

    // Read the result back, result is in bufferSrc (after final toggle)
    if (ok)
    {
        shrLog(LOGBOTH, 0, " clEnqueueReadBuffer\n");
        errnum = clEnqueueReadBuffer(commandQueue, bufferSrc, CL_TRUE, 0, volumeSize * sizeof(float), output, 0, NULL, NULL);
        if (errnum != CL_SUCCESS)
        {
            shrLog(LOGBOTH | ERRORMSG, 0.0, "clEnqueueReadBuffer bufferSrc (returned %d).\n", errnum);
            ok = false;
        }
    }

    // Report time
#ifdef GPU_PROFILING
    if (ok)
    {
        shrLog(LOGBOTH, 0, " clGetEventProfilingInfo\n");
        errnum = clGetEventProfilingInfo(kernelEvents[0], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &kernelEventStart, NULL);
        if (errnum != CL_SUCCESS)
        {
            shrLog(LOGBOTH | ERRORMSG, 0.0, "clGetEventProfilingInfo (returned %d).\n", errnum);
            ok = false;
        }
    }
    if (ok)
    {
        shrLog(LOGBOTH, 0, " clGetEventProfilingInfo\n\n");
        errnum = clGetEventProfilingInfo(kernelEvents[timesteps-1], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &kernelEventEnd, NULL);
        if (errnum != CL_SUCCESS)
        {
            shrLog(LOGBOTH | ERRORMSG, 0.0, "clGetEventProfilingInfo (returned %d).\n", errnum);
            ok = false;
        }
    }
    if (ok)
    {
        double elapsedTime = ((double)kernelEventEnd - (double)kernelEventStart) * 1.0e-9;  // convert nanoseconds to seconds
        double throughputM   = 1.0e-6 * ((double)timesteps * (double)dimx * (double)dimy * (double)dimz) / elapsedTime;
        shrLog(LOGBOTH | MASTER, 0, "oclFDTD3d, Throughput = %.4f, Time = %.5f, Size = %u, NumDevsUsed = %i, Workgroup = %u\n", 
            throughputM, elapsedTime/(double)timesteps, (dimx * dimy * dimz), 1, localWorkSize[0] * localWorkSize[1]); 
    }
#endif
    
    // Cleanup
    if (kernelEvents)
    {
        for (int it = 0 ; it < timesteps ; it++)
        {
            if (kernelEvents[it])
                clReleaseEvent(kernelEvents[it]);
        }
        free(kernelEvents);
    }
    if (kernel)
        clReleaseKernel(kernel);
    if (program)
        clReleaseProgram(program);
    if (cSourceCL)
        free(cSourceCL);
    if (cPathAndName)
        free(cPathAndName);
    if (bufferCoeff)
        clReleaseMemObject(bufferCoeff);
    if (bufferIn)
        clReleaseMemObject(bufferIn);
    if (bufferOut)
        clReleaseMemObject(bufferOut);
    if (commandQueue)
        clReleaseCommandQueue(commandQueue);
    if (devices)
        free(devices);
    if (context)
        clReleaseContext(context);
    return ok;
}
