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

///////////////////////////////////////////////////////////////////////////////
// This sample implements Niederreiter quasirandom number generator
// and Moro's Inverse Cumulative Normal Distribution generator
///////////////////////////////////////////////////////////////////////////////

// standard utilities and systems includes
#include <oclUtils.h>
#include "oclQuasirandomGenerator_common.h"

// forward declarations
extern "C" void initQuasirandomGenerator(
    unsigned int table[QRNG_DIMENSIONS][QRNG_RESOLUTION]
    );
extern "C" double getQuasirandomValue63(INT64 i, int dim);
extern "C" double MoroInvCNDcpu(double P);

// OpenCL wrappers
size_t QuasirandomGeneratorGPU(cl_command_queue cqCommandQueue,
			                 cl_kernel ckQuasirandomGenerator,
                             cl_mem d_Output,
                             cl_mem c_Table,
                             unsigned int seed,
                             unsigned int N);
size_t InverseCNDGPU(cl_command_queue cqCommandQueue, 
		   cl_kernel ckInverseCNDGPU, 
		   cl_mem d_Output, 
		   unsigned int pathN,
           unsigned int iDevice,
           unsigned int nDevice);

// size of output random array
const unsigned int N = 1048576;

///////////////////////////////////////////////////////////////////////////////
// Main function 
///////////////////////////////////////////////////////////////////////////////
int main(int argc, const char **argv)
{
    cl_context cxGPUContext;                          // OpenCL context
    cl_command_queue cqCommandQueue[MAX_GPU_COUNT];   // OpenCL command que
    cl_device_id *cdDevices;                          // OpenCL device list    
    cl_program cpProgram;                             // OpenCL program
    cl_kernel ckQuasirandomGenerator, ckInverseCNDGPU;// OpenCL kernel
    cl_mem *d_Output, *c_Table;                       // OpenCL buffers
    float *h_OutputGPU;
    cl_int ciErr1, ciErr2;                            // Error code var
    unsigned int dim, pos;
    double delta, ref, sumDelta, sumRef, L1norm;
    unsigned int tableCPU[QRNG_DIMENSIONS][QRNG_RESOLUTION];

    // Start logs 
    shrSetLogFileName("oclQuasirandomGenerator.txt");
    shrLog(LOGBOTH, 0, "%s Starting...\n\n", argv[0]); 

    shrLog(LOGBOTH, 0, "Create context...\n");
    cxGPUContext = clCreateContextFromType(0, CL_DEVICE_TYPE_GPU, NULL, NULL, &ciErr1);
    shrCheckErrorEX(ciErr1, CL_SUCCESS, NULL);

    shrLog(LOGBOTH, 0, "Get device info...\n");
    int nDevice = 0;
    size_t nDeviceBytes;
    ciErr1 |= clGetContextInfo(cxGPUContext, CL_CONTEXT_DEVICES, 0, NULL, &nDeviceBytes);
    cdDevices = (cl_device_id*)malloc(nDeviceBytes);
    ciErr1 |= clGetContextInfo(cxGPUContext, CL_CONTEXT_DEVICES, nDeviceBytes, cdDevices, NULL);
    nDevice = (int)(nDeviceBytes/sizeof(cl_device_id));
    
    int id_device;
    if(shrGetCmdLineArgumenti(argc, argv, "device", &id_device)) // Set up command queue(s) for GPU specified on the command line
    {
        // get & log device index # and name
        cl_device_id cdDevice = cdDevices[id_device];

        // create a command que
        cqCommandQueue[0] = clCreateCommandQueue(cxGPUContext, cdDevice, 0, &ciErr1);
        shrCheckErrorEX(ciErr1, CL_SUCCESS, NULL);
        oclPrintDevInfo(LOGBOTH, cdDevice);
        nDevice = 1;   
    }  
    else 
    { // create command queues for all available devices        
        for (int i = 0; i < nDevice; i++) 
        {
	  cqCommandQueue[i] = clCreateCommandQueue(cxGPUContext, cdDevices[i], 0, &ciErr1);
            shrCheckErrorEX(ciErr1, CL_SUCCESS, NULL);
        }
        for (int i = 0; i < nDevice; i++) oclPrintDevInfo(LOGBOTH, cdDevices[i]);
    }

    shrLog(LOGBOTH, 0, "\nUsing %d GPU(s)...\n\n", nDevice); 

    shrLog(LOGBOTH, 0, "Allocate memory...\n"); 
    d_Output = (cl_mem*)malloc(nDevice*sizeof(cl_mem));
    c_Table = (cl_mem*)malloc(nDevice*sizeof(cl_mem));
    for (int i = 0; i < nDevice; i++)
    {
        d_Output[i] = clCreateBuffer(cxGPUContext, CL_MEM_WRITE_ONLY, QRNG_DIMENSIONS * N / nDevice * sizeof(cl_float), NULL, &ciErr1);
        shrCheckErrorEX(ciErr1, CL_SUCCESS, NULL);
    }
    h_OutputGPU = (float *)malloc(QRNG_DIMENSIONS * N * sizeof(cl_float));

    shrLog(LOGBOTH, 0, "Initializing QRNG tables...\n");
    initQuasirandomGenerator(tableCPU);
    for (int i = 0; i < nDevice; i++)
    {
        c_Table[i] = clCreateBuffer(cxGPUContext, CL_MEM_READ_ONLY, QRNG_DIMENSIONS * QRNG_RESOLUTION * sizeof(unsigned int), 
    		     NULL, &ciErr2);
        ciErr1 |= ciErr2;
        ciErr1 |= clEnqueueWriteBuffer(cqCommandQueue[i], c_Table[i], CL_TRUE, 0, 
            QRNG_DIMENSIONS * QRNG_RESOLUTION * sizeof(unsigned int), tableCPU, 0, NULL, NULL);
    }
    shrCheckErrorEX(ciErr1, CL_SUCCESS, NULL);

    shrLog(LOGBOTH, 0, "Create and build program...\n");
    size_t szKernelLength; // Byte size of kernel code
    char *progSource = oclLoadProgSource(shrFindFilePath("QuasirandomGenerator.cl", argv[0]), "// My comment\n", &szKernelLength);
	shrCheckErrorEX(progSource == NULL, false, NULL);

    cpProgram = clCreateProgramWithSource(cxGPUContext, 1, (const char **)&progSource, &szKernelLength, &ciErr1);
    ciErr1 |= clBuildProgram(cpProgram, 0, NULL, NULL, NULL, NULL);
    if (ciErr1 != CL_SUCCESS)
    {
        // write out standard error, Build Log and PTX, then cleanup and exit
        shrLog(LOGBOTH | ERRORMSG, (double)ciErr1, STDERROR);
        oclLogBuildInfo(cpProgram, oclGetFirstDev(cxGPUContext));
        oclLogPtx(cpProgram, oclGetFirstDev(cxGPUContext), "QuasirandomGenerator.ptx");
        shrCheckError(ciErr1, CL_SUCCESS); 
    }

    shrLog(LOGBOTH, 0, "Create QuasirandomGenerator kernel...\n"); 
    ckQuasirandomGenerator = clCreateKernel(cpProgram, "QuasirandomGenerator", &ciErr1);
    shrCheckErrorEX(ciErr1, CL_SUCCESS, NULL); 

    shrLog(LOGBOTH, 0, "Create InverseCND kernel...\n\n"); 
    ckInverseCNDGPU = clCreateKernel(cpProgram, "InverseCND", &ciErr1);
    shrCheckErrorEX(ciErr1, CL_SUCCESS, NULL); 

    shrLog(LOGBOTH, 0, "Launch QuasirandomGenerator kernel...\n"); 

#ifdef GPU_PROFILING
    for (int iDevice = 0; iDevice < nDevice; iDevice++)
    {
        clFinish(cqCommandQueue[iDevice]);
    }
    shrDeltaT(1);
#endif

    int iCycles = 100;
    size_t szWorkgroup;    
    for (int i = 0; i< iCycles; i++)
    {
        for (int i = 0; i < nDevice; i++)
        {
            szWorkgroup = QuasirandomGeneratorGPU(cqCommandQueue[i], ckQuasirandomGenerator, d_Output[i], c_Table[i], 0, N/nDevice);    
        }
        for (int i = 0; i < nDevice; i++)
        {
            clFinish(cqCommandQueue[i]);
        }
    }

#ifdef GPU_PROFILING
    double gpuTime = shrDeltaT(1)/(double)iCycles;
    shrLog(LOGBOTH | MASTER, 0, "oclQuasirandomGenerator, Throughput = %.4f, Time = %.5f, Size = %u, NumDevsUsed = %u, Workgroup = %u\n", 
           (double)QRNG_DIMENSIONS * (double)N * 1.0E-9 / gpuTime, gpuTime, QRNG_DIMENSIONS * N, nDevice, szWorkgroup);
#endif

    shrLog(LOGBOTH, 0, "\nRead back results...\n"); 
    int offset = 0;
    for (int i = 0; i < nDevice; i++)
    {
        ciErr1 |= clEnqueueReadBuffer(cqCommandQueue[i], d_Output[i], CL_TRUE, 0, sizeof(cl_float) * QRNG_DIMENSIONS * N / nDevice, 
            h_OutputGPU + offset, 0, NULL, NULL);
        offset += QRNG_DIMENSIONS * N / nDevice;
    }
    shrCheckErrorEX(ciErr1, CL_SUCCESS, NULL); 

    shrLog(LOGBOTH, 0, "Comparing to the CPU results...\n");
    sumDelta = 0;
    sumRef   = 0;
    for (int i = 0; i < nDevice; i++)
    {
        for(dim = 0; dim < QRNG_DIMENSIONS; dim++)
        {
            for(pos = 0; pos < N / nDevice; pos++) 
            {
	            ref       = getQuasirandomValue63(pos, dim);
	            delta     = (double)h_OutputGPU[i*QRNG_DIMENSIONS*N/nDevice + dim * N / nDevice + pos] - ref;
	            sumDelta += fabs(delta);
	            sumRef   += fabs(ref);
	        }
        }
    }
    L1norm = sumDelta / sumRef;
    shrLog(LOGBOTH, 0, "L1 norm: %E\n\n", L1norm);
    shrLog(LOGBOTH, 0, "TEST %s\n\n", (L1norm < 1e-6) ? "PASSED" : "FAILED !!!");

    shrLog(LOGBOTH, 0, "Launch InverseCND kernel...\n"); 

#ifdef GPU_PROFILING
    for (int iDevice = 0; iDevice < nDevice; iDevice++)
    {
        clFinish(cqCommandQueue[iDevice]);
    }
    shrDeltaT(1);
#endif

    for (int i = 0; i< iCycles; i++)
    {
        for (int i = 0; i < nDevice; i++)
        {
            szWorkgroup = InverseCNDGPU(cqCommandQueue[i], ckInverseCNDGPU, d_Output[i], QRNG_DIMENSIONS * N / nDevice, i, nDevice);    
        }
        for (int i = 0; i < nDevice; i++)
        {
            clFinish(cqCommandQueue[i]);
        }
    }

#ifdef GPU_PROFILING
    gpuTime = shrDeltaT(1)/(double)iCycles;
    shrLog(LOGBOTH | MASTER, 0, "oclQuasirandomGenerator-inverse, Throughput = %.4f, Time = %.5f, Size = %u, NumDevsUsed = %u, Workgroup = %u\n", 
           (double)QRNG_DIMENSIONS * (double)N * 1.0E-9 / gpuTime, gpuTime, QRNG_DIMENSIONS * N, nDevice, szWorkgroup);    
#endif

    shrLog(LOGBOTH, 0, "Read back results...\n"); 
    offset = 0;
    for (int i = 0; i < nDevice; i++)
    {
        ciErr1 |= clEnqueueReadBuffer(cqCommandQueue[i], d_Output[i], CL_TRUE, 0, 
            sizeof(cl_float) * QRNG_DIMENSIONS * N / nDevice, h_OutputGPU + offset, 0, NULL, NULL);
        offset += QRNG_DIMENSIONS * N / nDevice;
        shrCheckErrorEX(ciErr1, CL_SUCCESS, NULL); 
    }

    shrLog(LOGBOTH, 0, "Comparing to the CPU results...\n");
    sumDelta = 0;
    sumRef   = 0;
    for(pos = 0; pos < QRNG_DIMENSIONS * N; pos++)
    {
        double  p = (double)(pos + 1) / (double)(QRNG_DIMENSIONS * N + 1);
	    ref       = MoroInvCNDcpu(p);
	    delta     = (double)h_OutputGPU[pos] - ref;
	    sumDelta += fabs(delta);
	    sumRef   += fabs(ref);
    }
    L1norm = sumDelta / sumRef;
    shrLog(LOGBOTH, 0, "L1 norm: %E\n\n", L1norm);
    shrLog(LOGBOTH, 0, "TEST %s\n\n", (L1norm < 1e-6) ? "PASSED" : "FAILED !!!");


    // NOTE:  Most properly this should be done at any of the exit points above, but it is omitted elsewhere for clarity.
    shrLog(LOGBOTH, 0, "Release CPU buffers and OpenCL objects...\n"); 
    free(h_OutputGPU); 
    free(progSource);
    free(cdDevices);
    for (int i = 0; i < nDevice; i++)
    {
        clReleaseMemObject(d_Output[i]);
        clReleaseMemObject(c_Table[i]);
        clReleaseCommandQueue(cqCommandQueue[i]);
    }
    clReleaseKernel(ckQuasirandomGenerator);
    clReleaseKernel(ckInverseCNDGPU);
    clReleaseProgram(cpProgram);
    clReleaseContext(cxGPUContext);

    shrEXIT(argc, argv);
}

///////////////////////////////////////////////////////////////////////////////
// Wrapper for OpenCL Niederreiter quasirandom number generator kernel
///////////////////////////////////////////////////////////////////////////////
size_t QuasirandomGeneratorGPU(cl_command_queue cqCommandQueue,
			     cl_kernel ckQuasirandomGenerator,
			     cl_mem d_Output,
			     cl_mem c_Table,
			     unsigned int seed,
			     unsigned int N)
{
    cl_int ciErr;
    size_t globalWorkSize[2] = {128*128, QRNG_DIMENSIONS};
    size_t localWorkSize[2] = {128, QRNG_DIMENSIONS};
    
    ciErr  = clSetKernelArg(ckQuasirandomGenerator, 0, sizeof(cl_mem),       (void*)&d_Output);
    ciErr |= clSetKernelArg(ckQuasirandomGenerator, 1, sizeof(cl_mem),       (void*)&c_Table );
    ciErr |= clSetKernelArg(ckQuasirandomGenerator, 2, sizeof(unsigned int), (void*)&seed    );
    ciErr |= clSetKernelArg(ckQuasirandomGenerator, 3, sizeof(unsigned int), (void*)&N       );
    ciErr = clEnqueueNDRangeKernel(cqCommandQueue, ckQuasirandomGenerator, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);  
    return (localWorkSize[0] * localWorkSize[1]);
}

///////////////////////////////////////////////////////////////////////////////
// Wrapper for OpenCL Inverse Cumulative Normal Distribution generator kernel
///////////////////////////////////////////////////////////////////////////////
size_t InverseCNDGPU(cl_command_queue cqCommandQueue, 
		   cl_kernel ckInverseCNDGPU, 
		   cl_mem d_Output, 
		   unsigned int pathN,
           unsigned int iDevice,
           unsigned int nDevice)
{
    cl_int ciErr;
    size_t globalWorkSize[1] = {128*128};
    size_t localWorkSize[1] = {128};

    ciErr  = clSetKernelArg(ckInverseCNDGPU, 0, sizeof(cl_mem),       (void*)&d_Output);
    ciErr |= clSetKernelArg(ckInverseCNDGPU, 1, sizeof(unsigned int), (void*)&pathN   );
    ciErr |= clSetKernelArg(ckInverseCNDGPU, 2, sizeof(unsigned int), (void*)&iDevice );
    ciErr |= clSetKernelArg(ckInverseCNDGPU, 3, sizeof(unsigned int), (void*)&nDevice );
    ciErr = clEnqueueNDRangeKernel(cqCommandQueue, ckInverseCNDGPU, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
    return localWorkSize[0];    
}
