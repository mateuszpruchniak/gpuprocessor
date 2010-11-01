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

/* Matrix multiplication: C = A * B.
 * Host code.
 *
 * This sample implements matrix multiplication with multi GPU support.
 * It has been written for clarity of exposition to illustrate various OpenCL
 * programming principles, not with the goal of providing the most
 * performant generic kernel for matrix multiplication.
 *
 * CUBLAS provides high-performance matrix multiplication.
 */

// standard utilities and system includes
#include <oclUtils.h>

// project include
#include "matrixMul.h"

// max GPU's to manage for multi-GPU parallel compute
const unsigned int MAX_GPU_COUNT = 8;

// global variables
cl_context cxGPUContext;
cl_kernel multiplicationKernel[MAX_GPU_COUNT];
cl_command_queue commandQueue[MAX_GPU_COUNT];

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
int runTest(int argc, const char** argv);
void printDiff(float*, float*, int, int);
void matrixMulGPU(cl_uint ciDeviceCount, cl_mem h_A, float* h_B_data, unsigned int mem_size_B, float* h_C );

extern "C"
void computeGold(float*, const float*, const float*, unsigned int, unsigned int, unsigned int);

////////////////////////////////////////////////////////////////////////////////
// Helper functions
////////////////////////////////////////////////////////////////////////////////

double executionTime(cl_event &event)
{
    cl_ulong start, end;
    
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
    
    return (double)1.0e-9 * (end - start); // convert nanoseconds to seconds on return
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, const char** argv)
{
    // start the logs
    shrSetLogFileName ("oclMatrixMul.txt");
    shrLog(LOGBOTH, 0, "%s Starting...\n\n", argv[0]); 

    // run the code
    if (runTest(argc, argv) != 0)
    {
        shrLog(LOGBOTH, 0, "TEST FAILED !!!\n\n");
    }

    // finish
    shrEXIT(argc, argv);
}

void matrixMulGPU(cl_uint ciDeviceCount, cl_mem h_A, float* h_B_data, unsigned int mem_size_B, float* h_C )
{
    cl_mem d_A[MAX_GPU_COUNT];
    cl_mem d_C[MAX_GPU_COUNT];
    cl_mem d_B[MAX_GPU_COUNT];

    cl_event GPUDone[MAX_GPU_COUNT];
    cl_event GPUExecution[MAX_GPU_COUNT];

    // Start the computation on each available GPU
    
    // Create buffers for each GPU
    // Each GPU will compute sizePerGPU rows of the result
    int sizePerGPU = HA / ciDeviceCount;

    int workOffset[MAX_GPU_COUNT];
    int workSize[MAX_GPU_COUNT];

    workOffset[0] = 0;
    for(unsigned int i=0; i < ciDeviceCount; ++i) 
    {
        // Input buffer
        workSize[i] = (i != (ciDeviceCount - 1)) ? sizePerGPU : (HA - workOffset[i]);        

        d_A[i] = clCreateBuffer(cxGPUContext, CL_MEM_READ_ONLY, workSize[i] * sizeof(float) * WA, NULL,NULL);

        // Copy only assigned rows from host to device
        clEnqueueCopyBuffer(commandQueue[i], h_A, d_A[i], workOffset[i] * sizeof(float) * WA, 
                            0, workSize[i] * sizeof(float) * WA, 0, NULL, NULL);        
        
        // create OpenCL buffer on device that will be initiatlize from the host memory on first use
        // on device
        d_B[i] = clCreateBuffer(cxGPUContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                mem_size_B, h_B_data, NULL);

        // Output buffer
        d_C[i] = clCreateBuffer(cxGPUContext, CL_MEM_WRITE_ONLY,  workSize[i] * WC * sizeof(float), NULL,NULL);
              
        // set the args values
        clSetKernelArg(multiplicationKernel[i], 0, sizeof(cl_mem), (void *) &d_C[i]);
        clSetKernelArg(multiplicationKernel[i], 1, sizeof(cl_mem), (void *) &d_A[i]);
        clSetKernelArg(multiplicationKernel[i], 2, sizeof(cl_mem), (void *) &d_B[i]);
        clSetKernelArg(multiplicationKernel[i], 3, sizeof(float) * BLOCK_SIZE *BLOCK_SIZE, 0 );
        clSetKernelArg(multiplicationKernel[i], 4, sizeof(float) * BLOCK_SIZE *BLOCK_SIZE, 0 );

        if(i+1 < ciDeviceCount)
            workOffset[i + 1] = workOffset[i] + workSize[i];
    }
    
    // Execute Multiplication on all GPUs in parallel
    size_t localWorkSize[] = {BLOCK_SIZE, BLOCK_SIZE};
    size_t globalWorkSize[] = {shrRoundUp(BLOCK_SIZE, WC), shrRoundUp(BLOCK_SIZE, workSize[0])};
    
    // Start timer and launch kernels on devices
    shrDeltaT(0);
    for(unsigned int i = 0; i < ciDeviceCount; i++) 
    {
        // Multiplication - non-blocking execution
        globalWorkSize[1] = shrRoundUp(BLOCK_SIZE, workSize[i]);

        clEnqueueNDRangeKernel(commandQueue[i], multiplicationKernel[i], 2, 0, globalWorkSize, localWorkSize,
                               0, NULL, &GPUExecution[i]);        
    }

    for(unsigned int i = 0; i < ciDeviceCount; i++) 
    {    
        // Non-blocking copy of result from device to host
        clEnqueueReadBuffer(commandQueue[i], d_C[i], CL_FALSE, 0, WC * sizeof(float) * workSize[i], 
                            h_C + workOffset[i] * WC, 0, NULL, &GPUDone[i]);
    }

    // CPU sync with GPU
    clWaitForEvents(ciDeviceCount, GPUDone);

    // stop and log timer 
    #ifdef GPU_PROFILING
        double dSeconds = shrDeltaT(0);
        double dSize = ((double)WA * (double)HA * (double)WB * (double)HB);
        shrLog(LOGBOTH | MASTER, 0, "oclMatrixMul, Throughput = %.4f, Time = %.5f, Size = %.0f, NumDevsUsed = %d, Workgroup = %u\n", 
                1.0e-9 * dSize/dSeconds, dSeconds, dSize, ciDeviceCount, localWorkSize[0] * localWorkSize[1]);

        // Print kernel timing per GPU
        for(unsigned int i = 0; i < ciDeviceCount; i++) 
        {    
            shrLog(LOGBOTH, 0, "  Kernel execution time on GPU %d \t: %.5f s\n", i, executionTime(GPUExecution[i]));
        }
        shrLog(LOGBOTH, 0, "\n");
     #endif

    // Release mem and event objects    
    for(unsigned int i = 0; i < ciDeviceCount; i++) 
    {
        clReleaseMemObject(d_A[i]);
        clReleaseMemObject(d_C[i]);
        clReleaseMemObject(d_B[i]);
	    clReleaseEvent(GPUExecution[i]);
	    clReleaseEvent(GPUDone[i]);
    }
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for 
////////////////////////////////////////////////////////////////////////////////
int runTest(int argc, const char** argv)
{
    cl_uint ciDeviceCount = 0;
    cl_int ciErrNum = CL_SUCCESS;

    // create the OpenCL context on available GPU devices
    cxGPUContext = clCreateContextFromType(0, CL_DEVICE_TYPE_GPU, NULL, NULL, &ciErrNum);
    if (ciErrNum != CL_SUCCESS)
    {
        shrLog(LOGBOTH, 0, "Error: Failed to create OpenCL context!\n");
        return ciErrNum;
    }

    if(shrCheckCmdLineFlag(argc, (const char**)argv, "device"))
    {
        // User specified GPUs
        char* deviceList;
        char* deviceStr;
        char* next_token;
        shrGetCmdLineArgumentstr(argc, (const char**)argv, "device", &deviceList);

        #ifdef WIN32
            deviceStr = strtok_s (deviceList," ,.-", &next_token);
        #else
            deviceStr = strtok (deviceList," ,.-");
        #endif   
        while(deviceStr != NULL) 
        {
            // get and print the device for this queue
            cl_device_id device = oclGetDev(cxGPUContext, atoi(deviceStr));
            shrLog(LOGBOTH, 0, "Device %d:\n", ciDeviceCount);
            oclPrintDevName(LOGBOTH, device);            
           
            // create command queue
            commandQueue[ciDeviceCount] = clCreateCommandQueue(cxGPUContext, device, 0, &ciErrNum);
            if (ciErrNum != CL_SUCCESS)
            {
                shrLog(LOGBOTH, 0, " Error %i in clCreateCommandQueue call !!!\n\n", ciErrNum);
                return ciErrNum;
            }

            #ifdef GPU_PROFILING
                ciErrNum = clSetCommandQueueProperty(commandQueue[ciDeviceCount], CL_QUEUE_PROFILING_ENABLE, CL_TRUE, NULL);
                if (ciErrNum != CL_SUCCESS)
                {
                    shrLog(LOGBOTH, 0, " Error %i in clSetCommandQueueProperty call !!!\n\n", ciErrNum);
                    return ciErrNum;
                }
            #endif
                
             ++ciDeviceCount;

            #ifdef WIN32
                deviceStr = strtok_s (NULL," ,.-", &next_token);
            #else            
                deviceStr = strtok (NULL," ,.-");
            #endif
        }

        free(deviceList);
    } 
    else 
    {
        // Find out how many GPU's to compute on all available GPUs
	    size_t nDeviceBytes;
	    ciErrNum |= clGetContextInfo(cxGPUContext, CL_CONTEXT_DEVICES, 0, NULL, &nDeviceBytes);
	    ciDeviceCount = (cl_uint)nDeviceBytes/sizeof(cl_device_id);

        if (ciErrNum != CL_SUCCESS)
        {
            shrLog(LOGBOTH, 0, " Error %i in clGetDeviceIDs call !!!\n\n", ciErrNum);
            return ciErrNum;
        }
        else if (ciDeviceCount == 0)
        {
            shrLog(LOGBOTH, 0, " There are no devices supporting OpenCL (return code %i)\n\n", ciErrNum);
            return -1;
        } 

        // create command-queues
        for(unsigned int i = 0; i < ciDeviceCount; ++i) 
        {
            // get and print the device for this queue
            cl_device_id device = oclGetDev(cxGPUContext, i);
            shrLog(LOGBOTH, 0, "Device %d:\n", i);
            oclPrintDevName(LOGBOTH, device);            
            
            // create command queue
            commandQueue[i] = clCreateCommandQueue(cxGPUContext, device, 0, &ciErrNum);
            if (ciErrNum != CL_SUCCESS)
            {
                shrLog(LOGBOTH, 0, " Error %i in clCreateCommandQueue call !!!\n\n", ciErrNum);
                return ciErrNum;
            }
            #ifdef GPU_PROFILING
                clSetCommandQueueProperty(commandQueue[i], CL_QUEUE_PROFILING_ENABLE, CL_TRUE, NULL);
            #endif
        }
    }

    // allocate host memory for matrices A and B
    unsigned int size_A = WA * HA;
    unsigned int mem_size_A = sizeof(float) * size_A;
    float* h_A_data = (float*) malloc(mem_size_A);
    unsigned int size_B = WB * HB;
    unsigned int mem_size_B = sizeof(float) * size_B;
    float* h_B_data = (float*) malloc(mem_size_B);

    // initialize host memory
    srand(2006);
    shrFillArray(h_A_data, size_A);
    shrFillArray(h_B_data, size_B);

    // allocate host memory for result
    unsigned int size_C = WC * HC;
    unsigned int mem_size_C = sizeof(float) * size_C;
    float* h_C = (float*) malloc(mem_size_C);

    // create OpenCL buffer pointing to the host memory
    cl_mem h_A = clCreateBuffer(cxGPUContext, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
				    mem_size_A, h_A_data, &ciErrNum);
    if (ciErrNum != CL_SUCCESS)
    {
        shrLog(LOGBOTH, 0, "Error: clCreateBuffer\n");
        return ciErrNum;
    }

    // Program Setup
    size_t program_length;
    const char* header_path = shrFindFilePath("matrixMul.h", argv[0]);
    char* header = oclLoadProgSource(header_path, "", &program_length);
    if(!header)
    {
        shrLog(LOGBOTH, 0, "Error: Failed to load the header %s!\n", header_path);
        return -1000;
    }
    const char* source_path = shrFindFilePath("matrixMul.cl", argv[0]);
    char *source = oclLoadProgSource(source_path, header, &program_length);
    if(!source)
    {
        shrLog(LOGBOTH, 0, "Error: Failed to load compute program %s!\n", source_path);
        return -2000;
    }

    // create the program
    cl_program cpProgram = clCreateProgramWithSource(cxGPUContext, 1, (const char **)&source, 
                                                    &program_length, &ciErrNum);
    if (ciErrNum != CL_SUCCESS)
    {
        shrLog(LOGBOTH, 0, "Error: Failed to create program\n");
        return ciErrNum;
    }
    free(header);
    free(source);
    
    // build the program
    ciErrNum = clBuildProgram(cpProgram, 0, NULL, "-cl-mad-enable", NULL, NULL);
    if (ciErrNum != CL_SUCCESS)
    {
        // write out standard error, Build Log and PTX, then return error
        shrLog(LOGBOTH | ERRORMSG, ciErrNum, STDERROR);
        oclLogBuildInfo(cpProgram, oclGetFirstDev(cxGPUContext));
        oclLogPtx(cpProgram, oclGetFirstDev(cxGPUContext), "oclMatrixMul.ptx");
        return ciErrNum;
    }

    // write out PTX if requested on the command line
    if(shrCheckCmdLineFlag(argc, argv, "dump-ptx") )
    {
        oclLogPtx(cpProgram, oclGetFirstDev(cxGPUContext), "oclMatrixMul.ptx");
    }

    // Create Kernel
    for(unsigned int i=0; i<ciDeviceCount; ++i) {
        multiplicationKernel[i] = clCreateKernel(cpProgram, "matrixMul", &ciErrNum);
        if (ciErrNum != CL_SUCCESS)
        {
            shrLog(LOGBOTH, 0, "Error: Failed to create kernel\n");
            return ciErrNum;
        }
    }
        
    // Run multiplication on 1..deviceCount GPUs to compare improvement
    shrLog(LOGBOTH, 0, "\nRunning Computations on 1 - %d GPU's...\n", ciDeviceCount);
    for(unsigned int k = 1; k <= ciDeviceCount; ++k) 
    {
        matrixMulGPU(k, h_A, h_B_data, mem_size_B, h_C);
    }

    // compute reference solution
    shrLog(LOGBOTH, 0, "\nComparing results with CPU computation... \n\n");
    float* reference = (float*) malloc(mem_size_C);
    computeGold(reference, h_A_data, h_B_data, HA, WA, WB);

    // check result
    shrBOOL res = shrCompareL2fe(reference, h_C, size_C, 1e-6f);
    shrLog(LOGBOTH, 0, "TEST %s \n\n", (1 == res) ? "PASSED" : "FAILED !!!");
    if (res != 1) 
    {
        printDiff(reference, h_C, WC, HC);
    }

    // clean up OCL resources
    clReleaseMemObject(h_A);
    for(unsigned int k = 0; k < ciDeviceCount; ++k) 
    {
        clReleaseKernel( multiplicationKernel[k] );
        clReleaseCommandQueue( commandQueue[k] );
    }
    clReleaseProgram(cpProgram);
    ciErrNum = clReleaseContext(cxGPUContext);
    if( ciErrNum != CL_SUCCESS) 
        shrLog(LOGBOTH, 0, "Error: Failed to release context: %d\n", ciErrNum);

    // clean up memory
    free(h_A_data);
    free(h_B_data);
    free(h_C);
    free(reference);
    
    return 0;
}

void printDiff(float *data1, float *data2, int width, int height)
{
  int i,j,k;
  int error_count=0;
  for (j=0; j<height; j++) {
    for (i=0; i<width; i++) {
      k = j*width+i;
      if ( fabs(data1[k] - data2[k]) < 1e-5) {
          shrLog(LOGBOTH, 0, "diff(%d,%d) CPU=%.4f, GPU=%.4f \n", i,j, data1[k], data2[k]);
          error_count++;
      }

    }
      shrLog(LOGBOTH, 0, "\n");
  }
  shrLog(LOGBOTH, 0, " \nTotal Errors = %d \n", error_count);
}
