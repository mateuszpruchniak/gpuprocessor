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

/*
 * This application demonstrates how to use multiple GPUs with OpenCL.
 * Event Objects are used to synchronize the CPU with multiple GPUs. 
 *
 * Note that in order to detect multiple GPUs in your system you have to disable
 * SLI in the nvidia control panel. Otherwise only one GPU is visible to the 
 * application. On the other side, you can still extend your desktop to screens 
 * attached to both GPUs.
 */

#include <oclUtils.h>

////////////////////////////////////////////////////////////////////////////////
// Data configuration
////////////////////////////////////////////////////////////////////////////////
const unsigned int MAX_GPU_COUNT = 8;
const unsigned int DATA_N = 1048576*24;
const unsigned int BLOCK_N = 128;
const unsigned int THREAD_N = 128;
const unsigned int ACCUM_N = BLOCK_N * THREAD_N;

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
int main(int argc, const char **argv)
{
    // start logs 
    shrSetLogFileName ("oclSimpleMultiGPU.txt");
    shrLog(LOGBOTH, 0, "%s Starting, Array = %u float values...\n\n", argv[0], DATA_N); 

    // OpenCL
    cl_context cxGPUContext;
    cl_device_id cdDevice;                          // GPU device
    int deviceNr[MAX_GPU_COUNT];
    cl_command_queue commandQueue[MAX_GPU_COUNT];
    cl_mem d_Data[MAX_GPU_COUNT];
    cl_mem d_Result[MAX_GPU_COUNT];
    cl_program cpProgram; 
    cl_kernel reduceKernel[MAX_GPU_COUNT];
    cl_event GPUDone[MAX_GPU_COUNT];
    cl_event GPUExecution[MAX_GPU_COUNT];
    cl_uint ciDeviceCount = 0;
    size_t programLength;
    cl_int ciErrNum;			               
    char cDeviceName [256];
    cl_mem h_DataBuffer;

    // Vars for reduction results
    float h_SumGPU[MAX_GPU_COUNT * ACCUM_N];   
    float *h_Data;
    double sumGPU;
    double sumCPU, dRelError;

    // allocate and init host buffer with with some random generated input data
    h_Data = (float *)malloc(DATA_N * sizeof(float));
    shrFillArray(h_Data, DATA_N);

    // start timer & logs 
    shrLog(LOGBOTH, 0, "Setting up OpenCL on the Host...\n\n"); 
    shrDeltaT(1);

    // Annotate profiling state
    #ifdef GPU_PROFILING
        shrLog(LOGBOTH, 0, "OpenCL Profiling is enabled...\n\n"); 
    #endif

    // create the OpenCL context on available GPU devices
    cxGPUContext = clCreateContextFromType(0, CL_DEVICE_TYPE_GPU, NULL, NULL, &ciErrNum);
    shrLog(LOGBOTH, 0, "clCreateContextFromType\n\n"); 

    // Set up command queue(s) for GPU's specified on the command line or all GPU's
    if(shrCheckCmdLineFlag(argc, argv, "device"))
    {
        // User specified GPUs
        char* deviceList;
        char* deviceStr;
        char* next_token;
        shrGetCmdLineArgumentstr(argc, argv, "device", &deviceList);

        #ifdef WIN32
            deviceStr = strtok_s (deviceList," ,.-", &next_token);
        #else
            deviceStr = strtok (deviceList," ,.-");
        #endif   

        // Create command queues for all Requested GPU's
        while(deviceStr != NULL) 
        {
            // get & log device index # and name
	        deviceNr[ciDeviceCount] = atoi(deviceStr);
	        cdDevice = oclGetDev(cxGPUContext, deviceNr[ciDeviceCount]);
            ciErrNum = clGetDeviceInfo(cdDevice, CL_DEVICE_NAME, sizeof(cDeviceName), cDeviceName, NULL);
            shrCheckError(ciErrNum, CL_SUCCESS);
            shrLog(LOGBOTH, 0, " Device %i: %s\n\n", ciDeviceCount, cDeviceName);

            // create a command que
            commandQueue[ciDeviceCount] = clCreateCommandQueue(cxGPUContext, cdDevice, 0, &ciErrNum);
            shrCheckError(ciErrNum, CL_SUCCESS);
            shrLog(LOGBOTH, 0, "clCreateCommandQueue\n"); 

            #ifdef GPU_PROFILING
                ciErrNum = clSetCommandQueueProperty(commandQueue[ciDeviceCount], CL_QUEUE_PROFILING_ENABLE, CL_TRUE, NULL);
                shrCheckError(ciErrNum, CL_SUCCESS);
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
	ciErrNum = clGetContextInfo(cxGPUContext, CL_CONTEXT_DEVICES, 0, NULL, &nDeviceBytes);
        shrCheckError(ciErrNum, CL_SUCCESS);
	ciDeviceCount = (cl_uint)nDeviceBytes/sizeof(cl_device_id);

        for(unsigned int i = 0; i < ciDeviceCount; ++i ) 
        {
            // get & log device index # and name
	        deviceNr[i] = i;
            cdDevice = oclGetDev(cxGPUContext, i);
            ciErrNum = clGetDeviceInfo(cdDevice, CL_DEVICE_NAME, sizeof(cDeviceName), cDeviceName, NULL);
            shrCheckError(ciErrNum, CL_SUCCESS);
            shrLog(LOGBOTH, 0, " Device %i: %s\n", i, cDeviceName);

            // create a command que
            commandQueue[i] = clCreateCommandQueue(cxGPUContext, cdDevice, 0, &ciErrNum);
            shrCheckError(ciErrNum, CL_SUCCESS);
            shrLog(LOGBOTH, 0, "clCreateCommandQueue\n\n"); 

            #ifdef GPU_PROFILING
                ciErrNum = clSetCommandQueueProperty(commandQueue[i], CL_QUEUE_PROFILING_ENABLE, CL_TRUE, NULL);
                shrCheckError(ciErrNum, CL_SUCCESS);
            #endif
        }
    }

    // Load the OpenCL source code from the .cl file 
    const char* source_path = shrFindFilePath("simpleMultiGPU.cl", argv[0]);
    char *source = oclLoadProgSource(source_path, "", &programLength);
    shrCheckError(source != NULL, shrTRUE);
    shrLog(LOGBOTH, 0, "oclLoadProgSource\n"); 

    // Create the program for all GPUs in the context
    cpProgram = clCreateProgramWithSource(cxGPUContext, 1, (const char **)&source, &programLength, &ciErrNum);
    shrCheckError(ciErrNum, CL_SUCCESS);
    shrLog(LOGBOTH, 0, "clCreateProgramWithSource\n"); 
    
    // build the program
    ciErrNum = clBuildProgram(cpProgram, 0, NULL, "-cl-mad-enable", NULL, NULL);
    if (ciErrNum != CL_SUCCESS)
    {
        // write out standard error, Build Log and PTX, then cleanup and exit
        shrLog(LOGBOTH | ERRORMSG, ciErrNum, STDERROR);
        oclLogBuildInfo(cpProgram, oclGetFirstDev(cxGPUContext));
        oclLogPtx(cpProgram, oclGetFirstDev(cxGPUContext), "oclSimpleMultiGPU.ptx");
        shrCheckError(ciErrNum, CL_SUCCESS); 
    }
    shrLog(LOGBOTH, 0, "clBuildProgram\n"); 

    // Create host buffer with page-locked memory
    h_DataBuffer = clCreateBuffer(cxGPUContext, CL_MEM_COPY_HOST_PTR | CL_MEM_ALLOC_HOST_PTR,
                                  DATA_N * sizeof(float), h_Data, &ciErrNum);
    shrCheckError(ciErrNum, CL_SUCCESS);
    shrLog(LOGBOTH, 0, "clCreateBuffer (Page-locked Host)\n\n"); 

    // Create buffers for each GPU, with data divided evenly among GPU's
    int sizePerGPU = DATA_N / ciDeviceCount;
    int workOffset[MAX_GPU_COUNT];
    int workSize[MAX_GPU_COUNT];
    workOffset[0] = 0;
    for(unsigned int i = 0; i < ciDeviceCount; ++i ) 
    {
        workSize[i] = (i != (ciDeviceCount - 1)) ? sizePerGPU : (DATA_N - workOffset[i]);        

        // Input buffer
        d_Data[i] = clCreateBuffer(cxGPUContext, CL_MEM_READ_ONLY, workSize[i] * sizeof(float), NULL, &ciErrNum);
        shrCheckError(ciErrNum, CL_SUCCESS);
        shrLog(LOGBOTH, 0, "clCreateBuffer (Input)\t\tDev %i\n", i); 

        // Copy data from host to device
        ciErrNum = clEnqueueCopyBuffer(commandQueue[i], h_DataBuffer, d_Data[i], workOffset[i] * sizeof(float), 
                                      0, workSize[i] * sizeof(float), 0, NULL, NULL);        
        shrLog(LOGBOTH, 0, "clEnqueueCopyBuffer (Input)\tDev %i\n", i);

        // Output buffer
        d_Result[i] = clCreateBuffer(cxGPUContext, CL_MEM_WRITE_ONLY, ACCUM_N * sizeof(float), NULL, &ciErrNum);
        shrCheckError(ciErrNum, CL_SUCCESS);
        shrLog(LOGBOTH, 0, "clCreateBuffer (Output)\t\tDev %i\n", i);
        
        // Create kernel
        reduceKernel[i] = clCreateKernel(cpProgram, "reduce", &ciErrNum);
        shrCheckError(ciErrNum, CL_SUCCESS);
        shrLog(LOGBOTH, 0, "clCreateKernel\t\t\tDev %i\n", i); 
        
        // Set the args values and check for errors
        ciErrNum |= clSetKernelArg(reduceKernel[i], 0, sizeof(cl_mem), &d_Result[i]);
        ciErrNum |= clSetKernelArg(reduceKernel[i], 1, sizeof(cl_mem), &d_Data[i]);
        ciErrNum |= clSetKernelArg(reduceKernel[i], 2, sizeof(int), &workSize[i]);
        shrCheckError(ciErrNum, CL_SUCCESS);
        shrLog(LOGBOTH, 0, "clSetKernelArg\t\t\tDev %i\n\n", i);

        workOffset[i + 1] = workOffset[i] + workSize[i];
    }

    // Set # of work items in work group and total in 1 dimensional range
    size_t localWorkSize[] = {THREAD_N};        
    size_t globalWorkSize[] = {ACCUM_N};        

    // Start timer and launch reduction kernel on each GPU, with data split between them 
    shrLog(LOGBOTH, 0, "Launching Kernels on GPU(s)...\n\n");
    for(unsigned int i = 0; i < ciDeviceCount; i++) 
    {        
        ciErrNum = clEnqueueNDRangeKernel(commandQueue[i], reduceKernel[i], 1, 0, globalWorkSize, localWorkSize,
                                         0, NULL, &GPUExecution[i]);
        shrCheckError(ciErrNum, CL_SUCCESS);
    }
    
    // Copy result from device to host for each device
    for(unsigned int i = 0; i < ciDeviceCount; i++) 
    {
        ciErrNum = clEnqueueReadBuffer(commandQueue[i], d_Result[i], CL_FALSE, 0, ACCUM_N * sizeof(float), 
                            h_SumGPU + i *  ACCUM_N, 0, NULL, &GPUDone[i]);
        shrCheckError(ciErrNum, CL_SUCCESS);
    }

    // Synchronize with the GPUs and do accumulated error check
    clWaitForEvents(ciDeviceCount, GPUDone);
    shrLog(LOGBOTH, 0, "clWaitForEvents complete...\n\n"); 

    // Aggregate results for multiple GPU's and stop/log processing time
    sumGPU = 0;
    for(unsigned int i = 0; i < ciDeviceCount * ACCUM_N; i++)
    {
         sumGPU += h_SumGPU[i];
    }

    // Print Execution Times for each GPU
    #ifdef GPU_PROFILING
        shrLog(LOGBOTH, 0, "Profiling Information for GPU Processing:\n\n");
        for(unsigned int i = 0; i < ciDeviceCount; i++) 
        {
	        cdDevice = oclGetDev(cxGPUContext, deviceNr[i]);
	        clGetDeviceInfo(cdDevice, CL_DEVICE_NAME, sizeof(cDeviceName), cDeviceName, NULL);
            shrLog(LOGBOTH, 0, "Device %i : %s\n", deviceNr[i], cDeviceName);
            shrLog(LOGBOTH, 0, "  Reduce Kernel     : %.5f s\n", executionTime(GPUExecution[i]));
            shrLog(LOGBOTH, 0, "  Copy Device->Host : %.5f s\n\n\n", executionTime(GPUDone[i]));
        }
    #endif

    // Run the computation on the Host CPU and log processing time 
    shrLog(LOGBOTH, 0, "Launching Host/CPU C++ Computation...\n\n");
    sumCPU = 0;
    for(unsigned int i = 0; i < DATA_N; i++)
    {
        sumCPU += h_Data[i];
    }

    // Check GPU result against CPU result 
    dRelError = 100.0 * fabs(sumCPU - sumGPU) / fabs(sumCPU);
    shrLog(LOGBOTH, 0, "Comparing against Host/C++ computation...\n"); 
    shrLog(LOGBOTH, 0, " GPU sum: %f\n CPU sum: %f\n", sumGPU, sumCPU);
    shrLog(LOGBOTH, 0, " Relative Error % (100.0 * Error / Golden) = %f \n\n", dRelError);
    shrLog(LOGBOTH, 0, "TEST %s\n\n", (dRelError < 1e-4) ? "PASSED" : "FAILED !!!");

    // cleanup 
    free(source);
    free(h_Data);
    for(unsigned int i = 0; i < ciDeviceCount; ++i ) 
    {
	    clReleaseKernel(reduceKernel[i]);
        clReleaseCommandQueue(commandQueue[i]);
    }
    clReleaseProgram(cpProgram);
    clReleaseContext(cxGPUContext);

    // finish
    shrEXIT(argc, argv);
  }
