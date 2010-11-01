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

/* Matrix-vector multiplication: W = M * V.
 * Host code.
 *
 * This sample implements matrix-vector multiplication.
 * It has been written for clarity of exposition to illustrate various OpenCL
 * programming principles and optimizatoins, not with the goal of providing
 * the most performant generic kernel for matrix-vector multiplication.
 *
 * CUBLAS provides high-performance matrix-vector multiplication on GPU.
 */

// standard utilities and systems includes
#include <oclUtils.h>

#ifndef _WIN32
    typedef uint64_t memsize_t;
#else
    typedef unsigned __int64 memsize_t;
#endif


// Name of the file with the source code for the computation kernel
// *********************************************************************
const char* cSourceFile = "oclMatVecMul.cl";

// Host buffers for demo
// *********************************************************************
float *M, *V, *W;               // Host buffers for M, V, and W
float* Golden;                  // Host buffer for host golden processing cross check

// OpenCL Vars
cl_context cxGPUContext;        // OpenCL context
cl_command_queue cqCommandQueue;// OpenCL command que
cl_device_id* cdDevices;        // OpenCL device list    
cl_uint targetDevice = 0;	// Device to compute on
cl_program cpProgram;           // OpenCL program
cl_kernel ckKernel;             // OpenCL kernel
cl_event ceEvent;               // OpenCL event
cl_mem cmM, cmV, cmW;           // OpenCL buffers for M, V, and W
size_t szGlobalWorkSize;        // Total # of work items in the 1D range
size_t szLocalWorkSize;         // # of work items in the 1D work group    
size_t szParmDataBytes;         // Byte size of context information
size_t szKernelLength;          // Byte size of kernel code
cl_int ciErrNum;                // Error code var
char* cPathAndName = NULL;      // var for full paths to data, src, etc.
char* cSourceCL = NULL;         // Buffer to hold source for compilation 

// demo config vars
int width = 1100;               // Matrix width
int height;             // Matrix height
shrBOOL bNoPrompt = shrFALSE;  

// Forward Declarations
// *********************************************************************
void MatVecMulHost(const float* M, const float* V, int width, int height, float* W);
bool getTargetDeviceGlobalMemSize(memsize_t* result, const int argc, const char **argv);
void Cleanup (int iExitCode);
void (*pCleanup)(int) = &Cleanup;

// Main function 
// *********************************************************************
int main(int argc, const char **argv)
{

    // get command line arg for quick test, if provided
    bNoPrompt = shrCheckCmdLineFlag(argc, (const char**)argv, "noprompt");

    // start logs
    shrSetLogFileName ("oclMatVecMul.txt");
    shrLog(LOGBOTH, 0, "%s Starting...\n\n", argv[0]); 

    // calculate matrix height given GPU memory
    shrLog(LOGBOTH, 0,  "Determining Matrix height from available GPU mem...\n");
    memsize_t memsize;
    getTargetDeviceGlobalMemSize(&memsize, argc, argv);
    height = memsize/width/16;
    shrLog(LOGBOTH, 0, " Matrix width\t= %u\n Matrix height\t= %u\n\n", width, height); 

    // Allocate and initialize host arrays
    shrLog(LOGBOTH, 0,  "Allocate and Init Host Mem...\n");
    unsigned int size = width * height;
    unsigned int mem_size_M = size * sizeof(float);
    M = (float*)malloc(mem_size_M);
    unsigned int mem_size_V = width * sizeof(float);
    V = (float*)malloc(mem_size_V);
    unsigned int mem_size_W = height * sizeof(float);
    W = (float*)malloc(mem_size_W);
    shrFillArray(M, size);
    shrFillArray(V, width);
    Golden = (float*)malloc(mem_size_W);
    MatVecMulHost(M, V, width, height, Golden);

    // Create the OpenCL context on a GPU device
    shrLog(LOGBOTH, 0, "clCreateContextFromType...\n"); 
    cxGPUContext = clCreateContextFromType(0, CL_DEVICE_TYPE_GPU, NULL, NULL, &ciErrNum);
    shrCheckErrorEX (ciErrNum, CL_SUCCESS, pCleanup);

    // Get the list of GPU devices associated with context
    shrLog(LOGBOTH, 0, "clGetContextInfo...\n"); 
    ciErrNum = clGetContextInfo(cxGPUContext, CL_CONTEXT_DEVICES, 0, NULL, &szParmDataBytes);
    shrCheckErrorEX (ciErrNum, CL_SUCCESS, pCleanup);
    cdDevices = (cl_device_id*)malloc(szParmDataBytes);
    ciErrNum = clGetContextInfo(cxGPUContext, CL_CONTEXT_DEVICES, szParmDataBytes, cdDevices, NULL);
    shrCheckErrorEX (ciErrNum, CL_SUCCESS, pCleanup);

    // Query number of compute units on device 0
    cl_uint num_compute_units;
    clGetDeviceInfo(cdDevices[targetDevice], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(num_compute_units), &num_compute_units, NULL);

    // Create a command-queue
    shrLog(LOGBOTH, 0, "clCreateCommandQueue...\n"); 
    cqCommandQueue = clCreateCommandQueue(cxGPUContext, cdDevices[targetDevice], 0, &ciErrNum);
    shrCheckErrorEX (ciErrNum, CL_SUCCESS, pCleanup);
#ifdef GPU_PROFILING
    ciErrNum = clSetCommandQueueProperty(cqCommandQueue, CL_QUEUE_PROFILING_ENABLE, CL_TRUE, NULL);
    shrCheckErrorEX (ciErrNum, CL_SUCCESS, pCleanup);
#endif

    // Allocate the OpenCL buffer memory objects for source and result on the device GMEM
    shrLog(LOGBOTH, 0, "clCreateBuffer (M, V and W in device global memory, mem_size_m = %u)...\n", mem_size_M); 
    cmM = clCreateBuffer(cxGPUContext, CL_MEM_READ_ONLY, mem_size_M, NULL, &ciErrNum);
    shrCheckErrorEX (ciErrNum, CL_SUCCESS, pCleanup);
    cmV = clCreateBuffer(cxGPUContext, CL_MEM_READ_ONLY, mem_size_V, NULL, &ciErrNum);
    shrCheckErrorEX (ciErrNum, CL_SUCCESS, pCleanup);
    cmW = clCreateBuffer(cxGPUContext, CL_MEM_WRITE_ONLY, mem_size_W, NULL, &ciErrNum);
    shrCheckErrorEX (ciErrNum, CL_SUCCESS, pCleanup);

    // Read the OpenCL kernel in from source file
    shrLog(LOGBOTH, 0, "oclLoadProgSource (%s)...\n", cSourceFile); 
    cPathAndName = shrFindFilePath(cSourceFile, argv[0]);
    shrCheckErrorEX (cPathAndName != NULL, shrTRUE, pCleanup);
    cSourceCL = oclLoadProgSource(cPathAndName, "", &szKernelLength);
    shrCheckErrorEX (cSourceCL != NULL, shrTRUE, pCleanup);

    // Create the program
    shrLog(LOGBOTH, 0, "clCreateProgramWithSource...\n"); 
    cpProgram = clCreateProgramWithSource(cxGPUContext, 1, (const char **)&cSourceCL, &szKernelLength, &ciErrNum);

    // Build the program
    shrLog(LOGBOTH, 0, "clBuildProgram...\n"); 
    ciErrNum = clBuildProgram(cpProgram, 0, NULL, "-cl-mad-enable", NULL, NULL);
    if (ciErrNum != CL_SUCCESS)
    {
        // write out standard error, Build Log and PTX, then cleanup and exit
        shrLog(LOGBOTH | ERRORMSG, ciErrNum, STDERROR);
        oclLogBuildInfo(cpProgram, oclGetFirstDev(cxGPUContext));
        oclLogPtx(cpProgram, oclGetFirstDev(cxGPUContext), "oclMatVecMul.ptx");
        Cleanup(EXIT_FAILURE); 
    }

    // --------------------------------------------------------
    // Core sequence... copy input data to GPU, compute, copy results back

    // Asynchronous write of data to GPU device
    shrLog(LOGBOTH, 0, "clEnqueueWriteBuffer (M and V)...\n\n"); 
    ciErrNum = clEnqueueWriteBuffer(cqCommandQueue, cmM, CL_FALSE, 0, mem_size_M, M, 0, NULL, NULL);
    ciErrNum |= clEnqueueWriteBuffer(cqCommandQueue, cmV, CL_FALSE, 0, mem_size_V, V, 0, NULL, NULL);
    shrCheckErrorEX (ciErrNum, CL_SUCCESS, pCleanup);

    // Kernels
    char* kernels[] = {
        "MatVecMulUncoalesced0",
        "MatVecMulUncoalesced1",
        "MatVecMulCoalesced0",
        "MatVecMulCoalesced1",
        "MatVecMulCoalesced2",
        "MatVecMulCoalesced3" };

    for (int k = 0; k < sizeof(kernels)/sizeof(char*); ++k) {

        // Clear result
        memset(W, 0, mem_size_W);
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue, cmW, CL_FALSE, 0, mem_size_W, W, 0, NULL, NULL);
        shrCheckErrorEX (ciErrNum, CL_SUCCESS, pCleanup);

        // Create the kernel
        shrLog(LOGBOTH, 0, "clCreateKernel (%s)...\n", kernels[k]); 
        if (ckKernel) {
            clReleaseKernel(ckKernel);
            ckKernel = 0;
        }
        ckKernel = clCreateKernel(cpProgram, kernels[k], &ciErrNum);
        shrCheckErrorEX (ciErrNum, CL_SUCCESS, pCleanup);

        // Set and log Global and Local work size dimensions
        szLocalWorkSize = 256;
        if (k == 0)
            szGlobalWorkSize = shrRoundUp((int)szLocalWorkSize, height);  // rounded up to the nearest multiple of the LocalWorkSize
        else
            // Some experiments should be done here for determining the best global work size for a given device
            // We will assume here that we can run 2 work-groups per compute unit
            szGlobalWorkSize = 2 * num_compute_units * szLocalWorkSize;
        shrLog(LOGBOTH, 0, "Global Work Size \t\t= %u\nLocal Work Size \t\t= %u\n# of Work Groups \t\t= %u\n", 
               szGlobalWorkSize, szLocalWorkSize, (szGlobalWorkSize % szLocalWorkSize + szGlobalWorkSize/szLocalWorkSize)); 

        // Set the Argument values
        shrLog(LOGBOTH, 0, "clSetKernelArg...\n\n");
        int n = 0;
        ciErrNum = clSetKernelArg(ckKernel,  n++, sizeof(cl_mem), (void*)&cmM);
        ciErrNum |= clSetKernelArg(ckKernel, n++, sizeof(cl_mem), (void*)&cmV);
        ciErrNum |= clSetKernelArg(ckKernel, n++, sizeof(cl_int), (void*)&width);
        ciErrNum |= clSetKernelArg(ckKernel, n++, sizeof(cl_int), (void*)&height);
        ciErrNum |= clSetKernelArg(ckKernel, n++, sizeof(cl_mem), (void*)&cmW);
        if (k > 1)
            ciErrNum |= clSetKernelArg(ckKernel, n++, szLocalWorkSize * sizeof(float), 0);    
        shrCheckErrorEX (ciErrNum, CL_SUCCESS, pCleanup);

        // Launch kernel
        shrLog(LOGBOTH, 0, "clEnqueueNDRangeKernel (%s)...\n", kernels[k]); 
        ciErrNum = clEnqueueNDRangeKernel(cqCommandQueue, ckKernel, 1, NULL, &szGlobalWorkSize, &szLocalWorkSize, 0, NULL, &ceEvent);
        shrCheckErrorEX (ciErrNum, CL_SUCCESS, pCleanup);

        // Read back results and check accumulated errors
        shrLog(LOGBOTH, 0, "clEnqueueReadBuffer (W)...\n"); 
        ciErrNum = clEnqueueReadBuffer(cqCommandQueue, cmW, CL_TRUE, 0, mem_size_W, W, 0, NULL, NULL);
        shrCheckErrorEX (ciErrNum, CL_SUCCESS, pCleanup);

    #ifdef GPU_PROFILING
        // Execution time
        ciErrNum = clWaitForEvents(1, &ceEvent);
        shrCheckErrorEX (ciErrNum, CL_SUCCESS, pCleanup);
        cl_ulong start, end;
        ciErrNum = clGetEventProfilingInfo(ceEvent, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
        ciErrNum |= clGetEventProfilingInfo(ceEvent, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
        shrCheckErrorEX (ciErrNum, CL_SUCCESS, pCleanup);
        double dSeconds = 1.0e-9 * (double)(end - start);
        shrLog(LOGBOTH, 0, "Kernel execution time: %.5f s\n\n", dSeconds);
    #endif

        // Compare results for golden-host and report errors and pass/fail
        shrLog(LOGBOTH, 0, "Comparing against Host/C++ computation...\n\n"); 
        shrBOOL res = shrCompareL2fe(Golden, W, height, 1e-6f);
        shrLog(LOGBOTH, 0, "TEST %s\n\n", (res == shrTRUE) ? "PASSED" : "FAILED !!!");

        // Release event
        ciErrNum = clReleaseEvent(ceEvent);
        shrCheckErrorEX (ciErrNum, CL_SUCCESS, pCleanup);
        ceEvent = 0;
    }

    // Cleanup and leave
    Cleanup (EXIT_SUCCESS);
}

// "Golden" Host processing matrix vector multiplication function for comparison purposes
// *********************************************************************
void MatVecMulHost(const float* M, const float* V, int width, int height, float* W)
{
    for (int i = 0; i < height; ++i) {
        double sum = 0;
        for (int j = 0; j < width; ++j) {
            double a = M[i * width + j];
            double b = V[j];
            sum += a * b;
        }
        W[i] = (float)sum;
    }
}

// Cleanup and exit code
// *********************************************************************
void Cleanup(int iExitCode)
{
    // Cleanup allocated objects
    shrLog(LOGBOTH, 0, "Starting Cleanup...\n\n");
    if(cdDevices)free(cdDevices);
    if(cPathAndName)free(cPathAndName);
    if(cSourceCL)free(cSourceCL);
    if(ceEvent)clReleaseEvent(ceEvent);  
    if(ckKernel)clReleaseKernel(ckKernel);  
    if(cpProgram)clReleaseProgram(cpProgram);
    if(cqCommandQueue)clReleaseCommandQueue(cqCommandQueue);
    if(cxGPUContext)clReleaseContext(cxGPUContext);
    if (cmM)clReleaseMemObject(cmM);
    if (cmV)clReleaseMemObject(cmV);
    if (cmW)clReleaseMemObject(cmW);

    // Free host memory
    free(M); 
    free(V);
    free(W);
    free(Golden);

    // finalize logs and leave
    if (bNoPrompt)
    {
        shrLog(LOGBOTH | CLOSELOG, 0, "oclMatVecMul.exe Exiting...\n");
    }
    else 
    {
        shrLog(LOGBOTH | CLOSELOG, 0, "oclMatVecMul.exe Exiting...\nPress <Enter> to Quit\n");
        getchar();
    }
    exit (iExitCode);
}

bool getTargetDeviceGlobalMemSize(memsize_t* result, const int argc, const char **argv)
{
    bool ok = true;
    cl_context        context      = 0;
    cl_device_id     *devices      = 0;
    cl_uint           deviceCount  = 0;
    cl_ulong          memsize      = 0;
    cl_int            errnum       = 0;

    // Create the OpenCL context
    if (ok)
    {
        shrLog(LOGBOTH, 0, " clCreateContextFromType...\n");
        context = clCreateContextFromType(0, CL_DEVICE_TYPE_GPU, NULL, NULL, &errnum);
        if (context == (cl_context)0) 
        {
            shrLog(LOGBOTH | ERRORMSG, errnum, "clCreateContextFromType (returned %d).\n", errnum);
            ok = false;
        }
    }

    // Get the list of GPU devices associated with the context
    if (ok)
    {
        shrLog(LOGBOTH, 0, " clGetContextInfo...\n"); 
        size_t szParmDataBytes;
        clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &szParmDataBytes);
        devices = (cl_device_id *)malloc(szParmDataBytes);
        clGetContextInfo(context, CL_CONTEXT_DEVICES, szParmDataBytes, devices, NULL);
        deviceCount = (cl_uint)(szParmDataBytes / sizeof(cl_device_id));
        if (deviceCount == 0)
        {
            shrLog(LOGBOTH | ERRORMSG, 0, "clGetContextInfo (no devices found).\n");
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
                shrLog(LOGBOTH | ERRORMSG, 0, "invalid target device specified on command line (device %d does not exist).\n", targetDevice);
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
        shrLog(LOGBOTH, 0, " clGetDeviceInfo...\n"); 
        errnum = clGetDeviceInfo(devices[targetDevice], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &memsize, NULL);
        if (errnum != CL_SUCCESS)
        {
            shrLog(LOGBOTH | ERRORMSG, errnum, "clGetDeviceInfo (returned %d).\n", errnum);
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
