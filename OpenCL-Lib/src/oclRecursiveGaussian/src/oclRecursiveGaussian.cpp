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

// Standard utilities and system includes, plus project specific items
//*****************************************************************************
#include "oclRecursiveGaussian.h"

// GLUT includes
#if defined (__APPLE__) || defined(MACOSX)
    #include <OpenGL/OpenGL.h>
    #include <GLUT/glut.h>
#else
    #include <GL/glut.h>
#endif

// Defines and globals for recursive gaussian processing demo
//*****************************************************************************
float fSigma = 10.0f;               // filter sigma (blur factor)
int iOrder = 0;                     // filter order
int BLOCK_DIM = 16;
int iNumThreads = 16;	            // number of threads per block for Gaussian

// Image data vars
const char* cImageFile = "StoneRGB.ppm";
unsigned int uiImageWidth = 1920;   // Image width
unsigned int uiImageHeight = 1080;  // Image height
unsigned int* uiInput = NULL;       // Host buffer to hold input image data
unsigned int* uiTemp = NULL;        // Host buffer to hold intermediate image data
unsigned int* uiOutput = NULL;      // Host buffer to hold output image data

// OpenGL, Display Window and GUI Globals
int iGLUTWindowHandle;              // handle to the GLUT window
int iGLUTMenuHandle;                // handle to the GLUT menu
int iGraphicsWinPosX = 0;           // GL Window X location
int iGraphicsWinPosY = 0;           // GL Window Y location
int iGraphicsWinWidth = 1024;       // GL Window width
int iGraphicsWinHeight = ((float)uiImageHeight / (float)uiImageWidth) * iGraphicsWinWidth;  // GL Windows height
float fZoom = 1.0f;                 // pixel display zoom   
int iFrameCount = 0;                // FPS count for averaging
int iFrameTrigger = 90;             // FPS trigger for sampling
int iFramesPerSec = 45;             // frames per second
bool bFullScreen = false;           // state var for full screen mode or not
bool bFilter = true;                // state var for whether filter is enaged or not

// app configuration parms
const char* cProcessor [] = {"OpenCL GPU", "Host C++ CPU"};
int iProcFlag = 0;                  // 0 = GPU, 1 = CPU
shrBOOL bNoPrompt = shrFALSE;		// false = normal GL loop, true = Finite period of GL loop (a few seconds)
shrBOOL bQATest = shrFALSE;			// false = normal GL loop, true = run No-GL test sequence
int iTestSets = 3;                  // # of loop set retriggers before auto exit when bNoPrompt = shrTrue  

// OpenCL vars
const char* clSourcefile = "RecursiveGaussian.cl";  // OpenCL kernel source file
char* cPathAndName = NULL;          // var for full paths to data, src, etc.
char* cSourceCL;                    // Buffer to hold source for compilation 
cl_context cxGPUContext;            // OpenCL context
cl_command_queue cqCommandQueue;    // OpenCL command que
cl_device_id* cdDevices;            // OpenCL device list
cl_uint uiNumDevices;               // Number of devices available/used
cl_program cpProgram;               // OpenCL program
cl_kernel ckSimpleRecursiveRGBA;    // OpenCL Kernel for simple recursion
cl_kernel ckRecursiveGaussianRGBA;  // OpenCL Kernel for gaussian recursion
cl_kernel ckTranspose;              // OpenCL for transpose
cl_mem cmDevBufIn;                  // OpenCL device memory input buffer object
cl_mem cmDevBufTemp;                // OpenCL device memory temp buffer object
cl_mem cmDevBufOut;                 // OpenCL device memory output buffer object
size_t szBuffBytes;                 // Size of main image buffers
size_t szGaussGlobalWork[1];        // global # of work items in single dimensional range
size_t szGaussLocalWork[1];         // work group # of work items in single dimensional range
size_t szTransposeGlobalWork[2];    // global # of work items in 2 dimensional range
size_t szTransposeLocalWork[2];     // work group # of work items in a 2 dimensional range
size_t szParmDataBytes;			    // Byte size of context information
size_t szKernelLength;			    // Byte size of kernel code
cl_int ciErrNum;		            // Error code var

// Forward Function declarations
//*****************************************************************************
// OpenCL functionality
void GPUGaussianFilterRGBA(GaussParms* pGP);
void GPUGaussianSetCommonArgs(GaussParms* pGP);

// OpenGL functionality
void InitGL(int argc, const char** argv);
void DisplayGL();
void Reshape(int w, int h);
void Idle(void);
void KeyboardGL(unsigned char key, int x, int y);
void MenuGL(int i);

// Helpers
void TestNoGL();
void TriggerFPSUpdate();
void Cleanup(int iExitCode);
void (*pCleanup)(int) = &Cleanup;

// Main program
//*****************************************************************************
int main(int argc, const char** argv)
{
    // Start logs 
    shrSetLogFileName ("oclRecursiveGaussian.txt");
    shrLog(LOGBOTH, 0, "%s Starting (using %s)...\n\n", argv[0], clSourcefile); 

    // Get command line args for quick test or QA test, if provided
    bNoPrompt = shrCheckCmdLineFlag(argc, argv, "noprompt");
    bQATest = shrCheckCmdLineFlag(argc, argv, "qatest");

    // Find the path from the exe to the image file and load the image
    cPathAndName = shrFindFilePath(cImageFile, argv[0]);
    shrCheckErrorEX (cPathAndName != NULL, shrTRUE, pCleanup);
    ciErrNum = shrLoadPPM4ub(cPathAndName, (unsigned char **)&uiInput, &uiImageWidth, &uiImageHeight);
    shrCheckErrorEX (ciErrNum, shrTRUE, pCleanup);
    shrLog(LOGBOTH, 0, "Image Width = %i, Height = %i, bpp = %i\n\n", uiImageWidth, uiImageHeight, sizeof(unsigned int)<<3);

    // Allocate intermediate and output host image buffers
    szBuffBytes = uiImageWidth * uiImageHeight * sizeof (unsigned int);
    uiTemp = (unsigned int*)malloc(szBuffBytes);
    uiOutput = (unsigned int*)malloc(szBuffBytes);
    shrLog(LOGBOTH, 0, "Allocate Host Image Buffers...\n"); 

    // Initialize OpenGL items (if not No-GL QA test)
	if (!(bQATest))
	{
		InitGL(argc, argv);
	}
	shrLog(LOGBOTH, 0, "%sInitGL...\n", bQATest ? "Skipping " : "Calling "); 

    // Create the OpenCL context on a GPU device
    cxGPUContext = clCreateContextFromType(0, CL_DEVICE_TYPE_GPU, NULL, NULL, &ciErrNum);
    shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    shrLog(LOGBOTH, 0, "clCreateContextFromType...\n"); 

    // Get the list of GPU devices associated with context
    ciErrNum = clGetContextInfo(cxGPUContext, CL_CONTEXT_DEVICES, 0, NULL, &szParmDataBytes);
    cdDevices = (cl_device_id *)malloc(szParmDataBytes);
    ciErrNum |= clGetContextInfo(cxGPUContext, CL_CONTEXT_DEVICES, szParmDataBytes, cdDevices, NULL);
    shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    uiNumDevices = (cl_uint)szParmDataBytes/sizeof(cl_device_id);
    shrLog(LOGBOTH, 0, "clGetContextInfo...\n\n"); 

    // List used device 
    shrLog(LOGBOTH, 0, "GPU Device being used:\n"); 
    oclPrintDevInfo(LOGBOTH, cdDevices[0]);

    // Create a command-queue
    cqCommandQueue = clCreateCommandQueue(cxGPUContext, cdDevices[0], 0, &ciErrNum);
    shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    shrLog(LOGBOTH, 0, "clCreateCommandQueue...\n"); 

    // Allocate the OpenCL source, intermediate and result buffer memory objects on the device GMEM
    cmDevBufIn = clCreateBuffer(cxGPUContext, CL_MEM_READ_ONLY, szBuffBytes, NULL, &ciErrNum);
    shrCheckErrorEX (ciErrNum, CL_SUCCESS, pCleanup);
    cmDevBufTemp = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, szBuffBytes, NULL, &ciErrNum);
    shrCheckErrorEX (ciErrNum, CL_SUCCESS, pCleanup);
    cmDevBufOut = clCreateBuffer(cxGPUContext, CL_MEM_WRITE_ONLY, szBuffBytes, NULL, &ciErrNum);
    shrCheckErrorEX (ciErrNum, CL_SUCCESS, pCleanup);
    shrLog(LOGBOTH, 0, "clCreateBuffer (Input and Output buffers, device GMEM)...\n"); 

    // Read the OpenCL kernel source in from file
    free(cPathAndName);
    cPathAndName = shrFindFilePath(clSourcefile, argv[0]);
    shrCheckErrorEX(cPathAndName != NULL, shrTRUE, pCleanup);
    cSourceCL = oclLoadProgSource(cPathAndName, "// My comment\n", &szKernelLength);
    shrCheckErrorEX(cSourceCL != NULL, shrTRUE, pCleanup);
    shrLog(LOGBOTH, 0, "oclLoadProgSource...\n"); 

    // Create the program 
    cpProgram = clCreateProgramWithSource(cxGPUContext, 1, (const char **)&cSourceCL, &szKernelLength, &ciErrNum);
    shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    shrLog(LOGBOTH, 0, "clCreateProgramWithSource...\n"); 

    // Build the program with 'mad' Optimization option
    ciErrNum = clBuildProgram(cpProgram, 0, NULL, "-cl-mad-enable", NULL, NULL);
    if (ciErrNum != CL_SUCCESS)
    {
        // On error: write out standard error, Build Log and PTX, then cleanup and exit
        shrLog(LOGBOTH | ERRORMSG, ciErrNum, STDERROR);
        oclLogBuildInfo(cpProgram, oclGetFirstDev(cxGPUContext));
        oclLogPtx(cpProgram, oclGetFirstDev(cxGPUContext), "oclRecursiveGaussian.ptx");
        Cleanup(EXIT_FAILURE);
    }
    shrLog(LOGBOTH, 0, "clBuildProgram\n"); 

    // Create kernels
    ckSimpleRecursiveRGBA = clCreateKernel(cpProgram, "SimpleRecursiveRGBA", &ciErrNum);
    shrCheckErrorEX (ciErrNum, CL_SUCCESS, pCleanup);
    ckRecursiveGaussianRGBA = clCreateKernel(cpProgram, "RecursiveGaussianRGBA", &ciErrNum);
    shrCheckErrorEX (ciErrNum, CL_SUCCESS, pCleanup);
    ckTranspose = clCreateKernel(cpProgram, "Transpose", &ciErrNum);
    shrCheckErrorEX (ciErrNum, CL_SUCCESS, pCleanup);
    shrLog(LOGBOTH, 0, "clCreateKernel (Rows, Columns, Transpose)\n\n"); 

    // check work group size
    size_t wgSize;
    ciErrNum = clGetKernelWorkGroupInfo(ckTranspose, cdDevices[0], CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &wgSize, NULL);
    if (wgSize == 64) 
      BLOCK_DIM = 8;

    // Set unchanging local work sizes for gaussian kernels and transpose kernel
    szGaussLocalWork[0] = iNumThreads;
    szTransposeLocalWork[0] = BLOCK_DIM;
    szTransposeLocalWork[1] = BLOCK_DIM;

    // init filter coefficients
    PreProcessGaussParms (fSigma, iOrder, &oclGP);

    // set common kernel args
    GPUGaussianSetCommonArgs (&oclGP);

    // init timers
    shrDeltaT(0);   // timer 0 used for function timing 
    shrDeltaT(1);   // timer 1 used for fps computation

    // Start main GLUT rendering loop for processing and rendering, 
	// or otherwise run No-GL Q/A test sequence
	if (!(bQATest))
	{
		glutMainLoop();
	}
	else 
	{
		TestNoGL();
	}

    // Normally unused return path
    Cleanup(EXIT_FAILURE);
}

// Function to set common kernel args that only change outside of GLUT loop
//*****************************************************************************
void GPUGaussianSetCommonArgs(GaussParms* pGP)
{
    // common Gaussian args
    #if USE_SIMPLE_FILTER
        // Set the Common Argument values for the simple Gaussian kernel
        ciErrNum |= clSetKernelArg(ckSimpleRecursiveRGBA, 4, sizeof(float), (void*)&pGP->ema);
        shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

    #else
        // Set the Common Argument values for the Gaussian kernel
        ciErrNum |= clSetKernelArg(ckRecursiveGaussianRGBA, 4, sizeof(float), (void*)&pGP->a0);
        ciErrNum |= clSetKernelArg(ckRecursiveGaussianRGBA, 5, sizeof(float), (void*)&pGP->a1);
        ciErrNum |= clSetKernelArg(ckRecursiveGaussianRGBA, 6, sizeof(float), (void*)&pGP->a2);
        ciErrNum |= clSetKernelArg(ckRecursiveGaussianRGBA, 7, sizeof(float), (void*)&pGP->a3);
        ciErrNum |= clSetKernelArg(ckRecursiveGaussianRGBA, 8, sizeof(float), (void*)&pGP->b1);
        ciErrNum |= clSetKernelArg(ckRecursiveGaussianRGBA, 9, sizeof(float), (void*)&pGP->b2);
        ciErrNum |= clSetKernelArg(ckRecursiveGaussianRGBA, 10, sizeof(float), (void*)&pGP->coefp);
        ciErrNum |= clSetKernelArg(ckRecursiveGaussianRGBA, 11, sizeof(float), (void*)&pGP->coefn);
        shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

     #endif

    // Set common transpose Argument values 
    ciErrNum |= clSetKernelArg(ckTranspose, 4, sizeof(unsigned int) * BLOCK_DIM * (BLOCK_DIM+1), NULL );
    shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
}

// 8-bit RGBA Gaussian filter for GPU on a 2D image using OpenCL
//*****************************************************************************
void GPUGaussianFilterRGBA(GaussParms* pGP)
{
    // Copy input data from host to device
    ciErrNum = clEnqueueWriteBuffer(cqCommandQueue, cmDevBufIn, CL_TRUE, 0, szBuffBytes, uiInput, 0, NULL, NULL);
    shrCheckErrorEX (ciErrNum, CL_SUCCESS, pCleanup);

    // Set Gaussian global work dimensions, then set variable args and process in 1st dimension
    szGaussGlobalWork[0] = shrRoundUp((int)szGaussLocalWork[0], uiImageWidth); 
    #if USE_SIMPLE_FILTER
        // Set simple Gaussian kernel variable arg values
        ciErrNum = clSetKernelArg(ckSimpleRecursiveRGBA, 0, sizeof(cl_mem), (void*)&cmDevBufIn);
        ciErrNum |= clSetKernelArg(ckSimpleRecursiveRGBA, 1, sizeof(cl_mem), (void*)&cmDevBufTemp);
        ciErrNum |= clSetKernelArg(ckSimpleRecursiveRGBA, 2, sizeof(unsigned int), (void*)&uiImageWidth);
        ciErrNum |= clSetKernelArg(ckSimpleRecursiveRGBA, 3, sizeof(unsigned int), (void*)&uiImageHeight);
        shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

        // Launch simple Gaussian kernel on the data in one dimension
        ciErrNum = clEnqueueNDRangeKernel(cqCommandQueue, ckSimpleRecursiveRGBA, 1, NULL, szGaussGlobalWork, szGaussLocalWork, 0, NULL, NULL);
        shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    #else
        // Set full Gaussian kernel variable arg values
        ciErrNum = clSetKernelArg(ckRecursiveGaussianRGBA, 0, sizeof(cl_mem), (void*)&cmDevBufIn);
        ciErrNum |= clSetKernelArg(ckRecursiveGaussianRGBA, 1, sizeof(cl_mem), (void*)&cmDevBufTemp);
        ciErrNum |= clSetKernelArg(ckRecursiveGaussianRGBA, 2, sizeof(unsigned int), (void*)&uiImageWidth);
        ciErrNum |= clSetKernelArg(ckRecursiveGaussianRGBA, 3, sizeof(unsigned int), (void*)&uiImageHeight);
        shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

        // Launch full Gaussian kernel on the data in one dimension
        ciErrNum = clEnqueueNDRangeKernel(cqCommandQueue, ckRecursiveGaussianRGBA, 1, NULL, szGaussGlobalWork, szGaussLocalWork, 0, NULL, NULL);
        shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
     #endif

    // Set transpose global work dimensions and variable args 
    szTransposeGlobalWork[0] = shrRoundUp((int)szTransposeLocalWork[0], uiImageWidth); 
    szTransposeGlobalWork[1] = shrRoundUp((int)szTransposeLocalWork[1], uiImageHeight); 
    ciErrNum = clSetKernelArg(ckTranspose, 0, sizeof(cl_mem), (void*)&cmDevBufTemp);
    ciErrNum |= clSetKernelArg(ckTranspose, 1, sizeof(cl_mem), (void*)&cmDevBufOut);
    ciErrNum |= clSetKernelArg(ckTranspose, 2, sizeof(unsigned int), (void*)&uiImageWidth);
    ciErrNum |= clSetKernelArg(ckTranspose, 3, sizeof(unsigned int), (void*)&uiImageHeight);
    shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

    // Launch transpose kernel in 1st direction
    ciErrNum = clEnqueueNDRangeKernel(cqCommandQueue, ckTranspose, 2, NULL, szTransposeGlobalWork, szTransposeLocalWork, 0, NULL, NULL);
    shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

    // Reset Gaussian global work dimensions and variable args, then process in 2nd dimension
    // note width and height parameters flipped due to transpose
    szGaussGlobalWork[0] = shrRoundUp((int)szGaussLocalWork[0], uiImageHeight); 
    #if USE_SIMPLE_FILTER
        // set simple Gaussian kernel arg values
        ciErrNum = clSetKernelArg(ckSimpleRecursiveRGBA, 0, sizeof(cl_mem), (void*)&cmDevBufOut);
        ciErrNum |= clSetKernelArg(ckSimpleRecursiveRGBA, 1, sizeof(cl_mem), (void*)&cmDevBufTemp);
        ciErrNum |= clSetKernelArg(ckSimpleRecursiveRGBA, 2, sizeof(unsigned int), (void*)&uiImageHeight);
        ciErrNum |= clSetKernelArg(ckSimpleRecursiveRGBA, 3, sizeof(unsigned int), (void*)&uiImageWidth);
        shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

        // Launch simple Gaussian kernel on the data in the other dimension
        ciErrNum = clEnqueueNDRangeKernel(cqCommandQueue, ckSimpleRecursiveRGBA, 1, NULL, szGaussGlobalWork, szGaussLocalWork, 0, NULL, NULL);
        shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    #else
        // Set full Gaussian kernel arg values
        ciErrNum = clSetKernelArg(ckRecursiveGaussianRGBA, 0, sizeof(cl_mem), (void*)&cmDevBufOut);
        ciErrNum |= clSetKernelArg(ckRecursiveGaussianRGBA, 1, sizeof(cl_mem), (void*)&cmDevBufTemp);
        ciErrNum |= clSetKernelArg(ckRecursiveGaussianRGBA, 2, sizeof(unsigned int), (void*)&uiImageHeight);
        ciErrNum |= clSetKernelArg(ckRecursiveGaussianRGBA, 3, sizeof(unsigned int), (void*)&uiImageWidth);
        shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
 
        // Launch full Gaussian kernel on the data in the other dimension
        ciErrNum = clEnqueueNDRangeKernel(cqCommandQueue, ckRecursiveGaussianRGBA, 1, NULL, szGaussGlobalWork, szGaussLocalWork, 0, NULL, NULL);
        shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
     #endif

    // Reset transpose global work dimensions and variable args 
    // note width and height parameters flipped due to 1st transpose
    szTransposeGlobalWork[0] = shrRoundUp((int)szTransposeLocalWork[0], uiImageHeight); 
    szTransposeGlobalWork[1] = shrRoundUp((int)szTransposeLocalWork[1], uiImageWidth); 
    ciErrNum = clSetKernelArg(ckTranspose, 0, sizeof(cl_mem), (void*)&cmDevBufTemp);
    ciErrNum |= clSetKernelArg(ckTranspose, 1, sizeof(cl_mem), (void*)&cmDevBufOut);
    ciErrNum |= clSetKernelArg(ckTranspose, 2, sizeof(unsigned int), (void*)&uiImageHeight);
    ciErrNum |= clSetKernelArg(ckTranspose, 3, sizeof(unsigned int), (void*)&uiImageWidth);
    shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

    // Launch transpose kernel in 2nd direction
    ciErrNum = clEnqueueNDRangeKernel(cqCommandQueue, ckTranspose, 2, NULL, szTransposeGlobalWork, szTransposeLocalWork, 0, NULL, NULL);
    shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

    // Copy results back to host, block until complete
    ciErrNum = clEnqueueReadBuffer(cqCommandQueue, cmDevBufOut, CL_TRUE, 0, szBuffBytes, uiOutput, 0, NULL, NULL);
    shrCheckErrorEX (ciErrNum, CL_SUCCESS, pCleanup);

    return;
}

// Initialize GL
//*****************************************************************************
void InitGL(int argc, const char **argv)
{
    // init GLUT and GLUT window
    glutInit(&argc, (char**)argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowPosition (glutGet(GLUT_SCREEN_WIDTH)/2 - iGraphicsWinWidth/2, 
                            glutGet(GLUT_SCREEN_HEIGHT)/2 - iGraphicsWinHeight/2);
    glutInitWindowSize(iGraphicsWinWidth, iGraphicsWinHeight);
    iGLUTWindowHandle = glutCreateWindow("OpenCL for GPU Recursive Gaussian");
#if defined (__APPLE__) || defined(MACOSX)
    long VBL = 0;
    CGLSetParameter(CGLGetCurrentContext(), kCGLCPSwapInterval, &VBL); // turn off Vsync
#endif

    // register GLUT callbacks
    glutKeyboardFunc(KeyboardGL);
    glutDisplayFunc(DisplayGL);
    glutReshapeFunc(Reshape);
    glutIdleFunc(Idle);

    // create GLUT menu
    iGLUTMenuHandle = glutCreateMenu(MenuGL);
    glutAddMenuEntry("Toggle Filter On/Off <spacebar>", ' ');
    glutAddMenuEntry("Toggle Processing between GPU and CPU [p]", 'p');
    glutAddMenuEntry("Toggle between Full Screen and Windowed [f]", 'f');
    glutAddMenuEntry("Increment Sigma [+]", '+');
    glutAddMenuEntry("Decrement Sigma [-]", '-');
    glutAddMenuEntry("Set Order 0 [0]", '0');
    glutAddMenuEntry("Set Order 1 [1]", '1');
    glutAddMenuEntry("Set Order 2 [2]", '2');
    glutAddMenuEntry("Quit <esc>", '\033');
    glutAttachMenu(GLUT_RIGHT_BUTTON);

    // Show help info, start timers
    shrLog(LOGBOTH, 0, "  <Right Click Mouse Button> for Menu\n\n"); 
    shrLog(LOGBOTH, 0, "Press:\n\n  <+> or <-> to change Sigma\n\n  \'0\', \'1\', or \'2\' keys to set Order\n\n");
    shrLog(LOGBOTH, 0, "  <spacebar> to toggle Filter On/Off\n\n  \'F\' key to toggle FullScreen On/Off\n\n");
    shrLog(LOGBOTH, 0, "  \'P\' key to toggle Processing between GPU and CPU\n\n  <esc> to Quit\n\n\n"); 

    // Zoom with fixed aspect ratio
    float fAspects[2] = {(float)glutGet(GLUT_WINDOW_WIDTH)/(float)uiImageWidth , (float)glutGet(GLUT_WINDOW_HEIGHT)/(float)uiImageHeight};
    fZoom = fAspects[0] > fAspects[1] ? fAspects[1] : fAspects[0];
    glPixelZoom(fZoom, fZoom);
}

// Display callback for GLUT main loop
//*****************************************************************************
void DisplayGL()
{
    double dProcessingTime = 0.0;
    if (bFilter)
    {
        // start timer 0 if it's update time
        if (iFrameCount >= iFrameTrigger)
        {
            shrDeltaT(0); 
        }

        // process on the GPU or the Host, depending on user toggle/flag
        if (iProcFlag == 0)
        {
            GPUGaussianFilterRGBA(&oclGP);
        }
        else 
        {
            HostRecursiveGaussianRGBA(uiInput, uiTemp, uiOutput, uiImageWidth, uiImageHeight, &oclGP);
        }

        // get processing time from timer 0, if it's update time
        if (iFrameCount >= iFrameTrigger)
        {
            dProcessingTime = shrDeltaT(0); 
        }

        // Draw processed image
        glDrawPixels(uiImageWidth, uiImageHeight, GL_RGBA, GL_UNSIGNED_BYTE, uiOutput); 
    }
    else 
    {
        // Skip processing and draw the raw input image
        glDrawPixels(uiImageWidth, uiImageHeight, GL_RGBA, GL_UNSIGNED_BYTE, uiInput); 
    }

    //  Flip backbuffer to screen
    glutSwapBuffers();
    glutPostRedisplay();

    // Increment the frame counter, and do fps stuff if it's time
    if (iFrameCount++ > iFrameTrigger)
    {
        // Set the display window title
        char cTitle[256];
        iFramesPerSec = (int)((double)iFrameCount/shrDeltaT(1));
#ifdef GPU_PROFILING
        if (bFilter)
        {
            if (!USE_SIMPLE_FILTER)
            {
                #ifdef _WIN32
                    sprintf_s(cTitle, 256, "%s Recursive Gaussian | W: %u  H: %u | Full Filter, Order %i, Sigma %.1f | %i fps | Proc. t = %.4f s | %.3f Mpix/s", 
                            cProcessor[iProcFlag], uiImageWidth, uiImageHeight, iOrder, fSigma, 
                            iFramesPerSec, dProcessingTime, (1.0e-6 * uiImageWidth * uiImageHeight)/dProcessingTime);  
                #else
                    sprintf(cTitle, "%s Recursive Gaussian | W: %u  H: %u | Full Filter, Order %i, Sigma %.1f | %i fps | Proc. t = %.4f s | %.3f Mpix/s", 
                            cProcessor[iProcFlag], uiImageWidth, uiImageHeight, iOrder, fSigma, 
                            iFramesPerSec, dProcessingTime, (1.0e-6 * uiImageWidth * uiImageHeight)/dProcessingTime);  
                #endif
            }
            else 
            {
                #ifdef _WIN32
                    sprintf_s(cTitle, 256, "%s Recursive Gaussian | W: %u  H: %u | Simple Filter, Sigma %.1f | %i fps | Proc. t = %.4f s | %.3f Mpix/s",  
                            cProcessor[iProcFlag], uiImageWidth, uiImageHeight, fSigma, 
                            iFramesPerSec, dProcessingTime, (1.0e-6 * uiImageWidth * uiImageHeight)/dProcessingTime);  
                #else
                        sprintf(cTitle, "%s Recursive Gaussian | W: %u  H: %u | Simple Filter, Sigma %.1f | %i fps | Proc. t = %.4f s | %.3f Mpix/s", 
                            cProcessor[iProcFlag], uiImageWidth, uiImageHeight, fSigma, 
                            iFramesPerSec, dProcessingTime, (1.0e-6 * uiImageWidth * uiImageHeight)/dProcessingTime);  
                #endif
            }
        }
        else 
        {
            #ifdef _WIN32
                sprintf_s(cTitle, 256, "Recursive Gaussian OFF | W: %u  H: %u | %i fps", 
	                        uiImageWidth, uiImageHeight, iFramesPerSec);  
            #else
                sprintf(cTitle, "Recursive Gaussian OFF | W: %u  H: %u | %i fps", 
	                        uiImageWidth, uiImageHeight, iFramesPerSec);  
            #endif
        }
#else
        if (bFilter)
        {
            if (!USE_SIMPLE_FILTER)
            {
                #ifdef _WIN32
                    sprintf_s(cTitle, 256, "%s Recursive Gaussian | W: %u  H: %u | Full Filter, Order %i, Sigma %.1f", 
                            cProcessor[iProcFlag], uiImageWidth, uiImageHeight, iOrder, fSigma);  
                #else
                    sprintf(cTitle, "%s Recursive Gaussian | W: %u  H: %u | Full Filter, Order %i, Sigma %.1f", 
                            cProcessor[iProcFlag], uiImageWidth, uiImageHeight, iOrder, fSigma);  
                #endif
            }
            else 
            {
                #ifdef _WIN32
                    sprintf_s(cTitle, 256, "%s Recursive Gaussian | W: %u  H: %u | Simple Filter, Sigma %.1f",  
                            cProcessor[iProcFlag], uiImageWidth, uiImageHeight, fSigma);  
                #else
                        sprintf(cTitle, "%s Recursive Gaussian | W: %u  H: %u | Simple Filter, Sigma %.1f", 
                            cProcessor[iProcFlag], uiImageWidth, uiImageHeight, fSigma);  
                #endif
            }
        }
        else 
        {
            #ifdef _WIN32
                sprintf_s(cTitle, 256, "Recursive Gaussian OFF | W: %u  H: %u", 
	                        uiImageWidth, uiImageHeight);  
            #else
                sprintf(cTitle, "Recursive Gaussian OFF | W: %u  H: %u", 
	                        uiImageWidth, uiImageHeight);  
            #endif
        }
#endif
        glutSetWindowTitle(cTitle);

        // Log fps and processing info to console and file 
        shrLog(LOGBOTH, 0, "%s\n", cTitle); 

        // if doing quick test, exit
        if ((bNoPrompt) && (!--iTestSets))
        {
            // Cleanup up and quit
            Cleanup(EXIT_SUCCESS);
        }

        // reset the frame counter and adjust trigger
        iFrameCount = 0; 
        iFrameTrigger = (iFramesPerSec > 1) ? iFramesPerSec * 2 : 1;
    }
}

// Window resize handler callback
//*****************************************************************************
void Reshape(int /*w*/, int /*h*/)
{
    // Zoom with fixed aspect ratio
    float fAspects[2] = {(float)glutGet(GLUT_WINDOW_WIDTH)/(float)uiImageWidth , (float)glutGet(GLUT_WINDOW_HEIGHT)/(float)uiImageHeight};
    fZoom = fAspects[0] > fAspects[1] ? fAspects[1] : fAspects[0];
    glPixelZoom(fZoom, fZoom);
}

// Keyboard event handler callback
//*****************************************************************************
void KeyboardGL(unsigned char key, int /*x*/, int /*y*/)
{
    switch(key) 
    {
        case 'P':   // P toggles Processing between CPU and GPU
        case 'p':   // p toggles Processing between CPU and GPU
            if (iProcFlag == 0)
            {
                iProcFlag = 1;
            }
            else 
            {
                iProcFlag = 0;
            }
            shrLog(LOGBOTH, 0, "\n%s Processing...\n", cProcessor[iProcFlag]);
            break;
        case 'F':   // F toggles main graphics display full screen
        case 'f':   // f toggles main graphics display full screen
            bFullScreen = !bFullScreen;
            if (bFullScreen)
            {
                iGraphicsWinPosX = glutGet(GLUT_WINDOW_X);
                iGraphicsWinPosY = glutGet(GLUT_WINDOW_Y);
                iGraphicsWinWidth = glutGet(GLUT_WINDOW_WIDTH); 
                iGraphicsWinHeight = glutGet(GLUT_WINDOW_HEIGHT);
                glutFullScreen();
            }
            else
            {
                glutReshapeWindow(iGraphicsWinWidth, iGraphicsWinHeight);
                glutPositionWindow(iGraphicsWinPosX, iGraphicsWinPosY);
            }
            shrLog(LOGBOTH, 0, "\nMain Graphics %s...\n", bFullScreen ? "FullScreen" : "Windowed");
            break;
        case ' ':   // space bar toggles filter on and off
            bFilter = !bFilter;
            shrLog(LOGBOTH, 0, "\nRecursive Gaussian Filter Toggled %s...\n", bFilter ? "ON" : "OFF");
            break;
        case '=':
            fSigma += 0.1f;
            break;
        case '-':
            fSigma -= 0.1f;
            break;
        case '+':
			fSigma += 1.0f;
			break;
		case '_':
			fSigma -= 1.0f;
			break;
        case '0':
            iOrder = 0;
            break;
        case '1':
            if (!USE_SIMPLE_FILTER)
            {
                iOrder = 1;
                fSigma = 2.0f;
            }
            break;
        case '2':
            if (!USE_SIMPLE_FILTER)
            {
                iOrder = 2;
                fSigma = 0.2f;
            }
            break;
        case '\033': // escape quits
        case '\015':// Enter quits    
        case 'Q':    // Q quits
        case 'q':    // q (or escape) quits
            // Cleanup up and quit
            Cleanup(EXIT_SUCCESS);
            break;
    }

    // range check order and sigma 
    iOrder = CLAMP (iOrder, 0, 2);
    fSigma = MAX (0.1f, fSigma);

    // pre-compute filter coefficients and set common kernel args
    PreProcessGaussParms (fSigma, iOrder, &oclGP);
    GPUGaussianSetCommonArgs (&oclGP);

    // Log filter params
    if (bFilter)
    {
        if (USE_SIMPLE_FILTER)
        {
            shrLog(LOGBOTH, 0, "Simple Filter, Sigma =  %.1f\n", fSigma);  
        }
        else 
        {
            shrLog(LOGBOTH, 0, "Full Filter, Order = %i, Sigma =  %.1f\n", iOrder, fSigma);  
        }
    }

    // Trigger fps update and call for refresh
    TriggerFPSUpdate();
    glutPostRedisplay();
}

// GLUT menu callback function
//*****************************************************************************
void MenuGL(int i)
{
    KeyboardGL((unsigned char) i, 0, 0);
}

// GL Idle time callback
//*****************************************************************************
void Idle(void)
{
    glutPostRedisplay();
}

// Helper to trigger reset of fps vars at transition 
//*****************************************************************************
void TriggerFPSUpdate()
{
    iFrameCount = 0; 
    shrDeltaT(1);
    iFramesPerSec = 1;
    iFrameTrigger = 2;
}

// Run a test sequence without any GL 
//*****************************************************************************
void TestNoGL()
{
    // Warmup call to assure OpenCL driver is awake
    GPUGaussianFilterRGBA(&oclGP);

    // Start timer 0 and process n loops on the GPU 
    clFinish(cqCommandQueue);
    int iCycles = 100;
	shrDeltaT(0); 
    for (int i = 0; i < iCycles; i++)
    {
        GPUGaussianFilterRGBA(&oclGP);
    }

    // Get elapsed time, and Log data
    double dAvgTime = shrDeltaT(0)/(double)iCycles;
    if (!USE_SIMPLE_FILTER)
    {
        shrLog(LOGBOTH | MASTER, 0, "oclRecursiveGaussian, Throughput = %.4f, Time = %.5f, Size = %u, NumDevsUsed = %u, Workgroup = %u\n", 
               (1.0e-6 * uiImageWidth * uiImageHeight)/dAvgTime, dAvgTime, (uiImageWidth * uiImageHeight), 1, szGaussLocalWork[0]); 
    }
    else 
    {
        shrLog(LOGBOTH | MASTER, 0, "oclRecursiveGaussian, Throughput = %.4f, Time = %.5f, Size = %u, NumDevsUsed = %u, Workgroup = %u\n", 
               (1.0e-6 * uiImageWidth * uiImageHeight)/dAvgTime, dAvgTime, (uiImageWidth * uiImageHeight), 1, szGaussLocalWork[0]); 
    }

    // Cleanup and exit
    Cleanup(EXIT_SUCCESS);
}

// Function to clean up and exit
//*****************************************************************************
void Cleanup(int iExitCode)
{
    // Cleanup allocated objects
    shrLog(LOGBOTH, 0, "\nStarting Cleanup...\n\n");
    if(cSourceCL)free(cSourceCL);
    if(cPathAndName)free(cPathAndName);
    if(ckSimpleRecursiveRGBA)clReleaseKernel(ckSimpleRecursiveRGBA);
    if(ckTranspose)clReleaseKernel(ckTranspose);
    if(ckRecursiveGaussianRGBA)clReleaseKernel(ckRecursiveGaussianRGBA);
    if(cpProgram)clReleaseProgram(cpProgram);
    if(cmDevBufIn)clReleaseMemObject(cmDevBufIn); 
    if(cmDevBufTemp)clReleaseMemObject(cmDevBufTemp); 
    if(cmDevBufOut)clReleaseMemObject(cmDevBufOut); 
    if(cqCommandQueue)clReleaseCommandQueue(cqCommandQueue);
    if(cdDevices)free(cdDevices);
    if(cxGPUContext)clReleaseContext(cxGPUContext);
    if(iGLUTMenuHandle)glutDestroyMenu(iGLUTMenuHandle);
    if(iGLUTWindowHandle)glutDestroyWindow(iGLUTWindowHandle);
    if (uiInput)free(uiInput);
    if (uiTemp)free(uiTemp);
    if (uiOutput)free(uiOutput);
    shrLog(LOGBOTH, 0, "TEST %s\n\n", iExitCode == 0 ? "PASSED" : "FAILED !!!"); 

    // finalize logs and leave
    if (bNoPrompt || bQATest)
    {
        shrLog(LOGBOTH | CLOSELOG, 0, "oclRecursiveGaussian.exe Exiting...\n");
    }
    else 
    {
        shrLog(LOGBOTH | CLOSELOG, 0, "oclRecursiveGaussian.exe Exiting...\nPress <Enter> to Quit\n");
        #ifdef WIN32
            getchar();
        #endif
    }
    exit (iExitCode);
}
