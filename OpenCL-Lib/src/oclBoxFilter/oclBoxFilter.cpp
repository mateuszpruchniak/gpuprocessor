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

// Includes
//*****************************************************************************

// standard utilities and systems includes
#include <oclUtils.h>

// GLUT includes
#if defined (__APPLE__) || defined(MACOSX)
    #include <GLUT/glut.h>
#else
    #include <GL/glut.h>
#endif

// Import host computation function for functional and perf comparison
extern "C" void BoxFilterHost(unsigned int* uiInputImage, unsigned int* uiTempImage, unsigned int* uiOutputImage, unsigned int uiWidth, unsigned int uiHeight, int r, float fScale);

// Defines and globals for box filter processing demo
//*****************************************************************************
#define MASK_RADIUS 8
#define MASK_RADIUS_ALIGNED 16
#define MASK_LENGTH (2 * MASK_RADIUS + 1)
#define ROWS_OUTPUT_WIDTH 128
#define COLUMNS_BLOCKDIMX 16
#define COLUMNS_BLOCKDIMY 16
#define COLUMNS_OUTPUT_HEIGHT 128
float fScale = 1.0f/MASK_LENGTH;

// Global declarations
//*****************************************************************************
// Image data vars
const char* cImageFile = "lenaRGB.ppm";
unsigned int uiImageWidth = 0;      // Image width
unsigned int uiImageHeight = 0;     // Image height
unsigned int* uiInput = NULL;       // Host buffer to hold input image data
unsigned int* uiTemp = NULL;        // Host buffer to hold intermediate image data
unsigned int* uiOutput = NULL;      // Host buffer to hold output image data

// OpenGL and Display Globals
int iGLUTWindowHandle;              // handle to the GLUT window
int iGLUTMenuHandle;                // handle to the GLUT menu
int iGraphicsWinPosX = 0;           // GL Window X location
int iGraphicsWinPosY = 0;           // GL Window Y location
int iGraphicsWinWidth = 768;        // GL Window width
int iGraphicsWinHeight = 768;       // GL Windows height
float fZoom = 1.0f;                 // pixel display zoom

// fps, quick test and qatest vars
const char* cProcessor [] = {"OpenCL GPU", "Host C++ CPU"};
int iFrameCount = 0;                // FPS count for averaging
int iFrameTrigger = 90;             // FPS trigger for sampling
int iFramesPerSec = 60;             // frames per second
bool bFullScreen = false;           // state var for full screen mode or not
bool bFilter = true;                // state var for whether filter is enaged or not
int iProcFlag = 0;                  // 0 = GPU, 1 = CPU
shrBOOL bNoPrompt = shrFALSE;		// false = normal GL loop, true = Finite period of GL loop (a few seconds)
shrBOOL bQATest = shrFALSE;			// false = normal GL loop, true = run No-GL test sequence
int iTestSets = 3;                  // # of loop set retriggers before auto exit when bNoPrompt = shrTrue  

// OpenCL vars
const char* clSourcefile = "BoxFilter.cl";  // OpenCL kernel source file
char* cPathAndName = NULL;          // var for full paths to data, src, etc.
char* cSourceCL = NULL;             // Buffer to hold source for compilation 
cl_context cxGPUContext;            // OpenCL context
cl_command_queue cqCommandQueue;    // OpenCL command que
cl_device_id* cdDevices;            // OpenCL device list
cl_uint uiNumDevices;               // Number of devices available/used
cl_program cpProgram;               // OpenCL program
cl_kernel ckBoxRows;                // OpenCL Kernel for row sum
cl_kernel ckBoxColumns;             // OpenCL for column sum and normalize
cl_mem cmDevBufIn;                  // OpenCL device memory input buffer object  
cl_mem cmDevBufTemp;                // OpenCL device memory temp buffer object  
cl_mem cmDevBufOut;                 // OpenCL device memory output buffer object
size_t szBuffBytes;                 // Size of main image buffers
size_t szGlobalWorkSize[2];         // global # of work items for 2 kernels
size_t szLocalWorkSize[2];          // work group # of work items for 2 kernels
size_t szParmDataBytes;			    // Byte size of context information
size_t szKernelLength;			    // Byte size of kernel code
cl_int ciErrNum;			        // Error code var

// Forward Function declarations
//*****************************************************************************
// OpenCL functionality
void BoxFilterGPU(unsigned int* uiInputImage, unsigned int* uiOutputImage, unsigned int uiWidth, unsigned int uiHeight, int r, float fScale);
void ResetKernelArgs(unsigned int uiWidth, unsigned int uiHeight, int r, float fScale);

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
unsigned int iDivUp(unsigned int dividend, unsigned int divisor);
void Cleanup(int iExitCode);
void (*pCleanup)(int) = &Cleanup;

// Main program
//*****************************************************************************
int main(int argc, const char** argv)
{
    // Start logs 
    shrSetLogFileName ("oclBoxFilter.txt");
    shrLog(LOGBOTH, 0, "%s Starting, using %s...\n\n", argv[0], clSourcefile); 

    // Get command line args for quick test or QA test, if provided
    bNoPrompt = shrCheckCmdLineFlag(argc, argv, "noprompt");
    bQATest = shrCheckCmdLineFlag(argc, argv, "qatest");

    // Find the path from the exe to the image file and load the image
    cPathAndName = shrFindFilePath(cImageFile, argv[0]);
    shrCheckErrorEX (cPathAndName != NULL, shrTRUE, pCleanup);
    ciErrNum = shrLoadPPM4ub(cPathAndName, (unsigned char **)&uiInput, &uiImageWidth, &uiImageHeight);
    shrCheckErrorEX (ciErrNum, shrTRUE, pCleanup);
    shrLog(LOGBOTH, 0, "Image Width = %i, Height = %i, bpp = %i, Mask Radius = %i\n\n", uiImageWidth, uiImageHeight, sizeof(unsigned int)<<3, MASK_RADIUS);

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
    shrCheckErrorEX (ciErrNum, CL_SUCCESS, pCleanup);
    shrLog(LOGBOTH, 0, "clCreateContextFromType...\n"); 

    // Get the list of GPU devices associated with context
    ciErrNum = clGetContextInfo(cxGPUContext, CL_CONTEXT_DEVICES, 0, NULL, &szParmDataBytes);
    cdDevices = (cl_device_id *)malloc(szParmDataBytes);
    ciErrNum |= clGetContextInfo(cxGPUContext, CL_CONTEXT_DEVICES, szParmDataBytes, cdDevices, NULL);
    shrCheckErrorEX (ciErrNum, CL_SUCCESS, pCleanup);
    uiNumDevices = (cl_uint)szParmDataBytes/sizeof(cl_device_id);
    shrLog(LOGBOTH, 0, "clGetContextInfo...\n\n"); 

    // List used device 
    shrLog(LOGBOTH, 0, "GPU Device being used:\n"); 
    oclPrintDevInfo(LOGBOTH, cdDevices[0]);

    // Create a command-queue
    cqCommandQueue = clCreateCommandQueue(cxGPUContext, cdDevices[0], 0, &ciErrNum);
    shrCheckErrorEX (ciErrNum, CL_SUCCESS, pCleanup);
    shrLog(LOGBOTH, 0, "clCreateCommandQueue...\n"); 

    // Allocate the OpenCL source, intermediate and result buffer memory objects on the device GMEM
    cmDevBufIn = clCreateBuffer(cxGPUContext, CL_MEM_READ_ONLY, szBuffBytes, NULL, &ciErrNum);
    shrCheckErrorEX (ciErrNum, CL_SUCCESS, pCleanup);
    cmDevBufTemp = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, szBuffBytes, NULL, &ciErrNum);
    shrCheckErrorEX (ciErrNum, CL_SUCCESS, pCleanup);
    cmDevBufOut = clCreateBuffer(cxGPUContext, CL_MEM_WRITE_ONLY, szBuffBytes, NULL, &ciErrNum);
    shrCheckErrorEX (ciErrNum, CL_SUCCESS, pCleanup);
    shrLog(LOGBOTH, 0, "clCreateBuffer (Input, Intermediate and Output buffers, device GMEM)...\n"); 

    // Read the OpenCL kernel source in from file
    free(cPathAndName);
    cPathAndName = shrFindFilePath(clSourcefile, argv[0]);
    shrCheckErrorEX (cPathAndName != NULL, shrTRUE, pCleanup);
    cSourceCL = oclLoadProgSource(cPathAndName, "// My comment\n", &szKernelLength);
    shrCheckErrorEX (cSourceCL != NULL, shrTRUE, pCleanup);
    shrLog(LOGBOTH, 0, "oclLoadProgSource...\n"); 

    // Create the program 
    cpProgram = clCreateProgramWithSource(cxGPUContext, 1, (const char **)&cSourceCL, &szKernelLength, &ciErrNum);
    shrCheckErrorEX (ciErrNum, CL_SUCCESS, pCleanup);
    shrLog(LOGBOTH, 0, "clCreateProgramWithSource...\n"); 

    // Build the program with 'mad' Optimization option
    ciErrNum = clBuildProgram(cpProgram, 0, NULL, "-cl-mad-enable", NULL, NULL);
    if (ciErrNum != CL_SUCCESS)
    {
        // write out standard error, Build Log and PTX, then cleanup and exit
        shrLog(LOGBOTH | ERRORMSG, ciErrNum, STDERROR);
        oclLogBuildInfo(cpProgram, oclGetFirstDev(cxGPUContext));
        oclLogPtx(cpProgram, oclGetFirstDev(cxGPUContext), "oclBoxFilter.ptx");
        Cleanup(EXIT_FAILURE);
    }
    shrLog(LOGBOTH, 0, "clBuildProgram...\n"); 

    // Create kernels
    ckBoxRows = clCreateKernel(cpProgram, "BoxRows", &ciErrNum);
    shrCheckErrorEX (ciErrNum, CL_SUCCESS, pCleanup);
    ckBoxColumns = clCreateKernel(cpProgram, "BoxColumns", &ciErrNum);
    shrCheckErrorEX (ciErrNum, CL_SUCCESS, pCleanup);
    shrLog(LOGBOTH, 0, "clCreateKernel (ckBoxRows and ckBoxColumns)...\n\n"); 

    // set the kernel args
    ResetKernelArgs(uiImageWidth, uiImageHeight, MASK_RADIUS, fScale);

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

// Function to set kernel args that only change outside of GLUT loop
//*****************************************************************************
void ResetKernelArgs(unsigned int uiWidth, unsigned int uiHeight, int r, float fScale)
{
    // Set the Argument values for the row kernel
    ciErrNum = clSetKernelArg(ckBoxRows, 0, sizeof(cl_mem), (void*)&cmDevBufIn);
    ciErrNum |= clSetKernelArg(ckBoxRows, 1, sizeof(cl_mem), (void*)&cmDevBufTemp);
    ciErrNum |= clSetKernelArg(ckBoxRows, 2, (MASK_RADIUS_ALIGNED + ROWS_OUTPUT_WIDTH + r) * sizeof(cl_uchar4), NULL);
    ciErrNum |= clSetKernelArg(ckBoxRows, 3, sizeof(unsigned int), (void*)&uiWidth);
    ciErrNum |= clSetKernelArg(ckBoxRows, 4, sizeof(unsigned int), (void*)&uiHeight);
    ciErrNum |= clSetKernelArg(ckBoxRows, 5, sizeof(float), (void*)&fScale);
    shrCheckErrorEX (ciErrNum, CL_SUCCESS, pCleanup);

    // Set the Argument values for the column kernel
    ciErrNum  = clSetKernelArg(ckBoxColumns, 0, sizeof(cl_mem), (void*)&cmDevBufTemp);
    ciErrNum |= clSetKernelArg(ckBoxColumns, 1, sizeof(cl_mem), (void*)&cmDevBufOut);
    ciErrNum |= clSetKernelArg(ckBoxColumns, 2, (r + COLUMNS_OUTPUT_HEIGHT + r) * COLUMNS_BLOCKDIMX * sizeof(cl_uchar4), NULL);
    ciErrNum |= clSetKernelArg(ckBoxColumns, 3, sizeof(unsigned int), (void*)&uiWidth);
    ciErrNum |= clSetKernelArg(ckBoxColumns, 4, sizeof(unsigned int), (void*)&uiHeight);
    ciErrNum |= clSetKernelArg(ckBoxColumns, 5, sizeof(float), (void*)&fScale);
    shrCheckErrorEX (ciErrNum, CL_SUCCESS, pCleanup);
}

// OpenCL computation function for GPU:  
// Copies input data to the device, runs kernel, copies output data back to host  
//*****************************************************************************
void BoxFilterGPU(unsigned int* uiInputImage, unsigned int* uiOutputImage, unsigned int uiWidth, unsigned int uiHeight, int r, float fScale)
{
    // Copy input data from host to device
    ciErrNum = clEnqueueWriteBuffer(cqCommandQueue, cmDevBufIn, CL_TRUE, 0, szBuffBytes, uiInputImage, 0, NULL, NULL);
    shrCheckErrorEX (ciErrNum, CL_SUCCESS, pCleanup);

    // Set global and local work sizes for row kernel
    szLocalWorkSize[0] = MASK_RADIUS_ALIGNED + ROWS_OUTPUT_WIDTH + r;
    szLocalWorkSize[1] = 1;
    szGlobalWorkSize[0] = iDivUp(uiWidth, ROWS_OUTPUT_WIDTH) * szLocalWorkSize[0];
    szGlobalWorkSize[1] = uiHeight;

    // Launch row kernel
    ciErrNum = clEnqueueNDRangeKernel(cqCommandQueue, ckBoxRows, 2, NULL, szGlobalWorkSize, szLocalWorkSize, 0, NULL, NULL);
    shrCheckErrorEX (ciErrNum, CL_SUCCESS, pCleanup);

    // Set global and local work sizes for column kernel
    szLocalWorkSize[0] = COLUMNS_BLOCKDIMX;
    szLocalWorkSize[1] = COLUMNS_BLOCKDIMY;
    szGlobalWorkSize[0] = uiWidth;
    szGlobalWorkSize[1] = iDivUp(uiHeight, COLUMNS_OUTPUT_HEIGHT) * szLocalWorkSize[1];

    // Launch column kernel
    ciErrNum = clEnqueueNDRangeKernel(cqCommandQueue, ckBoxColumns, 2, NULL, szGlobalWorkSize, szLocalWorkSize, 0, NULL, NULL);
    shrCheckErrorEX (ciErrNum, CL_SUCCESS, pCleanup);

    // Copy results back to host, block until complete
    ciErrNum = clEnqueueReadBuffer(cqCommandQueue, cmDevBufOut, CL_TRUE, 0, szBuffBytes, uiOutputImage, 0, NULL, NULL);
    shrCheckErrorEX (ciErrNum, CL_SUCCESS, pCleanup);

    return;
}

// Initialize GL
//*****************************************************************************
void InitGL(int argc, const char **argv)
{
    // initialize GLUT 
    glutInit(&argc, (char**)argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowPosition (glutGet(GLUT_SCREEN_WIDTH)/2 - iGraphicsWinWidth/2, 
                            glutGet(GLUT_SCREEN_HEIGHT)/2 - iGraphicsWinHeight/2);
    glutInitWindowSize(iGraphicsWinWidth, iGraphicsWinHeight);
    iGLUTWindowHandle = glutCreateWindow("OpenCL GPU BoxFilter Demo");

    // register glut callbacks
    glutKeyboardFunc(KeyboardGL);
    glutDisplayFunc(DisplayGL);
    glutReshapeFunc(Reshape);
    glutIdleFunc(Idle);

    // create GLUT menu
    iGLUTMenuHandle = glutCreateMenu(MenuGL);
    glutAddMenuEntry("Toggle Filter On/Off <spacebar>", ' ');
    glutAddMenuEntry("Toggle Processing between GPU and CPU [p]", 'p');
    glutAddMenuEntry("Toggle between Full Screen and Windowed [f]", 'f');
    glutAddMenuEntry("Quit <esc>", '\033');
    glutAttachMenu(GLUT_RIGHT_BUTTON);

    // Show help info in console / log
    shrLog(LOGBOTH, 0, "  <Right Click Mouse Button> for Menu\n\n"); 
    shrLog(LOGBOTH, 0, "Press:\n\n  <spacebar> to toggle Filter On/Off\n\n  \'F\' key to toggle FullScreen On/Off\n\n");
    shrLog(LOGBOTH, 0, "  \'P\' key to toggle Processing between GPU and CPU\n\n  <esc> to Quit\n\n"); 

    // Zoom with fixed aspect ratio
    float fAspects[2] = {(float)glutGet(GLUT_WINDOW_WIDTH)/(float)uiImageWidth , (float)glutGet(GLUT_WINDOW_HEIGHT)/(float)uiImageHeight};
    fZoom = fAspects[0] > fAspects[1] ? fAspects[1] : fAspects[0];
    glPixelZoom(fZoom, fZoom);
}

// Display callback for GLUT main loop
//*****************************************************************************
void DisplayGL()
{
    // Run OpenCL kernel to filter image (if toggled on), then render to backbuffer
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
            BoxFilterGPU (uiInput, uiOutput, uiImageWidth, uiImageHeight, MASK_RADIUS, fScale);
        }
        else 
        {
            BoxFilterHost (uiInput, uiTemp, uiOutput, uiImageWidth, uiImageHeight, MASK_RADIUS, fScale);
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

    //  Increment the frame counter, and do fps stuff if it's time
    if (iFrameCount++ > iFrameTrigger)
    {
        // Set the display window title
        char cTitle[256];
        iFramesPerSec = (int)((double)iFrameCount/ shrDeltaT(1));
#ifdef GPU_PROFILING
        if (bFilter)
        {
            #ifdef _WIN32
            sprintf_s(cTitle, 256, "%s RGBA BoxFilter ON | W %u , H %u | %i x %i Filter | %i fps | Proc. t = %.4f s | %.3f Mpix/s", 
                      cProcessor[iProcFlag], uiImageWidth, uiImageHeight, MASK_RADIUS, MASK_RADIUS, 
                      iFramesPerSec, dProcessingTime, (1.0e-6 * uiImageWidth * uiImageHeight)/dProcessingTime);  
            #else
             sprintf(cTitle, "%s RGBA BoxFilter | W %u , H %u | %i x %i Filter | %i fps | Proc. t = %.4f s | %.3f Mpix/s", 
 	                 cProcessor[iProcFlag],uiImageWidth, uiImageHeight,MASK_RADIUS, MASK_RADIUS, 
                     iFramesPerSec, dProcessingTime, (1.0e-6 * uiImageWidth * uiImageHeight)/dProcessingTime);  
            #endif
        }
        else 
        {
            #ifdef _WIN32
            sprintf_s(cTitle, 256, "%s RGBA BoxFilter OFF | W %u , H %u | %i fps", 
	                  cProcessor[iProcFlag], uiImageWidth, uiImageHeight, iFramesPerSec);  
            #else 
            sprintf(cTitle, "%s RGBA BoxFilter OFF | W %u , H %u | %i fps", 
                    cProcessor[iProcFlag], uiImageWidth, uiImageHeight, iFramesPerSec);  
            #endif
        }
#else
        if (bFilter)
        {
            #ifdef _WIN32
            sprintf_s(cTitle, 256, "%s RGBA BoxFilter ON | W %u , H %u | %i x %i Filter", 
                      cProcessor[iProcFlag], uiImageWidth, uiImageHeight, MASK_RADIUS, MASK_RADIUS);  
            #else
             sprintf(cTitle, "%s RGBA BoxFilter | W %u , H %u | %i x %i Filter", 
 	                 cProcessor[iProcFlag],uiImageWidth, uiImageHeight,MASK_RADIUS, MASK_RADIUS);  
            #endif
        }
        else 
        {
            #ifdef _WIN32
            sprintf_s(cTitle, 256, "%s RGBA BoxFilter OFF | W %u , H %u", 
	                cProcessor[iProcFlag], uiImageWidth, uiImageHeight);  
            #else 
            sprintf(cTitle, "%s RGBA BoxFilter OFF | W %u , H %u", 
                    cProcessor[iProcFlag], uiImageWidth, uiImageHeight);  
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
            shrLog(LOGBOTH, 0, "\n%s Processing...\n\n", cProcessor[iProcFlag]);
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
            shrLog(LOGBOTH, 0, "\nBoxFilter Toggled %s...\n", bFilter ? "ON" : "OFF");
            break;
        case '\033':// Escape quits    
        case '\015':// Enter quits    
        case 'Q':   // Q quits
        case 'q':   // q quits
            // Cleanup up and quit
            Cleanup(EXIT_SUCCESS);
            break;
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

// Helper to get next up value for integer division
//*****************************************************************************
unsigned int iDivUp(unsigned int dividend, unsigned int divisor)
{
    return (dividend % divisor == 0) ? (dividend / divisor) : (dividend / divisor + 1);
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
    BoxFilterGPU (uiInput, uiOutput, uiImageWidth, uiImageHeight, MASK_RADIUS, fScale);
    clFinish(cqCommandQueue);

	// Start timer 0 and process n loops on the GPU
    int iCycles = 100;
	shrDeltaT(0); 
    for (int i = 0; i < iCycles; i++)
    {
        BoxFilterGPU (uiInput, uiOutput, uiImageWidth, uiImageHeight, MASK_RADIUS, fScale);
    }

    // Get elapsed time and throughput, then log to sample and master logs
    double dAvgTime = shrDeltaT(0)/(double)iCycles;
    shrLog(LOGBOTH | MASTER, 0, "oclBoxFilter, Throughput = %.4f, Time = %.5f, Size = %u, NumDevsUsed = %u, Workgroup = %u\n", 
           (1.0e-6 * uiImageWidth * uiImageHeight)/dAvgTime, dAvgTime, (uiImageWidth * uiImageHeight), 1, (szLocalWorkSize[0] * szLocalWorkSize[1])); 

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
    if(uiInput)free(uiInput);
    if(uiTemp)free(uiTemp);
    if(uiOutput)free(uiOutput);
    if(ckBoxColumns)clReleaseKernel(ckBoxColumns);
    if(ckBoxRows)clReleaseKernel(ckBoxRows);
    if(cpProgram)clReleaseProgram(cpProgram);
    if(cmDevBufIn)clReleaseMemObject(cmDevBufIn);
    if(cmDevBufTemp)clReleaseMemObject(cmDevBufTemp);
    if(cmDevBufOut)clReleaseMemObject(cmDevBufOut);
    if(cqCommandQueue)clReleaseCommandQueue(cqCommandQueue);
    if(cdDevices)free(cdDevices);
    if(cxGPUContext)clReleaseContext(cxGPUContext);
    if(iGLUTMenuHandle)glutDestroyMenu(iGLUTMenuHandle);
    if(iGLUTWindowHandle)glutDestroyWindow(iGLUTWindowHandle);
    shrLog(LOGBOTH, 0, "TEST %s\n\n", iExitCode == 0 ? "PASSED" : "FAILED !!!"); 

    // finalize logs and leave
    if ((bNoPrompt)||(bQATest))
    {
        shrLog(LOGBOTH | CLOSELOG, 0, "oclBoxFilter.exe Exiting...\n");
    }
    else 
    {
        shrLog(LOGBOTH | CLOSELOG, 0, "oclBoxFilter.exe Exiting...\nPress <Enter> to Quit\n");
        #ifdef WIN32
            getchar();
        #endif
    }
    exit (iExitCode);
}
