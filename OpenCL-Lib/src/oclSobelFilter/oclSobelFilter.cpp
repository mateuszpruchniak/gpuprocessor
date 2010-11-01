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
    #include <OpenGL/OpenGL.h>
    #include <GLUT/glut.h>
#else
    #include <GL/glut.h>
#endif

// Import host computation function for functional and perf comparison
extern "C" void SobelFilterHost(unsigned int* uiInputImage, unsigned int* uiOutputImage, 
                                unsigned int uiWidth, unsigned int uiHeight, float fThresh);

// Defines and globals for Sobel filter processing demo
//*****************************************************************************
int iBlockDimX = 16;
int iBlockDimY = 4;
float fThresh = 80.0f;             // min display value for output 

// Global declarations
//*****************************************************************************
// Image data vars
const char* cImageFile = "StoneRGB.ppm";
unsigned int uiImageWidth = 1920;   // Image width
unsigned int uiImageHeight = 1080;  // Image height

// OpenGL and Display Globals
int iGLUTWindowHandle;              // handle to the GLUT window
int iGLUTMenuHandle;                // handle to the GLUT menu
int iGraphicsWinPosX = 0;           // GL Window X location
int iGraphicsWinPosY = 0;           // GL Window Y location
int iGraphicsWinWidth = 1024;       // GL Window width
int iGraphicsWinHeight = ((float)uiImageHeight / (float)uiImageWidth) * iGraphicsWinWidth;  // GL Windows height
float fZoom = 1.0f;                 // pixel display zoom   
int iFrameCount = 0;                // FPS count for averaging
int iFrameTrigger = 90;             // FPS trigger for sampling
int iFramesPerSec = 60;             // frames per second
bool bFullScreen = false;           // state var for full screen mode or not
bool bFilter = true;                // state var for whether filter is enaged or not

// app configuration parms
const char* cProcessor [] = {"OpenCL GPU", "Host C++ CPU"};
int iProcFlag = 0;                  // 0 = GPU, 1 = CPU
shrBOOL bNoPrompt = shrFALSE;		// false = normal GL loop, true = Finite period of GL loop (a few seconds)
shrBOOL bQATest = shrFALSE;			// false = normal GL loop, true = run No-GL test sequence
int iTestSets = 3;                  // # of loop set retriggers before auto exit when bNoPrompt = shrTrue     

// OpenCL vars
const char* clSourcefile = "SobelFilter.cl";  // OpenCL kernel source file
cl_context cxGPUContext;            // OpenCL context
char* cPathAndName = NULL;          // var for full paths to data, src, etc.
char* cSourceCL = NULL;             // Buffer to hold source for compilation 
cl_command_queue* cqCommandQueue;   // OpenCL command que array
cl_device_id* cdDevices;            // OpenCL device list
cl_uint uiNumDevices;               // Number of devices available/used
cl_int iSelectedDevice = 0;         // Selected device, if any
cl_uint* uiInHostPixOffsets;        // Host input buffer pixel offset for image portion worked on by each device 
cl_event* ceQueueFinish;            // OpenCL event list, 1 per queue
cl_program cpProgram;               // OpenCL program
cl_kernel* ckSobel;                  // OpenCL Kernel array for Sobel
cl_mem cmPinnedBufIn;               // OpenCL host memory input buffer object:  pinned 
cl_mem cmPinnedBufOut;              // OpenCL host memory output buffer object:  pinned
cl_mem* cmDevBufIn;                 // OpenCL device memory input buffer object  
cl_mem* cmDevBufOut;                // OpenCL device memory output buffer object
cl_uint* uiInput = NULL;            // Mapped Pointer to pinned Host input buffer for host processing
cl_uint* uiOutput = NULL;           // Mapped Pointer to pinned Host output buffer for host processing
size_t szBuffBytes;                 // Size of main image buffers
size_t* szAllocDevBytes;            // Array of Sizes of device buffers
cl_uint* uiDevImageHeight;          // Array of heights of Image portions for each device
size_t szGlobalWorkSize[2];         // 2D global work items (ND range) for Median kernel
size_t szLocalWorkSize[2];          // 2D local work items (work group) for Median kernel
size_t szParmDataBytes;			    // Byte size of context information
size_t szKernelLength;			    // Byte size of kernel code
cl_int ciErrNum;			        // Error code var

// Forward Function declarations
//*****************************************************************************
// OpenCL functionality
void SobelFilterGPU(unsigned int* uiInputImage, unsigned int* uiOutputImage, float fThresh);

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
size_t DivUp(size_t dividend, size_t divisor);
void GetDeviceLoadProportions(float* fLoadProportions, cl_device_id* cdDevices, cl_uint uiDevCount);
void Cleanup(int iExitCode);
void (*pCleanup)(int) = &Cleanup;

// Main program
//*****************************************************************************
int main(int argc, const char** argv)
{
    // Start logs 
    shrSetLogFileName ("oclSobelFilter.txt");
    shrLog(LOGBOTH, 0, "%s Starting (Using %s)...\n\n", argv[0], clSourcefile); 

    // Get command line args for quick test or QA test, if provided
    bNoPrompt = shrCheckCmdLineFlag(argc, argv, "noprompt");
    bQATest = shrCheckCmdLineFlag(argc, argv, "qatest");

    // Find the path from the exe to the image file 
    cPathAndName = shrFindFilePath(cImageFile, argv[0]);
    shrCheckErrorEX (cPathAndName != NULL, shrTRUE, pCleanup);
    shrLog(LOGBOTH, 0, "Image File: \t\t%s\nImage Dimensions:\t%u w x %u h x %u bpp\n\n", cPathAndName, uiImageWidth, uiImageHeight, sizeof(unsigned int)<<3);

    // Initialize OpenGL items (if not No-GL QA test)
	if (!(bQATest))
	{
		InitGL(argc, argv);
	}
	shrLog(LOGBOTH, 0, "%sInitGL...\n\n", bQATest ? "Skipping " : "Calling "); 

    // Create the OpenCL context on a GPU device
    cxGPUContext = clCreateContextFromType(0, CL_DEVICE_TYPE_GPU, NULL, NULL, &ciErrNum);
    shrCheckErrorEX (ciErrNum, CL_SUCCESS, pCleanup);
    shrLog(LOGBOTH, 0, "clCreateContextFromType...\n"); 

    // Get the list of GPU devices associated with context
    ciErrNum = clGetContextInfo(cxGPUContext, CL_CONTEXT_DEVICES, 0, NULL, &szParmDataBytes);
    cdDevices = (cl_device_id*)malloc(szParmDataBytes);
    ciErrNum |= clGetContextInfo(cxGPUContext, CL_CONTEXT_DEVICES, szParmDataBytes, cdDevices, NULL);
    shrCheckErrorEX (ciErrNum, CL_SUCCESS, pCleanup);
    uiNumDevices = (cl_uint)szParmDataBytes/sizeof(cl_device_id);
    shrCheckErrorEX (uiNumDevices > 0, true, pCleanup);
    shrLog(LOGBOTH, 0, "clGetContextInfo (%u Devices found)...\n\n", uiNumDevices); 

    // get selected device if there is one
    if(shrGetCmdLineArgumenti(argc, argv, "device", &iSelectedDevice)) 
    {
        // create a command que for the selected device, and print out the device info
        iSelectedDevice = CLAMP((cl_uint)iSelectedDevice, 0, (uiNumDevices - 1));
        uiNumDevices = 1;   
    } 

    // Allocate per-device OpenCL objects 
    shrLog(LOGBOTH, 0, "Using %u Device(s) with divided work for Median Filter Computation...\n", uiNumDevices); 
    cqCommandQueue = new cl_command_queue[uiNumDevices];
    ckSobel = new cl_kernel[uiNumDevices];
    cmDevBufIn = new cl_mem[uiNumDevices];
    cmDevBufOut = new cl_mem[uiNumDevices];
    szAllocDevBytes = new size_t[uiNumDevices];
    uiInHostPixOffsets = new cl_uint[uiNumDevices];
    uiDevImageHeight = new cl_uint[uiNumDevices];
    ceQueueFinish = new cl_event[uiNumDevices];

    // Determine device load proportions
    float* fLoadProportions = new float[uiNumDevices];
    GetDeviceLoadProportions(fLoadProportions, cdDevices, uiNumDevices);

    // Create command queue(s) for device(s)        
    for (cl_uint i = 0; i < uiNumDevices; i++) 
    {
        shrLog(LOGBOTH, 0, "\n  clCreateCommandQueue %u\n  Device Load Proportion = %.2f...\n ", i, fLoadProportions[i]); 
        cqCommandQueue[i] = clCreateCommandQueue(cxGPUContext, cdDevices[i + iSelectedDevice], 0, &ciErrNum);
        shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
        oclPrintDevName(LOGBOTH, cdDevices[i + iSelectedDevice]);
    }

    // Allocate pinned input and output host image buffers:  mem copy operations to/from pinned memory is much faster than paged memory
    szBuffBytes = uiImageWidth * uiImageHeight * sizeof (unsigned int);
    cmPinnedBufIn = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, szBuffBytes, NULL, &ciErrNum);
    shrCheckErrorEX (ciErrNum, CL_SUCCESS, pCleanup);
    cmPinnedBufOut = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, szBuffBytes, NULL, &ciErrNum);
    shrCheckErrorEX (ciErrNum, CL_SUCCESS, pCleanup);
    shrLog(LOGBOTH, 0, "\nclCreateBuffer (Input and Output Pinned Host buffers)...\n"); 

    // Get mapped pointers for writing to pinned input and output host image pointers 
    uiInput = (cl_uint*)clEnqueueMapBuffer(cqCommandQueue[0], cmPinnedBufIn, CL_TRUE, CL_MAP_WRITE, 0, szBuffBytes, 0, NULL, NULL, &ciErrNum);
    shrCheckErrorEX (ciErrNum, CL_SUCCESS, pCleanup);
    uiOutput = (cl_uint*)clEnqueueMapBuffer(cqCommandQueue[0], cmPinnedBufOut, CL_TRUE, CL_MAP_READ, 0, szBuffBytes, 0, NULL, NULL, &ciErrNum);
    shrCheckErrorEX (ciErrNum, CL_SUCCESS, pCleanup);
    shrLog(LOGBOTH, 0, "clEnqueueMapBuffer (Pointer to Input and Output pinned host buffers)...\n"); 

    // Load image data from file to pinned input host buffer
    ciErrNum = shrLoadPPM4ub(cPathAndName, (unsigned char **)&uiInput, &uiImageWidth, &uiImageHeight);
    shrCheckErrorEX (ciErrNum, shrTRUE, pCleanup);
    shrLog(LOGBOTH, 0, "Load Input Image to Input pinned host buffer...\n"); 

    // Read the kernel in from file
    free(cPathAndName);
    cPathAndName = shrFindFilePath(clSourcefile, argv[0]);
    shrCheckErrorEX (cPathAndName != NULL, shrTRUE, pCleanup);
    cSourceCL = oclLoadProgSource(cPathAndName, "// My comment\n", &szKernelLength);
    shrCheckErrorEX (cSourceCL != NULL, shrTRUE, pCleanup);
    shrLog(LOGBOTH, 0, "oclLoadProgSource...\n"); 

    // Create the program object
    cpProgram = clCreateProgramWithSource(cxGPUContext, 1, (const char **)&cSourceCL, &szKernelLength, &ciErrNum);
    shrCheckErrorEX (ciErrNum, CL_SUCCESS, pCleanup);
    shrLog(LOGBOTH, 0, "clCreateProgramWithSource...\n"); 

    // Build the program with 'mad' Optimization option
#ifdef MAC
    char *flags = "-cl-mad-enable -DMAC";
#else
    char *flags = "-cl-mad-enable";
#endif

    ciErrNum = clBuildProgram(cpProgram, 0, NULL, flags, NULL, NULL);
    if (ciErrNum != CL_SUCCESS)
    {
        // On error: write out standard error, Build Log and PTX, then cleanup and exit
        shrLog(LOGBOTH | ERRORMSG, ciErrNum, STDERROR);
        oclLogBuildInfo(cpProgram, oclGetFirstDev(cxGPUContext));
        oclLogPtx(cpProgram, oclGetFirstDev(cxGPUContext), "oclSobelFilter.ptx");
        Cleanup(EXIT_FAILURE);
    }
    shrLog(LOGBOTH, 0, "clBuildProgram...\n\n"); 

    // Determine, the size/shape of the image portions for each dev and create the device buffers
    unsigned uiSumHeight = 0;
    for (cl_uint i = 0; i < uiNumDevices; i++)
    {
        // Create kernel instance
        ckSobel[i] = clCreateKernel(cpProgram, "ckSobel", &ciErrNum);
        shrCheckErrorEX (ciErrNum, CL_SUCCESS, pCleanup);
        shrLog(LOGBOTH, 0, "clCreateKernel (ckMedian), Device %u...\n", i); 

        // Allocations and offsets for the portion of the image worked on by each device
        if (uiNumDevices == 1)
        {
            // One device processes the whole image with no offset 
            uiDevImageHeight[i] = uiImageHeight; 
            uiInHostPixOffsets[i] = 0;
            szAllocDevBytes[i] = uiDevImageHeight[i] * uiImageWidth * sizeof(cl_uint);
        }
        else if (i == 0)
        {
            // Multiple devices, top stripe zone including topmost row of image:  
            // Over-allocate on device by 1 row 
            // Set offset and size to copy extra 1 padding row H2D (below bottom of stripe)
            // Won't return the last row (dark/garbage row) D2H
            uiDevImageHeight[i] = (cl_uint)(fLoadProportions[i] * (float)uiImageHeight);     // height is proportional to dev perf 
            uiSumHeight += uiDevImageHeight[i];
            uiDevImageHeight[i] += 1;
            uiInHostPixOffsets[i] = 0;
            szAllocDevBytes[i] = uiDevImageHeight[i] * uiImageWidth * sizeof(cl_uint);
        }
        else if (i < (uiNumDevices - 1))
        {
            // Multiple devices, middle stripe zone:  
            // Over-allocate on device by 2 rows 
            // Set offset and size to copy extra 2 padding rows H2D (above top and below bottom of stripe)
            // Won't return the first and last rows (dark/garbage rows) D2H
            uiDevImageHeight[i] = (cl_uint)(fLoadProportions[i] * (float)uiImageHeight);     // height is proportional to dev perf 
            uiSumHeight += uiDevImageHeight[i];
            uiDevImageHeight[i] += 2;
            uiInHostPixOffsets[i] = (uiSumHeight - 1) * uiImageWidth;
            szAllocDevBytes[i] = uiDevImageHeight[i] * uiImageWidth * sizeof(cl_uint);
        }
        else 
        {
            // Multiple devices, last boundary tile:  
            // Over-allocate on device by 1 row 
            // Set offset and size to copy extra 1 padding row H2D (above top of stripe)
            // Won't return the first row (dark/garbage rows D2H 
            uiDevImageHeight[i] = uiImageHeight - uiSumHeight;                              // "leftover" rows 
            uiDevImageHeight[i] += 1;
            uiInHostPixOffsets[i] = (uiSumHeight - 1) * uiImageWidth;
            szAllocDevBytes[i] = uiDevImageHeight[i] * uiImageWidth * sizeof(cl_uint);
        }
        shrLog(LOGBOTH, 0, "Image Height (rows) for Device %u = %u...\n", i, uiDevImageHeight[i]); 

        // Create the device buffers in GMEM on each device
        cmDevBufIn[i] = clCreateBuffer(cxGPUContext, CL_MEM_READ_ONLY, szAllocDevBytes[i], NULL, &ciErrNum);
        shrCheckErrorEX (ciErrNum, CL_SUCCESS, pCleanup);
        cmDevBufOut[i] = clCreateBuffer(cxGPUContext, CL_MEM_WRITE_ONLY, szAllocDevBytes[i], NULL, &ciErrNum);
        shrCheckErrorEX (ciErrNum, CL_SUCCESS, pCleanup);
        shrLog(LOGBOTH, 0, "clCreateBuffer (Input and Output GMEM buffers, Device %u)...\n", i); 

        // Set the common argument values for the Median kernel instance for each device
        int iLocalPixPitch = iBlockDimX + 2;
        ciErrNum = clSetKernelArg(ckSobel[i], 0, sizeof(cl_mem), (void*)&cmDevBufIn[i]);
        ciErrNum |= clSetKernelArg(ckSobel[i], 1, sizeof(cl_mem), (void*)&cmDevBufOut[i]);
        ciErrNum |= clSetKernelArg(ckSobel[i], 2, (iLocalPixPitch * (iBlockDimY + 2) * sizeof(cl_uchar4)), NULL);
        ciErrNum |= clSetKernelArg(ckSobel[i], 3, sizeof(cl_int), (void*)&iLocalPixPitch);
        ciErrNum |= clSetKernelArg(ckSobel[i], 4, sizeof(cl_uint), (void*)&uiImageWidth);
        ciErrNum |= clSetKernelArg(ckSobel[i], 6, sizeof(cl_float), (void*)&fThresh);
        shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
        shrLog(LOGBOTH, 0, "clSetKernelArg (0-4,6), Device %u...\n\n", i); 
    }
    delete (fLoadProportions);  // no longer needed

    // Set common global and local work sizes for Sobel kernel
    szLocalWorkSize[0] = iBlockDimX;
    szLocalWorkSize[1] = iBlockDimY;
    szGlobalWorkSize[0] = shrRoundUp((int)szLocalWorkSize[0], uiImageWidth); 

    // Init timers
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

// OpenCL computation function for GPU:  
// Copies input data from pinned host buf to the device, runs kernel, copies output data back to pinned output host buf
//*****************************************************************************
void SobelFilterGPU(cl_uint* uiInputImage, cl_uint* uiOutputImage, float fThresh)
{
    // If this is a video application, fresh data in pinned host buffer is needed beyond here 
    //      This line could be a sync point assuring that an asynchronous acqusition is complete.
    //      That ascynchronous acquisition would do a map, update and unmap for the pinned input buffer
    //
    //      Otherwise a synchronous acquisition call ('get next frame') could be placed here, but that would be less optimal.

    // For each device: copy fresh input H2D, process, copy fresh output D2H
    for (cl_uint i = 0; i < uiNumDevices; i++)
    {
        // Determine configuration bytes, offsets and launch config, based on position of device region vertically in image
        size_t szReturnBytes;
        cl_uint uiOutHostPixOffset;
        cl_uint uiOutDevByteOffset;
        if (uiNumDevices == 1)
        {
            // One device processes the whole image with no offset tricks needed
            szReturnBytes = szBuffBytes;
            uiOutHostPixOffset = 0;
            uiOutDevByteOffset = 0;
            szGlobalWorkSize[1] = shrRoundUp((int)szLocalWorkSize[1], (int)uiDevImageHeight[i]);
        }
        else if (i == 0)
        {
            // Multiple devices, top boundary tile:  
            // Process whole device allocation, including extra row 
            // No offset, but don't return the last row (dark/garbage row) D2H 
            szReturnBytes = szAllocDevBytes[i] - (uiImageWidth * sizeof(cl_uint));
            uiOutHostPixOffset = uiInHostPixOffsets[i];
            uiOutDevByteOffset = 0;
            szGlobalWorkSize[1] = shrRoundUp((int)szLocalWorkSize[1], (int)uiDevImageHeight[i]);
        }
        else if (i < (uiNumDevices - 1))
        {
            // Multiple devices, middle tile:  
            // Process whole device allocation, including extra 2 rows 
            // Offset down by 1 row, and don't return the first and last rows (dark/garbage rows) D2H 
            szReturnBytes = szAllocDevBytes[i] - ((uiImageWidth * sizeof(cl_uint)) * 2);
            uiOutHostPixOffset = uiInHostPixOffsets[i] + uiImageWidth;
            uiOutDevByteOffset = uiImageWidth * sizeof(cl_uint);
            szGlobalWorkSize[1] = shrRoundUp((int)szLocalWorkSize[1], (int)uiDevImageHeight[i]);
        }
        else 
        {   
            // Multiple devices, last boundary tile:  
            // Process whole device allocation, including extra row 
            // Offset down by 1 row, and don't return the first row (dark/garbage row) D2H 
            szReturnBytes = szAllocDevBytes[i] - (uiImageWidth * sizeof(cl_uint));
            uiOutHostPixOffset = uiInHostPixOffsets[i] + uiImageWidth;
            uiOutDevByteOffset = uiImageWidth * sizeof(cl_uint);
            szGlobalWorkSize[1] = shrRoundUp((int)szLocalWorkSize[1], (int)uiDevImageHeight[i]);
        }

        // Pass in dev image height (# of rows worked on) for this device
        ciErrNum = clSetKernelArg(ckSobel[i], 5, sizeof(cl_uint), (void*)&uiDevImageHeight[i]);
        shrCheckError(ciErrNum, CL_SUCCESS);

        // Nonblocking Write of input image data from host to device
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue[i], cmDevBufIn[i], CL_FALSE, 0, szAllocDevBytes[i], (void*)&uiInputImage[uiInHostPixOffsets[i]], 0, NULL, NULL);
        shrCheckError(ciErrNum, CL_SUCCESS);

        // Launch Median kernel
        ciErrNum = clEnqueueNDRangeKernel(cqCommandQueue[i], ckSobel[i], 2, NULL, szGlobalWorkSize, szLocalWorkSize, 0, NULL, NULL);
        shrCheckErrorEX (ciErrNum, CL_SUCCESS, pCleanup);

        // Non Blocking Read of output image data from device to host
        ciErrNum = clEnqueueReadBuffer(cqCommandQueue[i], cmDevBufOut[i], CL_FALSE, uiOutDevByteOffset, szReturnBytes, (void*)&uiOutputImage[uiOutHostPixOffset], 0, NULL, &ceQueueFinish[i]);
        shrCheckError(ciErrNum, CL_SUCCESS);
    }

    // Finish all queues before returning
    ciErrNum = clWaitForEvents(uiNumDevices, ceQueueFinish);
    shrCheckError(ciErrNum, CL_SUCCESS);
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
    iGLUTWindowHandle = glutCreateWindow("OpenCL for GPU RGB Sobel Filter Demo");
#if defined (__APPLE__) || defined(MACOSX)
    long VBL = 0;
    CGLSetParameter(CGLGetCurrentContext(), kCGLCPSwapInterval, &VBL); // turn off Vsync
#endif

    // register glut callbacks
    glutKeyboardFunc(KeyboardGL);
    glutDisplayFunc(DisplayGL);
    glutReshapeFunc(Reshape);
    glutIdleFunc(Idle);

    // create GLUT menu
    iGLUTMenuHandle = glutCreateMenu(MenuGL);
    glutAddMenuEntry("Toggle Filter On/Off <spacebar>", ' ');
    glutAddMenuEntry("Toggle Processing between GPU and CPU [p]", 'p');
    glutAddMenuEntry("Increase Threshold [+]", '+');
    glutAddMenuEntry("Decrease Threshold [-]", '-');
    glutAddMenuEntry("Toggle Processing between GPU and CPU [p]", 'p');
    glutAddMenuEntry("Toggle between Full Screen and Windowed [f]", 'f');
    glutAddMenuEntry("Quit <esc>", '\033');
    glutAttachMenu(GLUT_RIGHT_BUTTON);

    // Show help info, start timers
    shrLog(LOGBOTH, 0, "  <Right Click Mouse Button> for Menu\n\n"); 
    shrLog(LOGBOTH, 0, "Press:\n  <Spacebar> to toggle Filter On/Off\n\n  \'F\' key to toggle FullScreen On/Off\n\n");
    shrLog(LOGBOTH, 0, "  \'P\' key to toggle Processing between GPU and CPU\n\n  <Esc> to Quit\n\n"); 

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
            SobelFilterGPU (uiInput, uiOutput, fThresh);
        }
        else 
        {
            SobelFilterHost (uiInput, uiOutput, uiImageWidth, uiImageHeight, fThresh);
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
            sprintf_s(cTitle, 256, "%s RGB Sobel Filter ON | W: %u , H: %u | Thresh. = %.1f | %i fps | Proc. t = %.4f s", 
	                        cProcessor[iProcFlag], uiImageWidth, uiImageHeight, fThresh, iFramesPerSec, dProcessingTime);  
            #else
                sprintf(cTitle, "%s RGB Sobel Filter ON | W: %u , H: %u | Thresh. = %.1f | %i fps | Proc. t = %.4f s", 
	                        cProcessor[iProcFlag], uiImageWidth, uiImageHeight, fThresh, iFramesPerSec, dProcessingTime);  
            #endif
        }
        else 
        {
            #ifdef _WIN32
                sprintf_s(cTitle, 256, "RGB Sobel Filter OFF | W: %u , H: %u | %i fps", 
	                        uiImageWidth, uiImageHeight, iFramesPerSec);  
            #else 
                sprintf(cTitle, "RGB Sobel Filter OFF | W: %u , H: %u | %i fps", 
	                        uiImageWidth, uiImageHeight, iFramesPerSec);  
            #endif
        }
#else
        if (bFilter)
        {
            #ifdef _WIN32
                sprintf_s(cTitle, 256, "%s RGB Sobel Filter ON | W: %u , H: %u | Thresh. = %.1f", 
	                        cProcessor[iProcFlag], uiImageWidth, uiImageHeight, fThresh);  
            #else
                sprintf(cTitle, "%s RGB Sobel Filter ON | W: %u , H: %u | Thresh. = %.1f", 
	                        cProcessor[iProcFlag], uiImageWidth, uiImageHeight, fThresh);  
            #endif
        }
        else 
        {
            #ifdef _WIN32
                sprintf_s(cTitle, 256, "RGB Sobel Filter OFF | W: %u , H: %u", 
	                        uiImageWidth, uiImageHeight);  
            #else 
                sprintf(cTitle, "RGB Sobel Filter OFF | W: %u , H: %u", 
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
            shrLog(LOGBOTH, 0, "\nSobel Filter Toggled %s...\n", bFilter ? "ON" : "OFF");
            break;
        case '+':   // + sign increases threshold
        case '=':   // = sign increases threshold
        case '-':   // - sign decreases threshold
        case '_':   // _ decreases threshold
            if(key == '+' || key == '=')
            {
                fThresh += 10.0f;
            }
            else
            {
                fThresh -= 10.0f;
            }

            // Clamp and reset the associated kernel arg, and log value
            fThresh = CLAMP(fThresh,0.0f, 255.0f);
            for (cl_uint i = 0; i < uiNumDevices; i++)
            {
                ciErrNum = clSetKernelArg(ckSobel[i], 6, sizeof(cl_float), (void*)&fThresh);
            }
            shrCheckErrorEX (ciErrNum, CL_SUCCESS, pCleanup);
            shrLog(LOGBOTH, 0, "\nThreshold changed to %.1f...\n", fThresh);
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

// Helper to trigger reset of fps vars at transition 
//*****************************************************************************
void TriggerFPSUpdate()
{
    iFrameCount = 0; 
    shrDeltaT(1);
    iFramesPerSec = 1;
    iFrameTrigger = 2;
}

// Helper to get next up value for integer division
//*****************************************************************************
size_t DivUp(size_t dividend, size_t divisor)
{
    return (dividend % divisor == 0) ? (dividend / divisor) : (dividend / divisor + 1);
}

// Helper determined balanced load proportions for multiGPU config using perf estimation
//*****************************************************************************
void GetDeviceLoadProportions(float* fLoadProportions, cl_device_id* cdDevices, cl_uint uiDevCount)
{
    // Estimate dev perf and total perf
    float fTotalPerf = 0.0f;
    float* fDevPerfs = new float[uiDevCount];
    for (cl_uint i = 0; i < uiDevCount; i++)
    {
        // CL_DEVICE_MAX_COMPUTE_UNITS
        cl_uint compute_units;
        clGetDeviceInfo(cdDevices[i], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(compute_units), &compute_units, NULL);

        // CL_DEVICE_MAX_CLOCK_FREQUENCY
        cl_uint clock_frequency;
        clGetDeviceInfo(cdDevices[i], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(clock_frequency), &clock_frequency, NULL);

        // Get individual device perf and accumulate
        // Note: To achieve better load proportions for each GPU an overhead penalty is subtracted
        // from the computed device perf 
        fDevPerfs[i] = (float)(compute_units * clock_frequency) - 15000.0f;

        fTotalPerf += fDevPerfs[i];
    }

    // Compute/assign load proportions
    for (cl_uint i = 0; i < uiDevCount; i++)
    {
        fLoadProportions[i] = fDevPerfs[i]/fTotalPerf;
    }

    // delete temp alloc
    delete (fDevPerfs);
}

// Run a test sequence without any GL 
//*****************************************************************************
void TestNoGL()
{
    // Warmup call to assure OpenCL driver is awake
    SobelFilterGPU (uiInput, uiOutput, fThresh);
    for (cl_uint i = 0; i < uiNumDevices; i++)
    {
        clFinish(cqCommandQueue[i]);
    }

	// Start timer 0 and process n loops on the GPU
    int iCycles = 100;
	shrDeltaT(0); 
    for (int i = 0; i < iCycles; i++)
    {
        SobelFilterGPU (uiInput, uiOutput, fThresh);
    }

    // Get elapsed time and throughput, then log to sample and master logs
    double dAvgTime = shrDeltaT(0)/(double)iCycles;
    shrLog(LOGBOTH | MASTER, 0, "oclSobelFilter, Throughput = %.4f, Time = %.5f, Size = %u, NumDevsUsed = %u, Workgroup = %u\n", 
           (1.0e-6 * uiImageWidth * uiImageHeight)/dAvgTime, dAvgTime, (uiImageWidth * uiImageHeight), uiNumDevices, (szLocalWorkSize[0] * szLocalWorkSize[1])); 

    // Cleanup and exit
    Cleanup(EXIT_SUCCESS);
}

// Function to clean up and exit
//*****************************************************************************
void Cleanup(int iExitCode)
{
    // Cleanup allocated objects
    shrLog(LOGBOTH, 0, "\nStarting Cleanup...\n\n");

    // Release all the OpenCL Objects
    if(cpProgram)clReleaseProgram(cpProgram);
    for (cl_uint i = 0; i < uiNumDevices; i++)
    {
        if(ceQueueFinish[i])clReleaseEvent(ceQueueFinish[i]);
        if(ckSobel[i])clReleaseKernel(ckSobel[i]);
        if(cmDevBufIn[i])clReleaseMemObject(cmDevBufIn[i]);
        if(cmDevBufOut[i])clReleaseMemObject(cmDevBufOut[i]);
    }
    if(uiInput)clEnqueueUnmapMemObject(cqCommandQueue[0], cmPinnedBufIn, (void*)uiInput, 0, NULL, NULL);
    if(uiOutput)clEnqueueUnmapMemObject(cqCommandQueue[0], cmPinnedBufOut, (void*)uiOutput, 0, NULL, NULL);
    if(cmPinnedBufIn)clReleaseMemObject(cmPinnedBufIn);
    if(cmPinnedBufOut)clReleaseMemObject(cmPinnedBufOut);
    for (cl_uint i = 0; i < uiNumDevices; i++)
    {
        if(cqCommandQueue[i])clReleaseCommandQueue(cqCommandQueue[i]);
    }
    if(cxGPUContext)clReleaseContext(cxGPUContext);

    // free the host allocs
    if(cSourceCL)free(cSourceCL);
    if(cPathAndName)free(cPathAndName);
    if(cmDevBufIn)delete(cmDevBufIn);
    if(cmDevBufOut)delete(cmDevBufOut);
    if(szAllocDevBytes)delete(szAllocDevBytes);
    if(uiInHostPixOffsets)delete(uiInHostPixOffsets);
    if(ceQueueFinish)delete(ceQueueFinish);
    if(uiDevImageHeight)delete(uiDevImageHeight);
    if(cdDevices)free(cdDevices);
    if(cqCommandQueue)free(cqCommandQueue);

    // cleanup GLUT objects
    if(iGLUTMenuHandle)glutDestroyMenu(iGLUTMenuHandle);
    if(iGLUTWindowHandle)glutDestroyWindow(iGLUTWindowHandle);

    // finalize logs and leave
    shrLog(LOGBOTH, 0, "TEST %s\n\n", iExitCode == 0 ? "PASSED" : "FAILED !!!"); 
    if (bNoPrompt || bQATest)
    {
        shrLog(LOGBOTH | CLOSELOG, 0, "oclSobelFilter.exe Exiting...\n");
    }
    else 
    {
        shrLog(LOGBOTH | CLOSELOG, 0, "oclSobelFilter.exe Exiting...\nPress <Enter> to Quit\n");
        #ifdef WIN32
            getchar();
        #endif
    }
    exit (iExitCode);
}
