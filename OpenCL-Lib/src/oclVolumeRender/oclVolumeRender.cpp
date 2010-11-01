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
    Volume rendering sample

    This sample loads a 3D volume from disk and displays it using
    ray marching and 3D textures.
*/

// Utilities and standard system includes
#include <oclUtils.h>

// GLEW and GLUT
#include <GL/glew.h>
#if defined (__APPLE__) || defined(MACOSX)
    #include <GLUT/glut.h>
#else
    #include <GL/glut.h>
#endif

// Extra CL/GL include
#include <CL/cl_gl.h>

// Constants, defines, typedefs and global declarations
//*****************************************************************************
// Uncomment this #define to enable CL/GL Interop
//#define GL_INTEROP

#define IMAGE_SUPPORT

typedef unsigned int uint;
typedef unsigned char uchar;

const char *volumeFilename = "Bucky.raw";
size_t volumeSize[3] = {32, 32, 32};

uint width = 512, height = 512;
size_t gridSize[2] = {width, height};

float viewRotation[3];
float viewTranslation[3] = {0.0, 0.0, -4.0f};
float invViewMatrix[12];

float density = 0.05f;
float brightness = 1.0f;
float transferOffset = 0.0f;
float transferScale = 1.0f;
bool linearFiltering = true;

GLuint pbo = 0;                 // OpenGL pixel buffer object
int iGLUTWindowHandle;          // handle to the GLUT window

// OpenCL vars
cl_context cxGPUContext;
cl_device_id device;
cl_command_queue cqCommandQueue;
cl_program cpProgram;
cl_kernel ckKernel;
cl_int ciErrNum;
cl_mem pbo_cl;
cl_mem d_volumeArray;
cl_mem d_transferFuncArray;
cl_mem d_invViewMatrix;
cl_sampler d_volumeSampler;
cl_sampler d_transferFuncSampler;
char* cPathAndName = NULL;          // var for full paths to data, src, etc.
char* cSourceCL;                    // Buffer to hold source for compilation 

int iFrameCount = 0;                // FPS count for averaging
int iFrameTrigger = 20;             // FPS trigger for sampling
int iFramesPerSec = 0;              // frames per second
int iTestSets = 3;
int g_Index = 0;
shrBOOL bNoPrompt = shrFALSE;		// false = normal GL loop, true = Finite period of GL loop (a few seconds)
shrBOOL bQATest = shrFALSE;			// false = normal GL loop, true = run No-GL test sequence  
bool g_bFBODisplay = false;
int ox, oy;                         // mouse location vars
int buttonState = 0;                

// Forward Function declarations
//*****************************************************************************
// OpenCL Functions
void initPixelBuffer();
void render();
void initOpenCL(uchar *h_volume);

// OpenGL functionality
void InitGL(int argc, const char** argv);
void DisplayGL();
void Reshape(int w, int h);
void Idle(void);
void KeyboardGL(unsigned char key, int x, int y);
void motion(int x, int y);
void mouse(int button, int state, int x, int y);

// Helpers
void Cleanup(int iExitCode);
void (*pCleanup)(int) = &Cleanup;
void TestNoGL();

// Main program
//*****************************************************************************
int main(int argc, const char** argv) 
{
    // start logs
    shrSetLogFileName ("oclVolumeRender.txt");
    shrLog(LOGBOTH, 0, "%s Starting...\n\n", argv[0]); 
    shrLog(LOGBOTH, 0, " Press '=' and '-' to change density\n"
                         "       ']' and '[' to change brightness\n"
                         "       ';' and ''' to modify transfer function offset\n"
                         "       '.' and ',' to modify transfer function scale\n\n");

    // get command line arg for quick test, if provided
    // process command line arguments
    if (argc > 1) 
    {
        bQATest = shrCheckCmdLineFlag(argc, argv, "qatest");
        bNoPrompt = shrCheckCmdLineFlag(argc, argv, "noprompt");
    }

    // First initialize OpenGL context, so we can properly setup the OpenGL / OpenCL interop.
    if(!bQATest) 
    {
        InitGL(argc, argv); 
    }

    // create the OpenCL context 
    cxGPUContext = clCreateContextFromType(0, CL_DEVICE_TYPE_GPU, NULL, NULL, &ciErrNum);
    shrCheckErrorEX (ciErrNum, CL_SUCCESS, pCleanup);

    // get and log device info
    if( shrCheckCmdLineFlag(argc, argv, "device") ) {
      int device_nr = 0;
      shrGetCmdLineArgumenti(argc, argv, "device", &device_nr);
      device = oclGetDev(cxGPUContext, device_nr);
    } else {
      device = oclGetMaxFlopsDev(cxGPUContext);
    }
    oclPrintDevInfo(LOGBOTH, device);

    // create a command-queue
    cqCommandQueue = clCreateCommandQueue(cxGPUContext, device, 0, &ciErrNum);
    shrCheckErrorEX (ciErrNum, CL_SUCCESS, pCleanup);

    // Program Setup
    size_t program_length;
    cPathAndName = shrFindFilePath("volumeRender.cl", argv[0]);
    shrCheckErrorEX (cPathAndName != NULL, shrTRUE, pCleanup);
    #ifdef IMAGE_SUPPORT
        cSourceCL = oclLoadProgSource(cPathAndName, "#define IMAGE_SUPPORT", &program_length);
    #else
        cSourceCL = oclLoadProgSource(cPathAndName, "", &program_length);
    #endif    
    shrCheckErrorEX (cSourceCL != NULL, shrTRUE, pCleanup);

    // create the program
    cpProgram = clCreateProgramWithSource(cxGPUContext, 1,
					  (const char **)&cSourceCL, &program_length, &ciErrNum);
    shrCheckErrorEX (ciErrNum, CL_SUCCESS, pCleanup);
    
    // build the program
    ciErrNum = clBuildProgram(cpProgram, 0, NULL, "-cl-mad-enable", NULL, NULL);
    if (ciErrNum != CL_SUCCESS)
    {
        // write out standard error, Build Log and PTX, then cleanup and return error
        shrLog(LOGBOTH | ERRORMSG, ciErrNum, STDERROR);
        oclLogBuildInfo(cpProgram, oclGetFirstDev(cxGPUContext));
        oclLogPtx(cpProgram, oclGetFirstDev(cxGPUContext), "oclVolumeRender.ptx");
        Cleanup(EXIT_FAILURE); 
    }

    // create the kernel
    ckKernel = clCreateKernel(cpProgram, "d_render", &ciErrNum);
    shrCheckErrorEX (ciErrNum, CL_SUCCESS, pCleanup);

    // parse arguments
    char *filename;
    if (shrGetCmdLineArgumentstr(argc, argv, "file", &filename)) {
        volumeFilename = filename;
    }
    int n;
    if (shrGetCmdLineArgumenti(argc, argv, "size", &n)) {
        volumeSize[0] = volumeSize[1] = volumeSize[2] = n;
    }
    if (shrGetCmdLineArgumenti(argc, argv, "xsize", &n)) {
        volumeSize[0] = n;
    }
    if (shrGetCmdLineArgumenti(argc, argv, "ysize", &n)) {
        volumeSize[1] = n;
    }
    if (shrGetCmdLineArgumenti(argc, argv, "zsize", &n)) {
         volumeSize[2] = n;
    }

    // load volume data
    free(cPathAndName);
    cPathAndName = shrFindFilePath(volumeFilename, argv[0]);
    shrCheckErrorEX (cPathAndName != NULL, shrTRUE, pCleanup);
    size_t size = volumeSize[0] * volumeSize[1] * volumeSize[2];
    uchar* h_volume = shrLoadRawFile(cPathAndName, size);
    shrCheckErrorEX (h_volume != NULL, true, pCleanup);
    shrLog(LOGBOTH, 0, " Raw file data loaded...\n\n");

    // Init OpenCL
    initOpenCL(h_volume);
    free (h_volume);

    // init timer 1 for fps measurement 
    shrDeltaT(1);  
    
    // Create buffers and textures, 
    // and then start main GLUT rendering loop for processing and rendering, 
	// or otherwise run No-GL Q/A test sequence
    if(!bQATest) 
    {
        initPixelBuffer();
        glutMainLoop();
    } 
    else 
    {
        TestNoGL();
    }

    // Normally unused return path
    Cleanup(EXIT_FAILURE);
}

// render image using OpenCL
//*****************************************************************************
void render()
{
    ciErrNum = CL_SUCCESS;

    // Transfer ownership of buffer from GL to CL
#ifdef GL_INTEROP
    // Acquire PBO for OpenCL writing
    ciErrNum |= clEnqueueAcquireGLObjects(cqCommandQueue, 1, &pbo_cl, 0, 0, 0);
#endif    
    ciErrNum |= clEnqueueWriteBuffer(cqCommandQueue,d_invViewMatrix,CL_FALSE, 0,12*sizeof(float), invViewMatrix, 0, 0, 0);

    // execute OpenCL kernel, writing results to PBO
    ciErrNum |= clSetKernelArg(ckKernel, 3, sizeof(float), &density);
    ciErrNum |= clSetKernelArg(ckKernel, 4, sizeof(float), &brightness);
    ciErrNum |= clSetKernelArg(ckKernel, 5, sizeof(float), &transferOffset);
    ciErrNum |= clSetKernelArg(ckKernel, 6, sizeof(float), &transferScale);
    ciErrNum |= clEnqueueNDRangeKernel(cqCommandQueue, ckKernel, 2, NULL, gridSize, NULL, 0, 0, 0);
    shrCheckErrorEX (ciErrNum, CL_SUCCESS, pCleanup);

#ifdef GL_INTEROP
    // Transfer ownership of buffer back from CL to GL    
    ciErrNum |= clEnqueueReleaseGLObjects(cqCommandQueue, 1, &pbo_cl, 0, 0, 0);
#else
    // Explicit Copy 
    // map the PBO to copy data from the CL buffer via host
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);    

    // map the buffer object into client's memory
    GLubyte* ptr = (GLubyte*)glMapBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB,
                                        GL_WRITE_ONLY_ARB);
    clEnqueueReadBuffer(cqCommandQueue, pbo_cl, CL_TRUE, 0, sizeof(unsigned int) * height * width, ptr, 0, NULL, NULL);        
    shrCheckErrorEX (ciErrNum, CL_SUCCESS, pCleanup);
    glUnmapBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB); 
#endif
}

// Display callback for GLUT main loop
//*****************************************************************************
void DisplayGL()
{
    // use OpenGL to build view matrix
    GLfloat modelView[16];
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();
    glRotatef(-viewRotation[0], 1.0, 0.0, 0.0);
    glRotatef(-viewRotation[1], 0.0, 1.0, 0.0);
    glTranslatef(-viewTranslation[0], -viewTranslation[1], -viewTranslation[2]);
    glGetFloatv(GL_MODELVIEW_MATRIX, modelView);
    glPopMatrix();

    invViewMatrix[0] = modelView[0]; invViewMatrix[1] = modelView[4]; invViewMatrix[2] = modelView[8]; invViewMatrix[3] = modelView[12];
    invViewMatrix[4] = modelView[1]; invViewMatrix[5] = modelView[5]; invViewMatrix[6] = modelView[9]; invViewMatrix[7] = modelView[13];
    invViewMatrix[8] = modelView[2]; invViewMatrix[9] = modelView[6]; invViewMatrix[10] = modelView[10]; invViewMatrix[11] = modelView[14];

    // start timer 0 if it's update time
    double dProcessingTime = 0.0;
    if (iFrameCount >= iFrameTrigger)
    {
        shrDeltaT(0); 
    }

     // process 
    render();

    // get processing time from timer 0, if it's update time
    if (iFrameCount >= iFrameTrigger)
    {
        dProcessingTime = shrDeltaT(0); 
    }

    // draw image from PBO
    glClear(GL_COLOR_BUFFER_BIT);
    glDisable(GL_DEPTH_TEST);
    glRasterPos2i(0, 0);
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
    glDrawPixels(width, height, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

    // flip backbuffer to screen
    glutSwapBuffers();
    glutPostRedisplay();

    // Increment the frame counter, and do fps and Q/A stuff if it's time
    if (iFrameCount++ > iFrameTrigger) 
    {
        // set GLUT Window Title
        char cFPS[256];
        iFramesPerSec = (int)((double)iFrameCount/shrDeltaT(1));
#ifdef GPU_PROFILING
        #ifdef _WIN32
            sprintf_s(cFPS, 256, "Volume Render %ux%u | %i fps | Proc.t = %.3f s | %.3f MP/s", 
                width, height, iFramesPerSec, dProcessingTime, (1.0e-6 * width * height)/dProcessingTime);
        #else 
            sprintf(cFPS, "Volume Render %ux%u |  %i fps | Proc. t = %.3f s | %.3f MP/s", 
                width, height, iFramesPerSec, dProcessingTime, (1.0e-6 * width * height)/dProcessingTime);
        #endif
#else
        #ifdef _WIN32
            sprintf_s(cFPS, 256, "Volume Render | W: %u  H: %u", width, height);
        #else 
            sprintf(cFPS, "Volume Render | W: %u  H: %u", width, height);
        #endif
#endif
        glutSetWindowTitle(cFPS);

        // Log fps and processing info to console and file 
        shrLog(LOGBOTH, 0, " %s\n", cFPS); 

        // if doing quick test, exit
        if ((bNoPrompt) && (!--iTestSets))
        {
            // Cleanup up and quit
            Cleanup(EXIT_SUCCESS);
        }

        // reset framecount, trigger and timer
        iFrameCount = 0; 
        iFrameTrigger = (iFramesPerSec > 1) ? iFramesPerSec * 2 : 1;
    }
}

// GL Idle time callback
//*****************************************************************************
void Idle()
{
    glutPostRedisplay();
}

// Keyboard event handler callback
//*****************************************************************************
void KeyboardGL(unsigned char key, int /*x*/, int /*y*/)
{
    switch(key) 
    {
        case '=':
            density += 0.01;
            break;
        case '-':
            density -= 0.01;
            break;
        case '+':
            density += 0.1;
            break;
        case '_':
            density -= 0.1;
            break;

        case ']':
            brightness += 0.1;
            break;
        case '[':
            brightness -= 0.1;
            break;

        case ';':
            transferOffset += 0.01;
            break;
        case '\'':
            transferOffset -= 0.01;
            break;

        case '.':
            transferScale += 0.01;
            break;
        case ',':
            transferScale -= 0.01;
            break;
        case '\033': // escape quits
        case '\015': // Enter quits    
        case 'Q':    // Q quits
        case 'q':    // q (or escape) quits
            // Cleanup up and quit
            Cleanup(EXIT_SUCCESS);
            break;
        default:
            break;
    }
    shrLog(LOGBOTH, 0, "density = %.2f, brightness = %.2f, transferOffset = %.2f, transferScale = %.2f\n", density, brightness, transferOffset, transferScale);
    glutPostRedisplay();
}

// Mouse event handlers
//*****************************************************************************
void mouse(int button, int state, int x, int y)
{
    if (state == GLUT_DOWN)
        buttonState |= 1<<button;
    else if (state == GLUT_UP)
        buttonState = 0;

    ox = x; 
    oy = y;
    glutPostRedisplay();
}

void motion(int x, int y)
{
    float dx, dy;
    dx = x - ox;
    dy = y - oy;

    if (buttonState == 3) {
        // left+middle = zoom
        viewTranslation[2] += dy / 100.0;
    } 
    else if (buttonState & 2) {
        // middle = translate
        viewTranslation[0] += dx / 100.0;
        viewTranslation[1] -= dy / 100.0;
    }
    else if (buttonState & 1) {
        // left = rotate
        viewRotation[0] += dy / 5.0;
        viewRotation[1] += dx / 5.0;
    }

    ox = x; 
    oy = y;
    glutPostRedisplay();
}

// Window resize handler callback
//*****************************************************************************
void Reshape(int x, int y)
{
    width = x; height = y;
    initPixelBuffer();

    glViewport(0, 0, x, y);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0); 
}

// Intitialize OpenCL
//*****************************************************************************
void initOpenCL(uchar *h_volume)
{
    ciErrNum = CL_SUCCESS;

#ifdef IMAGE_SUPPORT
    // create 3D array and copy data to device
    cl_image_format volume_format;
    volume_format.image_channel_order = CL_R;
    volume_format.image_channel_data_type = CL_UNORM_INT8;
    

    d_volumeArray = clCreateImage3D(cxGPUContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, &volume_format, 
                                    volumeSize[0],volumeSize[1], volumeSize[2],
                                    volumeSize[0],volumeSize[0] * volumeSize[1],
                                    h_volume, &ciErrNum);
    
    // create transfer function texture
    float transferFunc[] = {
         0.0, 0.0, 0.0, 0.0, 
         1.0, 0.0, 0.0, 1.0, 
         1.0, 0.5, 0.0, 1.0, 
         1.0, 1.0, 0.0, 1.0, 
         0.0, 1.0, 0.0, 1.0, 
         0.0, 1.0, 1.0, 1.0, 
         0.0, 0.0, 1.0, 1.0, 
         1.0, 0.0, 1.0, 1.0, 
         0.0, 0.0, 0.0, 0.0, 
    };

    cl_image_format transferFunc_format;
    transferFunc_format.image_channel_order = CL_RGBA;
    transferFunc_format.image_channel_data_type = CL_FLOAT;

    d_transferFuncArray = clCreateImage2D(cxGPUContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, &transferFunc_format,
                                          9,1, sizeof(float) * 9 * 4,
                                          transferFunc, &ciErrNum);                                          

    ciErrNum |= clSetKernelArg(ckKernel, 8, sizeof(cl_mem), (void *) &d_volumeArray);
    ciErrNum |= clSetKernelArg(ckKernel, 9, sizeof(cl_mem), (void *) &d_transferFuncArray);
    shrCheckErrorEX (ciErrNum, CL_SUCCESS, pCleanup);
#endif    

    // init invViewMatrix
    d_invViewMatrix = clCreateBuffer(cxGPUContext, CL_MEM_READ_ONLY, 12 * sizeof(float), 0, &ciErrNum);
    ciErrNum |= clSetKernelArg(ckKernel, 7, sizeof(cl_mem), (void *) &d_invViewMatrix);
    shrCheckErrorEX (ciErrNum, CL_SUCCESS, pCleanup);
}

// Initialize GL
//*****************************************************************************
void InitGL(int argc, const char **argv)
{
    // initialize GLUT 
    glutInit(&argc, (char **)argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
    glutInitWindowPosition (glutGet(GLUT_SCREEN_WIDTH)/2 - width/2, 
                            glutGet(GLUT_SCREEN_HEIGHT)/2 - height/2);
    glutInitWindowSize(width, height);
    iGLUTWindowHandle = glutCreateWindow("OpenCL volume rendering");

    // register glut callbacks
    glutDisplayFunc(DisplayGL);
    glutKeyboardFunc(KeyboardGL);
    glutMouseFunc(mouse);
    glutMotionFunc(motion);
    glutReshapeFunc(Reshape);
    glutIdleFunc(Idle);

	// initialize necessary OpenGL extensions
    glewInit();
    GLboolean bGLEW = glewIsSupported("GL_VERSION_2_0 GL_ARB_pixel_buffer_object"); 
    shrCheckErrorEX(bGLEW, shrTRUE, pCleanup);
}

// Initialize GL
//*****************************************************************************
void initPixelBuffer()
{
     ciErrNum = CL_SUCCESS;

    if (pbo) {
        // delete old buffer
        clReleaseMemObject(pbo_cl);
        glDeleteBuffersARB(1, &pbo);
    }

    // create pixel buffer object for display
    glGenBuffersARB(1, &pbo);
	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
	glBufferDataARB(GL_PIXEL_UNPACK_BUFFER_ARB, width * height * sizeof(GLubyte) * 4, 0, GL_STREAM_DRAW_ARB);
	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

#ifdef GL_INTEROP
    // create OpenCL buffer from GL PBO
    pbo_cl = clCreateFromGLBuffer(cxGPUContext,CL_MEM_WRITE_ONLY, pbo, &ciErrNum);
#else            
    pbo_cl = clCreateBuffer(cxGPUContext, CL_MEM_WRITE_ONLY, width * height * sizeof(GLubyte) * 4, NULL, &ciErrNum);
#endif

    // calculate new grid size
    gridSize[0] = width;
    gridSize[1] = height;

    ciErrNum |= clSetKernelArg(ckKernel, 0, sizeof(cl_mem), (void *) &pbo_cl);
    ciErrNum |= clSetKernelArg(ckKernel, 1, sizeof(unsigned int), &width);
    ciErrNum |= clSetKernelArg(ckKernel, 2, sizeof(unsigned int), &height);
    shrCheckErrorEX (ciErrNum, CL_SUCCESS, pCleanup);
}

// Run a test sequence without any GL 
//*****************************************************************************
void TestNoGL()
{
    // execute OpenCL kernel without GL interaction
    invViewMatrix[0] = 1; invViewMatrix[1] = 0; invViewMatrix[2] = 0; invViewMatrix[3] = 0;
    invViewMatrix[4] = 0; invViewMatrix[5] = 1; invViewMatrix[6] = 0; invViewMatrix[7] = 0;
    invViewMatrix[8] = 0; invViewMatrix[9] = 0; invViewMatrix[10]= 1; invViewMatrix[11]= -viewTranslation[2];

    pbo_cl = clCreateBuffer(cxGPUContext, CL_MEM_WRITE_ONLY,  width*height*sizeof(GLubyte)*4, NULL, &ciErrNum);
    ciErrNum |= clEnqueueWriteBuffer(cqCommandQueue,d_invViewMatrix,CL_FALSE, 0,12*sizeof(float), invViewMatrix, 0, 0, 0);

    gridSize[0] = width;
    gridSize[1] = height;

    ciErrNum |= clSetKernelArg(ckKernel, 0, sizeof(cl_mem), (void *) &pbo_cl);
    ciErrNum |= clSetKernelArg(ckKernel, 1, sizeof(unsigned int), &width);
    ciErrNum |= clSetKernelArg(ckKernel, 2, sizeof(unsigned int), &height);
    ciErrNum |= clSetKernelArg(ckKernel, 3, sizeof(float), &density);
    ciErrNum |= clSetKernelArg(ckKernel, 4, sizeof(float), &brightness);
    ciErrNum |= clSetKernelArg(ckKernel, 5, sizeof(float), &transferOffset);
    ciErrNum |= clSetKernelArg(ckKernel, 6, sizeof(float), &transferScale);
    
    // Warmup
    ciErrNum |= clEnqueueNDRangeKernel(cqCommandQueue, ckKernel, 2, NULL, gridSize, NULL, 0, 0, 0);
    shrCheckErrorEX (ciErrNum, CL_SUCCESS, pCleanup);
    clFinish(cqCommandQueue);
    
    // Start timer 0 and process n loops on the GPU 
    shrDeltaT(0); 
    for (int i = 0; i < 10; i++)
    {
        ciErrNum |= clEnqueueNDRangeKernel(cqCommandQueue, ckKernel, 2, NULL, gridSize, NULL, 0, 0, 0);
        shrCheckErrorEX (ciErrNum, CL_SUCCESS, pCleanup);
        clFinish(cqCommandQueue);
    }

    // Get elapsed time and throughput, then log to sample and master logs
    double dAvgTime = shrDeltaT(0)/10.0;
    shrLog(LOGBOTH | MASTER, 0, "oclVolumeRender, Throughput = %.4f, Time = %.5f, Size = %u, NumDevsUsed = %u, Workgroup = %u\n", 
           (1.0e-6 * width * height)/dAvgTime, dAvgTime, (width * height), 1, 0); 

    // Cleanup and exit
    Cleanup(EXIT_SUCCESS);
}

// Function to clean up and exit
//*****************************************************************************
void Cleanup(int iExitCode)
{
    // cleanup allocated objects
    shrLog(LOGBOTH, 0, "\nStarting Cleanup...\n\n");
    if(cPathAndName)free(cPathAndName);
    if(cSourceCL)free(cSourceCL);
	if(ckKernel)clReleaseKernel(ckKernel);  
    if(cpProgram)clReleaseProgram(cpProgram);
    if(d_volumeArray)clReleaseMemObject(d_volumeArray);
    if(d_transferFuncArray)clReleaseMemObject(d_transferFuncArray);
    if(pbo_cl)clReleaseMemObject(pbo_cl);    
    if(d_invViewMatrix)clReleaseMemObject(d_invViewMatrix);    
    if(cqCommandQueue)clReleaseCommandQueue(cqCommandQueue);
    if(cxGPUContext)clReleaseContext(cxGPUContext);
	if(!bQATest) 
    {
        glDeleteBuffersARB(1, &pbo);
        if(iGLUTWindowHandle)glutDestroyWindow(iGLUTWindowHandle);        
    }
    shrLog(LOGBOTH, 0, "TEST %s\n\n", iExitCode == 0 ? "PASSED" : "FAILED !!!"); 

    // finalize logs and leave
    if (bNoPrompt || bQATest)
    {
        shrLog(LOGBOTH | CLOSELOG, 0, "oclVolumeRender.exe Exiting...\n");
    }
    else 
    {
        shrLog(LOGBOTH | CLOSELOG, 0, "oclVolumeRender.exe Exiting...\nPress <Enter> to Quit\n");
        #ifdef WIN32
            getchar();
        #endif
    }
    exit (iExitCode);
}
