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
    This example demonstrates how to use the OpenCL/OpenGL interoperability to
    dynamically modify a vertex buffer using a OpenCL kernel.

    The steps are:
    1. Create an empty vertex buffer object (VBO)
    2. Create an OpenCL memory object from the vertex buffer object
    3. Acquire the VBO for writing from OpenCL
    4. Run OpenCL kernel to modify the vertex positions
    5. Release the VBO for returning ownership to OpenGL
    6. Render the results using OpenGL

    Host code
*/

// standard utility and system includes
#include <oclUtils.h>

// GLEW and GLUT includes
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

// Rendering window vars
const unsigned int window_width = 512;
const unsigned int window_height = 512;
const unsigned int mesh_width = 256;
const unsigned int mesh_height = 256;

// OpenCL vars
cl_context cxGPUContext;
cl_device_id cdDevice;
cl_command_queue cqCommandQueue;
cl_kernel ckKernel;
cl_mem vbo_cl;
cl_program cpProgram;
cl_int ciErrNum;
char* cPathAndName = NULL;          // var for full paths to data, src, etc.
char* cSourceCL = NULL;             // Buffer to hold source for compilation 

// vbo variables
GLuint vbo;
int iGLUTWindowHandle = 0;              // handle to the GLUT window

// mouse controls
int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
float rotate_x = 0.0, rotate_y = 0.0;
float translate_z = -3.0;

// Sim and Auto-Verification parameters 
float anim = 0.0;
int iFrameCount = 0;                // FPS count for averaging
int iFrameTrigger = 90;             // FPS trigger for sampling
int iFramesPerSec = 0;              // frames per second
int iTestSets = 3;
const int iRefFrameNumber = 4;
int g_Index = 0;
shrBOOL bQATest = shrFALSE;
shrBOOL g_bFBODisplay = shrFALSE;
shrBOOL bNoPrompt = shrFALSE;  

// Forward Function declarations
//*****************************************************************************
// OpenCL functionality
void runKernel();
void checkResultOpenCL(int argc, const char** argv, const GLuint& vbo);

// GL functionality
void InitGL(int argc, const char** argv);
void createVBO(GLuint* vbo);
void deleteVBO(GLuint* vbo);
void DisplayGL();
void KeyboardGL(unsigned char key, int x, int y);
void mouse(int button, int state, int x, int y);
void motion(int x, int y);

// Helpers
void TestNoGL();
void Cleanup(int iExitCode);
void (*pCleanup)(int) = &Cleanup;

// Main program
//*****************************************************************************
int main(int argc, const char **argv)
{
    // start logs 
    shrSetLogFileName ("oclSimpleGL.txt");
    shrLog(LOGBOTH, 0, "%s Starting...\n\n", argv[0]); 

    // check command line args
    if (argc > 1) 
    {
        bQATest = shrCheckCmdLineFlag(argc, argv, "qatest");
        bNoPrompt = shrCheckCmdLineFlag(argc, argv, "noprompt");
    }

    // Initialize OpenGL context, so we can properly set the GL for CL.
    if(!bQATest)
    {
        InitGL(argc, argv);
    } 

    // create the OpenCL context 
    cxGPUContext = clCreateContextFromType(0, CL_DEVICE_TYPE_GPU, NULL, NULL, &ciErrNum);
    shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

    // Get and log the device info
    if(shrCheckCmdLineFlag(argc, argv, "device"))
    {
      int device_nr = 0;
      shrGetCmdLineArgumenti(argc, argv, "device", &device_nr);
      cdDevice = oclGetDev(cxGPUContext, device_nr);
    } 
    else 
    {
      cdDevice = oclGetMaxFlopsDev(cxGPUContext);
    }
    oclPrintDevInfo(LOGBOTH, cdDevice);

    // create a command-queue
    cqCommandQueue = clCreateCommandQueue(cxGPUContext, cdDevice, 0, &ciErrNum);
    shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

    // Program Setup
    size_t program_length;
    cPathAndName = shrFindFilePath("simpleGL.cl", argv[0]);
    shrCheckErrorEX(cPathAndName != NULL, shrTRUE, pCleanup);
    cSourceCL = oclLoadProgSource(cPathAndName, "", &program_length);
    shrCheckErrorEX(cSourceCL != NULL, shrTRUE, pCleanup);

    // create the program
    cpProgram = clCreateProgramWithSource(cxGPUContext, 1,
					  (const char **) &cSourceCL, &program_length, &ciErrNum);
    shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

    // build the program
    ciErrNum = clBuildProgram(cpProgram, 0, NULL, "-cl-mad-enable", NULL, NULL);
    if (ciErrNum != CL_SUCCESS)
    {
        // write out standard error, Build Log and PTX, then cleanup and exit
        shrLog(LOGBOTH | ERRORMSG, ciErrNum, STDERROR);
        oclLogBuildInfo(cpProgram, oclGetFirstDev(cxGPUContext));
        oclLogPtx(cpProgram, oclGetFirstDev(cxGPUContext), "oclSimpleGL.ptx");
        Cleanup(EXIT_FAILURE); 
    }

    // create the kernel
    ckKernel = clCreateKernel(cpProgram, "sine_wave", &ciErrNum);
    shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

    // create VBO
    if(!bQATest)
    {
        createVBO(&vbo);
    }

    // set the args values 
    ciErrNum  = clSetKernelArg(ckKernel, 0, sizeof(cl_mem), (void *) &vbo_cl);
    ciErrNum |= clSetKernelArg(ckKernel, 1, sizeof(unsigned int), &mesh_width);
    ciErrNum |= clSetKernelArg(ckKernel, 2, sizeof(unsigned int), &mesh_height);
    shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

    // If specified, compute and save off data for regression tests
    if(shrCheckCmdLineFlag(argc, (const char**) argv, "regression")) 
    {
        // run OpenCL kernel once to generate vertex positions
        runKernel();
        checkResultOpenCL(argc, argv, vbo);
    }

    // init timer 1 for fps measurement 
    shrDeltaT(1);  

    // Start main GLUT rendering loop for processing and rendering, 
	// or otherwise run No-GL Q/A test sequence
    if(!bQATest) 
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

// Initialize GL
//*****************************************************************************
void InitGL(int argc, const char** argv)
{
    // initialize GLUT 
    glutInit(&argc, (char**)argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowPosition (glutGet(GLUT_SCREEN_WIDTH)/2 - window_width/2, 
                            glutGet(GLUT_SCREEN_HEIGHT)/2 - window_height/2);
    glutInitWindowSize(window_width, window_height);
    iGLUTWindowHandle = glutCreateWindow("OpenCL/GL Interop (VBO)");

    // register GLUT callback functions
    glutDisplayFunc(DisplayGL);
    glutKeyboardFunc(KeyboardGL);
    glutMouseFunc(mouse);
    glutMotionFunc(motion);

	// initialize necessary OpenGL extensions
    glewInit();
    GLboolean bGLEW = glewIsSupported("GL_VERSION_2_0 GL_ARB_pixel_buffer_object"); 
    shrCheckErrorEX(bGLEW, shrTRUE, pCleanup);

    // default initialization
    glClearColor(0.0, 0.0, 0.0, 1.0);
    glDisable(GL_DEPTH_TEST);

    // viewport
    glViewport(0, 0, window_width, window_height);

    // projection
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, (GLfloat)window_width / (GLfloat) window_height, 0.1, 10.0);

    // set view matrix
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(0.0, 0.0, translate_z);
    glRotatef(rotate_x, 1.0, 0.0, 0.0);
    glRotatef(rotate_y, 0.0, 1.0, 0.0);

    return;
}

// Run the OpenCL part of the computation
//*****************************************************************************
void runKernel()
{
    ciErrNum = CL_SUCCESS;
    size_t szGlobalWorkSize[2];
 
#ifdef GL_INTEROP   
    // map OpenGL buffer object for writing from OpenCL
    ciErrNum |= clEnqueueAcquireGLObjects(cqCommandQueue, 1, &vbo_cl, 0,0,0);
#endif

    // Set work size and execute the kernel
    szGlobalWorkSize[0] = mesh_width;
    szGlobalWorkSize[1] = mesh_height;
    ciErrNum |= clSetKernelArg(ckKernel, 3, sizeof(float), &anim);
    ciErrNum |= clEnqueueNDRangeKernel(cqCommandQueue, ckKernel, 2, NULL, szGlobalWorkSize, NULL, 0,0,0 );
    shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

#ifdef GL_INTEROP
    // unmap buffer object
    ciErrNum |= clEnqueueReleaseGLObjects(cqCommandQueue, 1, &vbo_cl, 0,0,0);
#else

    // Explicit Copy 
    // map the PBO to copy data from the CL buffer via host
    glBindBufferARB(GL_ARRAY_BUFFER, vbo);    

    // map the buffer object into client's memory
    void* ptr = glMapBufferARB(GL_ARRAY_BUFFER,
                               GL_WRITE_ONLY_ARB);

    ciErrNum |= clEnqueueReadBuffer(cqCommandQueue, vbo_cl, CL_TRUE, 0, sizeof(float) * 4 * mesh_height * mesh_width, ptr, 0, NULL, NULL);
    shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

    glUnmapBufferARB(GL_ARRAY_BUFFER); 
#endif
}

// Create VBO
//*****************************************************************************
void createVBO(GLuint* vbo)
{
    // create buffer object
    glGenBuffers(1, vbo);
    glBindBuffer(GL_ARRAY_BUFFER, *vbo);

    // initialize buffer object
    unsigned int size = mesh_width * mesh_height * 4 * sizeof(float);
    glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);

#ifdef GL_INTEROP
    // create OpenCL buffer from GL VBO
    vbo_cl = clCreateFromGLBuffer(cxGPUContext,CL_MEM_WRITE_ONLY, *vbo, NULL);
#else            
    vbo_cl = clCreateBuffer(cxGPUContext, CL_MEM_WRITE_ONLY, size, NULL, &ciErrNum);
#endif
    shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
}

// Delete VBO
//*****************************************************************************
void deleteVBO(GLuint* vbo) {
    clReleaseMemObject(vbo_cl);

    glBindBuffer(1, *vbo);
    glDeleteBuffers(1, vbo);

    *vbo = 0;
}

// Display callback
//*****************************************************************************
void DisplayGL()
{
    // increment the geometry computation parameter (or set to reference for Q/A check)
    if (iFrameCount < iFrameTrigger)
    {
        anim += 0.01f;
    }

    // start timer 0 if it's update time
    double dProcessingTime = 0.0;
    if (iFrameCount >= iFrameTrigger)
    {
        shrDeltaT(0); 
    }

    // run OpenCL kernel to generate vertex positions
    runKernel();

    // get processing time from timer 0, if it's update time
    if (iFrameCount >= iFrameTrigger)
    {
        dProcessingTime = shrDeltaT(0); 
    }

    // clear graphics then render from the vbo
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glVertexPointer(4, GL_FLOAT, 0, 0);
    glEnableClientState(GL_VERTEX_ARRAY);
    glColor3f(1.0, 0.0, 0.0);
    glDrawArrays(GL_POINTS, 0, mesh_width * mesh_height);
    glDisableClientState(GL_VERTEX_ARRAY);

    // flip backbuffer to screen
    glutSwapBuffers();
    glutPostRedisplay();

    // Increment the frame counter, and do fps if it's time
    if (iFrameCount++ > iFrameTrigger) 
    {
        // set GLUT Window Title
        char cTitle[256];
        iFramesPerSec = (int)((double)iFrameCount/shrDeltaT(1));
#ifdef GPU_PROFILING
        #ifdef _WIN32
            sprintf_s(cTitle, 256, "OpenCL Simple GL (VBO) | %u x %u | %i fps | Proc. t = %.3f s", 
                      mesh_width, mesh_height, iFramesPerSec, dProcessingTime);
        #else 
            sprintf(cTitle, "OpenCL Simple GL (VBO) | %u x %u | %i fps | Proc. t = %.3f s", 
                    mesh_width, mesh_height, iFramesPerSec, dProcessingTime);
        #endif
#else
        #ifdef _WIN32
            sprintf_s(cTitle, 256, "OpenCL Simple GL (VBO) | W: %u  H: %u", mesh_width, mesh_height );
        #else 
            sprintf(cTitle, "OpenCL Simple GL (VBO) | W: %u  H: %u", mesh_width, mesh_height);
        #endif
#endif
        glutSetWindowTitle(cTitle);

        // Log fps and processing info to console and file 
        shrLog(LOGBOTH, 0, " %s\n", cTitle); 

        // Cleanup up and quit if requested and counter is up
        iTestSets--;
        if (bNoPrompt && (!iTestSets)) 
        {
            Cleanup(EXIT_SUCCESS);
        }

        // reset framecount, trigger and timer
        iFrameCount = 0; 
        iFrameTrigger = (iFramesPerSec > 1) ? iFramesPerSec * 2 : 1;
    }
}

// Keyboard events handler
//*****************************************************************************
void KeyboardGL(unsigned char key, int x, int y)
{
    switch(key) 
    {
        case '\033': // escape quits
        case '\015': // Enter quits    
        case 'Q':    // Q quits
        case 'q':    // q (or escape) quits
            // Cleanup up and quit
	        Cleanup(EXIT_SUCCESS);
            break;
    }
}

// Mouse event handlers
//*****************************************************************************
void mouse(int button, int state, int x, int y)
{
    if (state == GLUT_DOWN) {
        mouse_buttons |= 1<<button;
    } else if (state == GLUT_UP) {
        mouse_buttons = 0;
    }

    mouse_old_x = x;
    mouse_old_y = y;

    glutPostRedisplay();
}

void motion(int x, int y)
{
    float dx, dy;
    dx = x - mouse_old_x;
    dy = y - mouse_old_y;

    if (mouse_buttons & 1) {
        rotate_x += dy * 0.2;
        rotate_y += dx * 0.2;
    } else if (mouse_buttons & 4) {
        translate_z += dy * 0.01;
    }

    mouse_old_x = x;
    mouse_old_y = y;

    // set view matrix
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(0.0, 0.0, translate_z);
    glRotatef(rotate_x, 1.0, 0.0, 0.0);
    glRotatef(rotate_y, 0.0, 1.0, 0.0);
    glutPostRedisplay();
}

// If specified, write data to file for external regression testing
//*****************************************************************************
void checkResultOpenCL(int argc, const char** argv, const GLuint& vbo)
{
    // map buffer object
    glBindBuffer(GL_ARRAY_BUFFER_ARB, vbo);
    float* data = (float*)glMapBuffer(GL_ARRAY_BUFFER, GL_READ_ONLY);

    // save data for regression testing result
    shrWriteFilef("./data/regression.dat",
                      data, mesh_width * mesh_height * 3, 0.0);

    // unmap GL buffer object
    if(!glUnmapBuffer(GL_ARRAY_BUFFER))
    {
        shrLog(LOGBOTH, 0, "Unmap buffer failed !\n");
    }
}

// Run a test sequence without any GL 
//*****************************************************************************
void TestNoGL()
{
    // Set work size and execute the kernel without GL 
    ciErrNum = CL_SUCCESS;
    size_t szGlobalWorkSize[2];
    szGlobalWorkSize[0] = mesh_width;
    szGlobalWorkSize[1] = mesh_height;

    unsigned int size = mesh_width * mesh_height * 4 * sizeof(float);
    vbo_cl = clCreateBuffer(cxGPUContext, CL_MEM_WRITE_ONLY, size, NULL, &ciErrNum);

    ciErrNum  = clSetKernelArg(ckKernel, 0, sizeof(cl_mem), (void *) &vbo_cl);
    ciErrNum |= clSetKernelArg(ckKernel, 1, sizeof(unsigned int), &mesh_width);
    ciErrNum |= clSetKernelArg(ckKernel, 2, sizeof(unsigned int), &mesh_height);
    ciErrNum |= clSetKernelArg(ckKernel, 3, sizeof(float), &anim);
    shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

    // Warmup call to assure OpenCL driver is awake
    ciErrNum = clEnqueueNDRangeKernel(cqCommandQueue, ckKernel, 2, NULL, szGlobalWorkSize, NULL, 0, 0, 0 );
    shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    clFinish(cqCommandQueue);

    // Start timer 0 and process n loops on the GPU 
    shrDeltaT(0); 
    for (int i = 0; i < 10; i++)
    {
        ciErrNum = clEnqueueNDRangeKernel(cqCommandQueue, ckKernel, 2, NULL, szGlobalWorkSize, NULL, 0, 0, 0 );
        shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
        
        clFinish(cqCommandQueue);
    }

    // Get elapsed time and throughput, then log to sample and master logs
    double dAvgTime = shrDeltaT(0)/10.0;
    shrLog(LOGBOTH | MASTER, 0, "oclSimpleGL, Throughput = %.4f, Time = %.5f, Size = %u, NumDevsUsed = %u\n", 
           (1.0e-6 * mesh_width * mesh_height)/dAvgTime, dAvgTime, (mesh_width * mesh_height), 1); 

    // Cleanup and exit
    Cleanup(EXIT_SUCCESS);
}

// Function to clean up and exit
//*****************************************************************************
void Cleanup(int iExitCode)
{
    // Cleanup allocated objects
    shrLog(LOGBOTH, 0, "\nStarting Cleanup...\n\n");
    if(cPathAndName)free(cPathAndName);
    if(cSourceCL)free(cSourceCL);
    if(vbo_cl && vbo)deleteVBO(&vbo);
	if(ckKernel)clReleaseKernel(ckKernel); 
    if(cpProgram)clReleaseProgram(cpProgram);
    if(cqCommandQueue)clReleaseCommandQueue(cqCommandQueue);
    if(cxGPUContext)clReleaseContext(cxGPUContext);
    if(iGLUTWindowHandle)glutDestroyWindow(iGLUTWindowHandle);
    shrLog(LOGBOTH, 0, "TEST %s\n\n", iExitCode == 0 ? "PASSED" : "FAILED !!!"); 

    // finalize logs and leave
    if (bQATest || bNoPrompt)
    {
        shrLog(LOGBOTH | CLOSELOG, 0, "oclSimpleGL.exe Exiting...\n");
    }
    else 
    {
        shrLog(LOGBOTH | CLOSELOG, 0, "oclSimpleGL.exe Exiting...\nPress <Enter> to Quit\n");
        #ifdef WIN32
            getchar();
        #endif
    }
    exit (iExitCode);
}
