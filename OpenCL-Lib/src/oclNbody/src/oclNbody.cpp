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

#include "oclUtils.h"
#include <GL/glew.h>
#include <GL/glut.h>
#include <paramgl.h>
#include <cstdlib>
#include <cstdio>
#include <algorithm>
#include "oclBodySystemOpencl.h"
#include "oclBodySystemCpu.h"
#include "oclRenderParticles.h"

// view, GLUT and display params
int ox = 0, oy = 0;
int buttonState          = 0;
float camera_trans[]     = {0, -2, -100};
float camera_rot[]       = {0, 0, 0};
float camera_trans_lag[] = {0, -2, -100};
float camera_rot_lag[]   = {0, 0, 0};
const float inertia      = 0.1;
ParamListGL *paramlist;      // parameter list
ParticleRenderer::DisplayMode displayMode = ParticleRenderer::PARTICLE_SPRITES_COLOR;
bool displayEnabled = true;
bool bPause = false;
bool bUsePBO = false;
bool bFullScreen = false;
bool bShowSliders = true;
int iGLUTWindowHandle;              // handle to the GLUT window
int iGraphicsWinPosX = 0;           // GLUT Window X location
int iGraphicsWinPosY = 0;           // GLUT Window Y location
int iGraphicsWinWidth = 1024;       // GLUT Window width
int iGraphicsWinHeight = 768;       // GL Window height

// Struct defintion for Nbody demo physical parameters
struct NBodyParams
{       
    float m_timestep;
    float m_clusterScale;
    float m_velocityScale;
    float m_softening;
    float m_damping;
    float m_pointSize;
    float m_x, m_y, m_z;

    void print()
    { 
        shrLog(LOGBOTH, 0, "{ %f, %f, %f, %f, %f, %f, %f, %f, %f },\n", 
                   m_timestep, m_clusterScale, m_velocityScale, 
                   m_softening, m_damping, m_pointSize, m_x, m_y, m_z); 
    }
};

// Array of structs of physical parameters to flip among
NBodyParams demoParams[] = 
{
    { 0.016f, 1.54f, 8.0f, 0.1f, 1.0f, 1.25f, 0, -2, -100},
    { 0.016f, 0.68f, 20.0f, 0.1f, 1.0f, 0.8f, 0, -2, -30},
    { 0.0006f, 0.16f, 1000.0f, 1.0f, 1.0f, 0.07f, 0, 0, -1.5f},
    { 0.0006f, 0.16f, 1000.0f, 1.0f, 1.0f, 0.07f, 0, 0, -1.5f},
    { 0.0019f, 0.32f, 276.0f, 1.0f, 1.0f, 0.07f, 0, 0, -5},
    { 0.0016f, 0.32f, 272.0f, 0.145f, 1.0f, 0.08f, 0, 0, -5},
    { 0.016000, 6.040000, 0.000000, 1.000000, 1.000000, 0.760000, 0, 0, -50},
};

// Basic simulation parameters
int numBodies = 30720;
int numDemos = sizeof(demoParams) / sizeof(NBodyParams);
int activeDemo = 0;
NBodyParams activeParams = demoParams[activeDemo];
BodySystem *nbody         = 0;
BodySystemOpenCL *nbodyGPU = 0;
BodySystemCPU  *nbodyCPU  = 0;
float* hPos = 0;
float* hVel = 0;
float* hColor = 0;
ParticleRenderer *renderer = 0;

// OpenCL vars
cl_context cxContext;
cl_device_id* cdDevices;
cl_uint uiNumDevices;           // Number of devices available/used
cl_command_queue cqCommandQueue;
size_t szParmDataBytes;			// Byte size of context information
const char* cExecutablePath;

// fps, quick test and qatest vars
#define DEMOTIME 0
#define FUNCTIME 1
#define FPSTIME 2

int iFrameCount = 0;                // FPS count for averaging
int iFrameTrigger = 90;             // FPS trigger for sampling
int iFramesPerSec = 60;             // frames per second
double dElapsedTime = 0.0;          // timing var to hold elapsed time in each phase of tour mode
double demoTime = 5.0;              // length of each demo phase in sec
shrBOOL bTour = shrTRUE;            // true = cycles between modes, false = stays on selected 1 mode (manually switchable)
shrBOOL bNoPrompt = shrFALSE;       // false = normal GL loop, true = Finite period of GL loop (a few seconds)
shrBOOL bQATest = shrFALSE;         // false = normal GL loop, true = run No-GL test sequence (checks against host and also does a perf test)
shrBOOL bRegression = shrFALSE;     // true = dump output data to a file
int iTestSets = 3;

// Forward Function declarations
//*****************************************************************************
// OpenGL (GLUT) functionality
void InitGL(int argc, char **argv);
void DisplayGL();
void ReshapeGL(int w, int h);
void IdleGL(void);
void KeyboardGL(unsigned char key, int x, int y);
void MouseGL(int button, int state, int x, int y);
void MotionGL(int x, int y);
void SpecialGL (int key, int x, int y);

// Simulation
void ResetSim(BodySystem *system, int numBodies, NBodyConfig config, bool useGL);
void InitNbody(cl_device_id dev, cl_context ctx, cl_command_queue cmdq,
               int numBodies, int p, int q, bool bUsePBO);
void SelectDemo(int index);
void CompareResults(shrBOOL bRegression, int numBodies);
void RunProfiling(int iterations);
void ComputePerfStats(double &dGigaInteractionsPerSecond, double &dGigaFlops, 
                      double dSeconds, int iterations);

// helpers
void Cleanup(int iExitCode);
void (*pCleanup)(int) = &Cleanup;
void TriggerFPSUpdate();

// Main program
//*****************************************************************************
int main(int argc, const char** argv) 
{
    // start logs 
    shrSetLogFileName ("oclNbody.txt");

#ifdef GPU_PROFILING
    shrLog(LOGBOTH, 0, "%s Starting...\n\nRun \"nbody --profiling -n=<numBodies>\" to measure performance\n\n", argv[0]); 
#else
    shrLog(LOGBOTH, 0, "%s Starting...\n\n", argv[0]); 
#endif

    // latch the executable path for other funcs to use, get command line args and set flags accordingly
    cExecutablePath = argv[0];
    int numIterations = 0;  // iterations for profiling    
    int p = 256;
    int q = 1;
    shrGetCmdLineArgumenti(argc, argv, "i", &numIterations);
    shrGetCmdLineArgumenti(argc, argv, "p", &p);
    shrGetCmdLineArgumenti(argc, argv, "q", &q);
    shrGetCmdLineArgumenti(argc, argv, "n", &numBodies);
    bNoPrompt = shrCheckCmdLineFlag(argc, argv, "noprompt");
    bRegression = shrCheckCmdLineFlag(argc, argv, "regression");
    bQATest = shrCheckCmdLineFlag(argc, argv, "qatest");
#ifdef GPU_PROFILING
    bool bProfilingMode = shrCheckCmdLineFlag(argc, argv, "profiling") || shrCheckCmdLineFlag(argc, argv, "qatest");
#else
    bool bProfilingMode = (false || shrCheckCmdLineFlag(argc, argv, "qatest"));
#endif
    bool bCompareToCPU = shrCheckCmdLineFlag(argc, argv, "compare") || shrCheckCmdLineFlag(argc, argv, "qatest");

    // Create the OpenCL context on a GPU device
    cl_int ciErrNum = CL_SUCCESS;
    cxContext = clCreateContextFromType(0, CL_DEVICE_TYPE_GPU, NULL, NULL, &ciErrNum);
    shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup); 
    shrLog(LOGBOTH, 0, "clCreateContextFromType\n"); 

    // Get the list of GPU devices associated with context
    ciErrNum = clGetContextInfo(cxContext, CL_CONTEXT_DEVICES, 0, NULL, &szParmDataBytes);
    cdDevices = (cl_device_id*)malloc(szParmDataBytes);
    ciErrNum |= clGetContextInfo(cxContext, CL_CONTEXT_DEVICES, szParmDataBytes, cdDevices, NULL);
    shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup); 
    shrLog(LOGBOTH, 0, "clGetContextInfo\n\n"); 
    uiNumDevices = (cl_uint)szParmDataBytes/sizeof(cl_device_id);
    oclPrintDevInfo(LOGBOTH, cdDevices[0]);

    // create a command-queue
    cqCommandQueue = clCreateCommandQueue(cxContext, cdDevices[0], 0, &ciErrNum);
    shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup); 
    shrLog(LOGBOTH, 0, "clCreateCommandQueue\n"); 

    switch (numBodies)
    {
        case 1024:
            activeParams.m_clusterScale = 1.52f;
            activeParams.m_velocityScale = 2.f;
            break;
        case 2048:
            activeParams.m_clusterScale = 1.56f;
            activeParams.m_velocityScale = 2.64f;
            break;
        case 4096:
            activeParams.m_clusterScale = 1.68f;
            activeParams.m_velocityScale = 2.98f;
            break;
        case 8192:
            activeParams.m_clusterScale = 1.98f;
            activeParams.m_velocityScale = 2.9f;
            break;
        default:
        case 16384:
            activeParams.m_clusterScale = 1.54f;
            activeParams.m_velocityScale = 8.f;
            break;
        case 32768:
            activeParams.m_clusterScale = 1.44f;
            activeParams.m_velocityScale = 11.f;
            break;
    }

    if ((q * p) > 256)
    {
        p = 256 / q;
        shrLog(LOGBOTH, 0, "Setting p=%d, q=%d to maintain %d threads per block\n", p, q, 256);
    }

    if ((q == 1) && (numBodies < p))
    {
        p = numBodies;
    }

    if (!bProfilingMode && !bCompareToCPU && !bRegression)
    {
        // init the GL objects
	    InitGL(argc, (char**)argv);
    }
	
    //GL interop disabled
    bUsePBO = 0 && !(bProfilingMode || bCompareToCPU);
    InitNbody(cdDevices[0], cxContext, cqCommandQueue, numBodies, p, q, bUsePBO);
    ResetSim(nbody, numBodies, NBODY_CONFIG_SHELL, !(bProfilingMode || bCompareToCPU));

    // init timers
    shrDeltaT(DEMOTIME); // timer 0 is for timing demo periods
    shrDeltaT(FUNCTIME); // timer 1 is for logging function delta t's
    shrDeltaT(FPSTIME);  // timer 2 is for fps measurement   

    // standard simulation
    if (!(bCompareToCPU || bRegression || bProfilingMode))
    {
        shrLog(LOGBOTH, 0, "\nRunning standard oclNbody simulation...\n\n"); 
        glutDisplayFunc(DisplayGL);
        glutReshapeFunc(ReshapeGL);
        glutMouseFunc(MouseGL);
        glutMotionFunc(MotionGL);
        glutKeyboardFunc(KeyboardGL);
        glutSpecialFunc(SpecialGL);
        glutIdleFunc(IdleGL);
        glutMainLoop();
    }
    
    // Compare to host
    if (bCompareToCPU || bRegression)
    {
        shrLog(LOGBOTH, 0, "\nRunning oclNbody Results Comparison...\n\n"); 
        CompareResults(bRegression, numBodies);
    }

    // Profiling mode
    if (bProfilingMode)
    {
        shrLog(LOGBOTH, 0, "\nProfiling oclNbody...\n\n"); 
        if (numIterations <= 0)
        {
            numIterations = 100;
        }
        RunProfiling(numIterations);
    }

    // Cleanup/exit 
    Cleanup(EXIT_SUCCESS);
}
// Setup function for GLUT parameters and loop
//*****************************************************************************
void InitGL(int argc, char **argv)
{  
    // init GLUT 
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_DEPTH | GLUT_DOUBLE);
    glutInitWindowPosition (glutGet(GLUT_SCREEN_WIDTH)/2 - iGraphicsWinWidth/2, 
                            glutGet(GLUT_SCREEN_HEIGHT)/2 - iGraphicsWinHeight/2);
    glutInitWindowSize(iGraphicsWinWidth, iGraphicsWinHeight);
    iGLUTWindowHandle = glutCreateWindow("OpenCL for GPU Nbody Demo");

    // init GLEW
    glewInit();
    GLboolean bGlew = glewIsSupported("GL_VERSION_2_0 "
                         "GL_VERSION_1_5 "
			             "GL_ARB_multitexture "
                         "GL_ARB_vertex_buffer_object"); 
    shrCheckErrorEX(bGlew, shrTRUE, pCleanup);
    glEnable(GL_DEPTH_TEST);
    glClearColor(0.0, 0.0, 0.0, 1.0);
    renderer = new ParticleRenderer;

    // check GL errors
    GLenum error;
    while ((error = glGetError()) != GL_NO_ERROR) 
    {
        shrLog(LOGBOTH, 0, "InitGL: error - %s\n", (char *)gluErrorString(error));
    }

    // create a new parameter list
    paramlist = new ParamListGL("sliders");
    paramlist->bar_col_outer[0] = 0.8f;
    paramlist->bar_col_outer[1] = 0.8f;
    paramlist->bar_col_outer[2] = 0.0f;
    paramlist->bar_col_inner[0] = 0.8f;
    paramlist->bar_col_inner[1] = 0.8f;
    paramlist->bar_col_inner[2] = 0.0f;
    
    // add parameters to the list

    // Point Size
    paramlist->AddParam(new Param<float>("Point Size", activeParams.m_pointSize, 
                    0.0f, 10.0f, 0.01f, &activeParams.m_pointSize));

    // Velocity Damping
    paramlist->AddParam(new Param<float>("Velocity Damping", activeParams.m_damping, 
                    0.5f, 1.0f, .0001f, &(activeParams.m_damping)));

    // Softening Factor
    paramlist->AddParam(new Param<float>("Softening Factor", activeParams.m_softening,
                    0.001f, 1.0f, .0001f, &(activeParams.m_softening)));

    // Time step size
    paramlist->AddParam(new Param<float>("Time Step", activeParams.m_timestep, 
                    0.0f, 1.0f, .0001f, &(activeParams.m_timestep)));

    // Cluster scale (only affects starting configuration
    paramlist->AddParam(new Param<float>("Cluster Scale", activeParams.m_clusterScale, 
                    0.0f, 10.0f, 0.01f, &(activeParams.m_clusterScale)));

    
    // Velocity scale (only affects starting configuration)
    paramlist->AddParam(new Param<float>("Velocity Scale", activeParams.m_velocityScale, 
                    0.0f, 1000.0f, 0.1f, &activeParams.m_velocityScale));
}

// Primary GLUT callback loop function
//*****************************************************************************
void DisplayGL()
{
    // update the simulation, unless paused
    double dProcessingTime = 0.0;
    if (!bPause)
    {
        // start timer FUNCTIME if it's update time
        if (iFrameCount >= iFrameTrigger)
        {
            shrDeltaT(FUNCTIME); 
        }

        // Run the simlation computations
        nbody->update(activeParams.m_timestep); 
        nbody->getArray(BodySystem::BODYSYSTEM_POSITION);

        // Make graphics work with or without CL/GL interop 
        if (bUsePBO) 
        {
            renderer->setPBO((unsigned int)nbody->getCurrentReadBuffer(), nbody->getNumBodies());
        } 
        else 
        { 
            renderer->setPositions((float*)nbody->getCurrentReadBuffer(), nbody->getNumBodies());
        }

        // get processing time from timer FUNCTIME, if it's update time
        if (iFrameCount >= iFrameTrigger)
        {
            dProcessingTime = shrDeltaT(FUNCTIME); 
        }
    }

    // Redraw main graphics display, if enabled
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);  
    if (displayEnabled)
    {
        // view transform
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
        for (int c = 0; c < 3; ++c)
        {
            camera_trans_lag[c] += (camera_trans[c] - camera_trans_lag[c]) * inertia;
            camera_rot_lag[c] += (camera_rot[c] - camera_rot_lag[c]) * inertia;
        }
        glTranslatef(camera_trans_lag[0], camera_trans_lag[1], camera_trans_lag[2]);
        glRotatef(camera_rot_lag[0], 1.0, 0.0, 0.0);
        glRotatef(camera_rot_lag[1], 0.0, 1.0, 0.0);
        renderer->setSpriteSize(activeParams.m_pointSize);
        renderer->display(displayMode);
    }

    // Display user interface if enabled
    if (bShowSliders)
    {
        glBlendFunc(GL_ONE_MINUS_DST_COLOR, GL_ZERO); // invert color
        glEnable(GL_BLEND);
	    paramlist->Render(0, 0);
        glDisable(GL_BLEND);
    }

    // Flip backbuffer to screen 
    glutSwapBuffers();

    //  If frame count has triggerd, increment the frame counter, and do fps stuff 
    if (iFrameCount++ > iFrameTrigger)
    {
        // If tour mode is enabled & interval has timed out, switch to next tour/demo mode
        dElapsedTime += shrDeltaT(DEMOTIME); 
        if (bTour && (dElapsedTime > demoTime))
        {
            dElapsedTime = 0.0;
            activeDemo = (activeDemo + 1) % numDemos;
            SelectDemo(activeDemo);
        }

        // get the perf and fps stats
        iFramesPerSec = (int)((double)iFrameCount/ shrDeltaT(FPSTIME));
        double dGigaInteractionsPerSecond = 0.0;
        double dGigaFlops = 0.0;
        ComputePerfStats(dGigaInteractionsPerSecond, dGigaFlops, dProcessingTime, 1);

        // If not paused, set the display window title, reset trigger and log info
        char cTitle[256];
        if(!bPause) 
        {
        #ifdef GPU_PROFILING
            #ifdef _WIN32
                sprintf_s(cTitle, 
                        "OpenCL for GPU Nbody Demo (%d bodies): %i fps | %0.4f BIPS | %0.4f GFLOP/s", 
		                numBodies, iFramesPerSec, dGigaInteractionsPerSecond, dGigaFlops);  
            #else 
                sprintf(cTitle, 
                        "OpenCL for GPU Nbody Demo (%d bodies): %i fps | %0.4f BIPS | %0.4f GFLOP/s", 
		                numBodies, iFramesPerSec, dGigaInteractionsPerSecond, dGigaFlops);  
            #endif
        #else
            #ifdef _WIN32
                sprintf_s(cTitle, 
                        "OpenCL for GPU Nbody Demo (%d bodies)", 
		                numBodies, iFramesPerSec, dGigaInteractionsPerSecond);  
            #else 
                sprintf(cTitle, 
                        "OpenCL for GPU Nbody Demo (%d bodies)", 
		                numBodies, iFramesPerSec, dGigaInteractionsPerSecond);  
            #endif
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

    glutReportErrors();
}

// GLUT key event handler
//*****************************************************************************
void KeyboardGL(unsigned char key, int /*x*/, int /*y*/)
{
    switch (key) 
    {
        case ' ': // space toggle computation flag on/off
            bPause = !bPause;
            shrLog(LOGBOTH, 0, "\nSim %s...\n\n", bPause ? "Paused" : "Running");
            break;
        case '`':   // Tilda toggles slider display
            bShowSliders = !bShowSliders;
            shrLog(LOGBOTH, 0, "\nSlider Display %s...\n\n", bShowSliders ? "ON" : "OFF");
            break;
        case 'p':   // 'p' falls through to 'P' 
        case 'P':   // p switched between points and blobs 
            displayMode = (ParticleRenderer::DisplayMode)((displayMode + 1) % ParticleRenderer::PARTICLE_NUM_MODES);
            break;
        case 'c':   // 'c' falls through to 'C'
        case 'C':   // c switches between cycle demo mode and fixed demo mode
            bTour = bTour ? shrFALSE : shrTRUE;
            shrLog(LOGBOTH, 0, "\nTour Mode %s...\n\n", bTour ? "ON" : "OFF");
            break;
        case '[':
            activeDemo = (activeDemo == 0) ? numDemos - 1 : (activeDemo - 1) % numDemos;
            SelectDemo(activeDemo);
            break;
        case ']':
            activeDemo = (activeDemo + 1) % numDemos;
            SelectDemo(activeDemo);
            break;
        case 'd':   // 'd' falls through to 'D'
        case 'D':   // d toggled main graphics display on/off
            displayEnabled = !displayEnabled;
            shrLog(LOGBOTH, 0, "\nMain Graphics Display %s...\n\n", displayEnabled ? "ON" : "OFF");
            break;
        case 'f':   // 'f' falls through to 'F'
        case 'F':   // f toggles main graphics display full screen
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
            shrLog(LOGBOTH, 0, "\nMain Graphics %s...\n\n", bFullScreen ? "FullScreen" : "Windowed");
            break;
        case 'o':   // 'o' falls through to 'O'
        case 'O':   // 'O' prints Nbody sim physical parameters
            activeParams.print();
            break;
        case 'T':   // Toggles from (T)our mode to standard mode and back
        case 't':   // Toggles from (t)our mode to standard mode and back
            bTour = bTour ? shrFALSE : shrTRUE;
            shrLog(LOGBOTH, 0, "\nTour Mode %s...\n", bTour ? "ON" : "OFF");
            break;
        case '1':
            ResetSim(nbody, numBodies, NBODY_CONFIG_SHELL, true);
            break;
        case '2':
            ResetSim(nbody, numBodies, NBODY_CONFIG_RANDOM, true);
            break;
        case '3':
            ResetSim(nbody, numBodies, NBODY_CONFIG_EXPAND, true);
            break;
        case '\033': // escape quits
        case '\015': // Enter quits    
        case 'Q':    // Q quits
        case 'q':    // q (or escape) quits
            // Cleanup up and quit
            Cleanup(EXIT_SUCCESS);
            break;
    }

    // Trigger fps update and call for refresh
    TriggerFPSUpdate();
    glutPostRedisplay();
}

//*****************************************************************************
void RunProfiling(int iterations)
{
    // once without timing to prime the GPU
    nbody->update(activeParams.m_timestep);
    clFinish(cqCommandQueue);

	// Start timer 0 and process n loops on the GPU
    shrDeltaT(FUNCTIME);  
    for (int i = 0; i < iterations; ++i)
    {
        nbody->update(activeParams.m_timestep);
    }
    nbody->synchronizeThreads();

    // Get elapsed time and throughput, then log to sample and master logs
    double dSeconds = shrDeltaT(FUNCTIME);
    double dGigaInteractionsPerSecond = 0.0;
    double dGigaFlops = 0.0;
    ComputePerfStats(dGigaInteractionsPerSecond, dGigaFlops, dSeconds, iterations);
     shrLog(LOGBOTH | MASTER, 0, "oclNBody, Throughput = %.4f, Time = %.5f, Size = %u, NumDevsUsed = %u\n", 
           dGigaFlops, dSeconds/(double)iterations, numBodies, 1); 
}

// Handler for GLUT window resize event
//*****************************************************************************
void ReshapeGL(int w, int h)
{
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, (float) w / (float) h, 0.1, 1000.0);
    glMatrixMode(GL_MODELVIEW);
    glViewport(0, 0, w, h);
}

// Handler for GLU Mouse events
//*****************************************************************************
void MouseGL(int button, int state, int x, int y)
{
    if (bShowSliders) 
    {
	    // call list mouse function
        if (paramlist->Mouse(x, y, button, state))
        {
            nbody->setSoftening(activeParams.m_softening);
            nbody->setDamping(activeParams.m_damping);
        }
    }
    
    int mods;

    if (state == GLUT_DOWN)
    {
        buttonState |= 1<<button;
    }
    else if (state == GLUT_UP)
    {
        buttonState = 0;
    }

    mods = glutGetModifiers();
    if (mods & GLUT_ACTIVE_SHIFT) 
    {
        buttonState = 2;
    } 
    else if (mods & GLUT_ACTIVE_CTRL) 
    {
        buttonState = 3;
    }

    ox = x; 
    oy = y;

    glutPostRedisplay();
}

//*****************************************************************************
void MotionGL(int x, int y)
{
    if (bShowSliders) 
    {
        // call parameter list motion function
        if (paramlist->Motion(x, y))
	    {
            nbody->setSoftening(activeParams.m_softening);
            nbody->setDamping(activeParams.m_damping);
            glutPostRedisplay();
	        return;
        }
    }

    float dx = x - ox;
    float dy = y - oy;

    if (buttonState == 3) 
    {
        // left+middle = zoom
        camera_trans[2] += (dy * 0.01) * 0.5 * fabs(camera_trans[2]);
    } 
    else if (buttonState & 2) 
    {
        // middle = translate
        camera_trans[0] += (dx * 0.01);
        camera_trans[1] -= (dy * 0.01);
    }
    else if (buttonState & 1) 
    {
        // left = rotate
        camera_rot[0] += (dy * 0.20);
        camera_rot[1] += (dx * 0.20);
    }
    
    ox = x; 
    oy = y;
    glutPostRedisplay();
}

//*****************************************************************************
void SpecialGL(int key, int x, int y)
{
    paramlist->Special(key, x, y);
    glutPostRedisplay();
}

//*****************************************************************************
void IdleGL(void)
{
    glutPostRedisplay();
}

// Helper to trigger reset of fps vars at transition 
//*****************************************************************************
void TriggerFPSUpdate()
{
    iFrameCount = 0; 
    shrDeltaT(FPSTIME);
    iFramesPerSec = 1;
    iFrameTrigger = 2;
}

//*****************************************************************************
void ResetSim(BodySystem *system, int numBodies, NBodyConfig config, bool useGL)
{
    shrLog(LOGBOTH, 0, "\nReset Nbody System...\n");

    // initalize the memory
    randomizeBodies(config, hPos, hVel, hColor, activeParams.m_clusterScale, 
		            activeParams.m_velocityScale, numBodies);

    system->setArray(BodySystem::BODYSYSTEM_POSITION, hPos);
    system->setArray(BodySystem::BODYSYSTEM_VELOCITY, hVel);
    if (useGL)
    {
        renderer->setColors(hColor, nbody->getNumBodies());
        renderer->setSpriteSize(activeParams.m_pointSize);
    }
}

//*****************************************************************************
void InitNbody(cl_device_id dev, cl_context ctx, cl_command_queue cmdq,
               int numBodies, int p, int q, bool bUsePBO)
{
    nbodyGPU = new BodySystemOpenCL(numBodies, dev, ctx, cmdq, p, q, bUsePBO);
    nbody = nbodyGPU;

    // allocate host memory
    hPos = new float[numBodies*4];
    hVel = new float[numBodies*4];
    hColor = new float[numBodies*4];

    nbody->setSoftening(activeParams.m_softening);
    nbody->setDamping(activeParams.m_damping);

    // Start the demo timer
    shrDeltaT(DEMOTIME);
}

//*****************************************************************************
void SelectDemo(int index)
{
    shrCheckErrorEX((index < numDemos), shrTRUE, pCleanup);

    activeParams = demoParams[index];
    camera_trans[0] = camera_trans_lag[0] = activeParams.m_x;
    camera_trans[1] = camera_trans_lag[1] = activeParams.m_y;
    camera_trans[2] = camera_trans_lag[2] = activeParams.m_z;
    ResetSim(nbody, numBodies, NBODY_CONFIG_SHELL, true);

    //Rest the demo timer
    shrDeltaT(DEMOTIME);
}

//*****************************************************************************
void CompareResults(shrBOOL bRegression, int numBodies)
{
    nbodyGPU->update(0.001f);

    // check result
    if(bRegression) 
    {
        // write file for bRegression test
        shrWriteFilef( "./data/bRegression.dat",
            nbodyGPU->getArray(BodySystem::BODYSYSTEM_POSITION), 
            numBodies, 0.0);
    }
    else
    {
        nbodyCPU = new BodySystemCPU(numBodies);
        nbodyCPU->setArray(BodySystem::BODYSYSTEM_POSITION, hPos);
        nbodyCPU->setArray(BodySystem::BODYSYSTEM_VELOCITY, hVel);
        nbodyCPU->update(0.001f);

        float* gpuPos = nbodyGPU->getArray(BodySystem::BODYSYSTEM_POSITION);
        float* cpuPos  = nbodyCPU->getArray(BodySystem::BODYSYSTEM_POSITION);

        // custom output handling when no bRegression test running
        // in this case check if the result is equivalent to the expected 
	    // solution
        shrBOOL res = shrComparefe( cpuPos, gpuPos, numBodies, .001f);
        shrLog(LOGBOTH, 0, "TEST %s\n\n", (1 == res) ? "PASSED" : "FAILED !!!");
    }
}

//*****************************************************************************
void ComputePerfStats(double &dGigaInteractionsPerSecond, double &dGigaFlops, double dSeconds, int iterations)
{
    dGigaInteractionsPerSecond = 1.0e-9 * (double)numBodies * (double)numBodies * (double)iterations / dSeconds;
    dGigaFlops = dGigaInteractionsPerSecond * 20.0; // assumes 20 flops per interactions
}

// Helper to clean up
//*****************************************************************************
void Cleanup(int iExitCode)
{
    // Cleanup allocated objects
    shrLog(LOGBOTH , 0.0, "\nStarting Cleanup...\n\n");
    if(cdDevices)free(cdDevices);
    if(nbodyCPU) delete nbodyCPU;
    if(nbodyGPU) delete nbodyGPU;
    if(cqCommandQueue)clReleaseCommandQueue(cqCommandQueue);
    if(cxContext) clReleaseContext(cxContext);
    if(hPos) delete [] hPos;
    if(hVel) delete [] hVel;
    if(hColor) delete [] hColor;
    if(iGLUTWindowHandle)glutDestroyWindow(iGLUTWindowHandle);

    // finalize logs and leave
    if ((bNoPrompt) || bQATest)
    {
        shrLog(LOGBOTH | CLOSELOG, 0, "oclNbody.exe Exiting...\n");
    }
    else 
    {
        shrLog(LOGBOTH | CLOSELOG, 0, "oclNbody.exe Exiting...\nPress <Enter> to Quit\n");
        #ifdef WIN32
            getchar();
        #endif
    }
    exit (iExitCode);
}
