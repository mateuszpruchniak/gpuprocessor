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

#ifndef __CL_BODYSYSTEMOPENCL_LAUNCH_H
    #define __CL_BODYSYSTEMOPENCL_LAUNCH_H

    #ifdef __cplusplus
        extern "C"
        {
    #endif
            #include <GL/glew.h>
            #include "CL/cl.h"
            #include "oclBodySystemOpencl.h"

                int  CreateProgramAndKernel(cl_context ctx, cl_device_id* cdDevices, const char* kernel_name, cl_kernel* kernel);
                void AllocateNBodyArrays(cl_context ctx, cl_mem* vel, int numBodies);
                void DeleteNBodyArrays(cl_mem* vel);


                void IntegrateNbodySystem(cl_command_queue cqCommandQueue,
                                          cl_kernel MT_kernel, cl_kernel noMT_kernel,
                                          cl_mem newPos, cl_mem newVel,
                                          cl_mem oldPos, cl_mem oldVel,
                                          cl_mem pboCLOldPos, cl_mem pboCLNewPos,
                                          float deltaTime, float damping, float softSq,
                                          int numBodies, int p, int q,
                                          int bUsePBO);
                void CopyArrayFromDevice(cl_command_queue cmdq, float *host, cl_mem device, cl_mem pboCL, int numBodies);
                void CopyArrayToDevice(cl_command_queue cmdq, cl_mem device, const float *host, int numBodies);
                cl_mem RegisterGLBufferObject(cl_context ctx, unsigned int pboGL);
                void UnregisterGLBufferObject(cl_mem pboCL);
                void ThreadSync(cl_command_queue cmdq);

    #ifdef __cplusplus
        }
    #endif

#endif
