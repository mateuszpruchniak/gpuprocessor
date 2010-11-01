/*
 * Copyright 1993-2009 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and 
 * proprietary rights in and to this software and related documentation. 
 * Any use, reproduction, disclosure, or distribution of this software 
 * and related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 * 
 */
 
 #ifndef __REDUCTION_H__
#define __REDUCTION_H__

template <class T>
void reduce_sm10(int size, int threads, int blocks, 
                 int whichKernel, T *d_idata, T *d_odata);

template <class T>
void reduce_sm13(int size, int threads, int blocks, 
                 int whichKernel, T *d_idata, T *d_odata);

// CL objects
cl_context cxGPUContext;
cl_command_queue cqCommandQueue;
cl_device_id device;
cl_int ciErrNum;
const char* source_path;
bool smallBlock = true;

#endif
