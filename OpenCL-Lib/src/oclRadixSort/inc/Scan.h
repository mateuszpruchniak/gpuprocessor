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
#ifndef _SCAN_H_
#define _SCAN_H_

#define SCAN2

#include <CL/cl.h>

#define MAX_WORKGROUP_INCLUSIVE_SCAN_SIZE 2048
#define MAX_LOCAL_GROUP_SIZE 512

class Scan
{
public:
	Scan(cl_context GPUContext,
		 cl_command_queue CommandQue,
		 unsigned int numElements, 
		 const char *path);
	Scan() {}
	~Scan();

	void scanExclusiveLarge(
						cl_mem d_Dst,
						cl_mem d_Src,
						unsigned int batchSize,
						unsigned int arrayLength);

private:
	cl_context cxGPUContext;             // OpenCL context
    cl_command_queue cqCommandQueue;     // OpenCL command que   
    cl_program cpProgram;                // OpenCL program
	cl_mem d_Buffer;                     // Memory objects for original keys and work space
	cl_kernel ckScanExclusiveLocal1, ckScanExclusiveLocal2, ckUniformUpdate;

	static const int WORKGROUP_SIZE = 512;
	static const unsigned int   MAX_BATCH_ELEMENTS = 64 * 1048576;
	static const unsigned int MIN_SHORT_ARRAY_SIZE = 4;
	static const unsigned int MAX_SHORT_ARRAY_SIZE = 4 * WORKGROUP_SIZE;
	static const unsigned int MIN_LARGE_ARRAY_SIZE = 8 * WORKGROUP_SIZE;
	static const unsigned int MAX_LARGE_ARRAY_SIZE = 4 * WORKGROUP_SIZE * WORKGROUP_SIZE;

	unsigned int  mNumElements;     // Number of elements of temp storage allocated

	// kernel wrappers

	void scanExclusiveLocal1(
			cl_mem d_Dst,
			cl_mem d_Src,
			unsigned int n,
			unsigned int size);
	void scanExclusiveLocal2(
			cl_mem d_Buffer,
			cl_mem d_Dst,
			cl_mem d_Src,
			unsigned int n,
			unsigned int size);
	void uniformUpdate(
			cl_mem d_Dst,
			cl_mem d_Buffer,
			unsigned int n);
	static unsigned int iSnapUp(unsigned int dividend, unsigned int divisor)
	{
		return ((dividend % divisor) == 0) ? dividend : (dividend - dividend % divisor + divisor);
	}
	unsigned int factorRadix2(unsigned int& log2L, unsigned int L)
	{
		if(!L)
		{
			log2L = 0;
		return 0;
		} else {
			for(log2L = 0; (L & 1) == 0; L >>= 1, log2L++);
			return L;
		}
	}
         
};
#endif