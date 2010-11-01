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

#ifndef _HMM_H_
#define _HMM_H_

#include <CL/cl.h>

class HMM
{
public:
    HMM(cl_context GPUContext,
        cl_command_queue CommandQue,
        float *initProb,
        float *mtState,
        float *mtEmit,
        int numState,
        int numEmit,
		int numObs,
        const char *path,
		int workgroupSize);
    HMM() {};
    ~HMM();

    size_t ViterbiSearch(cl_mem vProb, 
                       cl_mem vPath, 
                       int *obs);

private:
    cl_context cxGPUContext;             // OpenCL context
    cl_command_queue cqCommandQue;       // OpenCL command que 
    cl_program cpProgram;                // OpenCL program
    cl_kernel ckViterbiOneStep;
    cl_kernel ckViterbiPath;
    cl_mem d_maxProbNew;
    cl_mem d_maxProbOld;
    cl_mem d_mtState;
    cl_mem d_mtEmit;
    cl_mem d_path;
    int nState;
    int nEmit;
	int nObs;
    size_t ViterbiOneStep(const int &obs, const int &iObs);
    void ViterbiPath(cl_mem vProb, cl_mem vPath);
    double tKernel;
    int szLmem;
    bool smallBlock;
	int wgSize;
};

#endif
