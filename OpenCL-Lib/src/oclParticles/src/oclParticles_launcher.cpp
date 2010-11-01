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

//Standard utilities and systems includes
#include <oclUtils.h>

#include "particleSystem_common.h"
#include "particleSystem_engine.h"

////////////////////////////////////////////////////////////////////////////////
// OpenCL launcher for particles kernels
////////////////////////////////////////////////////////////////////////////////
//OpenCL particles program
static cl_program cpParticles;

//OpenCL particles kernels
static cl_kernel
    ckIntegrate,
    ckCalcHash,
    ckMemset,
    ckFindCellBoundsAndReorder,
    ckCollide;

//Default command queue for particles kernels
static cl_command_queue cqDefaultCommandQue;

//Simulation parameters
static cl_mem params;

extern "C" void initParticles(cl_context cxGPUContext, cl_command_queue cqParamCommandQue, const char **argv){
    cl_int ciErrNum;
    size_t kernelLength;

    shrLog(LOGBOTH, 0, "Loading Particles.cl...\n");
        char *cParticles = oclLoadProgSource(shrFindFilePath("Particles.cl", argv[0]), "// My comment\n", &kernelLength);
        shrCheckError(cParticles != NULL, shrTRUE);

    shrLog(LOGBOTH, 0, "Creating particles program...\n");
        cpParticles = clCreateProgramWithSource(cxGPUContext, 1, (const char **)&cParticles, &kernelLength, &ciErrNum);
        shrCheckError(ciErrNum, CL_SUCCESS);

    shrLog(LOGBOTH, 0, "Building particles program...\n");
        ciErrNum = clBuildProgram(cpParticles, 0, NULL, "-cl-mad-enable", NULL, NULL);
        shrCheckError(ciErrNum, CL_SUCCESS);

    shrLog(LOGBOTH, 0, "Creating particles kernels...\n\n");
        ckIntegrate = clCreateKernel(cpParticles, "integrate", &ciErrNum);
        shrCheckError(ciErrNum, CL_SUCCESS);
        ckCalcHash = clCreateKernel(cpParticles, "calcHash", &ciErrNum);
        shrCheckError(ciErrNum, CL_SUCCESS);
        ckMemset = clCreateKernel(cpParticles, "Memset", &ciErrNum);
        shrCheckError(ciErrNum, CL_SUCCESS);
        ckFindCellBoundsAndReorder = clCreateKernel(cpParticles, "findCellBoundsAndReorder", &ciErrNum);
        shrCheckError(ciErrNum, CL_SUCCESS);
        ckCollide = clCreateKernel(cpParticles, "collide", &ciErrNum);
        shrCheckError(ciErrNum, CL_SUCCESS);

    shrLog(LOGBOTH, 0, "Creating parameter GPU buffer...\n\n");
        allocateArray(&params, sizeof(simParams_t));

    //Save default command queue
    cqDefaultCommandQue = cqParamCommandQue;

    //Discard temp storage
    free(cParticles);

    //Save ptx code to separate file
    oclLogPtx(cpParticles, oclGetFirstDev(cxGPUContext), "Particles.ptx");
}

extern "C" void closeParticles(void){
    cl_int ciErrNum;
    ciErrNum  = clReleaseMemObject(params);
    ciErrNum |= clReleaseKernel(ckCollide);
    ciErrNum |= clReleaseKernel(ckFindCellBoundsAndReorder);
    ciErrNum |= clReleaseKernel(ckMemset);
    ciErrNum |= clReleaseKernel(ckCalcHash);
    ciErrNum |= clReleaseKernel(ckIntegrate);
    ciErrNum |= clReleaseProgram(cpParticles);
    shrCheckError(ciErrNum, CL_SUCCESS);
}

extern "C" void setParameters(simParams_t *m_params){
    copyArrayToDevice(params, m_params, 0, sizeof(simParams_t));
}

static size_t uSnap(size_t a, size_t b){
    return ((a % b) == 0) ? a : (a - (a % b) + b);
}

extern "C" void integrateSystem(
    memHandle_t d_Pos,
    memHandle_t d_Vel,
    float deltaTime,
    uint numParticles
){
    cl_int ciErrNum;

    size_t  localWorkSize = 256;
    size_t globalWorkSize = uSnap(numParticles, localWorkSize);

    ciErrNum  = clSetKernelArg(ckIntegrate, 0, sizeof(cl_mem), (void *)&d_Pos);
    ciErrNum |= clSetKernelArg(ckIntegrate, 1, sizeof(cl_mem), (void *)&d_Vel);
    ciErrNum |= clSetKernelArg(ckIntegrate, 2, sizeof(cl_mem), (void *)&params);
    ciErrNum |= clSetKernelArg(ckIntegrate, 3, sizeof(float), (void *)&deltaTime);
    ciErrNum |= clSetKernelArg(ckIntegrate, 4, sizeof(uint), (void *)&numParticles);
    shrCheckError(ciErrNum, CL_SUCCESS);

    assert( globalWorkSize % localWorkSize == 0 );
    ciErrNum = clEnqueueNDRangeKernel(cqDefaultCommandQue, ckIntegrate, 1, NULL, &globalWorkSize, &localWorkSize, 0, NULL, NULL);
    shrCheckError(ciErrNum, CL_SUCCESS);
}

extern "C" void calcHash(
    memHandle_t d_Hash,
    memHandle_t d_Index,
    memHandle_t d_Pos,
    int numParticles
){
    cl_int ciErrNum;

    size_t  localWorkSize = 256;
    size_t globalWorkSize = uSnap(numParticles, localWorkSize);

    ciErrNum  = clSetKernelArg(ckCalcHash, 0, sizeof(cl_mem), (void *)&d_Hash);
    ciErrNum |= clSetKernelArg(ckCalcHash, 1, sizeof(cl_mem), (void *)&d_Index);
    ciErrNum |= clSetKernelArg(ckCalcHash, 2, sizeof(cl_mem), (void *)&d_Pos);
    ciErrNum |= clSetKernelArg(ckCalcHash, 3, sizeof(cl_mem), (void *)&params);
    ciErrNum |= clSetKernelArg(ckCalcHash, 4,  sizeof(uint), (void *)&numParticles);
    shrCheckError(ciErrNum, CL_SUCCESS);

    ciErrNum = clEnqueueNDRangeKernel(cqDefaultCommandQue, ckCalcHash, 1, NULL, &globalWorkSize, &localWorkSize, 0, NULL, NULL);
    shrCheckError(ciErrNum, CL_SUCCESS);
}

static void memsetOCL(
    memHandle_t d_Data,
    uint val,
    uint N
){
    cl_int ciErrNum;

    size_t  localWorkSize = 256;
    size_t globalWorkSize = uSnap(N, localWorkSize);

    ciErrNum  = clSetKernelArg(ckMemset, 0, sizeof(cl_mem), (void *)&d_Data);
    ciErrNum |= clSetKernelArg(ckMemset, 1, sizeof(cl_uint), (void *)&val);
    ciErrNum |= clSetKernelArg(ckMemset, 2, sizeof(cl_uint), (void *)&N);
    shrCheckError(ciErrNum, CL_SUCCESS);

    ciErrNum = clEnqueueNDRangeKernel(cqDefaultCommandQue, ckMemset, 1, NULL, &globalWorkSize, &localWorkSize, 0, NULL, NULL);
    shrCheckError(ciErrNum, CL_SUCCESS);
}

extern "C" void findCellBoundsAndReorder(
    memHandle_t d_CellStart,
    memHandle_t d_CellEnd,
    memHandle_t d_ReorderedPos,
    memHandle_t d_ReorderedVel,
    memHandle_t d_Hash,
    memHandle_t d_Index,
    memHandle_t d_Pos,
    memHandle_t d_Vel,
    uint numParticles,
    uint numCells
){
    cl_int ciErrNum;
    memsetOCL(d_CellStart, 0xFFFFFFFFU, numCells);
    memsetOCL(d_CellEnd, 0xFFFFFFFFU, numCells);

    size_t  localWorkSize = 256;
    size_t globalWorkSize = uSnap(numParticles, localWorkSize);

    ciErrNum  = clSetKernelArg(ckFindCellBoundsAndReorder, 0, sizeof(cl_mem), (void *)&d_CellStart);
    ciErrNum |= clSetKernelArg(ckFindCellBoundsAndReorder, 1, sizeof(cl_mem), (void *)&d_CellEnd);
    ciErrNum |= clSetKernelArg(ckFindCellBoundsAndReorder, 2, sizeof(cl_mem), (void *)&d_ReorderedPos);
    ciErrNum |= clSetKernelArg(ckFindCellBoundsAndReorder, 3, sizeof(cl_mem), (void *)&d_ReorderedVel);
    ciErrNum |= clSetKernelArg(ckFindCellBoundsAndReorder, 4, sizeof(cl_mem), (void *)&d_Hash);
    ciErrNum |= clSetKernelArg(ckFindCellBoundsAndReorder, 5, sizeof(cl_mem), (void *)&d_Index);
    ciErrNum |= clSetKernelArg(ckFindCellBoundsAndReorder, 6, sizeof(cl_mem), (void *)&d_Pos);
    ciErrNum |= clSetKernelArg(ckFindCellBoundsAndReorder, 7, sizeof(cl_mem), (void *)&d_Vel);
    ciErrNum |= clSetKernelArg(ckFindCellBoundsAndReorder, 8, (localWorkSize + 1) * sizeof(cl_uint), NULL);
    ciErrNum |= clSetKernelArg(ckFindCellBoundsAndReorder, 9, sizeof(cl_uint), (void *)&numParticles);
    shrCheckError(ciErrNum, CL_SUCCESS);

    ciErrNum = clEnqueueNDRangeKernel(cqDefaultCommandQue, ckFindCellBoundsAndReorder, 1, NULL, &globalWorkSize, &localWorkSize, 0, NULL, NULL);
    shrCheckError(ciErrNum, CL_SUCCESS);
}

extern "C" void collide(
    memHandle_t d_Vel,
    memHandle_t d_ReorderedPos,
    memHandle_t d_ReorderedVel,
    memHandle_t d_Index,
    memHandle_t d_CellStart,
    memHandle_t d_CellEnd,
    uint   numParticles,
    uint   numCells
){
    cl_int ciErrNum;

    size_t  localWorkSize = 64;
    size_t globalWorkSize = uSnap(numParticles, localWorkSize);

    ciErrNum  = clSetKernelArg(ckCollide, 0, sizeof(cl_mem), (void *)&d_Vel);
    ciErrNum |= clSetKernelArg(ckCollide, 1, sizeof(cl_mem), (void *)&d_ReorderedPos);
    ciErrNum |= clSetKernelArg(ckCollide, 2, sizeof(cl_mem), (void *)&d_ReorderedVel);
    ciErrNum |= clSetKernelArg(ckCollide, 3, sizeof(cl_mem), (void *)&d_Index);
    ciErrNum |= clSetKernelArg(ckCollide, 4, sizeof(cl_mem), (void *)&d_CellStart);
    ciErrNum |= clSetKernelArg(ckCollide, 5, sizeof(cl_mem), (void *)&d_CellEnd);
    ciErrNum |= clSetKernelArg(ckCollide, 6, sizeof(cl_mem), (void *)&params);
    ciErrNum |= clSetKernelArg(ckCollide, 7, sizeof(uint),   (void *)&numParticles);
    shrCheckError(ciErrNum, CL_SUCCESS);

    ciErrNum = clEnqueueNDRangeKernel(cqDefaultCommandQue, ckCollide, 1, NULL, &globalWorkSize, &localWorkSize, 0, NULL, NULL);
    shrCheckError(ciErrNum, CL_SUCCESS);
}

