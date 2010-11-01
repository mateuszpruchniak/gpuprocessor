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
 
 #ifndef PARTICLESYSTEM_COMMON_H
#define PARTICLESYSTEM_COMMON_H

#include <oclUtils.h>
#include "vector_types.h"

////////////////////////////////////////////////////////////////////////////////
// CPU/GPU common types
////////////////////////////////////////////////////////////////////////////////
typedef unsigned int uint;
typedef cl_mem memHandle_t;

//Simulation parameters
typedef struct{
    float3 colliderPos;
    float  colliderRadius;

    float3 gravity;
    float globalDamping;
    float particleRadius;

    uint3 gridSize;
    uint numCells;
    float3 worldOrigin;
    float3 cellSize;

    uint numBodies;
    uint maxParticlesPerCell;

    float spring;
    float damping;
    float shear;
    float attraction;
    float boundaryDamping;
} simParams_t;

////////////////////////////////////////////////////////////////////////////////
// Sort of API-independent interface
////////////////////////////////////////////////////////////////////////////////
extern "C" void startupOpenCL(int argc, const char **argv);
extern "C" void shutdownOpenCL(void);

extern "C" void allocateArray(memHandle_t *memObj, size_t size);
extern "C" void freeArray(memHandle_t memObj);

extern "C" void copyArrayFromDevice(void *hostPtr, const memHandle_t memObj, unsigned int vbo, size_t size);
extern "C" void copyArrayToDevice(memHandle_t memObj, const void *hostPtr, size_t offset, size_t size);

extern "C" void registerGLBufferObject(unsigned int vbo);
extern "C" void unregisterGLBufferObject(unsigned int vbo);

extern "C" memHandle_t mapGLBufferObject(uint vbo);
extern "C" void unmapGLBufferObject(uint vbo);

#endif
