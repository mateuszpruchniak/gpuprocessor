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



#ifndef QUASIRANDOMGENERATOR_COMMON_H
#define QUASIRANDOMGENERATOR_COMMON_H



////////////////////////////////////////////////////////////////////////////////
// Global types and constants
////////////////////////////////////////////////////////////////////////////////
typedef long long int INT64;

#define QRNG_DIMENSIONS 3
#define QRNG_RESOLUTION 31
#define INT_SCALE (1.0f / (float)0x80000001U)

#define MAX_GPU_COUNT 8

#endif
