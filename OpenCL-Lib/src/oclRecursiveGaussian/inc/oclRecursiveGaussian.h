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

// Utililities and system includes
#include <oclUtils.h>

// 0 gives "fade-from-dark" edges (model where zero value pixels are being averaged in when within radius of edge)
// 1 gives "extrapolation" edges (model where pixels within radius are copied from the last value outside radius)
#define CLAMP_TO_EDGE 0

// 0 selects full version of filter function, 1 selects simple version 
#define USE_SIMPLE_FILTER 0

// custom type for Gaussian parameter precomputation
typedef struct _GaussParms
{
    float nsigma; 
    float alpha;
    float ema; 
    float ema2; 
    float b1; 
    float b2; 
    float a0; 
    float a1; 
    float a2; 
    float a3; 
    float coefp; 
    float coefn; 
} GaussParms, *pGaussParms;

// struct instance to hold all the Gaussian filter coefs
static GaussParms oclGP;               

// forward interface declaration for Gaussian parameter pre-processing function (for host and GPU proc)
extern "C" void PreProcessGaussParms (float fSigma, int iOrder, GaussParms* pGP);

// forward interface declaration for host Gaussian processing function
extern "C" void HostRecursiveGaussianRGBA(unsigned int* uiInputImage, unsigned int* uiTempImage, unsigned int* uiOutputImage, 
                                          int iWidth, int iHeight, GaussParms* pGP);
