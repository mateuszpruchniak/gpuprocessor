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

#include "oclConvolutionSeparable_common.h"

extern "C" void convolutionRowHost(
    float *h_Dst,
    float *h_Src,
    float *h_Kernel,
    int imageW,
    int imageH,
    int kernelR
){
    for(int y = 0; y < imageH; y++)
        for(int x = 0; x < imageW; x++){
            double sum = 0;
            for(int k = -kernelR; k <= kernelR; k++){
                int d = x + k;
                if(d >= 0 && d < imageW)
                    sum += h_Src[y * imageW + d] * h_Kernel[kernelR - k];
            }
            h_Dst[y * imageW + x] = (float)sum;
        }
}

extern "C" void convolutionColumnHost(
    float *h_Dst,
    float *h_Src,
    float *h_Kernel,
    int imageW,
    int imageH,
    int kernelR
){
    for(int y = 0; y < imageH; y++)
        for(int x = 0; x < imageW; x++){
            double sum = 0;
            for(int k = -kernelR; k <= kernelR; k++){
                int d = y + k;
                if(d >= 0 && d < imageH)
                    sum += h_Src[d * imageW + x] * h_Kernel[kernelR - k];
            }
            h_Dst[y * imageW + x] = (float)sum;
        }
}
