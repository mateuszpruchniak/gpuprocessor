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

// Defines for box filter processing demo
//*****************************************************************************
#define MASK_RADIUS 8
#define MASK_RADIUS_ALIGNED 16
#define MASK_LENGTH (2 * MASK_RADIUS + 1)
#define ROWS_OUTPUT_WIDTH 128
#define COLUMNS_BLOCKDIMX 16
#define COLUMNS_BLOCKDIMY 16
#define COLUMNS_OUTPUT_HEIGHT 128

// Inline device function to convert 32-bit unsigned integer to floating point rgba color 
//*****************************************************************
float4 rgbaUintToFloat4(unsigned int c)
{
    float4 rgba;
    rgba.x = c & 0xff;
    rgba.y = (c >> 8) & 0xff;
    rgba.z = (c >> 16) & 0xff;
    rgba.w = (c >> 24) & 0xff;
    return rgba;
}

// Inline device function to convert floating point rgba color to 32-bit unsigned integer
//*****************************************************************
unsigned int rgbaFloat4ToUint(float4 rgba, float fScale)
{
    unsigned int uiPackedPix = 0U;
    uiPackedPix |= 0x000000FF & (unsigned int)(rgba.x * fScale);
    uiPackedPix |= 0x0000FF00 & (((unsigned int)(rgba.y * fScale)) << 8);
    uiPackedPix |= 0x00FF0000 & (((unsigned int)(rgba.z * fScale)) << 16);
    uiPackedPix |= 0xFF000000 & (((unsigned int)(rgba.w * fScale)) << 24);
    return uiPackedPix;
}

// Row summation filter kernel with rescaling
//*****************************************************************
__kernel void BoxRows( __global const uchar4* uc4Source, __global unsigned int* uiDest,
                       __local uchar4* uc4LocalData,
                        unsigned int uiWidth, unsigned int uiHeight, float fScale)
{
    // Compute x and y pixel coordinates from group ID and local ID indexes
    int globalPosX = ((int)get_group_id(0) * ROWS_OUTPUT_WIDTH) + (int)get_local_id(0) - MASK_RADIUS_ALIGNED;
    int globalPosY = get_group_id(1);
    int iGlobalOffset = globalPosY * uiWidth + globalPosX;

    // Read global data into LMEM
    if (globalPosX >= 0 && globalPosX < uiWidth)
    {
        uc4LocalData[get_local_id(0)] = uc4Source[iGlobalOffset];
    }
    else 
    {
        uc4LocalData[get_local_id(0)].xyzw = (uchar4)0; 
    }

    // Synchronize the read into LMEM
    barrier(CLK_LOCAL_MEM_FENCE);

    // Compute (if pixel plus apron is within bounds)
    if((globalPosX >= 0) && (globalPosX < uiWidth) && (get_local_id(0) >= MASK_RADIUS_ALIGNED) && (get_local_id(0) < (MASK_RADIUS_ALIGNED + ROWS_OUTPUT_WIDTH)))
    {
        // Init summation registers to zero
        float4 f4Sum = (float4)0.0f;

        // Do summation, using inline function to break up uint value from LMEM into independent RGBA values
        int iOffsetX = (int)get_local_id(0) - MASK_RADIUS;
        int iLimit = iOffsetX + MASK_LENGTH;
        for(iOffsetX; iOffsetX < iLimit; iOffsetX++)
        {
            f4Sum.x += uc4LocalData[iOffsetX].x;
            f4Sum.y += uc4LocalData[iOffsetX].y;
            f4Sum.z += uc4LocalData[iOffsetX].z;
            f4Sum.w += uc4LocalData[iOffsetX].w; 
        }

        // Use inline function to scale and convert registers to packed RGBA values in a uchar4, and write back out to GMEM
        uiDest[iGlobalOffset] = rgbaFloat4ToUint(f4Sum, fScale);
    }
}

// Column summation filter kernel with rescaling
//*****************************************************************
__kernel void BoxColumns( __global const  uchar4* uc4Source, __global unsigned int* uiDest, 
                          __local uchar4* uc4LocalData,
                          unsigned int uiWidth, unsigned int uiHeight, float fScale )
{
    // Compute x and y pixel coordinates from group ID and local ID indexes
    int globalPosX = get_global_id(0);
    int globalPosY = ((int)get_group_id(1) * COLUMNS_OUTPUT_HEIGHT) + (int)get_local_id(1) - MASK_RADIUS;

    // Read global data into LMEM
    int iOffsetX = get_local_id(0);
    int iLimitY = COLUMNS_OUTPUT_HEIGHT + (MASK_RADIUS << 1);
    for(int ly = get_local_id(1), gy = globalPosY; ly < iLimitY; ly += COLUMNS_BLOCKDIMY, gy += COLUMNS_BLOCKDIMY)
    {
        if ((gy >= 0) && (gy < uiHeight)) 
        {
            uc4LocalData[ly*COLUMNS_BLOCKDIMX+iOffsetX] = uc4Source[gy * uiWidth + globalPosX];
        }
        else 
        {
            uc4LocalData[ly*COLUMNS_BLOCKDIMX+iOffsetX].xyzw = (uchar4)0; 
        }
    }

    // Synchronize the read into LMEM
    barrier(CLK_LOCAL_MEM_FENCE);

    // Compute (if and where pixel plus apron is within bounds)
    for(int ly = get_local_id(1), gy = globalPosY; ly < iLimitY; ly += COLUMNS_BLOCKDIMY, gy += COLUMNS_BLOCKDIMY)
    {
        if((gy >= 0) && (gy < uiHeight) && (ly >= MASK_RADIUS) && (ly < (MASK_RADIUS + COLUMNS_OUTPUT_HEIGHT)))
        {
            // Init summation registers to zero
            float4 f4Sum = (float4)0.0f;

            // Do summation, using inline function to break up uint value from LMEM into independent RGBA values
            int iOffsetY = ly - MASK_RADIUS;
            int iLimit = iOffsetY + MASK_LENGTH;
            for(iOffsetY; iOffsetY < iLimit; iOffsetY++)
            {
                f4Sum.x += uc4LocalData[iOffsetY*COLUMNS_BLOCKDIMX+iOffsetX].x; 
                f4Sum.y += uc4LocalData[iOffsetY*COLUMNS_BLOCKDIMX+iOffsetX].y; 
                f4Sum.z += uc4LocalData[iOffsetY*COLUMNS_BLOCKDIMX+iOffsetX].z; 
                f4Sum.w += uc4LocalData[iOffsetY*COLUMNS_BLOCKDIMX+iOffsetX].w; 
            }

            // Use inline function to scale and convert registers to packed RGBA values in a UINT, and write back out to GMEM
            uiDest[gy * uiWidth + globalPosX] = rgbaFloat4ToUint(f4Sum, fScale);
        }
    }
}
