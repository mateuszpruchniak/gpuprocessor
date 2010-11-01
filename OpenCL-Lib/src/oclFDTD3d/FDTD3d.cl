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

__kernel void FiniteDifferences(__global float *output,
                                __global const float *input,
                                __local float *tile,
                                __constant float *coeff,
                                const int dimx,
                                const int dimy,
                                const int dimz)
{
    int gtidx = get_global_id(0);
    int gtidy = get_global_id(1);
    int ltidx = get_local_id(0);
    int ltidy = get_local_id(1);
    int workx = get_local_size(0);
    int worky = get_local_size(1);
    int tilex = workx + 2 * RADIUS;
    // If it were required: int tiley = worky + 2 * RADIUS;

    int inputIndex  = (gtidy + RADIUS) * (dimx + 2 * RADIUS) + (gtidx + RADIUS);
    int outputIndex = 0;
    int stride_y    = dimx + 2 * RADIUS;
    int stride_z    = stride_y * (dimy + 2 * RADIUS);

    float infront[RADIUS];
    float behind[RADIUS];
    float current;

	int tx = ltidx + RADIUS;
	int ty = ltidy + RADIUS;

    // For simplicity we assume that the global size is equal to the actual
    // problem size; since the global size must be a multiple of the local size
    // this means the problem size must be a multiple of the local size (or
    // padded to meet this constraint).
    // Preload the "infront" and "behind" data
    for (int i = RADIUS - 2 ; i >= 0 ; i--)
    {
        behind[i] = input[inputIndex];
        inputIndex += stride_z;
    }

    current = input[inputIndex];
    outputIndex = inputIndex;
    inputIndex += stride_z;

    for (int i = 0 ; i < RADIUS ; i++)
    {
        infront[i] = input[inputIndex];
        inputIndex += stride_z;
    }

    // Step through the xy-planes
    for (int z = 0 ; z < dimz ; z++)
    {
        // Advance the slice (move the thread-front)
        for (int i = RADIUS - 1 ; i > 0 ; i--)
            behind[i] = behind[i - 1];
        behind[0] = current;
        current = infront[0];
        for (int i = 0 ; i < RADIUS - 1 ; i++)
            infront[i] = infront[i + 1];
        infront[RADIUS - 1] = input[inputIndex];

        inputIndex  += stride_z;
        outputIndex += stride_z;
        barrier(CLK_LOCAL_MEM_FENCE);

        // Update the data slice in the local tile
        // Halo above & below
        if (ltidy < RADIUS)
        {
            tile[ltidy * tilex + tx]                    = input[outputIndex - RADIUS * stride_y];
            tile[(ltidy + worky + RADIUS) * tilex + tx] = input[outputIndex + worky * stride_y];
        }
        // Halo left & right
        if (ltidx < RADIUS)
        {
            tile[ty * tilex + ltidx]                  = input[outputIndex - RADIUS];
            tile[ty * tilex + ltidx + workx + RADIUS] = input[outputIndex + workx];
        }
        tile[ty * tilex + tx] = current;
        barrier(CLK_LOCAL_MEM_FENCE);

        // Compute the output value
        float value  = coeff[0] * current;
        for (int i = 1 ; i <= RADIUS ; i++)
        {
            value += coeff[i] * (infront[i-1] + behind[i-1] + tile[(ty - i) * tilex + tx] + tile[(ty + i) * tilex + tx] + tile[ty * tilex + tx - i] + tile[ty * tilex + tx + i]);
        }

        // Store the output value
        output[outputIndex] = value;
    }
}
