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

#include "FDTD3dReference.h"

#include <cstdlib>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <shrUtils.h>

void generateRandomData(float *data, const int dimx, const int dimy, const int dimz, const int radius, const float lowerBound, const float upperBound)
{
	srand(0);

	for (int iz = 0 ; iz < dimz + 2 * radius ; iz++)
    {
		for (int iy = 0 ; iy < dimy + 2 * radius ; iy++)
        {
			for (int ix = 0 ; ix < dimx + 2 * radius ; ix++)
			{
				*data = (float)(lowerBound + ((float)rand() / (float)RAND_MAX) * (upperBound - lowerBound));
				++data;
			}
        }
    }
}

void generatePatternData(float *data, const int dimx, const int dimy, const int dimz, const int radius, const float lowerBound, const float upperBound)
{
    for (int iz = 0 ; iz < dimz + 2 * radius ; iz++)
    {
		for (int iy = 0 ; iy < dimy + 2 * radius ; iy++)
        {
			for (int ix = 0 ; ix < dimx + 2 * radius ; ix++)
			{
				*data = (float)(lowerBound + ((float)iz / (float)dimz) * (upperBound - lowerBound));
				++data;
			}
        }
    }
}

bool fdtdReference(float *output, const float *input, const float *coeff, const int dimx, const int dimy, const int dimz, const int radius, const int timesteps)
{
    bool ok = true;
    const size_t  volumeSize   = (dimx + 2 * radius) * (dimy + 2 * radius) * (dimz + 2 * radius);
    const int     stride_y     = dimx + 2 * radius;
    const int     stride_z     = stride_y * (dimy + 2 * radius);
    float        *intermediate = 0;
    const float  *bufsrc       = 0;
    float        *bufdst       = 0;
    float        *bufdstnext   = 0;

    // Allocate temporary buffer
    if (ok)
    {
        shrLog(LOGBOTH, 0, " calloc intermediate\n");
        if ((intermediate = (float *)calloc(volumeSize, sizeof(float))) == NULL)
        {
            shrLog(LOGBOTH | ERRORMSG, 0.0, "calloc.\n");
            shrLog(LOGBOTH, 0, "Insufficient memory, please try a smaller volume (use --help for syntax).\n");
            ok = false;
        }
    }

    // Decide which buffer to use first (result should end up in output)
    if (ok)
    {
        if ((timesteps % 2) == 0)
        {
            bufsrc     = input;
            bufdst     = intermediate;
            bufdstnext = output;
        }
        else
        {
            bufsrc     = input;
            bufdst     = output;
            bufdstnext = intermediate;
        }
    }

    // Run the FDTD (naive method)
    if (ok)
    {
        shrLog(LOGBOTH, 0, " Host FDTD loop\n");
        for (int it = 0 ; it < timesteps ; it++)
        {
            shrLog(LOGBOTH, 0, "\tt = %d\n", it);
            const float *src = bufsrc;
            float *dst       = bufdst;
            for (int iz = - radius ; iz < dimz + radius ; iz++)
            {
                for (int iy = - radius ; iy < dimy + radius ; iy++)
                {
                    for (int ix = - radius ; ix < dimx + radius ; ix++)
                    {
                        if (ix >= 0 && ix < dimx && iy >= 0 && iy < dimy && iz >= 0 && iz < dimz)
                        {
                            float value = (*src) * coeff[0];
                            for(int ir = 1 ; ir <= radius ; ir++)
                            {
                                value += coeff[ir] * (*(src + ir) + *(src - ir));                       // horizontal
                                value += coeff[ir] * (*(src + ir * stride_y) + *(src - ir * stride_y)); // vertical
                                value += coeff[ir] * (*(src + ir * stride_z) + *(src - ir * stride_z)); // in front & behind
                            }
                            *dst = value;
                        }
                        else
                        {
                            *dst = *src;
                        }
                        ++dst;
                        ++src;
                    }
                }
            }
            // Rotate buffers
            float *tmp = bufdst;
            bufdst     = bufdstnext;
            bufdstnext = tmp;
            bufsrc = (const float *)tmp;
        }
        shrLog(LOGBOTH, 0, "\n");
    }

    if (intermediate)
        free(intermediate);

    return ok;
}

bool compareData(const float *output, const float *reference, const int dimx, const int dimy, const int dimz, const int radius, const float tolerance)
{
    bool ok = true;

	for (int iz = - radius ; iz < dimz + radius ; iz++)
	{
		for (int iy = - radius ; iy < dimy + radius ; iy++)
		{
			for (int ix = - radius ; ix < dimx + radius ; ix++)
			{
                if (ix >= 0 && ix < dimx && iy >= 0 && iy < dimy && iz >= 0 && iz < dimz)
				{
                    // Determine the absolute difference
					float difference = abs(*reference - *output);
                    float error;

                    // Determine the relative error
                    if (*reference != 0)
                        error = difference / *reference;
                    else
                        error = difference;

                    // Check the error is within the tolerance
					if (error > tolerance)
					{
						ok = false;
                        shrLog(LOGBOTH, 0, "Data error at (%d,%d,%d)\t%f instead of %f\n", ix, iy, iz, *output, *reference);
                        return ok;
					}
				}
				++output;
				++reference;
			}
		}
	}

	return ok;
}
