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

#include "oclFDTD3d.h"

#include <oclUtils.h>
#include <iostream>
#include <iomanip>

#include "FDTD3dReference.h"
#include "FDTD3dGPU.h"

// Name of the file with the source code for the computation kernel
const char* clSourceFile = "FDTD3d.cl";

// Name of the log file
const char *shrLogFile = "oclFDTD3d.txt";

// Forward declarations
bool runTest(int argc, const char **argv);
void showHelp(const int argc, const char **argv);

int main(int argc, const char **argv)
{
    // Start the log
    shrSetLogFileName(shrLogFile);
    shrLog(LOGBOTH, 0, "%s Starting...\n\n", argv[0]);

    // Check help flag
    if (shrCheckCmdLineFlag(argc, argv, "help"))
    {
        shrLog(LOGFILE, 0.0, "Displaying help on console\n");
        showHelp(argc, argv);
    }
    else
    {
        // Execute
        bool result = runTest(argc, argv);
        shrCheckErrorEX(result, true, NULL);
    }

    // Finish
    shrEXIT(argc, argv);
}

void showHelp(const int argc, const char **argv)
{
    if (argc > 0)
        std::cout << std::endl << argv[0] << std::endl;
    std::cout << std::endl << "Syntax:" << std::endl;
    std::cout << std::left;
    std::cout << "    " << std::setw(20) << "--device=<device>" << "Specify device to use for execution" << std::endl;
    std::cout << "    " << std::setw(20) << "--dimx=<N>" << "Specify number of elements in x direction" << std::endl;
    std::cout << "    " << std::setw(20) << "--dimy=<N>" << "Specify number of elements in y direction" << std::endl;
    std::cout << "    " << std::setw(20) << "--dimz=<N>" << "Specify number of elements in z direction" << std::endl;
    std::cout << "    " << std::setw(20) << "--radius=<N>" << "Specify radius of stencil" << std::endl;
    std::cout << "    " << std::setw(20) << "--timesteps=<N>" << "Specify number of timesteps" << std::endl;
    std::cout << std::endl;
    std::cout << "    " << std::setw(20) << "--noprompt" << "Skip prompt before exit" << std::endl;
    std::cout << std::endl;
}

bool runTest(int argc, const char **argv)
{
    bool ok = true;

    float *host_output;
    float *device_output;
    float *input;
    float *coeff;

    int defaultDim;
    int dimx;
    int dimy;
    int dimz;
    int radius;
    int timesteps;
    size_t volumeSize;
    memsize_t memsize;

    const float lowerBound = 0.0f;
    const float upperBound = 1.0f;

    // Determine default dimensions
    shrLog(LOGBOTH, 0, "Set-up, based upon target device GMEM size...\n");
    if (ok)
    {
        // Get the memory size of the target device
        shrLog(LOGBOTH, 0, " getTargetDeviceGlobalMemSize\n");
        ok = getTargetDeviceGlobalMemSize(&memsize, argc, argv);
    }
    if (ok)
    {
        // We can never use all the memory so to keep things simple we aim to
        // use around half the total memory
        memsize /= 2;
        
        // Most of our memory use is taken up by the input and output buffers -
        // two buffers of equal size - and for simplicity the volume is a cube:
        //   dim = floor( (N/2)^(1/3) )
        defaultDim = floor(pow((memsize / (2.0 * sizeof(float))), 1.0/3.0));

        // By default, make the volume edge size a multiple of the work group width
        defaultDim = (defaultDim / k_localWorkX) * k_localWorkX;

        // Check dimension is valid
        if (defaultDim < k_dim_min)
        {
            shrLog(LOGBOTH | ERRORMSG, 0.0, "insufficient device memory (maximum volume on device is %d, must be between %d and %d).\n", defaultDim, k_dim_min, k_dim_max);
            ok = false;
        }
        else if (defaultDim > k_dim_max)
        {
            defaultDim = k_dim_default;
        }
    }

    // Parse command line arguments
    if (ok)
    {
        char *dim = 0;
        if (shrGetCmdLineArgumentstr(argc, argv, "dimx", &dim))
        {
            dimx = (int)atoi(dim);
            if (dimx < k_dim_min || dimx > k_dim_max)
            {
                shrLog(LOGBOTH | ERRORMSG, 0.0, "dimx out of range (%d requested, must be between %d and %d).\n", dimx, k_dim_min, k_dim_max);
                ok = false;
            }
        }
        else
        {
            dimx = defaultDim;
        }
        if (shrGetCmdLineArgumentstr(argc, argv, "dimy", &dim))
        {
            dimy = (int)atoi(dim);
            if (dimy < k_dim_min || dimy > k_dim_max)
            {
                shrLog(LOGBOTH | ERRORMSG, 0.0, "dimy out of range (%d requested, must be between %d and %d).\n", dimy, k_dim_min, k_dim_max);
                ok = false;
            }
        }
        else
        {
            dimy = defaultDim;
        }
        if (shrGetCmdLineArgumentstr(argc, argv, "dimz", &dim))
        {
            dimz = (int)atoi(dim);
            if (dimz < k_dim_min || dimz > k_dim_max)
            {
                shrLog(LOGBOTH | ERRORMSG, 0.0, "dimz out of range (%d requested, must be between %d and %d).\n", dimz, k_dim_min, k_dim_max);
                ok = false;
            }
        }
        else
        {
            dimz = defaultDim;
        }
        if (shrGetCmdLineArgumentstr(argc, argv, "radius", &dim))
        {
            radius = (int)atoi(dim);
            if (radius < k_radius_min || radius >= k_radius_max)
            {
                shrLog(LOGBOTH | ERRORMSG, 0.0, "radius out of range (%d requested, must be between %d and %d).\n", radius, k_radius_min, k_radius_max);
                ok = false;
            }
        }
        else
        {
            radius = k_radius_default;
        }
        if (shrGetCmdLineArgumentstr(argc, argv, "timesteps", &dim))
        {
            timesteps = (int)atoi(dim);
            if (timesteps < k_timesteps_min || radius >= k_timesteps_max)
            {
                shrLog(LOGBOTH | ERRORMSG, 0.0, "timesteps out of range (%d requested, must be between %d and %d).\n", timesteps, k_timesteps_min, k_timesteps_max);
                ok = false;
            }
        }
        else
        {
            timesteps = k_timesteps_default;
        }
        if (dim)
            free(dim);
    }

    // Determine volume size, allowing additional space for boundary
    if (ok)
        volumeSize = (dimx + 2 * radius) * (dimy + 2 * radius) * (dimz + 2 * radius);

    // Allocate memory
    if (ok)
    {
        shrLog(LOGBOTH, 0, " calloc host_output\n");
        if ((host_output = (float *)calloc(volumeSize, sizeof(float))) == NULL)
        {
            shrLog(LOGBOTH | ERRORMSG, 0.0, "calloc.\n");
            shrLog(LOGBOTH, 0, "Insufficient memory, please try a smaller volume (use --help for syntax).\n");
            ok = false;
        }
    }
    if (ok)
    {
        shrLog(LOGBOTH, 0, " calloc device_output\n");
        if ((device_output = (float *)calloc(volumeSize, sizeof(float))) == NULL)
        {
            shrLog(LOGBOTH | ERRORMSG, 0.0, "calloc.\n");
            shrLog(LOGBOTH, 0, "Insufficient memory, please try a smaller volume (use --help for syntax).\n");
            ok = false;
        }
    }
    if (ok)
    {
        shrLog(LOGBOTH, 0, " malloc input\n");
        if ((input = (float *)malloc(volumeSize * sizeof(float))) == NULL)
        {
            shrLog(LOGBOTH | ERRORMSG, 0.0, "malloc.\n");
            shrLog(LOGBOTH, 0, "Insufficient memory, please try a smaller volume (use --help for syntax).\n");
            ok = false;
        }
    }
    if (ok)
    {
        shrLog(LOGBOTH, 0, " malloc coeff\n");
        if ((coeff = (float *)malloc((radius + 1) * sizeof(float))) == NULL)
        {
            shrLog(LOGBOTH | ERRORMSG, 0.0, "malloc.\n");
            shrLog(LOGBOTH, 0, "Insufficient memory, please try a smaller volume (use --help for syntax).\n");
            ok = false;
        }
    }

    // Create coefficients
    if (ok)
    {
        for (int i = 0 ; i <= radius ; i++)
        {
            coeff[i] = 0.1f;
        }
    }

    // Generate data
    if (ok)
    {
        shrLog(LOGBOTH, 0, " generateRandomData\n\n");
        generateRandomData(input, dimx, dimy, dimz, radius, lowerBound, upperBound);
    }

    if (ok)
        shrLog(LOGBOTH, 0, "FDTD on %d x %d x %d volume with symmetric filter radius %d for %d timesteps...\n\n", dimx, dimy, dimz, radius, timesteps);

    // Execute on the host
    if (ok)
    {
        shrLog(LOGBOTH, 0, "fdtdReference...\n");
        ok = fdtdReference(host_output, input, coeff, dimx, dimy, dimz, radius, timesteps);
    }

    // Execute on the device
    if (ok)
    {
        shrLog(LOGBOTH, 0, "fdtdGPU...\n");
        ok = fdtdGPU(device_output, input, coeff, dimx, dimy, dimz, radius, timesteps, argc, argv);
    }

    // Compare the results
    if (ok)
    {
        float tolerance = 0.0001f;
        shrLog(LOGBOTH, 0, "\nCompareData (tolerance %f)...\n", tolerance);
        ok = compareData(device_output, host_output, dimx, dimy, dimz, radius, tolerance);
    }

    shrLog(LOGBOTH, 0, "\nTEST %s\n\n", (ok) ?  "PASSED" : "FAILED !!!"); 
    return ok;
}
