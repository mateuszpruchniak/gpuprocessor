#pragma once

#include <oclUtils.h>
#include <iostream>
#include "cv.h"
#include "cxmisc.h"
#include "highgui.h"
#include <vector>
#include <string>
#include <algorithm>
#include <stdio.h>
#include <ctype.h>
#include <time.h>
#include "GPUTransferManager.h"

using namespace std;


class Filter
{
	protected:
        int iBlockDimX;                     // work-group size
        int iBlockDimY;                     // work-group size
        cl_int GPUError;
        GPUTransferManager* GPUTransfer;    // pointer to instance of class GPUTransferManager
        char* SourceOpenCLFilter;                 // loaded .cl file
		char* SourceOpenCL;                 // loaded .cl file
        cl_program GPUProgram;              // Program Objects
        cl_kernel GPUFilter;                // Filter kernel
        size_t GPUGlobalWorkSize[2];        // global size of NDRange
        size_t GPULocalWorkSize[2];         // work-group size
        
    public:

		Filter(void);

        Filter(char* , cl_context GPUContext  ,GPUTransferManager*  ,char* );

        virtual ~Filter();

        virtual void process(cl_command_queue GPUCommandQueue){};
        
        void CheckError(int);
};

