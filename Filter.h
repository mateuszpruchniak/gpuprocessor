/*!
 * \file Filter.h
 * \brief Abstract class for all filters.
 *
 * \author Mateusz Pruchniak
 * \date 2010-05-05
 */

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

/*!
 * \class Filter
 * \brief Abstract class for all filters.
 * \author Mateusz Pruchniak
 * \date 2010-05-05
 */
class Filter
{
	protected:

		/*!
		 * Work-group size - dim X.
		 */
        int iBlockDimX;                    

		/*!
		 * Work-group size - dim Y.
		 */
        int iBlockDimY;                    

		/*!
		 * Error code.
		 */
        cl_int GPUError;

		/*!
		 * Pointer to instance of class GPUTransferManager.
		 */
        GPUTransferManager* GPUTransfer;

		/*!
		 * Loaded .cl file, which contain code responsible for image processing.
		 */
        char* SourceOpenCLFilter;                 

		/*!
		 * Loaded .cl file, which contain support function.
		 */
		char* SourceOpenCL;              

		/*!
		 * Program object.
		 */
        cl_program GPUProgram;              

		/*!
		 * Filter kernel.
		 */
        cl_kernel GPUFilter;               

		/*!
		 * Global size of NDRange.
		 */
        size_t GPUGlobalWorkSize[2];        

    public:

		/*!
		 * Default constructor.
		 */
		Filter(void);

		/*!
		 * Constructor, creates a program object for a context, loads the source code (.cl files) and build the program.
		 */
        Filter(char* , cl_context GPUContext  ,GPUTransferManager*  ,char* );

		/*!
		 * Destructor.
		 */
        virtual ~Filter();

		/*!
		 * Virtual methods, processing image.
		 */
        virtual void process(cl_command_queue GPUCommandQueue){};
        
		/*!
		 * Check error code.
		 */
        void CheckError(int);
};

