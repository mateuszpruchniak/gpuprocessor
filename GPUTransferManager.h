/*!
 * \file GPUTransferManager.h
 * \brief Class responsible for managing transfer to GPU.
 *
 * \author Mateusz Pruchniak
 * \date 2010-05-05
 */

#pragma once

#include <oclUtils.h>
#include <iostream>
#include <vector>
#include <string>
#include "cv.h"
#include "cxmisc.h"
#include "highgui.h"
#include <algorithm>
#include <stdio.h>
#include <ctype.h>
#include <time.h>

using namespace std;

/*!
 * \class GPUTransferManager
 * \brief Class responsible for managing transfer to GPU.
 * \author Mateusz Pruchniak
 * \date 2010-05-05
 */
class GPUTransferManager
{
	private:
                
		/*!
		 * OpenCL context of device to use.
		 */
        cl_context GPUContext;             

		/*!
		 * Mapped Pointer to pinned Host input and output buffer for host processing.
		 */
        cl_uint* GPUInputOutput;                 

		/*!
		 * OpenCL host memory input output buffer object:  pinned.
		 */
        cl_mem cmPinnedBuf;               

		/*!
		 * Command-queue for specific device.
		 */
        cl_command_queue GPUCommandQueue; 

		/*!
		 * The size in bytes of the buffer memory object to be allocated.
		 */
        size_t szBuffBytes;                

		/*!
		 * Error code.
		 */
        cl_int GPUError;

		/*!
		 * Image to load or return from buffer.
		 */
        IplImage* image;                   
		

    public:

		/*!
		 * OpenCL device memory input/output buffer object  
		 */
        cl_mem cmDevBuf;                 

		/*!
		 * OpenCL device memory input buffer for LUT table.
		 */
		cl_mem cmDevBufLUT;                 

		/*!
		 * OpenCL device memory input buffer for mask.
		 */
		cl_mem cmDevBufMask1;

		/*!
		 * OpenCL device memory input buffer for mask.
		 */
		cl_mem cmDevBufMask2;

		/*!
		 * Image width.
		 */
        unsigned int ImageWidth;   

		/*!
		 * Image height.
		 */
        unsigned int ImageHeight;  

		/*!
		 * Number of color channels.
		 */
		int nChannels;

		/*!
		 * Destructor.
		 */
        ~GPUTransferManager();

        /*!
		 * Constructor, create buffers.
		 */
        GPUTransferManager( cl_context , cl_command_queue , unsigned int , unsigned int,  int nChannels );

		/*!
		 * Default Constructor.
		 */
        GPUTransferManager();

        /*!
		 * Load image to buffers.
		 */
        void LoadImageToGPU( IplImage*  );

        /*!
		 * Get image from buffers.
		 */
        IplImage* GetImageFromGPU();
        
        
		/*!
		 * Release all buffors.
		 */
        void Cleanup();

        /*!
		 * Check error code.
		 */
        void CheckError(int );

		/*!
		 * Load lookup table to buffer.
		 */
		void LoadLookUpTable(int* lut, int count);

		/*!
		 * Load mask to buffer.
		 */
		void LoadMask1(int* mask, int count);

		/*!
		 * Load mask to buffer.
		 */
		void LoadMask2(int* mask, int count);

		
};

