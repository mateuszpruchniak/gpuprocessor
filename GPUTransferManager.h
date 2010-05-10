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

using namespace std;

class GPUTransferManager
{
	private:
                
        cl_context GPUContext;              // OpenCL context of device to use. 
        cl_uint* GPUInputOutput;                  // Mapped Pointer to pinned Host input and output buffer for host processing			
        cl_mem cmPinnedBuf;               // OpenCL host memory input output buffer object:  pinned 
        cl_command_queue GPUCommandQueue;   // command-queue for specific device.
        size_t szBuffBytes;                 // The size in bytes of the buffer memory object to be allocated.
        cl_int GPUError;
        IplImage* image;                    // image to load or return from buffer
		

    public:

        cl_mem cmDevBuf;                 // OpenCL device memory input buffer object  
		cl_mem cmDevBufLUT;                 // OpenCL device memory input buffer object  
		cl_mem cmDevBufMask1;
		cl_mem cmDevBufMask2;
        unsigned int ImageWidth;   
        unsigned int ImageHeight;  
		int nChannels;

        ~GPUTransferManager();

		GPUTransferManager();

        //constructor - create buffers

        GPUTransferManager( cl_context , cl_command_queue , unsigned int , unsigned int,  int nChannels );


        //load image to buffor
        void LoadImageToGPU( IplImage*  );

        //get image from buffor
        IplImage* GetImageFromGPU();
        
        //release all buffors
        void Cleanup();

        //display info about error - what kind of error is
        void CheckError(int );

		void LoadLookUpTable(int* lut, int count);

		void LoadMask1(int* mask, int count);

		void LoadMask2(int* mask, int count);

		void LoadImageToGPUTest( IplImage* imageToLoad );
};

