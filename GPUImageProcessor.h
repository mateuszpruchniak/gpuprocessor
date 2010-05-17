#pragma once
#include <oclUtils.h>
#include <iostream>
#include <vector>
#include <string>
#include <stdio.h>
#include <ctype.h>
#include <time.h>
#include "cv.h"
#include "cxmisc.h"
#include "highgui.h"
#include "GPUTransferManager.h"
#include "Filter.h"

using namespace std;


class GPUImageProcessor
{
    private:
        
        unsigned int ImageWidth;        // image width
        unsigned int ImageHeight;       // image height
        cl_device_id* GPUDevices;       // id_device
        cl_uint GPUNumberDevices;       // number of device
        cl_int GPUError;	
		cl_platform_id cpPlatform;      // OpenCL platform
		cl_device_id* cdDevices;		// OpenCL device list
		cl_uint uiDevCount;				// total # of devices available to the platform


        vector<Filter*> filters;        // list od filters
    
    public:

        cl_context GPUContext;                  // OpenCL context of device to use. 
        cl_command_queue GPUCommandQueue;       // command-queue for specific device.
		GPUTransferManager* Transfer;

        GPUImageProcessor(int width,int height,int nChannels);

        // start processing
        void process();

        // add additional filters
        void addFilter(Filter* filter);
        
        // 
        void CheckError(int code);

        ~GPUImageProcessor();
};

