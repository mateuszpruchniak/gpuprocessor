/*!
 * \file GPUImageProcessor.h
 * \brief Class responsible for managing image processing.
 *
 * \author Mateusz Pruchniak
 * \date 2010-06-05
 */


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

/*!
 * \class GPUImageProcessor
 * \brief Class responsible for managing image processing.
 * \author Mateusz Pruchniak
 * \date 2010-06-05
 */
class GPUImageProcessor
{
    private:
        
        /*!
		 * Image width.
		 */
        unsigned int ImageWidth;   

		/*!
		 * Image height.
		 */
        unsigned int ImageHeight;  

		/*!
		 * Device id.
		 */
        cl_device_id* GPUDevices;      

		/*!
		 * Number of device.
		 */
        cl_uint GPUNumberDevices;      

		/*!
		 * Number of device.
		 */
        cl_int GPUError;	

		/*!
		 * OpenCL platform.
		 */
		cl_platform_id cpPlatform;      

		/*!
		 * OpenCL device list.
		 */
		cl_device_id* cdDevices;	

		/*!
		 * Total number of devices available to the platform.
		 */
		cl_uint uiDevCount;			

		/*!
		 * List of filters.
		 */
        vector<Filter*> filters;        
    
    public:

		/*!
		 * Command-queue for specific device.
		 */
        cl_command_queue GPUCommandQueue; 

		/*!
		 * OpenCL context of device to use.
		 */
        cl_context GPUContext;    

		/*!
		 * Pointer to instance of class GPUTransferManager.
		 */
		GPUTransferManager* Transfer;

		/*!
		 * List of filters.
		 */
        GPUImageProcessor(int width,int height,int nChannels);

		/*!
		 * Start image processing.
		 */
        void process();

		/*!
		 * Add additional filters.
		 */
        void addFilter(Filter* filter);
        
        /*!
		 * Check error code.
		 */
        void CheckError(int code);

		 /*!
		 * Destructor.
		 */
        ~GPUImageProcessor();
};

