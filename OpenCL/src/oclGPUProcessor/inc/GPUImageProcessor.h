/*!
 * \file GPUImageProcessor.h
 * \brief File contains class responsible for managing image processing.
 *
 * \author Mateusz Pruchniak
 * \date 2010-06-05
 */


#pragma once
#include "oclUtils.h"
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
 * \brief Class responsible for managing image processing. Contain list of image processing objects. 
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
		 * Error code, only 0 is allowed.
		 */
        cl_int GPUError;	

		/*!
		 * Platforms are represented by a cl_platform_id, OpenCL framework allow an application to share resources and execute kernels on devices in the platform.
		 */
		cl_platform_id cpPlatform;

		/*!
		 * Device (GPU,CPU) are represented by cl_device_id. Each device has id.
		 */
		cl_device_id* cdDevices;	

		/*!
		 * Total number of devices available to the platform.
		 */
		cl_uint uiDevCount;			

		/*!
		 * List of pointer to processing objects.
		 */
        vector<Filter*> filters;        
    
    public:
 
		/*!
		 * OpenCL command-queue, is an object where OpenCL commands are enqueued to be executed by the device.
		 * "The command-queue is created on a specific device in a context [...] Having multiple command-queues allows applications to queue multiple independent commands without requiring synchronization." (OpenCL Specification).
		 */
        cl_command_queue GPUCommandQueue; 

		/*!
		 * Context defines the entire OpenCL environment, including OpenCL kernels, devices, memory management, command-queues, etc. Contexts in OpenCL are referenced by an cl_context object
		 */
        cl_context GPUContext;    

		/*!
		 * Pointer to instance of class GPUTransferManager.
		 */
		GPUTransferManager* Transfer;

		/*!
		 * Constructor , Get the number of GPU devices available to the platform, and create the device list.
		 * Create the OpenCL context on a GPU device and create command-queue.
		 */
        GPUImageProcessor(int width,int height,int nChannels);

		/*!
		 * Start image processing. For each element of the image processing list called method filter().
		 */
        void Process();

		/*!
		 * Add filters to image processing list.
		 */
        void AddProcessing(Filter* filter);
        
        /*!
		 * Check error code.
		 */
        void CheckError(int code);

		 /*!
		 * Destructor.
		 */
        ~GPUImageProcessor();
};

