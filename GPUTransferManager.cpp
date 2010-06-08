/*!
 * \file GPUTransferManager.cpp
 * \brief Class responsible for managing transfer to GPU.
 *
 * \author Mateusz Pruchniak
 * \date 2010-05-05
 */

#include "GPUTransferManager.h"

GPUTransferManager::~GPUTransferManager(void)
{
	 Cleanup();
}

GPUTransferManager::GPUTransferManager()
{
	cmDevBuf = NULL;
    GPUInputOutput = NULL;
    cmPinnedBuf = NULL;
}

GPUTransferManager::GPUTransferManager( cl_context GPUContextArg, cl_command_queue GPUCommandQueueArg, unsigned int width, unsigned int height, int channels )
{
    //cout << "data transfer konstr" << endl;
	
	nChannels = channels;
    GPUContext = GPUContextArg;
    ImageHeight = height;
    ImageWidth = width;
    GPUCommandQueue = GPUCommandQueueArg;

    // Allocate pinned input and output host image buffers:  mem copy operations to/from pinned memory is much faster than paged memory
    szBuffBytes = ImageWidth * ImageHeight * nChannels * sizeof (char);
    // This flag specifies that the application wants the OpenCL implementation to allocate memory from host accessible memory.
    cmPinnedBuf = clCreateBuffer(GPUContext, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, szBuffBytes, NULL, &GPUError);
    CheckError(GPUError);
    

    // Enqueues a command to map a region of the buffer object given by buffer into the host address space and returns a pointer to this mapped region.
    GPUInputOutput = (cl_uint*)clEnqueueMapBuffer(GPUCommandQueue, cmPinnedBuf, CL_TRUE, CL_MAP_WRITE, 0, szBuffBytes, 0, NULL, NULL, &GPUError);
    CheckError(GPUError);

    // Create the device buffers in GMEM on each device, for now we have one device :)
    cmDevBuf = clCreateBuffer(GPUContext, CL_MEM_READ_WRITE, szBuffBytes, NULL, &GPUError);
    CheckError(GPUError);
}





void GPUTransferManager::CheckError(int code)
{
    switch(code)
    {
    case CL_SUCCESS:
        return;
        break;
    case CL_INVALID_COMMAND_QUEUE:
        cout << "CL_INVALID_COMMAND_QUEUE" << endl;
        break;
    case CL_INVALID_CONTEXT:
        cout << "CL_INVALID_CONTEXT" << endl;
        break;
    case CL_INVALID_MEM_OBJECT:
        cout << "CL_INVALID_MEM_OBJECT" << endl;
        break;
    case CL_INVALID_VALUE:
        cout << "CL_INVALID_VALUE" << endl;
        break;
    case CL_INVALID_EVENT_WAIT_LIST:
        cout << "CL_INVALID_EVENT_WAIT_LIST" << endl;
        break;
    case CL_MEM_OBJECT_ALLOCATION_FAILURE:
        cout << "CL_MEM_OBJECT_ALLOCATION_FAILURE" << endl;
        break;
    case CL_OUT_OF_HOST_MEMORY:
        cout << "CL_OUT_OF_HOST_MEMORY" << endl;
        break;
    default:
         cout << "OTHERS ERROR" << endl;
    }
}

void GPUTransferManager::Cleanup()
{
    // Cleanup allocated objects
    //cout << "\nStarting Cleanup...\n\n";

    if(cmDevBuf)clReleaseMemObject(cmDevBuf);
	
}

IplImage* GPUTransferManager::ReceiveImage()
{
	szBuffBytes = ImageWidth * ImageHeight * nChannels * sizeof (char);
    GPUError = clEnqueueReadBuffer(GPUCommandQueue, cmDevBuf, CL_TRUE, 0, szBuffBytes, (void*)GPUInputOutput, 0, NULL, NULL);
    CheckError(GPUError);
    
    image->imageData = (char*)GPUInputOutput;

    return image;
}

void GPUTransferManager::SendImage( IplImage* imageToLoad )
{
	ImageHeight = imageToLoad->height;
    ImageWidth = imageToLoad->width;
	 szBuffBytes = ImageWidth * ImageHeight * nChannels * sizeof (char);
	image = imageToLoad;

    GPUError = clEnqueueWriteBuffer(GPUCommandQueue, cmDevBuf, CL_TRUE, 0, szBuffBytes, (void*)imageToLoad->imageData, 0, NULL, NULL);
    CheckError(GPUError);
}

