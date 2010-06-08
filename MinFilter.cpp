/*!
 * \file MinFilter.cpp
 * \brief Minimal filter.
 * \author Mateusz Pruchniak
 * \date 2010-05-05
 */

#include "MinFilter.h"


MinFilter::MinFilter(void)
{
}


MinFilter::~MinFilter(void)
{
}

MinFilter::MinFilter(cl_context GPUContext ,GPUTransferManager* transfer): NonLinearFilter("./OpenCL/MinFilter.cl",GPUContext,transfer,"ckMin")
{

}

bool MinFilter::filter(cl_command_queue GPUCommandQueue)
{
 
    int iLocalPixPitch = iBlockDimX + 2;
    GPUError = clSetKernelArg(GPUFilter, 0, sizeof(cl_mem), (void*)&GPUTransfer->cmDevBuf);
    GPUError |= clSetKernelArg(GPUFilter, 1, (iLocalPixPitch * (iBlockDimY + 2) *  GPUTransfer->nChannels * sizeof(cl_uchar)), NULL);
    GPUError |= clSetKernelArg(GPUFilter, 2, sizeof(cl_int), (void*)&iLocalPixPitch);
    GPUError |= clSetKernelArg(GPUFilter, 3, sizeof(cl_uint), (void*)&GPUTransfer->ImageWidth);
    GPUError |= clSetKernelArg(GPUFilter, 4, sizeof(cl_uint), (void*)&GPUTransfer->ImageHeight);
	GPUError |= clSetKernelArg(GPUFilter, 5, sizeof(cl_int), (void*)&GPUTransfer->nChannels);
    if(GPUError) return false;

	size_t GPULocalWorkSize[2]; 
    GPULocalWorkSize[0] = iBlockDimX;
    GPULocalWorkSize[1] = iBlockDimY;
    GPUGlobalWorkSize[0] = shrRoundUp((int)GPULocalWorkSize[0], GPUTransfer->ImageWidth); 

    GPUGlobalWorkSize[1] = shrRoundUp((int)GPULocalWorkSize[1], (int)GPUTransfer->ImageHeight);

    if(clEnqueueNDRangeKernel( GPUCommandQueue, GPUFilter, 2, NULL, GPUGlobalWorkSize, GPULocalWorkSize, 0, NULL, NULL)) return false;
	return true;
}