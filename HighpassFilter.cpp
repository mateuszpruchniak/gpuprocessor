/*!
 * \file HighpassFilter.cpp
 * \brief Highpass filters.
 * \author Mateusz Pruchniak
 * \date 2010-05-05
 */

#include "HighpassFilter.h"


HighpassFilter::HighpassFilter(void)
{
}


HighpassFilter::~HighpassFilter(void)
{
}

HighpassFilter::HighpassFilter(char* source, cl_context GPUContext ,GPUTransferManager* transfer,char* KernelName): LinearFilter(source,GPUContext,transfer,KernelName)
{

}

void HighpassFilter::process(cl_command_queue GPUCommandQueue)
{
	GPUTransfer->LoadMask1(maskH,9);
	GPUTransfer->LoadMask2(maskV,9);

    int iLocalPixPitch = iBlockDimX + 2;
    GPUError = clSetKernelArg(GPUFilter, 0, sizeof(cl_mem), (void*)&GPUTransfer->cmDevBuf);
	GPUError |= clSetKernelArg(GPUFilter, 1, sizeof(cl_mem), (void*)&GPUTransfer->cmDevBufMask1);
	GPUError |= clSetKernelArg(GPUFilter, 2, sizeof(cl_mem), (void*)&GPUTransfer->cmDevBufMask2);
    GPUError |= clSetKernelArg(GPUFilter, 3, (iLocalPixPitch * (iBlockDimY + 2) * GPUTransfer->nChannels * sizeof(cl_uchar)), NULL);
	GPUError |= clSetKernelArg(GPUFilter, 4, ( 9 * sizeof(int)), NULL);
	GPUError |= clSetKernelArg(GPUFilter, 5, ( 9 * sizeof(int)), NULL);
    GPUError |= clSetKernelArg(GPUFilter, 6, sizeof(cl_int), (void*)&iLocalPixPitch);
    GPUError |= clSetKernelArg(GPUFilter, 7, sizeof(cl_uint), (void*)&GPUTransfer->ImageWidth);
    GPUError |= clSetKernelArg(GPUFilter, 8, sizeof(cl_uint), (void*)&GPUTransfer->ImageHeight);
	GPUError |= clSetKernelArg(GPUFilter, 9, sizeof(cl_int), (void*)&GPUTransfer->nChannels);
    CheckError(GPUError);

	size_t GPULocalWorkSize[2]; 
    GPULocalWorkSize[0] = iBlockDimX;
    GPULocalWorkSize[1] = iBlockDimY;
    GPUGlobalWorkSize[0] = shrRoundUp((int)GPULocalWorkSize[0], GPUTransfer->ImageWidth); 

    GPUGlobalWorkSize[1] = shrRoundUp((int)GPULocalWorkSize[1], (int)GPUTransfer->ImageHeight);

    

    GPUError = clEnqueueNDRangeKernel( GPUCommandQueue, GPUFilter, 2, NULL, GPUGlobalWorkSize, GPULocalWorkSize, 0, NULL, NULL);
    CheckError(GPUError);
}