#include "LowpassFilter.h"


LowpassFilter::LowpassFilter(void)
{
}


LowpassFilter::~LowpassFilter(void)
{
}


void LowpassFilter::process(cl_command_queue GPUCommandQueue)
{
	GPUTransfer->LoadMask1(mask,9);

    int iLocalPixPitch = iBlockDimX + 2;
    GPUError = clSetKernelArg(GPUFilter, 0, sizeof(cl_mem), (void*)&GPUTransfer->cmDevBuf);
	GPUError |= clSetKernelArg(GPUFilter, 1, sizeof(cl_mem), (void*)&GPUTransfer->cmDevBufMask1);
    GPUError |= clSetKernelArg(GPUFilter, 2, (iLocalPixPitch * (iBlockDimY + 2) * GPUTransfer->nChannels * sizeof(cl_uchar)), NULL);
	GPUError |= clSetKernelArg(GPUFilter, 3, ( 9 * sizeof(int)), NULL);
    GPUError |= clSetKernelArg(GPUFilter, 4, sizeof(cl_int), (void*)&iLocalPixPitch);
    GPUError |= clSetKernelArg(GPUFilter, 5, sizeof(cl_uint), (void*)&GPUTransfer->ImageWidth);
    GPUError |= clSetKernelArg(GPUFilter, 6, sizeof(cl_uint), (void*)&GPUTransfer->ImageHeight);
	GPUError |= clSetKernelArg(GPUFilter, 7, sizeof(cl_int), (void*)&GPUTransfer->nChannels);
    CheckError(GPUError);

    GPULocalWorkSize[0] = iBlockDimX;
    GPULocalWorkSize[1] = iBlockDimY;
    GPUGlobalWorkSize[0] = shrRoundUp((int)GPULocalWorkSize[0], GPUTransfer->ImageWidth); 

    GPUGlobalWorkSize[1] = shrRoundUp((int)GPULocalWorkSize[1], (int)GPUTransfer->ImageHeight);

    

    GPUError = clEnqueueNDRangeKernel( GPUCommandQueue, GPUFilter, 2, NULL, GPUGlobalWorkSize, GPULocalWorkSize, 0, NULL, NULL);
    CheckError(GPUError);
}