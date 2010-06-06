/*!
 * \file RGB2HSV.cpp
 * \brief Color transformation class.
 * \author Mateusz Pruchniak
 * \date 2010-05-05
 */

#include "RGB2HSV.h"


RGB2HSV::RGB2HSV(void)
{
}


RGB2HSV::~RGB2HSV(void)
{
}



void RGB2HSV::process(cl_command_queue GPUCommandQueue)
{
	
    int iLocalPixPitch = iBlockDimX + 2;
    GPUError = clSetKernelArg(GPUFilter, 0, sizeof(cl_mem), (void*)&GPUTransfer->cmDevBuf);
    GPUError |= clSetKernelArg(GPUFilter, 1, sizeof(cl_uint), (void*)&GPUTransfer->ImageWidth);
    GPUError |= clSetKernelArg(GPUFilter, 2, sizeof(cl_uint), (void*)&GPUTransfer->ImageHeight);
    CheckError(GPUError);
	
    size_t GPULocalWorkSize[2];    
    GPULocalWorkSize[0] = iBlockDimX;
    GPULocalWorkSize[1] = iBlockDimY;
    GPUGlobalWorkSize[0] = shrRoundUp((int)GPULocalWorkSize[0], GPUTransfer->ImageWidth); 

    GPUGlobalWorkSize[1] = shrRoundUp((int)GPULocalWorkSize[1], (int)GPUTransfer->ImageHeight);

    GPUError = clEnqueueNDRangeKernel( GPUCommandQueue, GPUFilter, 2, NULL, GPUGlobalWorkSize, GPULocalWorkSize, 0, NULL, NULL);
    CheckError(GPUError);
}