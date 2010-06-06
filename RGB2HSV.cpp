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



bool RGB2HSV::filter(cl_command_queue GPUCommandQueue)
{
	
    int iLocalPixPitch = iBlockDimX + 2;
    GPUError = clSetKernelArg(GPUFilter, 0, sizeof(cl_mem), (void*)&GPUTransfer->cmDevBuf);
    GPUError |= clSetKernelArg(GPUFilter, 1, sizeof(cl_uint), (void*)&GPUTransfer->ImageWidth);
    GPUError |= clSetKernelArg(GPUFilter, 2, sizeof(cl_uint), (void*)&GPUTransfer->ImageHeight);
    
	if( GPUError != 0 ) return false;

    size_t GPULocalWorkSize[2];    
    GPULocalWorkSize[0] = iBlockDimX;
    GPULocalWorkSize[1] = iBlockDimY;
    GPUGlobalWorkSize[0] = shrRoundUp((int)GPULocalWorkSize[0], GPUTransfer->ImageWidth); 

    GPUGlobalWorkSize[1] = shrRoundUp((int)GPULocalWorkSize[1], (int)GPUTransfer->ImageHeight);


	
    if( clEnqueueNDRangeKernel( GPUCommandQueue, GPUFilter, 2, NULL, GPUGlobalWorkSize, GPULocalWorkSize, 0, NULL, NULL) ) return false;
    return true;
}