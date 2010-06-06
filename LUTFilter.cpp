/*!
 * \file LUTFilter.cpp
 * \brief Lookup table filter.
 * \author Mateusz Pruchniak
 * \date 2010-05-05
 */

#include "LUTFilter.h"


LUTFilter::LUTFilter(void)
{
}


LUTFilter::~LUTFilter(void)
{
	if(cmDevBufLUT)clReleaseMemObject(cmDevBufLUT);
}


bool LUTFilter::filter(cl_command_queue GPUCommandQueue)
{
	
    int iLocalPixPitch = iBlockDimX + 2;
    GPUError = clSetKernelArg(GPUFilter, 0, sizeof(cl_mem), (void*)&GPUTransfer->cmDevBuf);
	GPUError |= clSetKernelArg(GPUFilter, 1, sizeof(cl_mem), (void*)&cmDevBufLUT);
    GPUError |= clSetKernelArg(GPUFilter, 2, (iLocalPixPitch * (iBlockDimY + 2) *  GPUTransfer->nChannels * sizeof(cl_uchar)), NULL);
    GPUError |= clSetKernelArg(GPUFilter, 3, sizeof(cl_int), (void*)&iLocalPixPitch);
    GPUError |= clSetKernelArg(GPUFilter, 4, sizeof(cl_uint), (void*)&GPUTransfer->ImageWidth);
    GPUError |= clSetKernelArg(GPUFilter, 5, sizeof(cl_uint), (void*)&GPUTransfer->ImageHeight);
	GPUError |= clSetKernelArg(GPUFilter, 6, sizeof(cl_int), (void*)&GPUTransfer->nChannels);
    if(GPUError) return false;

	size_t GPULocalWorkSize[2]; 
    GPULocalWorkSize[0] = iBlockDimX;
    GPULocalWorkSize[1] = iBlockDimY;
    GPUGlobalWorkSize[0] = shrRoundUp((int)GPULocalWorkSize[0], GPUTransfer->ImageWidth); 

    GPUGlobalWorkSize[1] = shrRoundUp((int)GPULocalWorkSize[1], (int)GPUTransfer->ImageHeight);

    if( clEnqueueNDRangeKernel( GPUCommandQueue, GPUFilter, 2, NULL, GPUGlobalWorkSize, GPULocalWorkSize, 0, NULL, NULL) ) return false;
    return true;
}


void LUTFilter::LoadLookUpTable(int* lut,int count,GPUTransferManager* transfer)
{
	
	// Create the device buffers in GMEM on each device, for now we have one device :)
    cmDevBufLUT = clCreateBuffer(transfer->GPUContext, CL_MEM_READ_WRITE, count * sizeof (unsigned int), NULL, &GPUError);
    CheckError(GPUError);

    GPUError = clEnqueueWriteBuffer(transfer->GPUCommandQueue, cmDevBufLUT, CL_TRUE, 0, count * sizeof (unsigned int), (void*)lut, 0, NULL, NULL);
    CheckError(GPUError);
}