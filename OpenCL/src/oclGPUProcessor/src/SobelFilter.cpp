/*!
 * \file SobelFilter.cpp
 * \brief Sobel filter, edge detection algorithm.
 * \author Mateusz Pruchniak
 * \date 2010-05-05
 */

#include "SobelFilter.h"

SobelFilter::SobelFilter(cl_context GPUContext ,GPUTransferManager* transfer): HighpassFilter("/home/mateusz/Pulpit/GIT/gpuprocessor/OpenCL/src/oclGPUProcessor/src/OpenCL/HighpassFilter.cl",GPUContext,transfer,"ckGradient")
{
	cout << "Sobel kontr" << endl;
	maskH = new int[9];
	maskV = new int[9];
		
	maskV[0] = 1;
	maskV[1] = 2;
	maskV[2] = 1;
	maskV[3] = 0;
	maskV[4] = 0;
	maskV[5] = 0;
	maskV[6] = -1;
	maskV[7] = -2;
	maskV[8] = -1;

	maskH[0] = 1;
	maskH[1] = 0;
	maskH[2] = -1;
	maskH[3] = 2;
	maskH[4] = 0;
	maskH[5] = -2;
	maskH[6] = 1;
	maskH[7] = 0;
	maskH[8] = -1;

	LoadMask(&cmDevBufMaskH,maskH,9,transfer);
	LoadMask(&cmDevBufMaskV,maskV,9,transfer);
}


SobelFilter::~SobelFilter(void)
{
}
