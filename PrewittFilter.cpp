/*!
 * \file highpassfilter.cpp
 * \brief Prewitt filter, edge detection algorithm.
 * \author Mateusz Pruchniak
 * \date 2010-05-05
 */


#include "PrewittFilter.h"


PrewittFilter::PrewittFilter(void)
{
}

PrewittFilter::PrewittFilter(cl_context GPUContext ,GPUTransferManager* transfer): HighpassFilter("./OpenCL/HighpassFilter.cl",GPUContext,transfer,"ckGradient")
{
	maskH = new int[9];
	maskV = new int[9];
		
	maskH[0] = -1;
	maskH[1] = 0;
	maskH[2] = 1;
	maskH[3] = -1;
	maskH[4] = 0;
	maskH[5] = 1;
	maskH[6] = -1;
	maskH[7] = 0;
	maskH[8] = 1;

	maskV[0] = -1;
	maskV[1] = -1;
	maskV[2] = -1;
	maskV[3] = 0;
	maskV[4] = 0;
	maskV[5] = 0;
	maskV[6] = 1;
	maskV[7] = 1;
	maskV[8] = 1;
}



PrewittFilter::~PrewittFilter(void)
{
}

bool PrewittFilter::filter(cl_command_queue GPUCommandQueue)
{
	return HighpassFilter::filter(GPUCommandQueue);
}