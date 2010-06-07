/*!
 * \file MedianFilter.cpp
 * \brief Mean filter.
 * \author Mateusz Pruchniak
 * \date 2010-05-05
 */

#include "MeanFilter.h"


MeanFilter::MeanFilter(void)
{
}


MeanFilter::~MeanFilter(void)
{
}


	
MeanFilter::MeanFilter(cl_context GPUContext ,GPUTransferManager* transfer): LowpassFilter("./OpenCL/LowpassFilter.cl",GPUContext,transfer,"ckConv")
{
	maskSize = 9;
	mask = new int[9];
	for(int i = 0 ; i < 9 ; ++i )
	{
		mask[i] = 1;
	}
	LoadMask(mask,maskSize,transfer);
}


bool MeanFilter::filter(cl_command_queue GPUCommandQueue)
{
	return LowpassFilter::filter(GPUCommandQueue);
}

