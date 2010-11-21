/*!
 * \file MedianFilter.cpp
 * \brief Mean filter.
 * \author Mateusz Pruchniak
 * \date 2010-05-05
 */

#include "MeanFilter.h"



MeanFilter::~MeanFilter(void)
{
}


	
MeanFilter::MeanFilter(cl_context GPUContext ,GPUTransferManager* transfer): LowpassFilter("./OpenCL/LowpassFilter.cl",GPUContext,transfer,"ckConv")
{
	mask = new int[9];
	for(int i = 0 ; i < 9 ; ++i )
	{
		mask[i] = 1;
	}
	LoadMask(mask,9,transfer);
}
