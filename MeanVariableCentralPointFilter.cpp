/*!
 * \file MedianFilter.h
 * \brief Mean filter with variable central point.
 * \author Mateusz Pruchniak
 * \date 2010-05-05
 */


#include "MeanVariableCentralPointFilter.h"


MeanVariableCentralPointFilter::MeanVariableCentralPointFilter(void)
{
}


MeanVariableCentralPointFilter::~MeanVariableCentralPointFilter(void)
{
}

bool MeanVariableCentralPointFilter::filter(cl_command_queue GPUCommandQueue)
{
	return LowpassFilter::filter(GPUCommandQueue);
}

MeanVariableCentralPointFilter::MeanVariableCentralPointFilter(cl_context GPUContext ,GPUTransferManager* transfer,int central): LowpassFilter("./OpenCL/LowpassFilter.cl",GPUContext,transfer,"ckConv")
{
	mask = new int[9];
	for(int i = 0 ; i < 9 ; ++i )
	{
		mask[i] = 1;
	}

	mask[4] = central;
	LoadMask(mask,9,transfer);
}