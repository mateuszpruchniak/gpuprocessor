/*!
 * \file MeanVariableCentralPointFilter.h
 * \brief Mean filter with variable central point.
 * \author Mateusz Pruchniak
 * \date 2010-05-05
 */


#include "MeanVariableCentralPointFilter.h"


MeanVariableCentralPointFilter::~MeanVariableCentralPointFilter(void)
{
}


MeanVariableCentralPointFilter::MeanVariableCentralPointFilter(cl_context GPUContext ,GPUTransferManager* transfer,int central): LowpassFilter("/home/mateusz/Pulpit/GIT/gpuprocessor/OpenCL/src/oclGPUProcessor/src/OpenCL/LowpassFilter.cl",GPUContext,transfer,"ckConv")
{
	
	mask = new int[9];
	for(int i = 0 ; i < 9 ; ++i )
	{
		mask[i] = 1;
	}

	mask[4] = central;
	LoadMask(mask,9,transfer);
}
