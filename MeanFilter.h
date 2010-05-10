#pragma once
#include "lowpassfilter.h"
class MeanFilter :
	public LowpassFilter
{

public:
	MeanFilter(void);

	~MeanFilter(void);

	MeanFilter(cl_context GPUContext ,GPUTransferManager* transfer): LowpassFilter("./OpenCL/LowpassFilter.cl",GPUContext,transfer,"ckConv")
	{
		mask = new int[9];
		for(int i = 0 ; i < 9 ; ++i )
		{
			mask[i] = 1;
		}
	}

	
};

