#pragma once
#include "lowpassfilter.h"
class MeanVariableCentralPointFilter :
	public LowpassFilter
{

public:
	MeanVariableCentralPointFilter(void);

	~MeanVariableCentralPointFilter(void);

	MeanVariableCentralPointFilter(cl_context GPUContext ,GPUTransferManager* transfer,int* maskArg): LowpassFilter("./OpenCL/LowpassFilter.cl",GPUContext,transfer,"ckConv")
	{
		mask = maskArg;
	}

	
};

