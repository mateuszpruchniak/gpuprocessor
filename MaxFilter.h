#pragma once
#include "nonlinearfilter.h"
class MaxFilter :
	public NonLinearFilter
{
public:
	MaxFilter(void);

	~MaxFilter(void);

	MaxFilter(cl_context GPUContext ,GPUTransferManager* transfer): NonLinearFilter("./OpenCL/MaxFilter.cl",GPUContext,transfer,"ckMax")
	{}

	void process(cl_command_queue GPUCommandQueue);
};

