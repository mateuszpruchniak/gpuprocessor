#pragma once
#include "nonlinearfilter.h"
class MinFilter :
	public NonLinearFilter
{
public:
	MinFilter(void);

	~MinFilter(void);

	MinFilter(cl_context GPUContext ,GPUTransferManager* transfer): NonLinearFilter("./OpenCL/MinFilter.cl",GPUContext,transfer,"ckMin")
	{}

	void process(cl_command_queue GPUCommandQueue);
};

