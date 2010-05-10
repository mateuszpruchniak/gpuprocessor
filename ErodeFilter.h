#pragma once
#include "morphologyfilter.h"
class ErodeFilter :
	public MorphologyFilter
{
public:
	ErodeFilter(void);

	~ErodeFilter(void);

	ErodeFilter(cl_context GPUContext ,GPUTransferManager* transfer): MorphologyFilter("./OpenCl/ErodeFilter.cl",GPUContext,transfer,"ckErode")
	{}

	void process(cl_command_queue GPUCommandQueue);
};

