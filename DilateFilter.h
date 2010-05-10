#pragma once
#include "morphologyfilter.h"
class DilateFilter :
	public MorphologyFilter
{
public:
	DilateFilter(void);

	~DilateFilter(void);

	DilateFilter(cl_context GPUContext ,GPUTransferManager* transfer): MorphologyFilter("./OpenCL/DilateFilter.cl",GPUContext,transfer,"ckDilate")
	{}

	void process(cl_command_queue GPUCommandQueue);
};

