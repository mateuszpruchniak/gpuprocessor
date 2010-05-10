#pragma once
#include "nonlinearfilter.h"
class MedianFilter :
	public NonLinearFilter
{
public:

	MedianFilter(void);

	~MedianFilter(void);

	MedianFilter(cl_context GPUContext ,GPUTransferManager* transfer): NonLinearFilter("./OpenCL/MedianFilter.cl",GPUContext,transfer,"ckMedian")
	{}

	void process(cl_command_queue GPUCommandQueue);
};

