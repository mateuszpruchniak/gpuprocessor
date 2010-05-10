#pragma once
#include "morphologyfilter.h"
#include "DilateFilter.h"
#include "ErodeFilter.h"

class OpenFilter :
	public MorphologyFilter
{
private:
	DilateFilter* dilate;
	ErodeFilter* erode;

public:
	OpenFilter(void);

	~OpenFilter(void);

	OpenFilter(cl_context GPUContext ,GPUTransferManager* transfer);

	void process(cl_command_queue GPUCommandQueue);
};
