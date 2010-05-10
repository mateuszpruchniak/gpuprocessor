#pragma once
#include "morphologyfilter.h"
#include "DilateFilter.h"
#include "ErodeFilter.h"

class CloseFilter :
	public MorphologyFilter
{
private:
	DilateFilter* dilate;
	ErodeFilter* erode;

public:
	CloseFilter(void);

	~CloseFilter(void);

	CloseFilter(cl_context GPUContext ,GPUTransferManager* transfer);

	void process(cl_command_queue GPUCommandQueue);
};


