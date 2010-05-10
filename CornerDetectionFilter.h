#pragma once
#include "highpassfilter.h"
class CornerDetectionFilter :
	public HighpassFilter
{
public:
	CornerDetectionFilter(void);

	~CornerDetectionFilter(void);

	CornerDetectionFilter(cl_context GPUContext ,GPUTransferManager* transfer);

};

