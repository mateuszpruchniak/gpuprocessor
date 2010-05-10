#pragma once
#include "highpassfilter.h"
class RobertsFilter :
	public HighpassFilter
{
public:
	RobertsFilter(void);

	~RobertsFilter(void);

	RobertsFilter(cl_context GPUContext ,GPUTransferManager* transfer);

};

