#pragma once
#include "highpassfilter.h"
class SobelFilter :
	public HighpassFilter
{
public:
	SobelFilter(void);

	~SobelFilter(void);

	SobelFilter(cl_context GPUContext ,GPUTransferManager* transfer);
	
};

