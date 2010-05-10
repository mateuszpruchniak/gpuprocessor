#pragma once
#include "highpassfilter.h"
class LaplaceFilter :
	public HighpassFilter
{
public:
	LaplaceFilter(void);

	~LaplaceFilter(void);

	LaplaceFilter(cl_context GPUContext ,GPUTransferManager* transfer);

};

