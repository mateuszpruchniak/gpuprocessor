#pragma once
#include "highpassfilter.h"
class PrewittFilter :
	public HighpassFilter
{
public:
	PrewittFilter(void);

	~PrewittFilter(void);

	PrewittFilter(cl_context GPUContext ,GPUTransferManager* transfer);

};

