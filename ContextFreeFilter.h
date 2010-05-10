#pragma once
#include "filter.h"
class ContextFreeFilter :
	public Filter
{
public:
	ContextFreeFilter(void);

	~ContextFreeFilter(void);

	ContextFreeFilter(char* source, cl_context GPUContext ,GPUTransferManager* transfer,char* KernelName): Filter(source,GPUContext,transfer,KernelName)
    {
    }
};

