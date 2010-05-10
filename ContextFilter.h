#pragma once
#include "filter.h"
class ContextFilter :
	public Filter
{
public:
	ContextFilter(void)
	{}

	~ContextFilter(void);

	ContextFilter(char* source, cl_context GPUContext ,GPUTransferManager* transfer,char* KernelName): Filter(source,GPUContext,transfer,KernelName)
    {
    }
};

