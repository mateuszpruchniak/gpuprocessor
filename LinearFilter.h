#pragma once
#include "contextfilter.h"
class LinearFilter :
	public ContextFilter
{
public:
	LinearFilter(void);

	~LinearFilter(void);

	LinearFilter(char* source, cl_context GPUContext ,GPUTransferManager* transfer,char* KernelName): ContextFilter(source,GPUContext,transfer,KernelName)
	{}
};

