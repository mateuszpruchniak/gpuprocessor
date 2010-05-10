#pragma once
#include "contextfilter.h"
class MorphologyFilter :
	public ContextFilter
{
public:
	MorphologyFilter(void)
	{}

	~MorphologyFilter(void);

	MorphologyFilter(char* source, cl_context GPUContext ,GPUTransferManager* transfer,char* KernelName): ContextFilter(source,GPUContext,transfer,KernelName)
    {
    }
};

