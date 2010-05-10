#pragma once
#include "contextfilter.h"
class NonLinearFilter :
	public ContextFilter
{
public:
	NonLinearFilter(void);

	~NonLinearFilter(void);

	NonLinearFilter(char* source, cl_context GPUContext ,GPUTransferManager* transfer,char* KernelName): ContextFilter(source,GPUContext,transfer,KernelName)
    {
    }
};

